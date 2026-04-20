import csv
import math
import os
import re
import sys
import threading
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_PYTHONPATH = os.environ.get("PYTHONPATH", "")
_PYTHONPATH_ENTRIES = [entry for entry in _PYTHONPATH.split(os.pathsep) if entry]
if REPO_ROOT not in _PYTHONPATH_ENTRIES:
    os.environ["PYTHONPATH"] = REPO_ROOT if not _PYTHONPATH else REPO_ROOT + os.pathsep + _PYTHONPATH

try:
    import sitecustomize as _project_sitecustomize  # noqa: F401
except Exception:
    _project_sitecustomize = None

os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")
os.environ.setdefault("EVAL_TEAM0_MODULE", "cs8803drl.deployment.trained_team_ray_agent")

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from cs8803drl.branches.role_specialization import build_field_role_reward_shaping_config
from cs8803drl.branches.team_siamese import (
    TEAM_SIAMESE_CROSS_ATTENTION_MODEL_NAME,
    TEAM_SIAMESE_MODEL_NAME,
    TEAM_SIAMESE_TRANSFORMER_MODEL_NAME,
    TEAM_SIAMESE_TRANSFORMER_MHA_MODEL_NAME,
    TEAM_SIAMESE_TRANSFORMER_MIN_MODEL_NAME,
    TEAM_SIAMESE_CROSS_AGENT_ATTN_MODEL_NAME,
    register_team_siamese_cross_attention_model,
    register_team_siamese_model,
    register_team_siamese_transformer_model,
    register_team_siamese_transformer_mha_model,
    register_team_siamese_transformer_min_model,
    register_team_siamese_cross_agent_attn_model,
)
from cs8803drl.branches.team_siamese_distill import (
    TEAM_SIAMESE_DISTILL_MODEL_NAME,
    TEAM_SIAMESE_ENSEMBLE_DISTILL_MODEL_NAME,
    register_team_siamese_distill_model,
    register_team_siamese_ensemble_distill_model,
)
from cs8803drl.branches.team_action_aux import (
    TEAM_ACTION_AUX_MODEL_NAME,
    TEAM_ACTION_AUX_SYMMETRIC_MODEL_NAME,
    register_team_action_aux_model,
)
from cs8803drl.branches.imitation_bc import warmstart_team_level_policy_from_bc
from cs8803drl.core.checkpoint_utils import load_policy_weights, sanitize_checkpoint_for_restore
from cs8803drl.core.utils import create_rllib_env
from cs8803drl.training.train_ray_team_vs_random_shaping import (
    _build_reward_shaping_config,
    _console_print,
    _resolve_resume_checkpoint_env,
    _resolve_resume_timesteps_delta,
    _warmstart_on_resume_enabled,
    _env_bool,
    _env_float,
    _env_int,
    _env_layers,
    _find_progress_csv,
    _last_csv_row,
    _monitor_checkpoint_evaluations,
    _monitor_progress,
    _print_progress,
    _row_float,
    _row_int,
    summarize_training_progress,
    _write_training_loss_curve,
)


DEFAULT_TIMESTEPS_TOTAL = 20_000_000
DEFAULT_TIME_TOTAL_S = 0
DEFAULT_RUN_NAME = "PPO_team_vs_baseline_shaping"
DEFAULT_CHECKPOINT_FREQ = 10
DEFAULT_FRAMEWORK = "torch"
DEFAULT_NUM_GPUS = 1
DEFAULT_NUM_WORKERS = 8
DEFAULT_NUM_ENVS_PER_WORKER = 5
DEFAULT_ROLLOUT_FRAGMENT_LENGTH = 1_000
DEFAULT_NUM_SGD_ITER = 10
DEFAULT_BASE_PORT = 6_505
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_FCNET_HIDDENS = (512, 512)
DEFAULT_BASELINE_PROB = 1.0


class CurriculumUpdateCallback(DefaultCallbacks):
    """SNAPSHOT-058 (Tier A4) opponent-strength curriculum.

    Reads CURRICULUM_PHASES at trainer init, computes new opponent_mix weights
    after each train iter, syncs to all workers via foreach_worker(foreach_env).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from cs8803drl.branches.curriculum import (
            CurriculumPhaseScheduler,
            parse_curriculum_phases,
        )
        spec = os.environ.get("CURRICULUM_PHASES", "").strip()
        phases = parse_curriculum_phases(spec)
        # Tolerate missing env at instantiation time (e.g., during ckpt-load at
        # eval/deploy time). Only the on_train_result path actually uses the
        # scheduler; at eval time no training happens.
        if phases:
            self._scheduler = CurriculumPhaseScheduler(phases)
        else:
            self._scheduler = None
        self._last_baseline_prob = -1.0  # force first push

    def on_train_result(self, *, trainer, result, **kwargs):
        if self._scheduler is None:
            return  # eval/deploy context: no curriculum to apply
        cur_iter = int(result.get("training_iteration", 0))
        new_baseline_prob = self._scheduler.baseline_prob_for_iter(cur_iter)
        new_weights = self._scheduler.compute_weights_dict(cur_iter)
        # Only push if changed (avoid spam when within same phase)
        if abs(new_baseline_prob - self._last_baseline_prob) < 1e-6:
            # Still report current phase to result for logging
            result.setdefault("custom_metrics", {})["curriculum_baseline_prob"] = float(
                new_baseline_prob
            )
            return
        self._last_baseline_prob = new_baseline_prob

        from cs8803drl.branches.curriculum import update_env_curriculum_weights

        n_updated = 0
        n_total = 0

        def _apply(env):
            nonlocal n_updated, n_total
            n_total += 1
            if update_env_curriculum_weights(env, new_weights):
                n_updated += 1

        try:
            trainer.workers.foreach_worker(
                lambda w: w.foreach_env(_apply)
            )
        except Exception as exc:  # pragma: no cover
            print(f"[curriculum-update] foreach_env failed: {exc!r}")

        msg = (
            f"[curriculum-update] iter={cur_iter} baseline_prob={new_baseline_prob:.3f} "
            f"updated={n_updated}/{n_total}"
        )
        print(msg)
        result.setdefault("custom_metrics", {})["curriculum_baseline_prob"] = float(
            new_baseline_prob
        )


class TeamModelMetricsCallback(DefaultCallbacks):
    """Surface model-side aux metrics into learner_stats/progress.csv."""

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        try:
            policy = trainer.get_policy("default_policy") or trainer.get_policy("default")
        except Exception:
            policy = None
        if policy is None:
            return

        model = getattr(policy, "model", None)
        if model is None or not hasattr(model, "metrics"):
            return

        try:
            metrics = model.metrics()
        except Exception:
            return
        if not isinstance(metrics, dict) or not metrics:
            return

        learner_stats = (
            result.setdefault("info", {})
            .setdefault("learner", {})
            .setdefault("default_policy", {})
            .setdefault("learner_stats", {})
        )
        custom_metrics = result.setdefault("custom_metrics", {})
        for key, value in metrics.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            learner_stats[key] = scalar
            custom_metrics[key] = scalar
            result[f"aux_metrics/{key}"] = scalar


def _append_warmstart_summary(message, trainer=None):
    summary_path = os.environ.get("WARMSTART_SUMMARY_PATH", "").strip()
    if not summary_path and trainer is not None:
        logdir = getattr(trainer, "logdir", "") or ""
        if logdir:
            summary_path = str(Path(logdir).resolve().parent / "warmstart_summary.txt")
    if not summary_path:
        return
    try:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except Exception as exc:  # pragma: no cover - best-effort logging
        _console_print(f"[warmstart-summary] failed to write summary: {exc}")


def _validate_warmstart_args():
    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip()
    bc_warmstart_path = os.environ.get("BC_WARMSTART_CHECKPOINT", "").strip()
    if warmstart_path and bc_warmstart_path:
        raise ValueError(
            "BC_WARMSTART_CHECKPOINT and WARMSTART_CHECKPOINT are mutually exclusive. "
            "Set only one warm-start source."
        )


def _after_init_warmstart(trainer):
    resume_path = _resolve_resume_checkpoint_env()
    if resume_path and not _warmstart_on_resume_enabled(False):
        _append_warmstart_summary(
            f"status: skipped (resume_checkpoint set, WARMSTART_ON_RESUME=0)\n"
            f"resume_checkpoint: {resume_path}",
            trainer=trainer,
        )
        return

    bc_warmstart_path = os.environ.get("BC_WARMSTART_CHECKPOINT", "").strip()
    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip()
    if warmstart_path:
        load_policy_weights(warmstart_path, trainer, policy_name="default_policy")
        try:
            trainer.workers.sync_weights()
        except Exception:
            pass
        _console_print(f"[warmstart] loaded default_policy from checkpoint: {warmstart_path}")
        _append_warmstart_summary(
            "status: warmstart_applied\n"
            f"source_checkpoint: {warmstart_path}\n"
            "mode: load_policy_weights",
            trainer=trainer,
        )
        return

    if not bc_warmstart_path:
        _append_warmstart_summary("status: no_warmstart", trainer=trainer)
        return

    stats = warmstart_team_level_policy_from_bc(
        trainer,
        bc_warmstart_path,
        policy_name="default_policy",
    )
    _console_print(
        "[bc-warmstart] copied team-level BC weights into team-level PPO policy "
        f"(copied={stats['copied']}, adapted={stats['adapted']}, skipped={stats['skipped']})"
    )
    _append_warmstart_summary(
        "status: bc_warmstart_applied\n"
        f"source_checkpoint: {bc_warmstart_path}\n"
        f"copied: {stats['copied']}\n"
        f"adapted: {stats['adapted']}\n"
        f"skipped: {stats['skipped']}",
        trainer=trainer,
    )


TeamVsBaselineShapingPPOTrainer = PPOTrainer.with_updates(
    name="TeamVsBaselineShapingPPOTrainer",
    after_init=_after_init_warmstart,
)


def _checkpoint_iteration_from_path(restore_path):
    checkpoint_name = Path(restore_path).name
    match = re.match(r"^checkpoint-(\d+)", checkpoint_name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _restore_base_timesteps(restore_path):
    checkpoint_iteration = _checkpoint_iteration_from_path(restore_path)
    if checkpoint_iteration is None:
        return None

    trial_dir = Path(restore_path).resolve().parent.parent
    progress_csv = trial_dir / "progress.csv"
    if not progress_csv.exists():
        return None

    try:
        with progress_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    if int(float(row.get("training_iteration", 0))) != checkpoint_iteration:
                        continue
                    return int(float(row.get("timesteps_total", 0)))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return None
    return None


def _resolve_target_timesteps(*, restore_path, requested_total, requested_delta):
    if not restore_path or requested_delta <= 0:
        return requested_total, None

    restore_base_timesteps = _restore_base_timesteps(restore_path)
    if restore_base_timesteps is None:
        raise ValueError(
            "RESTORE_TIMESTEPS_DELTA is set, but the restored checkpoint's base timesteps "
            "could not be inferred from progress.csv. Set TIMESTEPS_TOTAL explicitly instead."
        )
    return restore_base_timesteps + requested_delta, restore_base_timesteps


def _validate_base_port(*, base_port, num_workers, num_envs_per_worker):
    # create_rllib_env derives worker_id from worker_index * num_envs_per_worker (+ vector slot),
    # so the highest bound socket will be somewhat above BASE_PORT. Keep generous headroom.
    worst_case_worker_offset = max(1, num_workers) * max(1, num_envs_per_worker) + max(1, num_envs_per_worker)
    max_required_port = base_port + worst_case_worker_offset
    if max_required_port > 65535:
        raise ValueError(
            f"BASE_PORT={base_port} is too high for {num_workers} workers x {num_envs_per_worker} envs. "
            f"Estimated max required port is {max_required_port} (> 65535). "
            "Lower BASE_PORT or reduce workers/envs."
        )


def _parse_opponent_pool_specs(raw: str):
    specs = []
    for idx, piece in enumerate((raw or "").split(";")):
        piece = piece.strip()
        if not piece:
            continue
        fields = [field.strip() for field in piece.split("|", 3)]
        if len(fields) == 4:
            name, kind, weight_raw, checkpoint_path = fields
        elif len(fields) == 3:
            kind, weight_raw, checkpoint_path = fields
            name = f"frozen_{idx}"
        else:
            raise ValueError(
                "OPPONENT_POOL_FRONTIER_SPECS entries must look like "
                "'kind|weight|checkpoint_path' or 'name|kind|weight|checkpoint_path'. "
                f"Got: {piece!r}"
            )
        checkpoint_path = checkpoint_path.strip()
        if not checkpoint_path:
            raise ValueError(f"Missing checkpoint path in opponent pool spec: {piece!r}")
        specs.append(
            {
                "name": name or f"frozen_{idx}",
                "kind": kind,
                "weight": float(weight_raw),
                "checkpoint_path": checkpoint_path,
            }
        )
    return specs


def _print_header(
    *,
    run_dir,
    resume_path,
    bc_warmstart_path,
    warmstart_path,
    timesteps_total,
    time_total_s,
    max_iterations,
    train_batch_size,
    rollout_fragment_length,
    sgd_minibatch_size,
    num_sgd_iter,
    num_workers,
    num_envs_per_worker,
    num_gpus,
    fcnet_hiddens,
    base_port,
    baseline_prob,
    opponent_pool_frontier_specs,
    opponent_pool_baseline_prob,
    reward_shaping_enabled,
    reward_shaping_config,
    learned_reward_config,
    field_role_binding_shaping,
    team_siamese_encoder,
    team_siamese_encoder_hiddens,
    team_siamese_merge_hiddens,
    team_cross_attention,
    team_cross_attention_tokens,
    team_cross_attention_dim,
    aux_team_action_head,
    aux_team_action_symmetric,
    aux_team_action_weight,
    aux_team_action_hidden,
    team_distill_kl,
    team_distill_teacher_checkpoint,
    team_distill_teacher_policy_id,
    team_distill_alpha_init,
    team_distill_alpha_final,
    team_distill_decay_updates,
    team_distill_temperature,
    team_distill_ensemble_kl,
    team_distill_teacher_ensemble_paths,
    resume_base_timesteps,
    resume_timesteps_delta,
):
    est_total_iterations = None
    if timesteps_total > 0 and train_batch_size > 0:
        est_total_iterations = int(math.ceil(float(timesteps_total) / float(train_batch_size)))

    _console_print("")
    _console_print("Training Configuration")
    _console_print(f"  run_dir:              {run_dir}")
    _console_print(f"  resume_checkpoint:    {resume_path or 'None'}")
    _console_print(f"  bc_warmstart_ckpt:    {bc_warmstart_path or 'None'}")
    _console_print(f"  warmstart_checkpoint: {warmstart_path or 'None'}")
    _console_print(f"  target_timesteps:     {timesteps_total:,}")
    if resume_base_timesteps is not None and resume_timesteps_delta > 0:
        _console_print(f"  resume_base_steps:    {resume_base_timesteps:,}")
        _console_print(f"  resume_step_delta:    +{resume_timesteps_delta:,}")
    _console_print(
        f"  time_limit:           {time_total_s if time_total_s > 0 else 'disabled'}"
        + ("s" if time_total_s > 0 else "")
    )
    _console_print(f"  max_iterations:       {max_iterations if max_iterations > 0 else 'disabled'}")
    _console_print(f"  workers/envs:         {num_workers} workers x {num_envs_per_worker} env")
    _console_print(f"  learner_gpus:         {num_gpus}")
    _console_print(f"  rollout_fragment:     {rollout_fragment_length}")
    _console_print(f"  train_batch_size:     {train_batch_size} env steps / iteration")
    _console_print(f"  sgd_minibatch_size:   {sgd_minibatch_size}")
    _console_print(f"  ppo_epochs_per_it:    {num_sgd_iter}")
    _console_print(f"  model_hidden:         {fcnet_hiddens}")
    _console_print(f"  env_variation:        {EnvType.team_vs_policy}")
    _console_print("  starter_alignment:    team_vs_policy joint-action team training")
    _console_print(f"  reward_shaping:       {'enabled' if reward_shaping_enabled else 'disabled'}")
    _console_print(
        f"  field_role_binding:   {'enabled' if field_role_binding_shaping else 'disabled'}"
    )
    _console_print(
        f"  siamese_encoder:      {'enabled' if team_siamese_encoder else 'disabled'}"
    )
    if team_siamese_encoder:
        _console_print(f"  siamese_enc_hidden:   {team_siamese_encoder_hiddens}")
        _console_print(f"  siamese_merge_hidden: {team_siamese_merge_hiddens}")
        _console_print(
            f"  cross_attention:      {'enabled' if team_cross_attention else 'disabled'}"
        )
        if team_cross_attention:
            _console_print(
                f"  cross_attn_tokens:    {team_cross_attention_tokens}"
            )
            _console_print(
                f"  cross_attn_head_dim:  {team_cross_attention_dim}"
            )
        _console_print(
            f"  teacher_kl_distill:   {'enabled' if team_distill_kl else 'disabled'}"
        )
        if team_distill_kl:
            _console_print(f"  distill_teacher_ckpt: {team_distill_teacher_checkpoint}")
            _console_print(f"  distill_teacher_pid:  {team_distill_teacher_policy_id}")
            _console_print(f"  distill_alpha_init:   {team_distill_alpha_init}")
            _console_print(f"  distill_alpha_final:  {team_distill_alpha_final}")
            _console_print(f"  distill_decay_updates:{team_distill_decay_updates}")
            _console_print(f"  distill_temperature:  {team_distill_temperature}")
        _console_print(
            f"  ensemble_distill_kl:  {'enabled' if team_distill_ensemble_kl else 'disabled'}"
        )
        if team_distill_ensemble_kl:
            n_t = len((team_distill_teacher_ensemble_paths or "").split(","))
            _console_print(f"  ensemble_n_teachers:  {n_t}")
            _console_print(f"  distill_alpha_init:   {team_distill_alpha_init}")
            _console_print(f"  distill_alpha_final:  {team_distill_alpha_final}")
            _console_print(f"  distill_decay_updates:{team_distill_decay_updates}")
            _console_print(f"  distill_temperature:  {team_distill_temperature}")
    _console_print(
        f"  aux_action_head:      {'enabled' if aux_team_action_head else 'disabled'}"
    )
    if aux_team_action_head:
        _console_print(
            f"  aux_action_mode:      {'symmetric' if aux_team_action_symmetric else 'one_sided'}"
        )
        _console_print(f"  aux_action_weight:    {aux_team_action_weight}")
        _console_print(f"  aux_action_hidden:    {aux_team_action_hidden}")
    if isinstance(reward_shaping_config, dict):
        _console_print(f"  shaping_time_penalty: {reward_shaping_config.get('time_penalty')}")
        _console_print(f"  shaping_ball_prog:    {reward_shaping_config.get('ball_progress_scale')}")
        _console_print(f"  shaping_goal_prox:    {reward_shaping_config.get('goal_proximity_scale')}")
        if reward_shaping_config.get("goal_proximity_scale", 0.0):
            _console_print(
                f"  shaping_goal_center:  ({reward_shaping_config.get('goal_center_x')}, "
                f"{reward_shaping_config.get('goal_center_y')}) gamma={reward_shaping_config.get('goal_proximity_gamma')}"
            )
        _console_print(f"  shaping_opp_prog:     {reward_shaping_config.get('opponent_progress_penalty_scale')}")
        _console_print(f"  shaping_pos_bonus:    {reward_shaping_config.get('possession_bonus')}")
        _console_print(f"  shaping_pos_dist:     {reward_shaping_config.get('possession_dist')}")
        _console_print(f"  shaping_prog_gate:    {reward_shaping_config.get('progress_requires_possession')}")
        _console_print(f"  shaping_dz_outer:     x<{reward_shaping_config.get('deep_zone_outer_threshold')} => -{reward_shaping_config.get('deep_zone_outer_penalty')}")
        _console_print(f"  shaping_dz_inner:     x<{reward_shaping_config.get('deep_zone_inner_threshold')} => -{reward_shaping_config.get('deep_zone_inner_penalty')}")
        _console_print(f"  shaping_def_surv:     opp-possession & x<{reward_shaping_config.get('defensive_survival_threshold')} => +{reward_shaping_config.get('defensive_survival_bonus')}")
        _console_print(f"  shaping_fast_loss:    lose before {reward_shaping_config.get('fast_loss_threshold_steps')} => -{reward_shaping_config.get('fast_loss_penalty_per_step')}/step shortfall")
        if reward_shaping_config.get("team_spacing_scale", 0.0) or reward_shaping_config.get(
            "team_coverage_scale", 0.0
        ):
            _console_print(
                "  shaping_team_coord:  "
                f"spacing={reward_shaping_config.get('team_spacing_scale')} "
                f"coverage={reward_shaping_config.get('team_coverage_scale')} "
                f"gamma={reward_shaping_config.get('team_potential_gamma')}"
            )
            _console_print(
                "  shaping_team_gate:   "
                f"near_ball<{reward_shaping_config.get('team_near_ball_threshold')} "
                f"spacing_range=[{reward_shaping_config.get('team_spacing_min')}, "
                f"{reward_shaping_config.get('team_spacing_max')}]"
            )
    _console_print(
        f"  learned_reward:       {'enabled' if isinstance(learned_reward_config, dict) else 'disabled'}"
    )
    if isinstance(learned_reward_config, dict):
        _console_print(f"  learned_model_path:   {learned_reward_config.get('model_path')}")
        _console_print(f"  learned_weight:       {learned_reward_config.get('weight')}")
        _console_print(f"  learned_team0_ids:    {learned_reward_config.get('team0_agent_ids')}")
        _console_print(f"  learned_apply_team1:  {learned_reward_config.get('apply_to_team1')}")
        _console_print(f"  learned_warmup:       {learned_reward_config.get('warmup_steps')}")
    _console_print(f"  training_mode:        {'resume continuation' if resume_path else 'scratch'}")
    if opponent_pool_frontier_specs:
        _console_print("  opponent_pool:        enabled")
        _console_print(f"  pool baseline p:      {opponent_pool_baseline_prob:.2f}")
        for spec in opponent_pool_frontier_specs:
            _console_print(
                "  pool frontier:        "
                f"{spec['name']} kind={spec['kind']} weight={float(spec['weight']):.2f} "
                f"ckpt={spec['checkpoint_path']}"
            )
    else:
        _console_print(f"  opponent baseline p:  {baseline_prob:.2f}")
    _console_print(f"  base_port:            {base_port}")
    if est_total_iterations is not None:
        _console_print(f"  estimated_iterations: {est_total_iterations}")
    _console_print("")


def main():
    _validate_warmstart_args()

    framework = os.environ.get("FRAMEWORK", DEFAULT_FRAMEWORK).strip() or DEFAULT_FRAMEWORK
    num_gpus = _env_int("NUM_GPUS", DEFAULT_NUM_GPUS)
    if framework == "torch" and num_gpus > 1:
        num_gpus = 1

    num_workers = _env_int("NUM_WORKERS", DEFAULT_NUM_WORKERS)
    num_envs_per_worker = _env_int("NUM_ENVS_PER_WORKER", DEFAULT_NUM_ENVS_PER_WORKER)
    rollout_fragment_length = _env_int("ROLLOUT_FRAGMENT_LENGTH", DEFAULT_ROLLOUT_FRAGMENT_LENGTH)
    train_batch_size_default = num_workers * num_envs_per_worker * rollout_fragment_length
    train_batch_size = _env_int("TRAIN_BATCH_SIZE", train_batch_size_default)
    sgd_minibatch_size_default = min(1024, max(512, train_batch_size // 8))
    sgd_minibatch_size = _env_int("SGD_MINIBATCH_SIZE", sgd_minibatch_size_default)
    num_sgd_iter = _env_int("NUM_SGD_ITER", DEFAULT_NUM_SGD_ITER)
    lr = _env_float("LR", 3e-4)
    gamma = _env_float("GAMMA", 0.99)
    gae_lambda = _env_float("GAE_LAMBDA", 0.95)
    clip_param = _env_float("CLIP_PARAM", 0.2)
    entropy_coeff = _env_float("ENTROPY_COEFF", 0.0)
    vf_share_layers = _env_bool("VF_SHARE_LAYERS", True)
    log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper() or DEFAULT_LOG_LEVEL
    log_sys_usage = _env_bool("LOG_SYS_USAGE", True)
    fcnet_hiddens = _env_layers("FCNET_HIDDENS", DEFAULT_FCNET_HIDDENS)
    fcnet_activation = os.environ.get("FCNET_ACTIVATION", "relu").strip() or "relu"
    baseline_prob = _env_float("BASELINE_PROB", DEFAULT_BASELINE_PROB)
    opponent_pool_frontier_specs = _parse_opponent_pool_specs(
        os.environ.get("OPPONENT_POOL_FRONTIER_SPECS", "").strip()
    )
    opponent_pool_baseline_prob_raw = os.environ.get("OPPONENT_POOL_BASELINE_PROB", "").strip()
    opponent_pool_baseline_prob = (
        float(opponent_pool_baseline_prob_raw)
        if opponent_pool_baseline_prob_raw
        else baseline_prob
    )
    use_reward_shaping = _env_bool("USE_REWARD_SHAPING", True)
    reward_shaping = _build_reward_shaping_config() if use_reward_shaping else False
    field_role_binding_shaping = _env_bool("SHAPING_FIELD_ROLE_BINDING", False)
    if field_role_binding_shaping:
        if not use_reward_shaping:
            raise ValueError("SHAPING_FIELD_ROLE_BINDING requires USE_REWARD_SHAPING=1.")
        reward_shaping = build_field_role_reward_shaping_config()
    learned_reward_model_path = os.environ.get("LEARNED_REWARD_MODEL_PATH", "").strip()
    learned_reward_config = None
    if learned_reward_model_path:
        learned_reward_config = {
            "model_path": learned_reward_model_path,
            "weight": _env_float("LEARNED_REWARD_SHAPING_WEIGHT", 0.01),
            "team0_agent_ids": (0, 1),
            "apply_to_team1": _env_bool("LEARNED_REWARD_APPLY_TO_TEAM1", False),
            "warmup_steps": _env_int("LEARNED_REWARD_WARMUP_STEPS", 0),
        }
    # A2 PBRS: outcome predictor (direction_1b_v3) → ΔV reward bonus
    outcome_pbrs_predictor_path = os.environ.get("OUTCOME_PBRS_PREDICTOR_PATH", "").strip()
    outcome_pbrs_config = None
    if outcome_pbrs_predictor_path:
        outcome_pbrs_config = {
            "predictor_path": outcome_pbrs_predictor_path,
            "weight": _env_float("OUTCOME_PBRS_WEIGHT", 0.01),
            "team0_agent_ids": (0, 1),
            "warmup_steps": _env_int("OUTCOME_PBRS_WARMUP_STEPS", 0),
            "max_buffer_steps": _env_int("OUTCOME_PBRS_MAX_BUFFER_STEPS", 80),
        }
    # SNAPSHOT-058 (Tier A4) opponent-strength curriculum
    curriculum_enabled = _env_bool("CURRICULUM_ENABLED", False)
    curriculum_phases_spec = (os.environ.get("CURRICULUM_PHASES", "").strip())
    curriculum_phases = None
    if curriculum_enabled:
        from cs8803drl.branches.curriculum import parse_curriculum_phases
        curriculum_phases = parse_curriculum_phases(curriculum_phases_spec)
        if not curriculum_phases:
            raise ValueError(
                "CURRICULUM_ENABLED=1 requires CURRICULUM_PHASES="
                "'iter:prob,iter:prob,...' (e.g., '0:0.0,200:0.3,500:0.7,1000:1.0')."
            )
    # SNAPSHOT-057 (Tier A3) RND intrinsic motivation
    rnd_enabled = _env_bool("RND_ENABLED", False)
    rnd_config = None
    if rnd_enabled:
        rnd_config = {
            "weight": _env_float("RND_INTRINSIC_BETA", 0.01),
            "team0_agent_ids": (0, 1),
            "hidden_dim": _env_int("RND_HIDDEN_DIM", 256),
            "embed_dim": _env_int("RND_EMBED_DIM", 64),
            "lr": _env_float("RND_LR", 1e-4),
            "train_every_steps": _env_int("RND_TRAIN_EVERY_STEPS", 16),
            "train_batch_size": _env_int("RND_TRAIN_BATCH_SIZE", 256),
            "warmup_steps": _env_int("RND_WARMUP_STEPS", 0),
            "device": "cpu",
            "random_seed": _env_int("RND_RANDOM_SEED", 1234),
        }
    team_siamese_encoder = _env_bool("TEAM_SIAMESE_ENCODER", False)
    team_siamese_encoder_hiddens = _env_layers("TEAM_SIAMESE_ENCODER_HIDDENS", (256, 256))
    team_siamese_merge_hiddens = _env_layers("TEAM_SIAMESE_MERGE_HIDDENS", (256, 128))
    team_cross_attention = _env_bool("TEAM_CROSS_ATTENTION", False)
    team_cross_attention_tokens = _env_int("TEAM_CROSS_ATTENTION_TOKENS", 4)
    team_cross_attention_dim = _env_int("TEAM_CROSS_ATTENTION_DIM", 64)
    team_cross_attention_heads = _env_int("TEAM_CROSS_ATTENTION_HEADS", 4)
    team_transformer = _env_bool("TEAM_TRANSFORMER", False)
    team_transformer_min = _env_bool("TEAM_TRANSFORMER_MIN", False)
    team_cross_agent_attn = _env_bool("TEAM_CROSS_AGENT_ATTN", False)
    team_cross_agent_attn_dim = _env_int("TEAM_CROSS_AGENT_ATTN_DIM", 64)
    team_transformer_mha = _env_bool("TEAM_TRANSFORMER_MHA", False)
    team_transformer_ffn_hidden = _env_int("TEAM_TRANSFORMER_FFN_HIDDEN", 512)
    team_transformer_ffn_activation = (
        os.environ.get("TEAM_TRANSFORMER_FFN_ACTIVATION", "gelu").strip().lower() or "gelu"
    )
    team_transformer_norm = (
        os.environ.get("TEAM_TRANSFORMER_NORM", "postnorm").strip().lower() or "postnorm"
    )
    team_distill_kl = _env_bool("TEAM_DISTILL_KL", False)
    team_distill_teacher_checkpoint = (
        os.environ.get("TEAM_DISTILL_TEACHER_CHECKPOINT", "").strip() or None
    )
    team_distill_teacher_policy_id = (
        os.environ.get("TEAM_DISTILL_TEACHER_POLICY_ID", "shared_cc_policy").strip()
        or "shared_cc_policy"
    )
    team_distill_alpha_init = _env_float("TEAM_DISTILL_ALPHA_INIT", 0.02)
    team_distill_alpha_final = _env_float("TEAM_DISTILL_ALPHA_FINAL", 0.0)
    team_distill_decay_updates = _env_int("TEAM_DISTILL_DECAY_UPDATES", 16000)
    team_distill_temperature = _env_float("TEAM_DISTILL_TEMPERATURE", 1.0)
    team_distill_ensemble_kl = _env_bool("TEAM_DISTILL_ENSEMBLE_KL", False)
    team_distill_teacher_ensemble_paths = (
        os.environ.get("TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS", "").strip() or None
    )
    aux_team_action_head = _env_bool("AUX_TEAM_ACTION_HEAD", False)
    aux_team_action_symmetric = _env_bool("AUX_TEAM_ACTION_SYMMETRIC", False)
    aux_team_action_weight = _env_float("AUX_TEAM_ACTION_WEIGHT", 0.05)
    aux_team_action_hidden = _env_int("AUX_TEAM_ACTION_HIDDEN", 256)
    if team_cross_attention and not team_siamese_encoder:
        raise ValueError(
            "TEAM_CROSS_ATTENTION=1 requires TEAM_SIAMESE_ENCODER=1."
        )
    transformer_variant_count = int(team_transformer) + int(team_transformer_min) + int(team_transformer_mha)
    if transformer_variant_count > 1:
        raise ValueError(
            "TEAM_TRANSFORMER, TEAM_TRANSFORMER_MIN, and TEAM_TRANSFORMER_MHA are mutually "
            "exclusive. Choose exactly one 031C variant."
        )
    if team_transformer and not team_siamese_encoder:
        raise ValueError("TEAM_TRANSFORMER=1 requires TEAM_SIAMESE_ENCODER=1.")
    if team_transformer and not team_cross_attention:
        raise ValueError(
            "TEAM_TRANSFORMER=1 requires TEAM_CROSS_ATTENTION=1 because the full 031C "
            "architecture is defined on top of the 031B attention path."
        )
    if team_transformer_min and not team_siamese_encoder:
        raise ValueError("TEAM_TRANSFORMER_MIN=1 requires TEAM_SIAMESE_ENCODER=1.")
    if team_transformer_min and not team_cross_attention:
        raise ValueError(
            "TEAM_TRANSFORMER_MIN=1 requires TEAM_CROSS_ATTENTION=1 because 031C-min "
            "is defined as a refinement on top of the 031B attention path."
        )
    if team_transformer_mha and not team_siamese_encoder:
        raise ValueError("TEAM_TRANSFORMER_MHA=1 requires TEAM_SIAMESE_ENCODER=1.")
    if team_transformer_mha and not team_cross_attention:
        raise ValueError(
            "TEAM_TRANSFORMER_MHA=1 requires TEAM_CROSS_ATTENTION=1 because 031C-mha "
            "is defined as a refinement on top of the 031B attention path."
        )
    if team_distill_kl and not team_siamese_encoder:
        raise ValueError("TEAM_DISTILL_KL=1 requires TEAM_SIAMESE_ENCODER=1.")
    if team_distill_kl and team_transformer_min:
        raise ValueError(
            "TEAM_DISTILL_KL is not yet composable with TEAM_TRANSFORMER_MIN."
        )
    if team_distill_kl and team_transformer:
        raise ValueError(
            "TEAM_DISTILL_KL is not yet composable with TEAM_TRANSFORMER."
        )
    if team_distill_kl and team_transformer_mha:
        raise ValueError(
            "TEAM_DISTILL_KL is not yet composable with TEAM_TRANSFORMER_MHA."
        )
    if team_distill_kl and team_cross_attention:
        raise ValueError(
            "TEAM_DISTILL_KL is only implemented for the base Siamese encoder path, "
            "not the cross-attention variant."
        )
    if team_siamese_encoder and aux_team_action_head:
        raise ValueError(
            "TEAM_SIAMESE_ENCODER and AUX_TEAM_ACTION_HEAD are not yet composable. "
            "Enable only one custom team-level model path at a time."
        )
    # Ensemble distillation (snapshot-055): mutually exclusive with TEAM_DISTILL_KL
    if team_distill_ensemble_kl and team_distill_kl:
        raise ValueError(
            "TEAM_DISTILL_ENSEMBLE_KL and TEAM_DISTILL_KL are mutually exclusive."
        )
    if team_distill_ensemble_kl and not team_siamese_encoder:
        raise ValueError("TEAM_DISTILL_ENSEMBLE_KL=1 requires TEAM_SIAMESE_ENCODER=1.")
    if team_distill_ensemble_kl and not team_cross_attention:
        raise ValueError(
            "TEAM_DISTILL_ENSEMBLE_KL=1 requires TEAM_CROSS_ATTENTION=1 (031B-arch student)."
        )
    if team_distill_ensemble_kl and not team_distill_teacher_ensemble_paths:
        raise ValueError(
            "TEAM_DISTILL_ENSEMBLE_KL=1 requires "
            "TEAM_DISTILL_TEACHER_ENSEMBLE_CHECKPOINTS (comma-separated)."
        )
    if team_distill_ensemble_kl and (
        team_transformer or team_transformer_min or team_transformer_mha
        or team_cross_agent_attn
    ):
        raise ValueError(
            "TEAM_DISTILL_ENSEMBLE_KL is only composable with the 031B "
            "(siamese + cross-attention) student arch, not transformer/MAT variants."
        )
    if team_distill_kl and not team_distill_teacher_checkpoint:
        raise ValueError(
            "TEAM_DISTILL_KL=1 requires TEAM_DISTILL_TEACHER_CHECKPOINT to be set."
        )

    bc_warmstart_path = os.environ.get("BC_WARMSTART_CHECKPOINT", "").strip() or None
    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip() or None
    resume_path = _resolve_resume_checkpoint_env() or None
    if resume_path:
        resume_path = sanitize_checkpoint_for_restore(resume_path)

    requested_timesteps_total = _env_int("TIMESTEPS_TOTAL", DEFAULT_TIMESTEPS_TOTAL)
    restore_timesteps_delta = _resolve_resume_timesteps_delta(0)
    timesteps_total, restore_base_timesteps = _resolve_target_timesteps(
        restore_path=resume_path,
        requested_total=requested_timesteps_total,
        requested_delta=restore_timesteps_delta,
    )
    time_total_s = _env_int("TIME_TOTAL_S", DEFAULT_TIME_TOTAL_S)
    max_iterations = _env_int("MAX_ITERATIONS", 0)
    run_name = os.environ.get("RUN_NAME", DEFAULT_RUN_NAME).strip() or DEFAULT_RUN_NAME
    checkpoint_freq = _env_int("CHECKPOINT_FREQ", DEFAULT_CHECKPOINT_FREQ)
    base_port = _env_int("BASE_PORT", DEFAULT_BASE_PORT)
    local_dir = os.environ.get("LOCAL_DIR", "./ray_results").strip() or "./ray_results"
    _validate_base_port(
        base_port=base_port,
        num_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
    )

    ray_init_kwargs = {"include_dashboard": False}
    ray_address_override = os.environ.get("RAY_ADDRESS_OVERRIDE", "").strip()
    if ray_address_override:
        if ray_address_override.lower() == "local":
            # Ray 1.4 does not accept address="local"; treat it as our
            # sentinel for "start a fresh local runtime" and avoid inheriting
            # any ambient cluster address from the shell.
            os.environ.pop("RAY_ADDRESS", None)
        else:
            ray_init_kwargs["address"] = ray_address_override
    else:
        os.environ.pop("RAY_ADDRESS", None)
    ray_tmp_override = os.environ.get("RAY_SESSION_TMPDIR_OVERRIDE", "").strip()
    if ray_tmp_override:
        ray_init_kwargs["_temp_dir"] = ray_tmp_override

    ray.init(**ray_init_kwargs)
    tune.registry.register_env("Soccer", create_rllib_env)
    register_team_siamese_model()
    register_team_siamese_cross_attention_model()
    register_team_siamese_transformer_model()
    register_team_siamese_transformer_mha_model()
    register_team_siamese_transformer_min_model()
    register_team_siamese_cross_agent_attn_model()
    register_team_siamese_distill_model()
    register_team_siamese_ensemble_distill_model()
    register_team_action_aux_model()

    custom_model_name = None
    model_config = {
        "vf_share_layers": vf_share_layers,
        "fcnet_hiddens": fcnet_hiddens,
        "fcnet_activation": fcnet_activation,
    }
    if team_siamese_encoder:
        if team_transformer:
            custom_model_name = TEAM_SIAMESE_TRANSFORMER_MODEL_NAME
            model_config["custom_model"] = custom_model_name
            model_config["custom_model_config"] = {
                "encoder_hiddens": team_siamese_encoder_hiddens,
                "merge_hiddens": team_siamese_merge_hiddens,
                "attention_n_tokens": team_cross_attention_tokens,
                "attention_head_dim": team_cross_attention_dim,
                "attention_num_heads": team_cross_attention_heads,
                "transformer_ffn_hidden": team_transformer_ffn_hidden,
                "transformer_ffn_activation": team_transformer_ffn_activation,
                "transformer_norm": team_transformer_norm,
            }
        elif team_transformer_mha:
            custom_model_name = TEAM_SIAMESE_TRANSFORMER_MHA_MODEL_NAME
            model_config["custom_model"] = custom_model_name
            model_config["custom_model_config"] = {
                "encoder_hiddens": team_siamese_encoder_hiddens,
                "merge_hiddens": team_siamese_merge_hiddens,
                "attention_n_tokens": team_cross_attention_tokens,
                "attention_head_dim": team_cross_attention_dim,
                "attention_num_heads": team_cross_attention_heads,
                "transformer_ffn_hidden": team_transformer_ffn_hidden,
                "transformer_ffn_activation": team_transformer_ffn_activation,
                "transformer_norm": team_transformer_norm,
            }
        elif team_transformer_min:
            custom_model_name = TEAM_SIAMESE_TRANSFORMER_MIN_MODEL_NAME
            model_config["custom_model"] = custom_model_name
            model_config["custom_model_config"] = {
                "encoder_hiddens": team_siamese_encoder_hiddens,
                "merge_hiddens": team_siamese_merge_hiddens,
                "attention_n_tokens": team_cross_attention_tokens,
                "attention_head_dim": team_cross_attention_dim,
                "transformer_ffn_hidden": team_transformer_ffn_hidden,
                "transformer_ffn_activation": team_transformer_ffn_activation,
                "transformer_norm": team_transformer_norm,
            }
        elif team_distill_kl:
            custom_model_name = TEAM_SIAMESE_DISTILL_MODEL_NAME
            model_config["custom_model"] = custom_model_name
            model_config["custom_model_config"] = {
                "encoder_hiddens": team_siamese_encoder_hiddens,
                "merge_hiddens": team_siamese_merge_hiddens,
                "teacher_checkpoint": team_distill_teacher_checkpoint,
                "teacher_policy_id": team_distill_teacher_policy_id,
                "distill_alpha_init": team_distill_alpha_init,
                "distill_alpha_final": team_distill_alpha_final,
                "distill_decay_updates": team_distill_decay_updates,
                "distill_temperature": team_distill_temperature,
            }
        elif team_distill_ensemble_kl:
            custom_model_name = TEAM_SIAMESE_ENSEMBLE_DISTILL_MODEL_NAME
            model_config["custom_model"] = custom_model_name
            model_config["custom_model_config"] = {
                "encoder_hiddens": team_siamese_encoder_hiddens,
                "merge_hiddens": team_siamese_merge_hiddens,
                "attention_n_tokens": team_cross_attention_tokens,
                "attention_head_dim": team_cross_attention_dim,
                "teacher_ensemble_checkpoints": team_distill_teacher_ensemble_paths,
                "distill_alpha_init": team_distill_alpha_init,
                "distill_alpha_final": team_distill_alpha_final,
                "distill_decay_updates": team_distill_decay_updates,
                "distill_temperature": team_distill_temperature,
            }
        elif team_cross_agent_attn:
            # 054 MAT-min: 031B + cross-agent attention residual block (no FFN/LN)
            custom_model_name = TEAM_SIAMESE_CROSS_AGENT_ATTN_MODEL_NAME
            model_config["custom_model"] = custom_model_name
            model_config["custom_model_config"] = {
                "encoder_hiddens": team_siamese_encoder_hiddens,
                "merge_hiddens": team_siamese_merge_hiddens,
                "attention_n_tokens": team_cross_attention_tokens,
                "attention_head_dim": team_cross_attention_dim,
                "cross_agent_attn_dim": team_cross_agent_attn_dim,
            }
        elif team_cross_attention:
            custom_model_name = TEAM_SIAMESE_CROSS_ATTENTION_MODEL_NAME
            model_config["custom_model"] = custom_model_name
            model_config["custom_model_config"] = {
                "encoder_hiddens": team_siamese_encoder_hiddens,
                "merge_hiddens": team_siamese_merge_hiddens,
                "attention_n_tokens": team_cross_attention_tokens,
                "attention_head_dim": team_cross_attention_dim,
            }
        else:
            custom_model_name = TEAM_SIAMESE_MODEL_NAME
            model_config["custom_model"] = custom_model_name
            model_config["custom_model_config"] = {
                "encoder_hiddens": team_siamese_encoder_hiddens,
                "merge_hiddens": team_siamese_merge_hiddens,
            }
    elif aux_team_action_head:
        custom_model_name = (
            TEAM_ACTION_AUX_SYMMETRIC_MODEL_NAME
            if aux_team_action_symmetric
            else TEAM_ACTION_AUX_MODEL_NAME
        )
        model_config["custom_model"] = custom_model_name
        model_config["custom_model_config"] = {
            "aux_weight": aux_team_action_weight,
            "aux_hidden_size": aux_team_action_hidden,
        }

    # snapshot-046: optional frozen team-level checkpoint as opponent. When
    # set, fully replaces the baseline/random opponent_mix (mutually exclusive).
    team_opponent_checkpoint = os.environ.get("TEAM_OPPONENT_CHECKPOINT", "").strip() or None
    if team_opponent_checkpoint and opponent_pool_frontier_specs:
        raise ValueError(
            "TEAM_OPPONENT_CHECKPOINT and OPPONENT_POOL_FRONTIER_SPECS are mutually "
            "exclusive. Use either a single frozen opponent (046-style) or the "
            "baseline+frontier pool (043A' style)."
        )

    env_config = {
        "num_envs_per_worker": num_envs_per_worker,
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "base_port": base_port,
        "reward_shaping": reward_shaping,
    }
    if team_opponent_checkpoint:
        env_config["team_opponent_checkpoint"] = team_opponent_checkpoint
        _console_print(
            "[snapshot-046] TEAM_OPPONENT_CHECKPOINT set — overriding baseline_prob; "
            f"using frozen team-level opponent: {team_opponent_checkpoint}"
        )
    else:
        if opponent_pool_frontier_specs:
            env_config["opponent_mix"] = {
                "baseline_prob": opponent_pool_baseline_prob,
                "frozen_opponents": opponent_pool_frontier_specs,
            }
        elif curriculum_enabled:
            # Initial baseline_prob = phase 0 value (will be updated by callback)
            from cs8803drl.branches.curriculum import CurriculumPhaseScheduler
            _initial_scheduler = CurriculumPhaseScheduler(curriculum_phases)
            env_config["opponent_mix"] = {
                "baseline_prob": _initial_scheduler.baseline_prob_for_iter(0),
                "curriculum_enabled": True,
            }
        else:
            env_config["opponent_mix"] = {"baseline_prob": baseline_prob}
    if learned_reward_config is not None:
        env_config["learned_reward_shaping"] = learned_reward_config
    if outcome_pbrs_config is not None:
        env_config["outcome_pbrs_shaping"] = outcome_pbrs_config
    if rnd_config is not None:
        env_config["rnd_shaping"] = rnd_config

    config = {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": log_level,
        "log_sys_usage": log_sys_usage,
        "framework": framework,
        "env": "Soccer",
        "env_config": env_config,
        "model": model_config,
        "rollout_fragment_length": rollout_fragment_length,
        "batch_mode": os.environ.get("BATCH_MODE", "truncate_episodes"),
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "num_sgd_iter": num_sgd_iter,
        "lr": lr,
        "gamma": gamma,
        "lambda": gae_lambda,
        "clip_param": clip_param,
        "entropy_coeff": entropy_coeff,
    }
    # Combine callbacks if multiple needed (curriculum + model-metrics)
    needs_metrics = aux_team_action_head or team_distill_kl or team_distill_ensemble_kl
    if needs_metrics and curriculum_enabled:
        # Combined: both callbacks active. Multi-inherit so on_train_result fires both.
        class _CombinedCallback(CurriculumUpdateCallback, TeamModelMetricsCallback):
            def on_train_result(self, *, trainer, result, **kw):
                CurriculumUpdateCallback.on_train_result(self, trainer=trainer, result=result, **kw)
                TeamModelMetricsCallback.on_train_result(self, trainer=trainer, result=result, **kw)
        config["callbacks"] = _CombinedCallback
    elif needs_metrics:
        config["callbacks"] = TeamModelMetricsCallback
    elif curriculum_enabled:
        config["callbacks"] = CurriculumUpdateCallback

    local_dir = os.path.abspath(local_dir)
    run_dir = os.path.join(local_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    _print_header(
        run_dir=run_dir,
        resume_path=resume_path,
        bc_warmstart_path=bc_warmstart_path,
        warmstart_path=warmstart_path,
        timesteps_total=timesteps_total,
        time_total_s=time_total_s,
        max_iterations=max_iterations,
        train_batch_size=train_batch_size,
        rollout_fragment_length=rollout_fragment_length,
        sgd_minibatch_size=sgd_minibatch_size,
        num_sgd_iter=num_sgd_iter,
        num_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
        num_gpus=num_gpus,
        fcnet_hiddens=fcnet_hiddens,
        base_port=base_port,
        baseline_prob=baseline_prob,
        opponent_pool_frontier_specs=opponent_pool_frontier_specs,
        opponent_pool_baseline_prob=opponent_pool_baseline_prob,
        reward_shaping_enabled=bool(use_reward_shaping),
        reward_shaping_config=reward_shaping,
        learned_reward_config=learned_reward_config,
        field_role_binding_shaping=field_role_binding_shaping,
        team_siamese_encoder=team_siamese_encoder,
        team_siamese_encoder_hiddens=team_siamese_encoder_hiddens,
        team_siamese_merge_hiddens=team_siamese_merge_hiddens,
        team_cross_attention=team_cross_attention,
        team_cross_attention_tokens=team_cross_attention_tokens,
        team_cross_attention_dim=team_cross_attention_dim,
        aux_team_action_head=aux_team_action_head,
        aux_team_action_symmetric=aux_team_action_symmetric,
        aux_team_action_weight=aux_team_action_weight,
        aux_team_action_hidden=aux_team_action_hidden,
        team_distill_kl=team_distill_kl,
        team_distill_teacher_checkpoint=team_distill_teacher_checkpoint,
        team_distill_teacher_policy_id=team_distill_teacher_policy_id,
        team_distill_alpha_init=team_distill_alpha_init,
        team_distill_alpha_final=team_distill_alpha_final,
        team_distill_ensemble_kl=team_distill_ensemble_kl,
        team_distill_teacher_ensemble_paths=team_distill_teacher_ensemble_paths,
        team_distill_decay_updates=team_distill_decay_updates,
        team_distill_temperature=team_distill_temperature,
        resume_base_timesteps=restore_base_timesteps,
        resume_timesteps_delta=restore_timesteps_delta,
    )

    warmstart_summary_path = os.path.join(run_dir, "warmstart_summary.txt")
    os.environ["WARMSTART_SUMMARY_PATH"] = warmstart_summary_path
    with open(warmstart_summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"resume_checkpoint: {resume_path or 'None'}\n")
        handle.write(f"bc_warmstart_ckpt: {bc_warmstart_path or 'None'}\n")
        handle.write(f"warmstart_checkpoint: {warmstart_path or 'None'}\n")

    final_checkpoint = None
    best_checkpoint = None
    best_reward = None
    best_iteration = 0
    stop_reason = "unknown"
    monitor_state = {"last_printed_iteration": -1, "best_eval": None}
    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=_monitor_progress,
        args=(run_dir, monitor_stop),
        kwargs={
            "target_timesteps": timesteps_total,
            "max_iterations": max_iterations,
            "state": monitor_state,
        },
        daemon=True,
    )
    eval_thread = threading.Thread(
        target=_monitor_checkpoint_evaluations,
        args=(run_dir, monitor_stop),
        kwargs={"state": monitor_state},
        daemon=True,
    )

    try:
        monitor_thread.start()
        eval_thread.start()

        stop = {}
        if timesteps_total > 0:
            stop["timesteps_total"] = timesteps_total
        if time_total_s > 0:
            stop["time_total_s"] = time_total_s
        if max_iterations > 0:
            stop["training_iteration"] = max_iterations

        analysis = tune.run(
            TeamVsBaselineShapingPPOTrainer,
            name=run_name,
            config=config,
            stop=stop,
            checkpoint_freq=checkpoint_freq,
            checkpoint_at_end=True,
            local_dir=local_dir,
            restore=resume_path,
            verbose=0,
            raise_on_failed_trial=False,
        )

        progress_summary = summarize_training_progress(run_dir)
        final_row = progress_summary.get("final_row") if progress_summary else None
        if final_row is not None:
            final_iteration = _row_int(final_row, "training_iteration", 0)
            if final_iteration > int(monitor_state.get("last_printed_iteration", -1)):
                _print_progress(
                    final_row,
                    target_timesteps=timesteps_total,
                    max_iterations=max_iterations,
                    previous_steps=None,
                    previous_elapsed=None,
                )

        best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
        if best_trial is not None:
            try:
                best_checkpoint = analysis.get_best_checkpoint(
                    trial=best_trial,
                    metric="episode_reward_mean",
                    mode="max",
                )
            except Exception:
                best_checkpoint = None

        trials = list(getattr(analysis, "trials", []))
        if trials:
            last_trial = trials[0]
            final_checkpoint = getattr(last_trial, "checkpoint", None) or best_checkpoint
            trial_status = getattr(last_trial, "status", None) or "UNKNOWN"
            stop_reason = f"Trial status: {trial_status}"
        else:
            stop_reason = "No trial metadata returned"

    except KeyboardInterrupt:
        stop_reason = "Interrupted by user"
        _console_print("")
        _console_print("Training interrupted by user.")
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=5.0)
        eval_thread.join(timeout=5.0)
        ray.shutdown()

    progress_summary = summarize_training_progress(run_dir)
    if progress_summary:
        best_row = progress_summary.get("best_row")
        if best_row is not None:
            best_reward = _row_float(best_row, "episode_reward_mean", best_reward if best_reward is not None else 0.0)
            best_iteration = _row_int(best_row, "training_iteration", best_iteration)

    best_eval = monitor_state.get("best_eval")
    loss_curve_file = _write_training_loss_curve(run_dir)
    _console_print("")
    _console_print("Training Summary")
    _console_print(f"  stop_reason:      {stop_reason}")
    _console_print(f"  run_dir:          {run_dir}")
    if best_reward is not None:
        _console_print(f"  best_reward_mean: {best_reward:+.4f} @ iteration {best_iteration}")
    else:
        _console_print("  best_reward_mean: unavailable")
    _console_print(f"  best_checkpoint:  {best_checkpoint or 'unavailable'}")
    _console_print(f"  final_checkpoint: {final_checkpoint or 'unavailable'}")
    if best_eval:
        baseline = best_eval.get("baseline") or {}
        random_row = best_eval.get("random") or {}
        _console_print(
            "  best_eval_checkpoint: "
            f"{best_eval.get('checkpoint_file') or best_eval.get('checkpoint_dir')}"
        )
        _console_print(
            "  best_eval_baseline:  "
            f"{float(baseline.get('win_rate', 0.0)):.3f} "
            f"({baseline.get('wins', '?')}W-{baseline.get('losses', '?')}L-{baseline.get('ties', '?')}T) "
            f"@ iteration {best_eval.get('checkpoint_iteration', '?')}"
        )
        if random_row:
            _console_print(
                "  best_eval_random:    "
                f"{float(random_row.get('win_rate', 0.0)):.3f} "
                f"({random_row.get('wins', '?')}W-{random_row.get('losses', '?')}L-{random_row.get('ties', '?')}T)"
            )
        _console_print(f"  eval_results_csv:    {os.path.join(run_dir, 'checkpoint_eval.csv')}")
    if loss_curve_file:
        _console_print(f"  loss_curve_file:    {loss_curve_file}")
    _console_print("Done training")


if __name__ == "__main__":
    main()

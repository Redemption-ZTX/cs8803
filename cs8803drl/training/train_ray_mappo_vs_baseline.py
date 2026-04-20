import os
import sys
from pathlib import Path
import threading

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
os.environ.setdefault("EVAL_TEAM0_MODULE", "cs8803drl.deployment.trained_shared_cc_agent")

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from cs8803drl.branches.role_specialization import BASELINE_POLICY_ID, FrozenBaselinePolicy
from cs8803drl.branches.role_specialization import (
    build_field_role_reward_shaping_config,
    build_role_reward_shaping_config,
)
from cs8803drl.branches.shared_central_critic import (
    FillInTeammateActions,
    SHARED_CC_MODEL_NAME,
    SHARED_CC_POLICY_ID,
    build_cc_obs_space,
    register_shared_cc_model,
    shared_cc_observer,
    shared_cc_policy_mapping_fn,
    warmstart_shared_cc_policy,
    warmstart_shared_cc_policy_from_bc_player,
)
from cs8803drl.branches.teammate_aux_head import (
    FillInTeammateActionsAndAuxLabels,
    TEAMMATE_AUX_MODEL_NAME,
    register_shared_cc_teammate_aux_model,
)
from cs8803drl.core.checkpoint_utils import sanitize_checkpoint_for_restore
from cs8803drl.core.utils import create_rllib_env
from cs8803drl.training.train_ray_team_vs_random_shaping import (
    _build_reward_shaping_config,
    _console_print,
    _resolve_resume_checkpoint_env,
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
DEFAULT_TIME_TOTAL_S = 28_000
DEFAULT_RUN_NAME = "PPO_mappo_vs_baseline"
DEFAULT_CHECKPOINT_FREQ = 10
DEFAULT_FRAMEWORK = "torch"
DEFAULT_NUM_GPUS = 1
DEFAULT_NUM_WORKERS = 8
DEFAULT_NUM_ENVS_PER_WORKER = 5
DEFAULT_ROLLOUT_FRAGMENT_LENGTH = 1_000
DEFAULT_NUM_SGD_ITER = 10
DEFAULT_BASE_PORT = 65_105
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_FCNET_HIDDENS = (512, 512)


def _append_warmstart_summary(message):
    summary_path = os.environ.get("WARMSTART_SUMMARY_PATH", "").strip()
    if not summary_path:
        return
    try:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
    except Exception as exc:  # pragma: no cover - best-effort logging
        _console_print(f"[warmstart-summary] failed to write summary: {exc}")


def _after_init_warmstart(trainer):
    resume_path = _resolve_resume_checkpoint_env()
    if resume_path and not _warmstart_on_resume_enabled(False):
        _append_warmstart_summary(
            f"status: skipped (resume_checkpoint set, WARMSTART_ON_RESUME=0)\n"
            f"resume_checkpoint: {resume_path}"
        )
        return

    bc_warmstart_path = os.environ.get("BC_WARMSTART_CHECKPOINT", "").strip()
    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip()
    if bc_warmstart_path and warmstart_path:
        raise ValueError(
            "BC_WARMSTART_CHECKPOINT and WARMSTART_CHECKPOINT are mutually exclusive. "
            "Set only one warm-start source."
        )
    if bc_warmstart_path:
        stats = warmstart_shared_cc_policy_from_bc_player(
            trainer,
            bc_warmstart_path,
            policy_id=SHARED_CC_POLICY_ID,
        )
        _console_print(
            "[bc-warmstart] copied player-level BC weights into shared centralized-critic policy "
            f"(copied={stats['copied']}, adapted={stats['adapted']}, skipped={stats['skipped']})"
        )
        _append_warmstart_summary(
            "status: bc_warmstart_applied\n"
            f"source_checkpoint: {bc_warmstart_path}\n"
            f"copied: {stats['copied']}\n"
            f"adapted: {stats['adapted']}\n"
            f"skipped: {stats['skipped']}"
        )
        return

    if not warmstart_path:
        _append_warmstart_summary("status: no_warmstart")
        return

    stats = warmstart_shared_cc_policy(trainer, warmstart_path, policy_id=SHARED_CC_POLICY_ID)
    _console_print(
        "[warmstart] copied source weights into shared centralized-critic policy "
        f"(copied={stats['copied']}, adapted={stats['adapted']}, skipped={stats['skipped']})"
    )
    _append_warmstart_summary(
        "status: warmstart_applied\n"
        f"source_checkpoint: {warmstart_path}\n"
        f"copied: {stats['copied']}\n"
        f"adapted: {stats['adapted']}\n"
        f"skipped: {stats['skipped']}"
    )


MAPPOVsBaselineTrainer = PPOTrainer.with_updates(
    name="MAPPOVsBaselineTrainer",
    after_init=_after_init_warmstart,
)


def _validate_base_port(*, base_port, num_workers, num_envs_per_worker):
    worst_case_worker_offset = max(1, num_workers) * max(1, num_envs_per_worker) + max(1, num_envs_per_worker)
    max_required_port = base_port + worst_case_worker_offset
    if max_required_port > 65535:
        raise ValueError(
            f"BASE_PORT={base_port} is too high for {num_workers} workers x {num_envs_per_worker} envs. "
            f"Estimated max required port is {max_required_port} (> 65535). "
            "Lower BASE_PORT or reduce workers/envs."
        )


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
    reward_shaping_enabled,
    reward_shaping_config,
    obs_include_teammate,
    obs_include_time,
    teammate_state_scale,
    role_differentiated_shaping,
    field_role_binding_shaping,
    aux_teammate_head,
    aux_teammate_weight,
    aux_teammate_hidden,
    aux_teammate_label_scale,
):
    _console_print("")
    _console_print("Training Configuration")
    _console_print(f"  run_dir:              {run_dir}")
    _console_print(f"  resume_checkpoint:    {resume_path or 'None'}")
    _console_print(f"  bc_warmstart_ckpt:    {bc_warmstart_path or 'None'}")
    _console_print(f"  warmstart_checkpoint: {warmstart_path or 'None'}")
    _console_print(f"  target_timesteps:     {timesteps_total:,}")
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
    _console_print(f"  env_variation:        {EnvType.multiagent_player}")
    _console_print("  actor_structure:      shared-policy multiagent team0 actor")
    _console_print("  critic_type:          centralized critic (own_obs + teammate_obs + teammate_action)")
    _console_print("  opponent:             fixed baseline on team1")
    _console_print(f"  reward_shaping:       {'enabled' if reward_shaping_enabled else 'disabled'}")
    _console_print(f"  teammate_obs:         {'enabled' if obs_include_teammate else 'disabled'}")
    _console_print(f"  time_obs:             {'enabled' if obs_include_time else 'disabled'}")
    if obs_include_teammate:
        _console_print(f"  teammate_state_scale: {teammate_state_scale or 'raw/unscaled'}")
    _console_print(
        f"  role_diff_shaping:    {'enabled' if role_differentiated_shaping else 'disabled'}"
    )
    _console_print(
        f"  field_role_binding:   {'enabled' if field_role_binding_shaping else 'disabled'}"
    )
    _console_print(
        f"  aux_teammate_head:    {'enabled' if aux_teammate_head else 'disabled'}"
    )
    if aux_teammate_head:
        _console_print(f"  aux_weight:           {aux_teammate_weight}")
        _console_print(f"  aux_hidden:           {aux_teammate_hidden}")
        _console_print(f"  aux_label_scale:      {aux_teammate_label_scale}")
    if isinstance(reward_shaping_config, dict):
        _console_print(f"  shaping_time_penalty: {reward_shaping_config.get('time_penalty')}")
        _console_print(f"  shaping_ball_prog:    {reward_shaping_config.get('ball_progress_scale')}")
        _console_print(f"  shaping_goal_prox:   {reward_shaping_config.get('goal_proximity_scale')}")
        if reward_shaping_config.get("goal_proximity_scale", 0.0):
            _console_print(
                f"  shaping_goal_center:  ({reward_shaping_config.get('goal_center_x')}, "
                f"{reward_shaping_config.get('goal_center_y')}) gamma={reward_shaping_config.get('goal_proximity_gamma')}"
            )
        _console_print(f"  shaping_opp_prog:     {reward_shaping_config.get('opponent_progress_penalty_scale')}")
        _console_print(f"  shaping_pos_bonus:    {reward_shaping_config.get('possession_bonus')}")
        _console_print(f"  shaping_pos_dist:     {reward_shaping_config.get('possession_dist')}")
        _console_print(f"  shaping_prog_gate:    {reward_shaping_config.get('progress_requires_possession')}")
        if any(
            float(reward_shaping_config.get(key, 0.0) or 0.0) != 0.0
            for key in ("event_shot_reward", "event_tackle_reward", "event_clearance_reward")
        ):
            _console_print(
                "  shaping_events:      "
                f"shot={reward_shaping_config.get('event_shot_reward')} "
                f"tackle={reward_shaping_config.get('event_tackle_reward')} "
                f"clearance={reward_shaping_config.get('event_clearance_reward')} "
                f"cooldown={reward_shaping_config.get('event_cooldown_steps')}"
            )
            _console_print(
                "  shaping_event_cfg:   "
                f"shot_x>{reward_shaping_config.get('shot_x_threshold')} "
                f"shot_dx>{reward_shaping_config.get('shot_ball_dx_min')} "
                f"clearance {reward_shaping_config.get('clearance_from_x')} -> "
                f"{reward_shaping_config.get('clearance_to_x')}"
            )
        _console_print(
            f"  shaping_dz_outer:     x<{reward_shaping_config.get('deep_zone_outer_threshold')} "
            f"=> -{reward_shaping_config.get('deep_zone_outer_penalty')}"
        )
        _console_print(
            f"  shaping_dz_inner:     x<{reward_shaping_config.get('deep_zone_inner_threshold')} "
            f"=> -{reward_shaping_config.get('deep_zone_inner_penalty')}"
        )
    _console_print(f"  base_port:            {base_port}")
    _console_print("")


def main():
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
    use_reward_shaping = _env_bool("USE_REWARD_SHAPING", False)
    reward_shaping = _build_reward_shaping_config() if use_reward_shaping else False
    role_differentiated_shaping = _env_bool("SHAPING_ROLE_DIFFERENTIATED", False)
    field_role_binding_shaping = _env_bool("SHAPING_FIELD_ROLE_BINDING", False)
    if role_differentiated_shaping and field_role_binding_shaping:
        raise ValueError(
            "SHAPING_ROLE_DIFFERENTIATED and SHAPING_FIELD_ROLE_BINDING are mutually exclusive."
        )
    if role_differentiated_shaping:
        if not use_reward_shaping:
            raise ValueError("SHAPING_ROLE_DIFFERENTIATED requires USE_REWARD_SHAPING=1.")
        reward_shaping = build_role_reward_shaping_config()
    if field_role_binding_shaping:
        if not use_reward_shaping:
            raise ValueError("SHAPING_FIELD_ROLE_BINDING requires USE_REWARD_SHAPING=1.")
        reward_shaping = build_field_role_reward_shaping_config()
    obs_include_teammate = _env_bool("OBS_INCLUDE_TEAMMATE", False)
    obs_include_time = _env_bool("OBS_INCLUDE_TIME", False)
    teammate_time_max_steps = _env_int("TEAMMATE_TIME_MAX_STEPS", 1500)
    teammate_state_scale = os.environ.get("TEAMMATE_STATE_SCALE", "").strip()
    aux_teammate_head = _env_bool("AUX_TEAMMATE_HEAD", False)
    aux_teammate_weight = _env_float("AUX_TEAMMATE_WEIGHT", 0.1)
    aux_teammate_hidden = _env_int("AUX_TEAMMATE_HIDDEN", 128)
    aux_teammate_label_scale = os.environ.get("AUX_TEAMMATE_LABEL_SCALE", "15.0,7.0,5.0,5.0").strip() or "15.0,7.0,5.0,5.0"

    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip() or None
    bc_warmstart_path = os.environ.get("BC_WARMSTART_CHECKPOINT", "").strip() or None
    resume_path = _resolve_resume_checkpoint_env() or None
    if resume_path:
        resume_path = sanitize_checkpoint_for_restore(resume_path)

    timesteps_total = _env_int("TIMESTEPS_TOTAL", DEFAULT_TIMESTEPS_TOTAL)
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

    base_env_config = {
        "variation": EnvType.multiagent_player,
        "multiagent": True,
        "flatten_branched": True,
    }
    if obs_include_teammate:
        base_env_config["teammate_state_obs"] = {
            "include_time": obs_include_time,
            "max_steps": teammate_time_max_steps,
            "state_scale": teammate_state_scale,
        }
    if reward_shaping:
        base_env_config["reward_shaping"] = reward_shaping

    # snapshot-036 Path C: optional learned reward shaping layered on top of v2.
    # Activated via env var LEARNED_REWARD_MODEL_PATH (plus optional WEIGHT, APPLY_TO_TEAM1).
    # snapshot-036D adds LEARNED_REWARD_WARMUP_STEPS for per-env step-counter warmup
    # (skip shaping for first N env steps; see snapshot-036d §2.3).
    learned_reward_model_path = os.environ.get("LEARNED_REWARD_MODEL_PATH", "").strip()
    if learned_reward_model_path:
        base_env_config["learned_reward_shaping"] = {
            "model_path": learned_reward_model_path,
            "weight": _env_float("LEARNED_REWARD_SHAPING_WEIGHT", 0.01),
            "team0_agent_ids": (0, 1),
            "apply_to_team1": _env_bool("LEARNED_REWARD_APPLY_TO_TEAM1", False),
            "warmup_steps": _env_int("LEARNED_REWARD_WARMUP_STEPS", 0),
        }
        _console_print(
            f"[learned-reward] enabled: {learned_reward_model_path} "
            f"(weight={base_env_config['learned_reward_shaping']['weight']}, "
            f"warmup_steps={base_env_config['learned_reward_shaping']['warmup_steps']})"
        )

    temp_env = create_rllib_env(
        {
            **base_env_config,
            "base_port": base_port,
        }
    )
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    cc_obs_space = build_cc_obs_space(obs_space, act_space)
    register_shared_cc_model()
    register_shared_cc_teammate_aux_model()

    ray.init(include_dashboard=False)
    tune.registry.register_env("Soccer", create_rllib_env)

    baseline_policy_config = {}
    if obs_include_teammate:
        baseline_policy_config["strip_obs_tail_dims"] = 4 + (1 if obs_include_time else 0)

    callbacks_cls = FillInTeammateActionsAndAuxLabels if aux_teammate_head else FillInTeammateActions
    custom_model_name = TEAMMATE_AUX_MODEL_NAME if aux_teammate_head else SHARED_CC_MODEL_NAME

    # snapshot-039: opt-in adaptive reward refresh. When enabled, compose
    # AdaptiveRewardCallback with the existing teammate-action callback via
    # multi-inheritance. Methods that both define chain through MRO.
    adaptive_refresh_enabled = _env_bool("LEARNED_REWARD_ADAPTIVE_REFRESH", False)
    if adaptive_refresh_enabled and learned_reward_model_path:
        from cs8803drl.imitation.adaptive_reward_callback import AdaptiveRewardCallback

        class CombinedCallback(AdaptiveRewardCallback, callbacks_cls):
            pass

        callbacks_cls = CombinedCallback
        _console_print(
            f"[adaptive-reward] enabled: refresh_every="
            f"{os.environ.get('ADAPTIVE_REFRESH_EVERY', '30')} "
            f"loss={os.environ.get('ADAPTIVE_REFRESH_LOSS', 'bt')}"
        )

    config = {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": log_level,
        "log_sys_usage": log_sys_usage,
        "framework": framework,
        "callbacks": callbacks_cls,
        "env": "Soccer",
        "env_config": {
            **base_env_config,
            "num_envs_per_worker": num_envs_per_worker,
            "base_port": base_port,
        },
        "multiagent": {
            "policies": {
                SHARED_CC_POLICY_ID: (None, cc_obs_space, act_space, {}),
                BASELINE_POLICY_ID: (FrozenBaselinePolicy, obs_space, act_space, baseline_policy_config),
            },
            "policy_mapping_fn": tune.function(shared_cc_policy_mapping_fn),
            "policies_to_train": [SHARED_CC_POLICY_ID],
            "observation_fn": shared_cc_observer,
        },
        "model": {
            "custom_model": custom_model_name,
            "vf_share_layers": vf_share_layers,
            "fcnet_hiddens": fcnet_hiddens,
            "fcnet_activation": fcnet_activation,
            "custom_model_config": {
                "aux_weight": aux_teammate_weight,
                "aux_hidden_size": aux_teammate_hidden,
                "aux_label_scale": aux_teammate_label_scale,
            },
        },
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

    local_dir = os.path.abspath(local_dir)
    run_dir = os.path.join(local_dir, run_name)
    warmstart_summary_path = os.path.join(run_dir, "warmstart_summary.txt")
    os.environ["WARMSTART_SUMMARY_PATH"] = warmstart_summary_path
    os.makedirs(run_dir, exist_ok=True)
    with open(warmstart_summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"run_dir: {run_dir}\n")
        handle.write(f"resume_checkpoint: {resume_path or 'None'}\n")
        handle.write(f"bc_warmstart_ckpt: {bc_warmstart_path or 'None'}\n")
        handle.write(f"warmstart_checkpoint: {warmstart_path or 'None'}\n")
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
        reward_shaping_enabled=bool(use_reward_shaping),
        reward_shaping_config=reward_shaping,
        obs_include_teammate=obs_include_teammate,
        obs_include_time=obs_include_time,
        teammate_state_scale=teammate_state_scale,
        role_differentiated_shaping=role_differentiated_shaping,
        field_role_binding_shaping=field_role_binding_shaping,
        aux_teammate_head=aux_teammate_head,
        aux_teammate_weight=aux_teammate_weight,
        aux_teammate_hidden=aux_teammate_hidden,
        aux_teammate_label_scale=aux_teammate_label_scale,
    )

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
            MAPPOVsBaselineTrainer,
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

    loss_curve_path = _write_training_loss_curve(run_dir)
    best_eval = monitor_state.get("best_eval")

    _console_print("")
    _console_print("Training Summary")
    _console_print(f"  stop_reason:      {stop_reason}")
    _console_print(f"  run_dir:          {run_dir}")
    if best_reward is not None:
        _console_print(f"  best_reward_mean: {best_reward:+.4f} @ iteration {best_iteration}")
    else:
        _console_print("  best_reward_mean: unavailable")
    if best_checkpoint is not None:
        _console_print(f"  best_checkpoint:  {best_checkpoint}")
    else:
        _console_print("  best_checkpoint:  unavailable")
    _console_print(f"  final_checkpoint: {final_checkpoint}")
    if best_eval:
        baseline = best_eval.get("baseline") or {}
        random_row = best_eval.get("random") or {}
        checkpoint_file = best_eval.get("checkpoint_file") or best_eval.get("checkpoint_dir")
        _console_print(f"  best_eval_checkpoint: {checkpoint_file}")
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
    if loss_curve_path:
        _console_print(f"  loss_curve_file:    {loss_curve_path}")


if __name__ == "__main__":
    main()

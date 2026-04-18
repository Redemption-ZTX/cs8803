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

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from cs8803drl.branches.role_specialization import BASELINE_POLICY_ID, FrozenBaselinePolicy
from cs8803drl.branches.shared_central_critic import (
    FillInTeammateActions,
    SHARED_CC_MODEL_NAME,
    SHARED_CC_POLICY_ID,
    build_cc_obs_space,
    load_shared_cc_policy_from_checkpoint,
    register_shared_cc_model,
    shared_cc_observer_all,
    team0_agent_ids,
    warmstart_shared_cc_policy_from_bc_player,
)
from cs8803drl.core.checkpoint_utils import sanitize_checkpoint_for_restore
from cs8803drl.core.utils import create_rllib_env
from cs8803drl.training.train_ray_team_vs_random_shaping import (
    _build_reward_shaping_config,
    _console_print,
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


DEFAULT_TIMESTEPS_TOTAL = 12_000_000
DEFAULT_TIME_TOTAL_S = 20_000
DEFAULT_RUN_NAME = "PPO_mappo_v2_opponent_pool"
DEFAULT_CHECKPOINT_FREQ = 10
DEFAULT_FRAMEWORK = "torch"
DEFAULT_NUM_GPUS = 1
DEFAULT_NUM_WORKERS = 8
DEFAULT_NUM_ENVS_PER_WORKER = 5
DEFAULT_ROLLOUT_FRAGMENT_LENGTH = 1_000
DEFAULT_NUM_SGD_ITER = 10
DEFAULT_BASE_PORT = 63_105
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_FCNET_HIDDENS = (512, 512)

OPPONENT_ANCHOR_POLICY_ID = "opponent_anchor"
OPPONENT_V1_POLICY_ID = "opponent_v1"
OPPONENT_BS0_POLICY_ID = "opponent_bs0"


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


def _normalize_pool(ids, probs):
    filtered = [(pid, float(prob)) for pid, prob in zip(ids, probs) if float(prob) > 0.0]
    if not filtered:
        return [BASELINE_POLICY_ID], [1.0]
    ids = [pid for pid, _ in filtered]
    probs = np.asarray([prob for _, prob in filtered], dtype=np.float64)
    probs = probs / probs.sum()
    return ids, probs.tolist()


def _pool_ids_and_probs():
    ids = [
        BASELINE_POLICY_ID,
        OPPONENT_ANCHOR_POLICY_ID,
        OPPONENT_V1_POLICY_ID,
        OPPONENT_BS0_POLICY_ID,
    ]
    probs = [
        float(os.environ.get("POOL_BASELINE_PROB", "0.60")),
        float(os.environ.get("POOL_ANCHOR_PROB", "0.15")),
        float(os.environ.get("POOL_V1_PROB", "0.15")),
        float(os.environ.get("POOL_BS0_PROB", "0.10")),
    ]
    return _normalize_pool(ids, probs)


def _find_episode(args, kwargs):
    for arg in args:
        if hasattr(arg, "user_data"):
            return arg
    episode = kwargs.get("episode")
    if hasattr(episode, "user_data"):
        return episode
    return None


def opponent_pool_policy_mapping_fn(agent_id, *args, **kwargs):
    if int(agent_id) in set(team0_agent_ids()):
        return SHARED_CC_POLICY_ID

    ids, probs = _pool_ids_and_probs()
    episode = _find_episode(args, kwargs)
    if episode is None:
        return str(np.random.choice(ids, p=probs))

    key = "team1_pool_policy_id"
    if key not in episode.user_data:
        episode.user_data[key] = str(np.random.choice(ids, p=probs))
    return episode.user_data[key]


def _load_pool_opponents(trainer):
    loaded = []
    anchor_ckpt = os.environ.get("POOL_ANCHOR_CHECKPOINT", "").strip()
    v1_ckpt = os.environ.get("POOL_V1_CHECKPOINT", "").strip()
    bs0_ckpt = os.environ.get("POOL_BS0_CHECKPOINT", "").strip()

    if anchor_ckpt:
        load_shared_cc_policy_from_checkpoint(
            trainer,
            anchor_ckpt,
            source_policy_id=SHARED_CC_POLICY_ID,
            target_policy_id=OPPONENT_ANCHOR_POLICY_ID,
        )
        loaded.append((OPPONENT_ANCHOR_POLICY_ID, anchor_ckpt))
    if v1_ckpt:
        load_shared_cc_policy_from_checkpoint(
            trainer,
            v1_ckpt,
            source_policy_id=SHARED_CC_POLICY_ID,
            target_policy_id=OPPONENT_V1_POLICY_ID,
        )
        loaded.append((OPPONENT_V1_POLICY_ID, v1_ckpt))
    if bs0_ckpt:
        load_shared_cc_policy_from_checkpoint(
            trainer,
            bs0_ckpt,
            source_policy_id=SHARED_CC_POLICY_ID,
            target_policy_id=OPPONENT_BS0_POLICY_ID,
        )
        loaded.append((OPPONENT_BS0_POLICY_ID, bs0_ckpt))

    return loaded


def _after_init_pool_warmstart(trainer):
    restore_path = os.environ.get("RESTORE_CHECKPOINT", "").strip()
    if restore_path and not _env_bool("WARMSTART_ON_RESTORE", False):
        _append_warmstart_summary(
            f"status: skipped (restore_checkpoint set, WARMSTART_ON_RESTORE=0)\n"
            f"restore_checkpoint: {restore_path}"
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
    elif warmstart_path:
        load_shared_cc_policy_from_checkpoint(
            trainer,
            warmstart_path,
            source_policy_id=SHARED_CC_POLICY_ID,
            target_policy_id=SHARED_CC_POLICY_ID,
        )
        _console_print(f"[warmstart] loaded shared_cc_policy from checkpoint: {warmstart_path}")
        _append_warmstart_summary(
            "status: warmstart_applied\n"
            f"source_checkpoint: {warmstart_path}\n"
            "mode: load_shared_cc_policy_from_checkpoint"
        )
    else:
        _append_warmstart_summary("status: no_warmstart")

    loaded = _load_pool_opponents(trainer)
    if loaded:
        summary = ", ".join(f"{policy_id} <- {Path(path).name}" for policy_id, path in loaded)
        _console_print(f"[opponent-pool] loaded frozen opponent checkpoints: {summary}")
        _append_warmstart_summary(
            "opponent_pool:\n" + "\n".join(
                f"  {policy_id}: {path}" for policy_id, path in loaded
            )
        )


MAPPOVsOpponentPoolTrainer = PPOTrainer.with_updates(
    name="MAPPOVsOpponentPoolTrainer",
    after_init=_after_init_pool_warmstart,
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
    restore_path,
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
):
    ids, probs = _pool_ids_and_probs()
    _console_print("")
    _console_print("Training Configuration")
    _console_print(f"  run_dir:              {run_dir}")
    _console_print(f"  restore_checkpoint:   {restore_path or 'None'}")
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
    _console_print(
        "  opponent_pool:        "
        + ", ".join(f"{pid}:{prob:.2f}" for pid, prob in zip(ids, probs))
    )
    _console_print(f"  reward_shaping:       {'enabled' if reward_shaping_enabled else 'disabled'}")
    if isinstance(reward_shaping_config, dict):
        _console_print(f"  shaping_time_penalty: {reward_shaping_config.get('time_penalty')}")
        _console_print(f"  shaping_ball_prog:    {reward_shaping_config.get('ball_progress_scale')}")
        _console_print(f"  shaping_opp_prog:     {reward_shaping_config.get('opponent_progress_penalty_scale')}")
        _console_print(f"  shaping_pos_bonus:    {reward_shaping_config.get('possession_bonus')}")
        _console_print(f"  shaping_pos_dist:     {reward_shaping_config.get('possession_dist')}")
        _console_print(f"  shaping_prog_gate:    {reward_shaping_config.get('progress_requires_possession')}")
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
    use_reward_shaping = _env_bool("USE_REWARD_SHAPING", True)
    reward_shaping = _build_reward_shaping_config() if use_reward_shaping else False

    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip() or None
    bc_warmstart_path = os.environ.get("BC_WARMSTART_CHECKPOINT", "").strip() or None
    restore_path = os.environ.get("RESTORE_CHECKPOINT", "").strip() or None
    if restore_path:
        restore_path = sanitize_checkpoint_for_restore(restore_path)

    timesteps_total = _env_int("TIMESTEPS_TOTAL", DEFAULT_TIMESTEPS_TOTAL)
    time_total_s = _env_int("TIME_TOTAL_S", DEFAULT_TIME_TOTAL_S)
    max_iterations = _env_int("MAX_ITERATIONS", 0)
    run_name = os.environ.get("RUN_NAME", DEFAULT_RUN_NAME).strip() or DEFAULT_RUN_NAME
    checkpoint_freq = _env_int("CHECKPOINT_FREQ", DEFAULT_CHECKPOINT_FREQ)
    base_port = _env_int("BASE_PORT", DEFAULT_BASE_PORT)
    local_dir = os.environ.get("LOCAL_DIR", "./ray_results").strip() or "./ray_results"
    _validate_base_port(base_port=base_port, num_workers=num_workers, num_envs_per_worker=num_envs_per_worker)

    base_env_config = {
        "variation": EnvType.multiagent_player,
        "multiagent": True,
        "flatten_branched": True,
    }
    if reward_shaping:
        base_env_config["reward_shaping"] = reward_shaping

    temp_env = create_rllib_env(base_env_config)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    cc_obs_space = build_cc_obs_space(obs_space, act_space)
    register_shared_cc_model()

    flat_obs_dim = int(np.prod(obs_space.shape))
    action_dim = int(act_space.n) if hasattr(act_space, "n") else int(np.prod(act_space.nvec))
    baseline_strip_tail_dims = flat_obs_dim + action_dim

    ray.init(include_dashboard=False)
    tune.registry.register_env("Soccer", create_rllib_env)

    config = {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": log_level,
        "log_sys_usage": log_sys_usage,
        "framework": framework,
        "callbacks": FillInTeammateActions,
        "env": "Soccer",
        "env_config": {
            **base_env_config,
            "num_envs_per_worker": num_envs_per_worker,
            "base_port": base_port,
        },
        "multiagent": {
            "policies": {
                SHARED_CC_POLICY_ID: (None, cc_obs_space, act_space, {}),
                BASELINE_POLICY_ID: (
                    FrozenBaselinePolicy,
                    cc_obs_space,
                    act_space,
                    {"strip_obs_tail_dims": baseline_strip_tail_dims},
                ),
                OPPONENT_ANCHOR_POLICY_ID: (None, cc_obs_space, act_space, {}),
                OPPONENT_V1_POLICY_ID: (None, cc_obs_space, act_space, {}),
                OPPONENT_BS0_POLICY_ID: (None, cc_obs_space, act_space, {}),
            },
            "policy_mapping_fn": tune.function(opponent_pool_policy_mapping_fn),
            "policies_to_train": [SHARED_CC_POLICY_ID],
            "observation_fn": shared_cc_observer_all,
        },
        "model": {
            "custom_model": SHARED_CC_MODEL_NAME,
            "vf_share_layers": vf_share_layers,
            "fcnet_hiddens": fcnet_hiddens,
            "fcnet_activation": fcnet_activation,
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
        handle.write(f"restore_checkpoint: {restore_path or 'None'}\n")
        handle.write(f"bc_warmstart_ckpt: {bc_warmstart_path or 'None'}\n")
        handle.write(f"warmstart_checkpoint: {warmstart_path or 'None'}\n")
    _print_header(
        run_dir=run_dir,
        restore_path=restore_path,
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
        kwargs={"target_timesteps": timesteps_total, "max_iterations": max_iterations, "state": monitor_state},
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
            MAPPOVsOpponentPoolTrainer,
            name=run_name,
            config=config,
            stop=stop,
            checkpoint_freq=checkpoint_freq,
            checkpoint_at_end=True,
            local_dir=local_dir,
            restore=restore_path,
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

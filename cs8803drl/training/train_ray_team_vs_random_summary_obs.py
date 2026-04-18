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
os.environ.setdefault("EVAL_TEAM0_MODULE", "cs8803drl.deployment.trained_summary_ray_agent")

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from cs8803drl.core.checkpoint_utils import sanitize_checkpoint_for_restore
from cs8803drl.branches.obs_summary import warmstart_summary_policy
from cs8803drl.training.train_ray_team_vs_random_shaping import (
    DEFAULT_BASE_PORT,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_FCNET_HIDDENS,
    DEFAULT_FRAMEWORK,
    DEFAULT_LOG_LEVEL,
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
    _print_run_header,
    _row_float,
    _row_int,
    _write_training_loss_curve,
)
from cs8803drl.core.utils import create_rllib_env


DEFAULT_TIMESTEPS_TOTAL = 3_000_000
DEFAULT_TIME_TOTAL_S = 7_200
DEFAULT_RUN_NAME = "PPO_team_summary_obs"
DEFAULT_NUM_GPUS = 1
DEFAULT_NUM_WORKERS = 24
DEFAULT_NUM_ENVS_PER_WORKER = 1
DEFAULT_ROLLOUT_FRAGMENT_LENGTH = 1_000
DEFAULT_NUM_SGD_ITER = 4
DEFAULT_BASELINE_PROB = 1.0


def _after_init_warmstart(trainer):
    restore_path = os.environ.get("RESTORE_CHECKPOINT", "").strip()
    if restore_path and not _env_bool("WARMSTART_ON_RESTORE", False):
        return

    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip()
    if not warmstart_path:
        return

    stats = warmstart_summary_policy(trainer, warmstart_path, policy_name="default_policy")
    _console_print(
        "[warmstart] copied source weights into default_policy "
        f"(copied={stats['copied']}, adapted={stats['adapted']}, skipped={stats['skipped']})"
    )


SummaryObsPPOTrainer = PPOTrainer.with_updates(
    name="SummaryObsPPOTrainer",
    after_init=_after_init_warmstart,
)


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
    lr = _env_float("LR", 1e-4)
    gamma = _env_float("GAMMA", 0.99)
    gae_lambda = _env_float("GAE_LAMBDA", 0.95)
    clip_param = _env_float("CLIP_PARAM", 0.2)
    entropy_coeff = _env_float("ENTROPY_COEFF", 0.0)
    vf_share_layers = _env_bool("VF_SHARE_LAYERS", True)
    simple_optimizer = _env_bool("SIMPLE_OPTIMIZER", True)
    log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper() or DEFAULT_LOG_LEVEL
    log_sys_usage = _env_bool("LOG_SYS_USAGE", True)
    fcnet_hiddens = _env_layers("FCNET_HIDDENS", DEFAULT_FCNET_HIDDENS)
    fcnet_activation = os.environ.get("FCNET_ACTIVATION", "relu").strip() or "relu"
    use_reward_shaping = _env_bool("USE_REWARD_SHAPING", False)

    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip() or None
    restore_path = os.environ.get("RESTORE_CHECKPOINT", "").strip() or None
    if warmstart_path and not _env_bool("ALLOW_SUMMARY_RESTORE", False):
        restore_path = None
    if restore_path:
        restore_path = sanitize_checkpoint_for_restore(restore_path)

    timesteps_total = _env_int("TIMESTEPS_TOTAL", DEFAULT_TIMESTEPS_TOTAL)
    time_total_s = _env_int("TIME_TOTAL_S", DEFAULT_TIME_TOTAL_S)
    max_iterations = _env_int("MAX_ITERATIONS", 0)
    run_name = os.environ.get("RUN_NAME", DEFAULT_RUN_NAME).strip() or DEFAULT_RUN_NAME
    checkpoint_freq = _env_int("CHECKPOINT_FREQ", DEFAULT_CHECKPOINT_FREQ)
    base_port = _env_int("BASE_PORT", DEFAULT_BASE_PORT)
    baseline_prob = _env_float("BASELINE_PROB", DEFAULT_BASELINE_PROB)
    local_dir = os.environ.get("LOCAL_DIR", "./ray_results").strip() or "./ray_results"

    ray.init(include_dashboard=False)
    tune.registry.register_env("Soccer", create_rllib_env)

    reward_shaping = _build_reward_shaping_config() if use_reward_shaping else False

    config = {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": log_level,
        "log_sys_usage": log_sys_usage,
        "framework": framework,
        "simple_optimizer": simple_optimizer,
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": num_envs_per_worker,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "single_player": True,
            "flatten_branched": True,
            "ray_summary_obs": True,
            "reward_shaping": reward_shaping,
            "opponent_mix": {"baseline_prob": baseline_prob},
            "base_port": base_port,
        },
        "model": {
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
    _print_run_header(
        run_dir=run_dir,
        restore_path=restore_path,
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
    )
    _console_print("Observation Summary Configuration")
    _console_print(f"  warmstart_checkpoint: {warmstart_path or 'None'}")
    _console_print("  observation_mod:     ray summary features (+28 dims)")
    _console_print(f"  reward_shaping:      {'enabled' if use_reward_shaping else 'disabled'}")
    if warmstart_path and not restore_path:
        _console_print("  restore_checkpoint:  disabled (warm-start mode)")
    _console_print("")

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
            SummaryObsPPOTrainer,
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

        progress_csv = _find_progress_csv(run_dir)
        final_row = _last_csv_row(progress_csv) if progress_csv is not None else None
        if final_row is not None:
            best_reward = _row_float(final_row, "episode_reward_mean", 0.0)
            best_iteration = _row_int(final_row, "training_iteration", 0)
            if best_iteration > int(monitor_state.get("last_printed_iteration", -1)):
                _print_progress(
                    final_row,
                    target_timesteps=timesteps_total,
                    max_iterations=max_iterations,
                    previous_steps=None,
                    previous_elapsed=None,
                )

        best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
        if best_trial is not None:
            if getattr(best_trial, "last_result", None):
                best_reward = float(best_trial.last_result.get("episode_reward_mean", best_reward or 0.0))
                best_iteration = int(best_trial.last_result.get("training_iteration", best_iteration))
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
        eval_thread.join()
        ray.shutdown()

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

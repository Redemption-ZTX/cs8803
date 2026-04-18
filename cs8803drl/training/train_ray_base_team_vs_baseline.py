import math
import os
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
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from cs8803drl.core.utils import create_rllib_env
from cs8803drl.training.train_ray_team_vs_random_shaping import (
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
    _write_training_loss_curve,
)


DEFAULT_TIMESTEPS_TOTAL = 20_000_000
DEFAULT_TIME_TOTAL_S = 0
DEFAULT_RUN_NAME = "PPO_base_team_vs_baseline"
DEFAULT_CHECKPOINT_FREQ = 10
DEFAULT_FRAMEWORK = "torch"
DEFAULT_NUM_GPUS = 1
DEFAULT_NUM_WORKERS = 8
DEFAULT_NUM_ENVS_PER_WORKER = 5
DEFAULT_ROLLOUT_FRAGMENT_LENGTH = 1_000
DEFAULT_NUM_SGD_ITER = 10
DEFAULT_BASE_PORT = 5_405
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_FCNET_HIDDENS = (512, 512)
DEFAULT_BASELINE_PROB = 1.0


BaseTeamVsBaselinePPOTrainer = PPOTrainer.with_updates(
    name="BaseTeamVsBaselinePPOTrainer",
)


def _assert_scratch_only():
    restore_path = os.environ.get("RESTORE_CHECKPOINT", "").strip()
    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip()
    if restore_path or warmstart_path:
        raise ValueError(
            "Base model lane must run from scratch. "
            "Unset RESTORE_CHECKPOINT and WARMSTART_CHECKPOINT."
        )


def _print_base_header(
    *,
    run_dir,
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
):
    est_total_iterations = None
    if timesteps_total > 0 and train_batch_size > 0:
        est_total_iterations = int(math.ceil(float(timesteps_total) / float(train_batch_size)))

    _console_print("")
    _console_print("Training Configuration")
    _console_print(f"  run_dir:              {run_dir}")
    _console_print("  restore_checkpoint:   None")
    _console_print("  warmstart_checkpoint: None")
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
    _console_print(f"  env_variation:        {EnvType.team_vs_policy}")
    _console_print("  starter_alignment:    example_ray_team_vs_random.py (opponent swapped to baseline)")
    _console_print("  reward_shaping:       disabled")
    _console_print("  training_mode:        scratch base-model lane")
    _console_print(f"  opponent baseline p:  {baseline_prob:.2f}")
    _console_print(f"  base_port:            {base_port}")
    if est_total_iterations is not None:
        _console_print(f"  estimated_iterations: {est_total_iterations}")
    _console_print("")


def main():
    _assert_scratch_only()

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

    timesteps_total = _env_int("TIMESTEPS_TOTAL", DEFAULT_TIMESTEPS_TOTAL)
    time_total_s = _env_int("TIME_TOTAL_S", DEFAULT_TIME_TOTAL_S)
    max_iterations = _env_int("MAX_ITERATIONS", 0)
    run_name = os.environ.get("RUN_NAME", DEFAULT_RUN_NAME).strip() or DEFAULT_RUN_NAME
    checkpoint_freq = _env_int("CHECKPOINT_FREQ", DEFAULT_CHECKPOINT_FREQ)
    base_port = _env_int("BASE_PORT", DEFAULT_BASE_PORT)
    local_dir = os.environ.get("LOCAL_DIR", "./ray_results").strip() or "./ray_results"

    ray.init(include_dashboard=False)
    tune.registry.register_env("Soccer", create_rllib_env)

    config = {
        "num_gpus": num_gpus,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "log_level": log_level,
        "log_sys_usage": log_sys_usage,
        "framework": framework,
        "env": "Soccer",
        "env_config": {
            "num_envs_per_worker": num_envs_per_worker,
            "variation": EnvType.team_vs_policy,
            "multiagent": False,
            "base_port": base_port,
            "opponent_mix": {"baseline_prob": baseline_prob},
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
    _print_base_header(
        run_dir=run_dir,
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
            BaseTeamVsBaselinePPOTrainer,
            name=run_name,
            config=config,
            stop=stop,
            checkpoint_freq=checkpoint_freq,
            checkpoint_at_end=True,
            local_dir=local_dir,
            verbose=0,
            raise_on_failed_trial=False,
        )

        try:
            best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
            if best_trial is not None:
                best_reward = best_trial.metric_analysis["episode_reward_mean"]["max"]
                best_iteration = best_trial.metric_analysis["training_iteration"]["max"]
                best_checkpoint = analysis.get_best_checkpoint(
                    trial=best_trial,
                    metric="episode_reward_mean",
                    mode="max",
                )
        except Exception:
            best_trial = None

        try:
            checkpoints = analysis.get_trial_checkpoints_paths(
                analysis.trials[0],
                metric="episode_reward_mean",
            )
            if checkpoints:
                final_checkpoint = checkpoints[-1][0]
        except Exception:
            final_checkpoint = None

        if analysis.trials:
            status = analysis.trials[0].status
            stop_reason = f"Trial status: {status}"
    finally:
        monitor_stop.set()
        try:
            monitor_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            eval_thread.join(timeout=2.0)
        except Exception:
            pass

    progress_csv = _find_progress_csv(run_dir)
    final_row = _last_csv_row(progress_csv) if progress_csv is not None else None
    if final_row is not None:
        best_reward = _row_float(final_row, "episode_reward_mean", best_reward or 0.0)
        best_iteration = _row_int(final_row, "training_iteration", best_iteration)
        if best_iteration > int(monitor_state.get("last_printed_iteration", -1)):
            _print_progress(
                final_row,
                target_timesteps=timesteps_total,
                max_iterations=max_iterations,
                previous_steps=None,
                previous_elapsed=None,
            )

    loss_curve_path = _write_training_loss_curve(run_dir)

    _console_print("")
    _console_print("Training Summary")
    _console_print(f"  stop_reason:      {stop_reason}")
    _console_print(f"  run_dir:          {run_dir}")
    _console_print(
        f"  best_reward_mean: {best_reward if best_reward is not None else 'unavailable'}"
        + (f" @ iteration {best_iteration}" if best_reward is not None else "")
    )
    _console_print(f"  best_checkpoint:  {best_checkpoint if best_checkpoint is not None else 'unavailable'}")
    _console_print(
        f"  final_checkpoint: {final_checkpoint if final_checkpoint is not None else 'Checkpoint(persistent, None)'}"
    )

    eval_csv = Path(run_dir) / "checkpoint_eval.csv"
    best_eval_txt = Path(run_dir) / "best_checkpoint_by_eval.txt"
    if eval_csv.exists():
        _console_print(f"  eval_results_csv:    {eval_csv}")
    if best_eval_txt.exists():
        lines = [line.strip() for line in best_eval_txt.read_text().splitlines() if line.strip()]
        if len(lines) >= 2:
            _console_print(f"  best_eval_checkpoint: {lines[0]}")
            _console_print(f"  best_eval_baseline:  {lines[1]}")
    if loss_curve_path is not None:
        _console_print(f"  loss_curve_file:    {loss_curve_path}")
    _console_print("Done training")


if __name__ == "__main__":
    main()

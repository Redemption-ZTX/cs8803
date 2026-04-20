import csv
import math
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

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

_QUIET_CONSOLE = os.environ.get("QUIET_CONSOLE", "").strip().lower() in {"1", "true", "yes", "y", "on"}
_CONSOLE_STREAM = sys.stdout
_DEVNULL_STREAM = None

if _QUIET_CONSOLE:
    _CONSOLE_STREAM = os.fdopen(os.dup(sys.__stdout__.fileno()), "w", buffering=1)
    _DEVNULL_STREAM = open(os.devnull, "w", buffering=1)
    os.dup2(_DEVNULL_STREAM.fileno(), sys.__stdout__.fileno())
    os.dup2(_DEVNULL_STREAM.fileno(), sys.__stderr__.fileno())
    sys.stdout = _DEVNULL_STREAM
    sys.stderr = _DEVNULL_STREAM


def _console_print(*args, **kwargs):
    kwargs.setdefault("file", _CONSOLE_STREAM)
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.ppo import PPOTrainer
from soccer_twos import EnvType

from cs8803drl.core.checkpoint_utils import load_policy_weights, resolve_checkpoint_file, sanitize_checkpoint_for_restore
from cs8803drl.core.soccer_info import extract_score_from_info, extract_winner_from_info
from cs8803drl.core.training_plots import summarize_training_progress, write_training_loss_curve_png
from cs8803drl.core.utils import create_rllib_env


DEFAULT_TIMESTEPS_TOTAL = 15_000_000
DEFAULT_TIME_TOTAL_S = 7_200
DEFAULT_RUN_NAME = "PPO_team_vs_random_shaping"
DEFAULT_CHECKPOINT_FREQ = 20
DEFAULT_FRAMEWORK = "torch"
DEFAULT_NUM_GPUS = 1
DEFAULT_NUM_WORKERS = 8
DEFAULT_NUM_ENVS_PER_WORKER = 1
DEFAULT_ROLLOUT_FRAGMENT_LENGTH = 1_000
DEFAULT_NUM_SGD_ITER = 10
DEFAULT_BASE_PORT = 5_005
DEFAULT_LOG_LEVEL = "WARN"
DEFAULT_BASELINE_PROB = 0.9
DEFAULT_FCNET_HIDDENS = (512, 512)
_EVAL_SUMMARY_PATTERN = re.compile(
    r"^(team0_module|team1_module|episodes|team0_wins|team1_wins|ties|team0_win_rate|"
    r"team0_non_loss_rate|team0_fast_wins|team0_fast_win_threshold|team0_fast_win_rate):\s*(.+)$"
)


def _resolve_success_metric():
    """`EVAL_SUCCESS_METRIC` env var → metric used for best_eval selection."""
    val = (os.environ.get("EVAL_SUCCESS_METRIC", "") or "").strip().lower()
    if val in ("", "win_rate"):
        return "win_rate"
    if val in ("non_loss_rate", "fast_win_rate"):
        return val
    return "win_rate"


def _after_init_warmstart(trainer):
    resume_path = _resolve_resume_checkpoint_env()
    if resume_path and not _warmstart_on_resume_enabled(False):
        return

    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip()
    if not warmstart_path:
        return

    load_policy_weights(warmstart_path, trainer, policy_name="default_policy")
    try:
        trainer.workers.sync_weights()
    except Exception:
        pass
    _console_print(f"[warmstart] loaded default_policy from {warmstart_path}")


WarmstartPPOTrainer = PPOTrainer.with_updates(
    name="WarmstartPPOTrainer",
    after_init=_after_init_warmstart,
)


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return int(default)
    return int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_resume_checkpoint_env():
    resume_path = os.environ.get("RESUME_CHECKPOINT", "").strip()
    restore_path = os.environ.get("RESTORE_CHECKPOINT", "").strip()
    if resume_path and restore_path and resume_path != restore_path:
        raise ValueError(
            "RESUME_CHECKPOINT and RESTORE_CHECKPOINT are both set with different values. "
            "Use RESUME_CHECKPOINT going forward, or keep both identical."
        )
    return resume_path or restore_path


def _resolve_resume_timesteps_delta(default=0):
    resume_raw = os.environ.get("RESUME_TIMESTEPS_DELTA", "").strip()
    restore_raw = os.environ.get("RESTORE_TIMESTEPS_DELTA", "").strip()
    if resume_raw and restore_raw and int(resume_raw) != int(restore_raw):
        raise ValueError(
            "RESUME_TIMESTEPS_DELTA and RESTORE_TIMESTEPS_DELTA are both set with different "
            "values. Use RESUME_TIMESTEPS_DELTA going forward, or keep both identical."
        )
    if resume_raw:
        return int(resume_raw)
    if restore_raw:
        return int(restore_raw)
    return int(default)


def _warmstart_on_resume_enabled(default=False):
    if os.environ.get("WARMSTART_ON_RESUME", "").strip():
        return _env_bool("WARMSTART_ON_RESUME", default)
    return _env_bool("WARMSTART_ON_RESTORE", default)


def _allow_resume_with_warmstart(default=False):
    if os.environ.get("ALLOW_RESUME_WITH_WARMSTART", "").strip():
        return _env_bool("ALLOW_RESUME_WITH_WARMSTART", default)
    return _env_bool("ALLOW_RESTORE_WITH_WARMSTART", default)


def _env_layers(name, default):
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default)
    layers = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        layers.append(int(piece))
    if not layers:
        raise ValueError(f"{name} must contain at least one hidden size")
    return layers


def _format_count(value):
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _render_bar(fraction, width=28):
    fraction = 0.0 if fraction is None else max(0.0, min(1.0, float(fraction)))
    filled = int(round(fraction * width))
    return "#" * filled + "-" * (width - filled)


def _build_reward_shaping_config():
    debug_info = _env_bool("REWARD_SHAPING_DEBUG", False)
    if not any(
        name in os.environ
        for name in (
            "SHAPING_TIME_PENALTY",
            "SHAPING_BALL_PROGRESS",
            "SHAPING_GOAL_PROXIMITY_SCALE",
            "SHAPING_GOAL_PROXIMITY_GAMMA",
            "SHAPING_GOAL_CENTER_X",
            "SHAPING_GOAL_CENTER_Y",
            "SHAPING_EVENT_SHOT_REWARD",
            "SHAPING_EVENT_TACKLE_REWARD",
            "SHAPING_EVENT_CLEARANCE_REWARD",
            "SHAPING_EVENT_COOLDOWN_STEPS",
            "SHAPING_SHOT_X_THRESHOLD",
            "SHAPING_SHOT_BALL_DX_MIN",
            "SHAPING_CLEARANCE_FROM_X",
            "SHAPING_CLEARANCE_TO_X",
            "SHAPING_OPP_PROGRESS_PENALTY",
            "SHAPING_POSSESSION_DIST",
            "SHAPING_POSSESSION_BONUS",
            "SHAPING_PROGRESS_REQUIRES_POSSESSION",
            "SHAPING_DEEP_ZONE_OUTER_THRESHOLD",
            "SHAPING_DEEP_ZONE_OUTER_PENALTY",
            "SHAPING_DEEP_ZONE_INNER_THRESHOLD",
            "SHAPING_DEEP_ZONE_INNER_PENALTY",
            "SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD",
            "SHAPING_DEFENSIVE_SURVIVAL_BONUS",
            "SHAPING_FAST_LOSS_THRESHOLD_STEPS",
            "SHAPING_FAST_LOSS_PENALTY_PER_STEP",
            "SHAPING_TEAM_SPACING_SCALE",
            "SHAPING_TEAM_COVERAGE_SCALE",
            "SHAPING_TEAM_POTENTIAL_GAMMA",
            "SHAPING_TEAM_NEAR_BALL_THRESHOLD",
            "SHAPING_TEAM_SPACING_MIN",
            "SHAPING_TEAM_SPACING_MAX",
            "SPECIALIST_MODE",
            "FAST_WIN_THRESHOLD",
            "REWARD_SHAPING_DEBUG",
        )
    ):
        return {"debug_info": debug_info} if debug_info else True

    return {
        "time_penalty": _env_float("SHAPING_TIME_PENALTY", 0.001),
        "ball_progress_scale": _env_float("SHAPING_BALL_PROGRESS", 0.01),
        "goal_proximity_scale": _env_float("SHAPING_GOAL_PROXIMITY_SCALE", 0.0),
        "goal_proximity_gamma": _env_float("SHAPING_GOAL_PROXIMITY_GAMMA", 0.99),
        "goal_center_x": _env_float("SHAPING_GOAL_CENTER_X", 15.0),
        "goal_center_y": _env_float("SHAPING_GOAL_CENTER_Y", 0.0),
        "event_shot_reward": _env_float("SHAPING_EVENT_SHOT_REWARD", 0.0),
        "event_tackle_reward": _env_float("SHAPING_EVENT_TACKLE_REWARD", 0.0),
        "event_clearance_reward": _env_float("SHAPING_EVENT_CLEARANCE_REWARD", 0.0),
        "event_cooldown_steps": _env_int("SHAPING_EVENT_COOLDOWN_STEPS", 10),
        "shot_x_threshold": _env_float("SHAPING_SHOT_X_THRESHOLD", 10.0),
        "shot_ball_dx_min": _env_float("SHAPING_SHOT_BALL_DX_MIN", 0.5),
        "clearance_from_x": _env_float("SHAPING_CLEARANCE_FROM_X", -8.0),
        "clearance_to_x": _env_float("SHAPING_CLEARANCE_TO_X", -4.0),
        "opponent_progress_penalty_scale": _env_float("SHAPING_OPP_PROGRESS_PENALTY", 0.0),
        "possession_dist": _env_float("SHAPING_POSSESSION_DIST", 1.25),
        "possession_bonus": _env_float("SHAPING_POSSESSION_BONUS", 0.002),
        "progress_requires_possession": _env_bool("SHAPING_PROGRESS_REQUIRES_POSSESSION", False),
        "deep_zone_outer_threshold": _env_float("SHAPING_DEEP_ZONE_OUTER_THRESHOLD", 0.0),
        "deep_zone_outer_penalty": _env_float("SHAPING_DEEP_ZONE_OUTER_PENALTY", 0.0),
        "deep_zone_inner_threshold": _env_float("SHAPING_DEEP_ZONE_INNER_THRESHOLD", 0.0),
        "deep_zone_inner_penalty": _env_float("SHAPING_DEEP_ZONE_INNER_PENALTY", 0.0),
        "defensive_survival_threshold": _env_float("SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD", 0.0),
        "defensive_survival_bonus": _env_float("SHAPING_DEFENSIVE_SURVIVAL_BONUS", 0.0),
        "fast_loss_threshold_steps": _env_int("SHAPING_FAST_LOSS_THRESHOLD_STEPS", 0),
        "fast_loss_penalty_per_step": _env_float("SHAPING_FAST_LOSS_PENALTY_PER_STEP", 0.0),
        "team_spacing_scale": _env_float("SHAPING_TEAM_SPACING_SCALE", 0.0),
        "team_coverage_scale": _env_float("SHAPING_TEAM_COVERAGE_SCALE", 0.0),
        "team_potential_gamma": _env_float("SHAPING_TEAM_POTENTIAL_GAMMA", 0.99),
        "team_near_ball_threshold": _env_float("SHAPING_TEAM_NEAR_BALL_THRESHOLD", 3.0),
        "team_spacing_min": _env_float("SHAPING_TEAM_SPACING_MIN", 2.0),
        "team_spacing_max": _env_float("SHAPING_TEAM_SPACING_MAX", 6.0),
        "specialist_mode": os.environ.get("SPECIALIST_MODE", "none").strip().lower() or "none",
        "fast_win_threshold": _env_int("FAST_WIN_THRESHOLD", 100),
        "debug_info": debug_info,
    }


def _print_run_header(
    *,
    run_dir,
    resume_path,
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
    scenario_reset=None,
    teammate_checkpoint=None,
):
    est_total_iterations = None
    if timesteps_total > 0 and train_batch_size > 0:
        est_total_iterations = int(math.ceil(float(timesteps_total) / float(train_batch_size)))

    _console_print("")
    _console_print("Training Configuration")
    _console_print(f"  run_dir:              {run_dir}")
    _console_print(f"  resume_checkpoint:    {resume_path or 'None'}")
    _console_print(f"  target_timesteps:     {_format_count(timesteps_total)}")
    _console_print(f"  time_limit:           {_format_seconds(time_total_s) if time_total_s > 0 else 'disabled'}")
    _console_print(f"  max_iterations:       {max_iterations if max_iterations > 0 else 'disabled'}")
    _console_print(f"  workers/envs:         {num_workers} workers x {num_envs_per_worker} env")
    _console_print(f"  learner_gpus:         {num_gpus}")
    _console_print(f"  rollout_fragment:     {rollout_fragment_length}")
    _console_print(f"  train_batch_size:     {train_batch_size} env steps / iteration")
    _console_print(f"  sgd_minibatch_size:   {sgd_minibatch_size}")
    _console_print(f"  ppo_epochs_per_it:    {num_sgd_iter}")
    _console_print(f"  model_hidden:         {fcnet_hiddens}")
    _console_print(f"  opponent baseline p:  {baseline_prob:.2f}")
    _console_print(f"  base_port:            {base_port}")
    if scenario_reset:
        _console_print(f"  scenario_reset:       {scenario_reset}")
    if teammate_checkpoint:
        _console_print(f"  teammate_checkpoint:  {teammate_checkpoint}")
    if est_total_iterations is not None:
        _console_print(f"  estimated_iterations: {est_total_iterations}")
    _console_print("")


def _last_csv_row(progress_csv):
    with progress_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return rows[-1]


def _loss_curve_path(run_root):
    return str(Path(run_root) / "training_loss_curve.png")


def _write_training_loss_curve(run_root):
    return write_training_loss_curve_png(run_root, output_name="training_loss_curve.png")


def _row_float(row, key, default=0.0):
    if row is None:
        return float(default)
    value = row.get(key, default)
    if value in (None, ""):
        return float(default)
    return float(value)


def _row_int(row, key, default=0):
    return int(_row_float(row, key, default))


def _print_progress(row, *, target_timesteps, max_iterations, previous_steps, previous_elapsed):
    it = _row_int(row, "training_iteration", 0)
    steps = _row_int(row, "timesteps_total", 0)
    elapsed = _row_float(row, "time_total_s", 0.0)
    reward_mean = _row_float(row, "episode_reward_mean", 0.0)
    reward_max = _row_float(row, "episode_reward_max", 0.0)
    reward_min = _row_float(row, "episode_reward_min", 0.0)
    episodes_total = _row_int(row, "episodes_total", 0)
    episodes_this_iter = _row_int(row, "episodes_this_iter", 0)
    num_healthy_workers = _row_int(row, "num_healthy_workers", 0)

    sample_time_ms = _row_float(row, "timers/sample_time_ms", 0.0)
    learn_time_ms = _row_float(row, "timers/learn_time_ms", 0.0)
    sample_throughput = _row_float(row, "timers/sample_throughput", 0.0)
    learn_throughput = _row_float(row, "timers/learn_throughput", 0.0)
    sampled_total = _row_float(row, "info/num_steps_sampled", steps)

    iter_steps = steps if previous_steps is None else max(0, steps - previous_steps)
    iter_elapsed = (
        float(row.get("time_this_iter_s") or 0.0)
        if previous_elapsed is None
        else max(0.0, elapsed - previous_elapsed)
    )
    iter_steps_per_s = float(iter_steps) / iter_elapsed if iter_elapsed > 0 else 0.0
    it_per_s = 1.0 / iter_elapsed if iter_elapsed > 0 else 0.0

    step_fraction = None
    eta_seconds = None
    if target_timesteps > 0:
        step_fraction = min(1.0, float(steps) / float(target_timesteps))
        global_steps_per_s = float(steps) / elapsed if elapsed > 0 else 0.0
        if global_steps_per_s > 0 and steps < target_timesteps:
            eta_seconds = (target_timesteps - steps) / global_steps_per_s

    iter_fraction = None
    if max_iterations > 0:
        iter_fraction = min(1.0, float(it) / float(max_iterations))

    fraction = step_fraction if step_fraction is not None else iter_fraction
    bar = _render_bar(fraction)
    pct = 100.0 * fraction if fraction is not None else 0.0

    eta_text = _format_seconds(eta_seconds) if eta_seconds is not None else "--"
    step_goal_text = f"{_format_count(steps)}/{_format_count(target_timesteps)}" if target_timesteps > 0 else f"{_format_count(steps)}/--"
    iter_goal_text = f"{it}/{max_iterations}" if max_iterations > 0 else f"{it}/--"

    _console_print(
        f"[{bar}] {pct:5.1f}% | it {iter_goal_text} | steps {step_goal_text} | "
        f"iter_rate {it_per_s:.4f} it/s | iter_steps {iter_steps_per_s:7.1f}/s | "
        f"sample {sample_throughput:7.1f}/s | learn {learn_throughput:7.1f}/s | "
        f"reward {reward_mean:+.3f} [{reward_min:+.3f}, {reward_max:+.3f}] | "
        f"episodes +{episodes_this_iter}/{episodes_total} | workers {num_healthy_workers} | "
        f"sample_ms {sample_time_ms:8.1f} | learn_ms {learn_time_ms:7.1f} | "
        f"sampled {_format_count(sampled_total)} | eta {eta_text}"
    )


def _find_progress_csv(run_root):
    run_root = Path(run_root)
    if not run_root.exists():
        return None

    candidates = []
    for child in run_root.iterdir():
        if not child.is_dir():
            continue
        progress = child / "progress.csv"
        if progress.exists():
            candidates.append(progress)
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime)
    if candidates:
        return candidates[-1]

    direct_progress = run_root / "progress.csv"
    if direct_progress.exists():
        return direct_progress
    return None


def _parse_eval_opponents(raw):
    raw = (raw or "baseline").strip()
    if not raw:
        return []

    mapping = {
        "baseline": "ceia_baseline_agent",
        "random": "example_player_agent.agent_random",
    }
    opponents = []
    for piece in raw.split(","):
        name = piece.strip()
        if not name:
            continue
        module = mapping.get(name, name)
        label = name if name in mapping else module
        opponents.append((label, module))
    return opponents


def _checkpoint_iteration(checkpoint_dir):
    name = Path(checkpoint_dir).name
    if name.startswith("checkpoint_"):
        try:
            return int(name.split("_", 1)[1])
        except Exception:
            return -1
    return -1


def _find_checkpoint_dirs(run_root):
    run_root = Path(run_root)
    if not run_root.exists():
        return []

    candidates = []
    for child in run_root.iterdir():
        if not child.is_dir():
            continue
        for path in child.glob("checkpoint_*"):
            if path.is_dir():
                candidates.append(path)
    return sorted(candidates, key=lambda path: (_checkpoint_iteration(path), str(path)))


def _load_eval_rows(eval_csv):
    eval_csv = Path(eval_csv)
    if not eval_csv.exists():
        return []
    with eval_csv.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _append_eval_row(eval_csv, row):
    eval_csv = Path(eval_csv)
    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "checkpoint_iteration",
        "checkpoint_dir",
        "checkpoint_file",
        "opponent",
        "episodes",
        "wins",
        "losses",
        "ties",
        "win_rate",
        "non_loss_rate",
        "fast_wins",
        "fast_win_threshold",
        "fast_win_rate",
        "status",
        "log_path",
    ]
    existing_rows = []
    rewrite = False
    if eval_csv.exists():
        with eval_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_header = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_header != fieldnames:
            rewrite = True

    if rewrite:
        with eval_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                normalized = {key: existing_row.get(key, "") for key in fieldnames}
                writer.writerow(normalized)

    with eval_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if handle.tell() == 0:
            writer.writeheader()
        normalized = {key: row.get(key, "") for key in fieldnames}
        writer.writerow(normalized)


def _failed_eval_row(*, checkpoint_dir, checkpoint_iteration, opponent_name, episodes, log_path):
    checkpoint_dir = Path(checkpoint_dir)
    try:
        checkpoint_file = resolve_checkpoint_file(str(checkpoint_dir))
    except Exception:
        checkpoint_file = ""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": checkpoint_file,
        "checkpoint_iteration": int(checkpoint_iteration),
        "opponent": opponent_name,
        "episodes": int(episodes),
        "wins": "",
        "losses": "",
        "ties": "",
        "win_rate": "",
        "status": "failed",
        "log_path": str(log_path),
    }


def _parse_match_summary(output_text):
    summary = {}
    in_summary = False
    for line in output_text.splitlines():
        stripped = line.strip()
        if stripped == "---- Summary ----":
            in_summary = True
            continue
        if not in_summary:
            continue
        match = _EVAL_SUMMARY_PATTERN.match(stripped)
        if match:
            summary[match.group(1)] = match.group(2)
    return summary


def _evaluate_checkpoint_once(
    *,
    checkpoint_dir,
    team0_module,
    opponent_name,
    opponent_module,
    episodes,
    max_steps,
    base_port,
    python_bin,
    log_dir,
    ray_tmp_dir=None,
):
    checkpoint_file = resolve_checkpoint_file(str(checkpoint_dir))
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT if not _PYTHONPATH else REPO_ROOT + os.pathsep + _PYTHONPATH
    env["TRAINED_RAY_CHECKPOINT"] = checkpoint_file
    env.setdefault("RAY_DISABLE_DASHBOARD", "1")
    env.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    if ray_tmp_dir:
        env["RAY_SESSION_TMPDIR_OVERRIDE"] = str(ray_tmp_dir)
        os.makedirs(ray_tmp_dir, exist_ok=True)

    fast_win_threshold = _env_int("FAST_WIN_THRESHOLD", 100)
    cmd = [
        python_bin,
        "-m",
        "cs8803drl.evaluation.evaluate_matches",
        "-m1",
        team0_module,
        "-m2",
        opponent_module,
        "-n",
        str(int(episodes)),
        "--max_steps",
        str(int(max_steps)),
        "--base_port",
        str(int(base_port)),
        "--fast-win-threshold",
        str(int(fast_win_threshold)),
    ]

    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")

    checkpoint_name = Path(checkpoint_dir).name
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{checkpoint_name}_{opponent_name}.log"
    log_path.write_text(output, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(
            f"-m cs8803drl.evaluation.evaluate_matches failed for {checkpoint_name} vs {opponent_name} "
            f"(exit {proc.returncode}). See {log_path}"
        )

    summary = _parse_match_summary(output)
    if not summary:
        raise RuntimeError(
            f"Could not parse evaluation summary for {checkpoint_name} vs {opponent_name}. "
            f"See {log_path}"
        )

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": checkpoint_file,
        "checkpoint_iteration": _checkpoint_iteration(checkpoint_dir),
        "opponent": opponent_name,
        "episodes": int(summary.get("episodes", episodes)),
        "wins": int(summary.get("team0_wins", 0)),
        "losses": int(summary.get("team1_wins", 0)),
        "ties": int(summary.get("ties", 0)),
        "win_rate": float(summary.get("team0_win_rate", 0.0)),
        "non_loss_rate": float(summary.get("team0_non_loss_rate", 0.0)),
        "fast_win_rate": float(summary.get("team0_fast_win_rate", 0.0)),
        "fast_wins": int(summary.get("team0_fast_wins", 0)),
        "fast_win_threshold": int(summary.get("team0_fast_win_threshold", 0)),
        "status": "ok",
        "log_path": str(log_path),
    }


def _select_best_eval(rows):
    grouped = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        checkpoint_dir = row.get("checkpoint_dir")
        if not checkpoint_dir:
            continue
        grouped.setdefault(checkpoint_dir, {})[row.get("opponent")] = row

    metric_key = _resolve_success_metric()    # win_rate | non_loss_rate | fast_win_rate
    best = None
    best_key = None
    for checkpoint_dir, result_map in grouped.items():
        baseline = result_map.get("baseline")
        if baseline is None:
            continue
        random_row = result_map.get("random")
        baseline_rate = float(baseline.get(metric_key, 0.0))
        random_rate = float(random_row.get(metric_key, -1.0)) if random_row is not None else -1.0
        iteration = int(baseline.get("checkpoint_iteration", -1))
        key = (baseline_rate, random_rate, iteration)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_file": baseline.get("checkpoint_file"),
                "checkpoint_iteration": iteration,
                "baseline": baseline,
                "random": random_row,
            }
    return best


def _write_best_eval_summary(run_root, best_eval):
    run_root = Path(run_root)
    if not best_eval:
        return

    txt_path = run_root / "best_checkpoint_by_eval.txt"
    lines = [
        f"checkpoint_dir={best_eval['checkpoint_dir']}",
        f"checkpoint_file={best_eval['checkpoint_file']}",
        f"checkpoint_iteration={best_eval['checkpoint_iteration']}",
    ]

    baseline = best_eval.get("baseline") or {}
    lines.append(
        "baseline_win_rate="
        f"{float(baseline.get('win_rate', 0.0)):.3f}"
    )
    lines.append(
        "baseline_record="
        f"{baseline.get('wins', '?')}W-{baseline.get('losses', '?')}L-{baseline.get('ties', '?')}T"
    )

    random_row = best_eval.get("random") or {}
    if random_row:
        lines.append(
            "random_win_rate="
            f"{float(random_row.get('win_rate', 0.0)):.3f}"
        )
        lines.append(
            "random_record="
            f"{random_row.get('wins', '?')}W-{random_row.get('losses', '?')}L-{random_row.get('ties', '?')}T"
        )

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_EVAL_PORT_STEP = 20
_EVAL_MAX_BASE_PORT = 65505


def _safe_eval_base_port(base_port, offset=0):
    base_port = int(base_port)
    offset = int(offset)
    if base_port < 1024 or base_port > _EVAL_MAX_BASE_PORT:
        raise ValueError(
            f"EVAL_BASE_PORT must be in [1024, {_EVAL_MAX_BASE_PORT}], got {base_port}"
        )

    span = ((_EVAL_MAX_BASE_PORT - base_port) // _EVAL_PORT_STEP) + 1
    return base_port + (offset % span) * _EVAL_PORT_STEP


def _monitor_checkpoint_evaluations(run_root, stop_event, *, state, poll_s=5.0):
    eval_interval = _env_int("EVAL_INTERVAL", 0)
    if eval_interval <= 0:
        return

    opponents = _parse_eval_opponents(os.environ.get("EVAL_OPPONENTS", "baseline"))
    if not opponents:
        return

    eval_episodes = _env_int("EVAL_EPISODES", 20)
    eval_max_steps = _env_int("EVAL_MAX_STEPS", 1500)
    eval_max_retries = max(0, _env_int("EVAL_MAX_RETRIES_PER_MATCHUP", 3))
    python_bin = os.environ.get("EVAL_PYTHON_BIN", sys.executable).strip() or sys.executable
    team0_module = os.environ.get("EVAL_TEAM0_MODULE", "cs8803drl.deployment.trained_ray_agent").strip() or "cs8803drl.deployment.trained_ray_agent"
    eval_csv = Path(run_root) / "checkpoint_eval.csv"
    eval_log_dir = Path(run_root) / "checkpoint_eval_logs"

    rows = _load_eval_rows(eval_csv)
    seen = {
        (row.get("checkpoint_dir"), row.get("opponent"))
        for row in rows
        if row.get("checkpoint_dir") and row.get("opponent") and row.get("status") in {"ok", "failed"}
    }
    best_eval = _select_best_eval(rows)
    if best_eval is not None:
        state["best_eval"] = best_eval
        _write_best_eval_summary(run_root, best_eval)

    configured_eval_base_port = _env_int("EVAL_BASE_PORT", 7005)
    next_port_offset = 0
    logged_failures = set()
    failure_counts = {}

    while True:
        checkpoint_dirs = _find_checkpoint_dirs(run_root)
        pending = []
        final_iteration = _checkpoint_iteration(checkpoint_dirs[-1]) if checkpoint_dirs else -1
        force_final = stop_event.is_set()

        for checkpoint_dir in checkpoint_dirs:
            iteration = _checkpoint_iteration(checkpoint_dir)
            should_eval = (iteration > 0 and iteration % eval_interval == 0) or (
                force_final and iteration == final_iteration
            )
            if not should_eval:
                continue
            for opponent_name, opponent_module in opponents:
                key = (str(checkpoint_dir), opponent_name)
                if key not in seen:
                    pending.append((checkpoint_dir, iteration, opponent_name, opponent_module))

        if not pending and stop_event.is_set():
            break

        if not pending:
            stop_event.wait(poll_s)
            continue

        checkpoint_dir, iteration, opponent_name, opponent_module = pending[0]
        try:
            result = _evaluate_checkpoint_once(
                checkpoint_dir=checkpoint_dir,
                team0_module=team0_module,
                opponent_name=opponent_name,
                opponent_module=opponent_module,
                episodes=eval_episodes,
                max_steps=eval_max_steps,
                base_port=_safe_eval_base_port(configured_eval_base_port, next_port_offset),
                python_bin=python_bin,
                log_dir=eval_log_dir,
            )
            next_port_offset += 1
            result["timestamp"] = datetime.now().isoformat(timespec="seconds")
            _append_eval_row(eval_csv, result)
            rows.append(result)
            seen.add((str(checkpoint_dir), opponent_name))

            best_before = state.get("best_eval")
            best_eval = _select_best_eval(rows)
            state["best_eval"] = best_eval
            if best_eval is not None:
                _write_best_eval_summary(run_root, best_eval)

            suffix = ""
            if best_eval is not None and (
                best_before is None
                or best_eval.get("checkpoint_dir") != best_before.get("checkpoint_dir")
                or float(best_eval.get("baseline", {}).get("win_rate", 0.0))
                != float(best_before.get("baseline", {}).get("win_rate", 0.0))
            ):
                if best_eval.get("checkpoint_dir") == str(checkpoint_dir):
                    suffix = " | NEW_BEST"

            _console_print(
                f"[checkpoint-eval] it {iteration} | {opponent_name} "
                f"{result['wins']}W-{result['losses']}L-{result['ties']}T "
                f"(win_rate={result['win_rate']:.3f}){suffix}"
            )
        except Exception as exc:
            key = (str(checkpoint_dir), opponent_name)
            failure_counts[key] = failure_counts.get(key, 0) + 1
            attempts = failure_counts[key]
            checkpoint_name = Path(checkpoint_dir).name
            log_path = eval_log_dir / f"{checkpoint_name}_{opponent_name}.log"

            if attempts <= eval_max_retries:
                if key not in logged_failures:
                    _console_print(
                        f"[checkpoint-eval] pending retry for {checkpoint_name} vs {opponent_name}: "
                        f"{exc} (attempt {attempts}/{eval_max_retries})"
                    )
                    logged_failures.add(key)
            else:
                failed_row = _failed_eval_row(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_iteration=iteration,
                    opponent_name=opponent_name,
                    episodes=eval_episodes,
                    log_path=log_path,
                )
                _append_eval_row(eval_csv, failed_row)
                rows.append(failed_row)
                seen.add(key)
                _console_print(
                    f"[checkpoint-eval] marked failed after {attempts - 1} retries for "
                    f"{checkpoint_name} vs {opponent_name}: {exc}"
                )
            stop_event.wait(poll_s)


def _monitor_progress(run_root, stop_event, *, target_timesteps, max_iterations, state, poll_s=2.0):
    previous_steps = None
    previous_elapsed = None
    last_printed_iteration = -1

    while not stop_event.is_set():
        progress_csv = _find_progress_csv(run_root)
        if progress_csv is None:
            stop_event.wait(poll_s)
            continue

        try:
            row = _last_csv_row(progress_csv)
        except Exception:
            stop_event.wait(poll_s)
            continue

        if not row:
            stop_event.wait(poll_s)
            continue

        current_iteration = _row_int(row, "training_iteration", 0)
        if current_iteration > last_printed_iteration:
            _print_progress(
                row,
                target_timesteps=target_timesteps,
                max_iterations=max_iterations,
                previous_steps=previous_steps,
                previous_elapsed=previous_elapsed,
            )
            previous_steps = _row_int(row, "timesteps_total", 0)
            previous_elapsed = _row_float(row, "time_total_s", 0.0)
            last_printed_iteration = current_iteration
            state["last_printed_iteration"] = current_iteration

        stop_event.wait(poll_s)


class BaselineEvalCallbacks(DefaultCallbacks):
    @staticmethod
    def _extract_score_from_info(info):
        return extract_score_from_info(info)

    @staticmethod
    def _normalize_single_player_action(action):
        # soccer_twos with flatten_branched=True expects a hashable scalar (Discrete).
        a = action
        for _ in range(3):
            if isinstance(a, (np.ndarray,)):
                if a.shape == ():
                    a = a.item()
                    continue
                if a.size == 1:
                    a = a.reshape(()).item()
                    continue
            if isinstance(a, (list, tuple)):
                if len(a) == 1:
                    a = a[0]
                    continue
            break
        if isinstance(a, (np.generic,)):
            a = a.item()
        if isinstance(a, bool):
            return int(a)
        if isinstance(a, (int, float)):
            return int(a)
        return a

    @staticmethod
    def _normalize_branched_action(action):
        a = action
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if isinstance(a, (list, tuple)):
            out = []
            for x in a:
                if isinstance(x, np.generic):
                    x = x.item()
                out.append(int(x))
            return out
        if isinstance(a, (int, float, np.generic, bool)):
            return [int(a)]
        return a

    @staticmethod
    def _find_action_lookup(env):
        cur = env
        for _ in range(15):
            if hasattr(cur, "_flattener") and hasattr(cur._flattener, "action_lookup"):
                return cur._flattener.action_lookup
            if not hasattr(cur, "env"):
                break
            cur = cur.env
        return None

    @staticmethod
    def _coerce_to_discrete_action(action, action_lookup):
        # Convert policy output to a discrete scalar action compatible with
        # soccer_twos wrapper's ActionFlattener.lookup_action().
        a = action

        # RLlib often returns (action, state_out, info) from compute_single_action.
        if isinstance(a, tuple) and len(a) >= 1:
            a = a[0]

        # Fast-path: scalar-ish.
        a_norm = BaselineEvalCallbacks._normalize_single_player_action(a)
        if isinstance(a_norm, int):
            return a_norm

        # If model outputs a branched vector (list/ndarray), map it to the
        # flattened discrete id using the env's action_lookup inverse.
        if action_lookup is None:
            raise ValueError("Could not locate env action_lookup for action conversion")

        def _unwrap(x):
            cur = x
            for _ in range(5):
                if isinstance(cur, np.ndarray):
                    cur = cur.tolist()
                    continue
                if isinstance(cur, (list, tuple)) and len(cur) == 1:
                    cur = cur[0]
                    continue
                # Sometimes we still have (action, state_out, info) nested.
                if isinstance(cur, tuple) and len(cur) >= 1:
                    cur = cur[0]
                    continue
                break
            return cur

        a_norm = _unwrap(a_norm)
        if isinstance(a_norm, (np.generic,)):
            a_norm = a_norm.item()

        if isinstance(a_norm, (int, float, bool)):
            return int(a_norm)

        if not isinstance(a_norm, (list, tuple)):
            raise ValueError(f"Unsupported action type for discrete conversion: {type(a_norm)}")

        flat = []
        for x in a_norm:
            x = _unwrap(x)
            if isinstance(x, dict):
                raise ValueError(f"Unsupported dict element in action vector: keys={list(x.keys())[:5]}")
            if isinstance(x, (list, tuple)):
                # Flatten one additional nesting level.
                for y in x:
                    y = _unwrap(y)
                    if isinstance(y, np.generic):
                        y = y.item()
                    if isinstance(y, dict):
                        raise ValueError(f"Unsupported dict element in action vector: keys={list(y.keys())[:5]}")
                    flat.append(int(y))
            else:
                if isinstance(x, np.generic):
                    x = x.item()
                flat.append(int(x))

        key = tuple(flat)
        # Build inverse map lazily.
        inv = {tuple(v): k for k, v in action_lookup.items()}
        if key not in inv:
            raise ValueError(f"Action vector {key} not found in action_lookup")
        return int(inv[key])

    @staticmethod
    def _extract_winner_from_info(info):
        return extract_winner_from_info(info)

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        eval_interval = int(os.environ.get("EVAL_INTERVAL", "0"))
        if eval_interval <= 0:
            return

        it = int(result.get("training_iteration") or 0)
        if it % eval_interval != 0:
            return

        eval_episodes = int(os.environ.get("EVAL_EPISODES", "10"))
        eval_base_port = int(os.environ.get("EVAL_BASE_PORT", "7005"))
        eval_max_steps = int(os.environ.get("EVAL_MAX_STEPS", "1500"))

        policy = trainer.get_policy("default_policy") or trainer.get_policy("default")
        if policy is None:
            return

        # Manual evaluation to avoid RLlib evaluation workers syncing optimizer state.
        # Use the same flatten_branched=True setting as training to ensure obs shapes
        # match the policy model input.
        eval_env = create_rllib_env(
            {
                "variation": EnvType.team_vs_policy,
                "multiagent": False,
                "single_player": True,
                "flatten_branched": True,
                "reward_shaping": False,
                "opponent_mix": {"baseline_prob": 1.0},
                "base_port": eval_base_port,
            }
        )
        action_lookup = self._find_action_lookup(eval_env)
        wins = 0
        ties = 0
        total = 0

        try:
            for _ in range(max(eval_episodes, 0)):
                obs = eval_env.reset()
                ep_reward = 0.0
                last_info = None
                for _t in range(eval_max_steps):
                    act = policy.compute_single_action(obs, explore=False)
                    if isinstance(act, tuple) and len(act) >= 1:
                        act = act[0]
                    act = self._coerce_to_discrete_action(act, action_lookup)
                    obs, r, done, info = eval_env.step(act)
                    ep_reward += float(r)
                    last_info = info
                    if bool(done):
                        break

                score = self._extract_score_from_info(last_info) if isinstance(last_info, dict) else None
                winner = self._extract_winner_from_info(last_info) if isinstance(last_info, dict) else None

                if score is not None:
                    s0, s1 = score
                    if s0 > s1:
                        wins += 1
                    elif s0 == s1:
                        ties += 1
                elif winner is not None:
                    if winner == 0:
                        wins += 1
                else:
                    # Fallback: sparse env reward in Soccer-Twos is typically win/loss signal.
                    if ep_reward > 0:
                        wins += 1
                    elif ep_reward == 0:
                        ties += 1

                total += 1
        finally:
            try:
                eval_env.close()
            except Exception:
                pass

        if total <= 0:
            return

        cm = result.setdefault("custom_metrics", {})
        cm["win_vs_baseline"] = float(wins) / float(total)
        cm["tie_vs_baseline"] = float(ties) / float(total)
        cm["eval_episodes_vs_baseline"] = float(total)


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
    simple_optimizer = _env_bool("SIMPLE_OPTIMIZER", True)
    log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).strip().upper() or DEFAULT_LOG_LEVEL
    log_sys_usage = _env_bool("LOG_SYS_USAGE", True)
    fcnet_hiddens = _env_layers("FCNET_HIDDENS", DEFAULT_FCNET_HIDDENS)
    fcnet_activation = os.environ.get("FCNET_ACTIVATION", "relu").strip() or "relu"
    use_reward_shaping = _env_bool("USE_REWARD_SHAPING", True)

    warmstart_path = os.environ.get("WARMSTART_CHECKPOINT", "").strip() or None
    resume_path = _resolve_resume_checkpoint_env() or None
    if warmstart_path and not _allow_resume_with_warmstart(False):
        resume_path = None
    if resume_path:
        resume_path = sanitize_checkpoint_for_restore(resume_path)

    timesteps_total = _env_int("TIMESTEPS_TOTAL", DEFAULT_TIMESTEPS_TOTAL)
    time_total_s = _env_int("TIME_TOTAL_S", DEFAULT_TIME_TOTAL_S)
    max_iterations = _env_int("MAX_ITERATIONS", 0)
    run_name = os.environ.get("RUN_NAME", DEFAULT_RUN_NAME).strip() or DEFAULT_RUN_NAME
    checkpoint_freq = _env_int("CHECKPOINT_FREQ", DEFAULT_CHECKPOINT_FREQ)
    base_port = _env_int("BASE_PORT", DEFAULT_BASE_PORT)
    baseline_prob = _env_float("BASELINE_PROB", DEFAULT_BASELINE_PROB)
    scenario_reset = os.environ.get("SCENARIO_RESET", "").strip() or None
    teammate_checkpoint = os.environ.get("TEAMMATE_CHECKPOINT", "").strip() or None
    local_dir = os.environ.get("LOCAL_DIR", "./ray_results").strip() or "./ray_results"

    ray.init(include_dashboard=False)
    tune.registry.register_env("Soccer", create_rllib_env)

    base_env_config = {
        "num_envs_per_worker": num_envs_per_worker,
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": True,
        "flatten_branched": True,
    }
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
            **base_env_config,
            "reward_shaping": reward_shaping,
            "opponent_mix": {"baseline_prob": baseline_prob},
            "base_port": base_port,
            "scenario_reset": scenario_reset,
            "teammate_checkpoint": teammate_checkpoint,
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
        resume_path=resume_path,
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
        scenario_reset=scenario_reset,
        teammate_checkpoint=teammate_checkpoint,
    )
    if warmstart_path:
        _console_print(f"  warmstart_checkpoint: {warmstart_path}")
        if not resume_path:
            _console_print("  resume_checkpoint:   disabled (warm-start mode)")
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
            WarmstartPPOTrainer,
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
        eval_thread.join()
        ray.shutdown()

    progress_summary = summarize_training_progress(run_dir)
    if progress_summary:
        best_row = progress_summary.get("best_row")
        final_row = progress_summary.get("final_row")
        if best_row is not None:
            best_reward = _row_float(best_row, "episode_reward_mean", best_reward if best_reward is not None else 0.0)
            best_iteration = _row_int(best_row, "training_iteration", best_iteration)
        if final_row is not None and final_checkpoint in (None, "unavailable"):
            final_iteration = _row_int(final_row, "training_iteration", 0)
            if final_iteration > 0 and best_checkpoint is None:
                best_checkpoint = None

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
        success_metric = _resolve_success_metric()
        _console_print(
            "  best_eval_baseline:  "
            f"{float(baseline.get('win_rate', 0.0)):.3f} "
            f"({baseline.get('wins', '?')}W-{baseline.get('losses', '?')}L-{baseline.get('ties', '?')}T) "
            f"@ iteration {best_eval.get('checkpoint_iteration', '?')}"
        )
        if success_metric != "win_rate":
            specialist_val = float(baseline.get(success_metric, 0.0))
            extra_info = ""
            if success_metric == "fast_win_rate":
                extra_info = (
                    f" [fast_wins={baseline.get('fast_wins', '?')}, "
                    f"threshold={baseline.get('fast_win_threshold', '?')}]"
                )
            _console_print(
                f"  best_eval_baseline_{success_metric}: {specialist_val:.3f}{extra_info}"
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

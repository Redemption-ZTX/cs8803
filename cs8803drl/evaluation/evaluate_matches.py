import argparse
import importlib
import inspect
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soccer_twos
import gym

try:
    from gym_unity.envs import ActionFlattener
except Exception:  # pragma: no cover
    ActionFlattener = None

try:
    from mlagents_envs.exception import UnityWorkerInUseException
except Exception:  # pragma: no cover
    UnityWorkerInUseException = None


if not hasattr(np, "bool"):
    np.bool = bool

from cs8803drl.core.utils import RewardShapingWrapper
from cs8803drl.evaluation.failure_cases import (
    EpisodeFailureRecorder,
    infer_team0_outcome,
    save_episode_record,
)


def _find_agent_class(module) -> type:
    candidates = []
    for _, obj in inspect.getmembers(module):
        if not inspect.isclass(obj):
            continue
        if hasattr(obj, "act"):
            candidates.append(obj)

    if not candidates:
        raise ValueError(
            f"No agent class with an act() method found in module {module.__name__}"
        )

    # Prefer the first concrete class (not AgentInterface itself)
    for cls in candidates:
        if cls.__name__ in {"AgentInterface", "BaseAgent"}:
            continue
        return cls

    return candidates[0]


def load_agent(module_name: str, env) -> Any:
    module = importlib.import_module(module_name)

    # Common convention: module exports Agent or AgentClass
    for attr in ("Agent", "agent", "AgentClass"):
        if hasattr(module, attr) and inspect.isclass(getattr(module, attr)):
            return getattr(module, attr)(env)

    cls = _find_agent_class(module)
    return cls(env)


def _team_local_obs(obs: Dict[int, Any], global_ids) -> Dict[int, Any]:
    """Convert global env ids into team-local ids expected by starter evaluators."""
    return {local_id: obs[global_id] for local_id, global_id in enumerate(global_ids)}


def _remap_team_actions(action: Dict[int, Any], global_ids) -> Dict[int, Any]:
    """Map team-local action keys {0,1} back to env-global ids."""
    global_ids = tuple(int(i) for i in global_ids)
    local_to_global = {local_id: global_id for local_id, global_id in enumerate(global_ids)}
    out = {}
    for key, value in action.items():
        try:
            key_int = int(key)
        except Exception:
            key_int = key

        if key_int in local_to_global:
            out[local_to_global[key_int]] = value
        elif key_int in global_ids:
            out[key_int] = value
        else:
            out[key] = value
    return out


def accumulate_team_reward(team_reward: float, reward: Any, team_ids) -> float:
    if isinstance(reward, dict):
        for i in team_ids:
            if i in reward:
                team_reward += float(reward[i])
        return team_reward

    # list/tuple/np.ndarray
    try:
        for i in team_ids:
            team_reward += float(reward[i])
    except Exception:
        pass
    return team_reward


def _team_reward(reward: Any, team_ids) -> float:
    return accumulate_team_reward(0.0, reward, team_ids)


def _step_stats(values: List[int]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    ordered = sorted(int(v) for v in values)
    idx_75 = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.75)))
    return {
        "count": int(len(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(statistics.median(ordered)),
        "p75": float(ordered[idx_75]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _print_step_stats(label: str, values: List[int]) -> None:
    stats = _step_stats(values)
    if stats is None:
        return
    print(
        f"{label}: "
        f"mean={stats['mean']:.1f} median={stats['median']:.1f} "
        f"p75={stats['p75']:.1f} min={stats['min']:.0f} max={stats['max']:.0f}"
    )


def episode_done(done: Any) -> bool:
    if isinstance(done, dict):
        return bool(max(done.values())) if done else False
    return bool(done)


def normalize_actions(action: Dict[int, Any]) -> Dict[int, Any]:
    out = {}
    for pid, a in action.items():
        a = np.asarray(a) if isinstance(a, (np.ndarray, np.generic, list, tuple)) else a

        if isinstance(a, np.ndarray):
            if a.size == 1:
                out[pid] = int(a.reshape(-1)[0])
            else:
                out[pid] = [int(x) for x in a.reshape(-1).tolist()]
        elif isinstance(a, (np.integer,)):
            out[pid] = int(a)
        else:
            out[pid] = a
    return out


def adapt_actions_to_env(action: Dict[int, Any], env: gym.Env) -> Dict[int, Any]:
    action = normalize_actions(action)

    space = getattr(env, "action_space", None)
    if space is None:
        return action

    # If env expects branched actions (MultiDiscrete), convert scalar discrete actions into
    # branched vectors using ActionFlattener.
    if isinstance(space, gym.spaces.MultiDiscrete) and ActionFlattener is not None:
        flattener = ActionFlattener(space.nvec)
        out = {}
        for pid, a in action.items():
            if isinstance(a, (int, np.integer)):
                out[pid] = flattener.lookup_action(int(a))
            else:
                out[pid] = a
        return out

    return action


def make_env_with_retry(*, base_port: int, max_tries: int = 20, port_step: int = 10):
    last_err = None
    for i in range(max_tries):
        port = int(base_port) + i * int(port_step)
        try:
            return (
                soccer_twos.make(
                    render=False,
                    base_port=port,
                ),
                port,
            )
        except Exception as e:
            last_err = e
            if UnityWorkerInUseException is not None and isinstance(
                e, UnityWorkerInUseException
            ):
                continue
            msg = str(e)
            if "Address already in use" in msg or "still in use" in msg:
                continue
            raise

    raise RuntimeError(
        f"Could not create Soccer-Twos env after {max_tries} tries starting from base_port={base_port}. Last error: {last_err}"
    )


def _parse_bool_arg(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--team0_module",
        "-m1",
        required=True,
        help="Python module for team0 agent (e.g., example_player_agent)",
    )
    parser.add_argument(
        "--team1_module",
        "-m2",
        required=True,
        help="Python module for team1 agent (e.g., ceia_baseline_agent)",
    )
    parser.add_argument("--episodes", "-n", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--base_port", type=int, default=8500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-episodes-dir",
        default="",
        help="Optional directory to store structured episode traces as JSON.",
    )
    parser.add_argument(
        "--save-mode",
        choices=("losses", "nonwins", "all"),
        default="losses",
        help="Which episodes to save when --save-episodes-dir is set. Default: losses",
    )
    parser.add_argument(
        "--max-saved-episodes",
        type=int,
        default=50,
        help="Maximum number of episode JSON files to save. Default: 50",
    )
    parser.add_argument(
        "--trace-stride",
        type=int,
        default=5,
        help="Keep one sampled trace step every N env steps. Default: 5",
    )
    parser.add_argument(
        "--trace-tail-steps",
        type=int,
        default=40,
        help="Always keep the last N step summaries for each saved episode. Default: 40",
    )
    parser.add_argument(
        "--reward-shaping-debug",
        action="store_true",
        help="Wrap the eval env with RewardShapingWrapper(debug_info=True) so shaping diagnostics are logged.",
    )
    parser.add_argument("--time-penalty", type=float, default=0.001)
    parser.add_argument("--ball-progress-scale", type=float, default=0.01)
    parser.add_argument("--goal-proximity-scale", type=float, default=0.0)
    parser.add_argument("--goal-proximity-gamma", type=float, default=0.99)
    parser.add_argument("--goal-center-x", type=float, default=15.0)
    parser.add_argument("--goal-center-y", type=float, default=0.0)
    parser.add_argument("--opponent-progress-penalty-scale", type=float, default=0.0)
    parser.add_argument("--possession-dist", type=float, default=1.25)
    parser.add_argument("--possession-bonus", type=float, default=0.002)
    parser.add_argument(
        "--progress-requires-possession",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool_arg,
    )
    parser.add_argument("--deep-zone-outer-threshold", type=float, default=0.0)
    parser.add_argument("--deep-zone-outer-penalty", type=float, default=0.0)
    parser.add_argument("--deep-zone-inner-threshold", type=float, default=0.0)
    parser.add_argument("--deep-zone-inner-penalty", type=float, default=0.0)
    parser.add_argument("--defensive-survival-threshold", type=float, default=0.0)
    parser.add_argument("--defensive-survival-bonus", type=float, default=0.0)
    parser.add_argument("--fast-loss-threshold-steps", type=int, default=0)
    parser.add_argument("--fast-loss-penalty-per-step", type=float, default=0.0)
    parser.add_argument("--event-shot-reward", type=float, default=0.0)
    parser.add_argument("--event-tackle-reward", type=float, default=0.0)
    parser.add_argument("--event-clearance-reward", type=float, default=0.0)
    parser.add_argument("--event-cooldown-steps", type=int, default=10)
    args = parser.parse_args()

    env, used_port = make_env_with_retry(base_port=args.base_port)
    if used_port != args.base_port:
        print(f"[evaluate_matches] base_port {args.base_port} in use; using base_port={used_port}")

    if args.reward_shaping_debug:
        env = RewardShapingWrapper(
            env,
            time_penalty=float(args.time_penalty),
            ball_progress_scale=float(args.ball_progress_scale),
            goal_proximity_scale=float(args.goal_proximity_scale),
            goal_proximity_gamma=float(args.goal_proximity_gamma),
            goal_center_x=float(args.goal_center_x),
            goal_center_y=float(args.goal_center_y),
            event_shot_reward=float(args.event_shot_reward),
            event_tackle_reward=float(args.event_tackle_reward),
            event_clearance_reward=float(args.event_clearance_reward),
            event_cooldown_steps=int(args.event_cooldown_steps),
            opponent_progress_penalty_scale=float(args.opponent_progress_penalty_scale),
            possession_dist=float(args.possession_dist),
            possession_bonus=float(args.possession_bonus),
            progress_requires_possession=bool(args.progress_requires_possession),
            deep_zone_outer_threshold=float(args.deep_zone_outer_threshold),
            deep_zone_outer_penalty=float(args.deep_zone_outer_penalty),
            deep_zone_inner_threshold=float(args.deep_zone_inner_threshold),
            deep_zone_inner_penalty=float(args.deep_zone_inner_penalty),
            defensive_survival_threshold=float(args.defensive_survival_threshold),
            defensive_survival_bonus=float(args.defensive_survival_bonus),
            fast_loss_threshold_steps=int(args.fast_loss_threshold_steps),
            fast_loss_penalty_per_step=float(args.fast_loss_penalty_per_step),
            debug_info=True,
        )

    agent0 = load_agent(args.team0_module, env)
    agent1 = load_agent(args.team1_module, env)

    wins0 = 0
    wins1 = 0
    ties = 0
    saved_episode_paths = []
    save_dir = Path(args.save_episodes_dir).resolve() if args.save_episodes_dir else None
    all_steps: List[int] = []
    team0_win_steps: List[int] = []
    team1_win_steps: List[int] = []
    tie_steps: List[int] = []

    for ep in range(args.episodes):
        obs = env.reset()
        if ep % 2 == 0:
            team0_ids = (0, 1)
            team1_ids = (2, 3)
        else:
            team0_ids = (2, 3)
            team1_ids = (0, 1)

        team0_reward_total = 0.0
        team1_reward_total = 0.0
        final_team0_step_reward = 0.0
        final_team1_step_reward = 0.0
        recorder = EpisodeFailureRecorder(
            trace_stride=int(args.trace_stride),
            tail_steps=int(args.trace_tail_steps),
        )
        total_steps = 0
        info: Any = {}

        for step_idx in range(args.max_steps):
            obs0 = _team_local_obs(obs, team0_ids)
            obs1 = _team_local_obs(obs, team1_ids)
            act0 = _remap_team_actions(agent0.act(obs0), team0_ids)
            act1 = _remap_team_actions(agent1.act(obs1), team1_ids)

            act0 = adapt_actions_to_env(act0, env)
            act1 = adapt_actions_to_env(act1, env)

            action = {}
            action.update(act0)
            action.update(act1)

            obs, reward, done, info = env.step(action)
            final_team0_step_reward = _team_reward(reward, team0_ids)
            final_team1_step_reward = _team_reward(reward, team1_ids)
            team0_reward_total += final_team0_step_reward
            team1_reward_total += final_team1_step_reward
            recorder.record_step(
                step_index=step_idx,
                reward=reward,
                info=info,
                team0_ids=team0_ids,
                team1_ids=team1_ids,
            )
            total_steps = step_idx + 1

            if episode_done(done):
                break

        outcome, final_score, final_winner = infer_team0_outcome(
            final_info=info if isinstance(info, dict) else {},
            final_team0_step_reward=final_team0_step_reward,
            final_team1_step_reward=final_team1_step_reward,
            cumulative_team0_reward=team0_reward_total,
            cumulative_team1_reward=team1_reward_total,
        )

        if outcome == "team0_win":
            wins0 += 1
            outcome = "team0_win"
            team0_win_steps.append(int(total_steps))
        elif outcome == "team1_win":
            wins1 += 1
            outcome = "team1_win"
            team1_win_steps.append(int(total_steps))
        else:
            ties += 1
            outcome = "tie"
            tie_steps.append(int(total_steps))
        all_steps.append(int(total_steps))

        should_save = False
        if save_dir is not None and len(saved_episode_paths) < int(args.max_saved_episodes):
            if args.save_mode == "all":
                should_save = True
            elif args.save_mode == "nonwins":
                should_save = outcome != "team0_win"
            elif args.save_mode == "losses":
                should_save = outcome == "team1_win"

        if should_save:
            record = recorder.build_episode_record(
                episode_index=ep,
                team0_module=args.team0_module,
                team1_module=args.team1_module,
                outcome=outcome,
                final_score=final_score,
                final_winner=final_winner,
                cumulative_team0_reward=team0_reward_total,
                cumulative_team1_reward=team1_reward_total,
                final_team0_step_reward=final_team0_step_reward,
                final_team1_step_reward=final_team1_step_reward,
                total_steps=total_steps,
            )
            saved_episode_paths.append(save_episode_record(record, save_dir))

        print(
            f"Episode {ep}: {outcome} | "
            f"team0_reward={team0_reward_total:.4f} team1_reward={team1_reward_total:.4f}"
        )

    print("---- Summary ----")
    print(f"team0_module: {args.team0_module}")
    print(f"team1_module: {args.team1_module}")
    print(f"episodes: {args.episodes}")
    print(f"team0_wins: {wins0}")
    print(f"team1_wins: {wins1}")
    print(f"ties: {ties}")
    print(f"team0_win_rate: {wins0 / max(args.episodes, 1):.3f}")
    _print_step_stats("episode_steps_all", all_steps)
    _print_step_stats("episode_steps_team0_win", team0_win_steps)
    _print_step_stats("episode_steps_team1_win", team1_win_steps)
    _print_step_stats("episode_steps_tie", tie_steps)
    if save_dir is not None:
        print(f"saved_episodes_dir: {save_dir}")
        print(f"saved_episode_count: {len(saved_episode_paths)}")
        summary_path = save_dir / "summary.json"
        summary_payload = {
            "team0_module": args.team0_module,
            "team1_module": args.team1_module,
            "episodes": int(args.episodes),
            "team0_wins": int(wins0),
            "team1_wins": int(wins1),
            "ties": int(ties),
            "team0_win_rate": float(wins0 / max(args.episodes, 1)),
            "step_stats": {
                "all": _step_stats(all_steps),
                "team0_win": _step_stats(team0_win_steps),
                "team1_win": _step_stats(team1_win_steps),
                "tie": _step_stats(tie_steps),
            },
            "saved_episodes_dir": str(save_dir),
            "saved_episode_count": int(len(saved_episode_paths)),
            "saved_episode_paths": [str(path) for path in saved_episode_paths],
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2))
        print(f"summary_json: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()

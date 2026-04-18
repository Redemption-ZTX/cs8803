#!/usr/bin/env python
"""Local diagnostic eval for teammate-obs checkpoints using true env info.

This is intentionally not official-eval aligned. It exists to answer the
021b question: if training and evaluation both use the same true teammate/time
semantics, does the policy recover non-degenerate behavior?
"""

from __future__ import annotations

import argparse
import os
import pickle
import statistics
import sys
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import get_trainable_cls

from cs8803drl.branches.shared_central_critic import (
    SHARED_CC_POLICY_ID,
    build_cc_obs_space,
    register_shared_cc_model,
)
from cs8803drl.core.checkpoint_utils import infer_action_dim_from_checkpoint, load_policy_weights
from cs8803drl.core.obs_teammate import (
    augment_observation_dict_with_teammate_state,
    build_teammate_obs_space,
    extract_player_state_by_agent,
    parse_teammate_state_scale,
    normalized_episode_time,
)
from cs8803drl.core.soccer_info import (
    compute_shaping_components,
    extract_ball_position,
    extract_score_from_info,
    extract_winner_from_info,
)
from cs8803drl.deployment.trained_shared_cc_agent import (
    ALGORITHM,
    _DummyMultiAgentEnv,
    _coerce_int_action,
    _unflatten_discrete_to_multidiscrete,
    _unwrap_action,
)
from cs8803drl.evaluation.failure_cases import EpisodeFailureRecorder, save_episode_record
from cs8803drl.evaluation.evaluate_matches import (
    _remap_team_actions,
    _team_local_obs,
    adapt_actions_to_env,
    episode_done,
    load_agent,
    make_env_with_retry,
)

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:
    try:
        from gym_unity.envs import ActionFlattener
    except Exception:
        ActionFlattener = None


_OPPONENT_ALIAS = {
    "baseline": "ceia_baseline_agent",
    "random": "example_player_agent.agent_random",
}


def _resolve_opponent_module(value: str) -> str:
    key = str(value or "baseline").strip()
    if not key:
        key = "baseline"
    return _OPPONENT_ALIAS.get(key, key)


def _step_stats(values: Iterable[int]) -> Optional[Dict[str, float]]:
    values = [int(v) for v in values]
    if not values:
        return None
    ordered = sorted(values)
    idx_75 = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.75)))
    return {
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(statistics.median(ordered)),
        "p75": float(ordered[idx_75]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _print_step_stats(label: str, values: Iterable[int]) -> None:
    stats = _step_stats(values)
    if stats is None:
        return
    print(
        f"{label}: mean={stats['mean']:.1f} median={stats['median']:.1f} "
        f"p75={stats['p75']:.1f} min={stats['min']:.0f} max={stats['max']:.0f}"
    )


def _infer_eval_outcome(
    *,
    eval_team_on_blue: bool,
    final_info: Any,
    cumulative_blue_reward: float,
    cumulative_orange_reward: float,
    final_blue_step_reward: float,
    final_orange_step_reward: float,
) -> str:
    score = extract_score_from_info(final_info)
    if score is not None:
        blue_score = float(score[0])
        orange_score = float(score[1])
        eval_score, opp_score = (
            (blue_score, orange_score) if eval_team_on_blue else (orange_score, blue_score)
        )
        if eval_score > opp_score:
            return "team0_win"
        if opp_score > eval_score:
            return "team1_win"
        return "tie"

    winner = extract_winner_from_info(final_info)
    if winner is not None:
        eval_winner_id = 0 if eval_team_on_blue else 1
        return "team0_win" if int(winner) == eval_winner_id else "team1_win"

    eval_reward, opp_reward = (
        (cumulative_blue_reward, cumulative_orange_reward)
        if eval_team_on_blue
        else (cumulative_orange_reward, cumulative_blue_reward)
    )
    if eval_reward > opp_reward:
        return "team0_win"
    if opp_reward > eval_reward:
        return "team1_win"

    eval_final_step, opp_final_step = (
        (final_blue_step_reward, final_orange_step_reward)
        if eval_team_on_blue
        else (final_orange_step_reward, final_blue_step_reward)
    )
    if eval_final_step > opp_final_step:
        return "team0_win"
    if opp_final_step > eval_final_step:
        return "team1_win"
    return "tie"


def _accumulate_team_reward(total_reward: float, reward: Any, team_ids: Iterable[int]) -> float:
    if isinstance(reward, dict):
        for agent_id in team_ids:
            if agent_id in reward:
                total_reward += float(reward[agent_id])
    return total_reward


class LocalSharedCCTeammateInfoPolicy:
    def __init__(
        self,
        env: gym.Env,
        *,
        checkpoint_path: str,
        include_time: bool = True,
        time_max_steps: int = 1500,
        teammate_state_scale=None,
    ):
        self._checkpoint_path = str(Path(checkpoint_path).resolve())
        self._include_time = bool(include_time)
        self._time_max_steps = max(int(time_max_steps), 1)
        self._state_scale = parse_teammate_state_scale(teammate_state_scale)

        inferred_action_dim = infer_action_dim_from_checkpoint(self._checkpoint_path)

        os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
        os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
        os.environ.setdefault("RAY_GRAFANA_HOST", "")
        os.environ.setdefault("RAY_PROMETHEUS_HOST", "")
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            local_mode=True,
            num_cpus=1,
            log_to_driver=False,
        )

        config_dir = os.path.dirname(self._checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            raise ValueError(f"Could not find params.pkl near checkpoint: {self._checkpoint_path}")

        with open(config_path, "rb") as handle:
            config = pickle.load(handle)
        config["num_workers"] = 0
        config["num_gpus"] = 0

        raw_obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        if raw_obs_space is None or act_space is None:
            raise ValueError("Env must expose observation_space and action_space.")

        self._aug_obs_space = build_teammate_obs_space(raw_obs_space, include_time=self._include_time)
        cc_obs_space = build_cc_obs_space(self._aug_obs_space, act_space)
        self._cc_preprocessor = ModelCatalog.get_preprocessor_for_space(cc_obs_space)

        trainer_action_space = act_space
        if inferred_action_dim is not None:
            trainer_action_space = gym.spaces.Discrete(int(inferred_action_dim))
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            trainer_action_space = gym.spaces.Discrete(int(np.prod(act_space.nvec)))

        register_shared_cc_model()
        env_name = f"DummyEnvLocalSharedCCTeammateInfoPolicy_{os.getpid()}"
        tune.registry.register_env(
            env_name,
            lambda *_: _DummyMultiAgentEnv(cc_obs_space, self._aug_obs_space, trainer_action_space),
        )
        config["env"] = env_name
        config["env_config"] = {}
        if "multiagent" in config and "policies" in config["multiagent"]:
            config["multiagent"]["policies"][SHARED_CC_POLICY_ID] = (
                None,
                cc_obs_space,
                trainer_action_space,
                {},
            )

        self._env_action_space = act_space
        self._action_flattener = None
        if ActionFlattener is not None and isinstance(self._env_action_space, gym.spaces.MultiDiscrete):
            try:
                self._action_flattener = ActionFlattener(self._env_action_space.nvec)
            except Exception:
                self._action_flattener = None

        cls = get_trainable_cls(ALGORITHM)
        trainer = cls(env=config["env"], config=config)
        self._trainer = trainer
        self._shared_policy = load_policy_weights(self._checkpoint_path, trainer, SHARED_CC_POLICY_ID)

    def act(
        self,
        *,
        local_observation: Dict[int, np.ndarray],
        local_info_by_agent: Optional[Dict[int, np.ndarray]],
        episode_step: int,
    ) -> Dict[int, Any]:
        time_tail = (
            normalized_episode_time(episode_step=episode_step, max_steps=self._time_max_steps)
            if self._include_time
            else None
        )
        augmented_obs = augment_observation_dict_with_teammate_state(
            local_observation,
            local_info_by_agent or {},
            include_time=self._include_time,
            normalized_time=time_tail,
        )

        actions: Dict[int, Any] = {}
        zero_obs = None
        for player_id, obs in augmented_obs.items():
            if zero_obs is None:
                zero_obs = np.zeros_like(np.asarray(obs, dtype=np.float32).reshape(-1))
            mate_id = 1 if int(player_id) == 0 else 0
            mate_obs = np.asarray(augmented_obs.get(mate_id, zero_obs), dtype=np.float32).reshape(-1)
            cc_obs = {
                "own_obs": np.asarray(obs, dtype=np.float32).reshape(-1),
                "teammate_obs": mate_obs,
                "teammate_action": 0,
            }
            flat_cc_obs = self._cc_preprocessor.transform(cc_obs)
            action = _unwrap_action(self._shared_policy.compute_single_action(flat_cc_obs, explore=False))

            if isinstance(self._env_action_space, gym.spaces.MultiDiscrete):
                if isinstance(action, (list, tuple, np.ndarray)):
                    arr = np.asarray(action)
                    if arr.ndim == 1 and arr.size == len(self._env_action_space.nvec):
                        actions[player_id] = arr.astype(np.int64)
                        continue

                flat = _coerce_int_action(action)
                if self._action_flattener is not None:
                    actions[player_id] = np.asarray(
                        self._action_flattener.lookup_action(int(flat)),
                        dtype=np.int64,
                    )
                else:
                    actions[player_id] = _unflatten_discrete_to_multidiscrete(
                        flat,
                        np.asarray(self._env_action_space.nvec),
                    )
            else:
                actions[player_id] = _coerce_int_action(action)
        return actions


def _local_state_by_agent(
    last_info: Any,
    team_global_ids: Tuple[int, int],
    *,
    teammate_state_scale=None,
) -> Dict[int, np.ndarray]:
    state_by_global = extract_player_state_by_agent(
        last_info,
        state_scale=parse_teammate_state_scale(teammate_state_scale),
    )
    out: Dict[int, np.ndarray] = {}
    for local_id, global_id in enumerate(team_global_ids):
        if int(global_id) in state_by_global:
            out[int(local_id)] = np.asarray(state_by_global[int(global_id)], dtype=np.float32)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or checkpoint-* file.")
    parser.add_argument(
        "--team1-module",
        default="ceia_baseline_agent",
        help="Opponent module or alias baseline/random.",
    )
    parser.add_argument("-n", "--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--base-port", type=int, default=64005)
    parser.add_argument("--include-time", type=int, default=1)
    parser.add_argument("--time-max-steps", type=int, default=1500)
    parser.add_argument(
        "--teammate-state-scale",
        default="",
        help="Optional comma-separated scale for teammate x,y,vx,vy tail normalization.",
    )
    parser.add_argument("--side-swap", type=int, default=1, help="1 to alternate blue/orange each episode.")
    parser.add_argument("--possession-dist", type=float, default=1.25)
    parser.add_argument("--save-episodes-dir", default="", help="Optional directory to save structured episode records.")
    parser.add_argument("--save-mode", choices=("losses", "nonwins", "all"), default="losses")
    parser.add_argument("--max-saved-episodes", type=int, default=50)
    parser.add_argument("--trace-stride", type=int, default=5)
    parser.add_argument("--trace-tail-steps", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    team1_module = _resolve_opponent_module(args.team1_module)
    env, used_port = make_env_with_retry(base_port=int(args.base_port))
    try:
        eval_policy = LocalSharedCCTeammateInfoPolicy(
            env,
            checkpoint_path=str(args.checkpoint),
            include_time=bool(int(args.include_time)),
            time_max_steps=int(args.time_max_steps),
            teammate_state_scale=args.teammate_state_scale,
        )
        opponent_agent = load_agent(team1_module, env)

        team0_wins = 0
        team1_wins = 0
        ties = 0
        all_steps = []
        team0_win_steps = []
        team1_win_steps = []
        tie_steps = []
        save_dir = Path(args.save_episodes_dir).resolve() if args.save_episodes_dir else None
        saved_episode_paths = []

        for ep in range(max(int(args.episodes), 0)):
            obs = env.reset()
            last_info = None
            step_idx = 0
            recorder_prev_ball_x = None
            cumulative_blue_reward = 0.0
            cumulative_orange_reward = 0.0
            final_blue_step_reward = 0.0
            final_orange_step_reward = 0.0
            eval_team_on_blue = not bool(int(args.side_swap)) or (ep % 2 == 0)
            recorder = EpisodeFailureRecorder(
                trace_stride=int(args.trace_stride),
                tail_steps=int(args.trace_tail_steps),
            )

            while True:
                blue_obs = _team_local_obs(obs, (0, 1))
                orange_obs = _team_local_obs(obs, (2, 3))

                if eval_team_on_blue:
                    eval_local_obs = blue_obs
                    opp_local_obs = orange_obs
                    eval_global_ids = (0, 1)
                    opp_global_ids = (2, 3)
                else:
                    eval_local_obs = orange_obs
                    opp_local_obs = blue_obs
                    eval_global_ids = (2, 3)
                    opp_global_ids = (0, 1)

                local_info_by_agent = (
                    _local_state_by_agent(
                        last_info,
                        eval_global_ids,
                        teammate_state_scale=args.teammate_state_scale,
                    )
                    if last_info is not None
                    else {}
                )
                eval_actions_local = eval_policy.act(
                    local_observation=eval_local_obs,
                    local_info_by_agent=local_info_by_agent,
                    episode_step=step_idx,
                )
                opp_actions_local = opponent_agent.act(opp_local_obs)

                if eval_team_on_blue:
                    merged_actions = {}
                    merged_actions.update(_remap_team_actions(eval_actions_local, eval_global_ids))
                    merged_actions.update(_remap_team_actions(opp_actions_local, opp_global_ids))
                else:
                    merged_actions = {}
                    merged_actions.update(_remap_team_actions(opp_actions_local, opp_global_ids))
                    merged_actions.update(_remap_team_actions(eval_actions_local, eval_global_ids))

                env_actions = adapt_actions_to_env(merged_actions, env)
                obs, reward, done, info = env.step(env_actions)

                cumulative_blue_reward = _accumulate_team_reward(cumulative_blue_reward, reward, (0, 1))
                cumulative_orange_reward = _accumulate_team_reward(cumulative_orange_reward, reward, (2, 3))
                final_blue_step_reward = _accumulate_team_reward(0.0, reward, (0, 1))
                final_orange_step_reward = _accumulate_team_reward(0.0, reward, (2, 3))
                info_for_record = info if isinstance(info, dict) else {}
                if isinstance(info, dict):
                    _, shaping_debug = compute_shaping_components(
                        info,
                        recorder_prev_ball_x,
                        ball_progress_scale=0.0,
                        opponent_progress_penalty_scale=0.0,
                        possession_dist=float(args.possession_dist),
                        possession_bonus=0.0,
                        progress_requires_possession=False,
                    )
                    info_for_record = dict(info)
                    info_for_record["_reward_shaping"] = {
                        **shaping_debug,
                        "applied_reward": {},
                        "scalar_reward_delta": 0.0,
                        "episode_steps": int(step_idx + 1),
                    }
                    ball_pos = extract_ball_position(info)
                    if ball_pos is not None:
                        recorder_prev_ball_x = float(ball_pos[0])
                recorder.record_step(
                    step_index=step_idx,
                    reward=reward,
                    info=info_for_record,
                    team0_ids=eval_global_ids,
                    team1_ids=opp_global_ids,
                )

                step_idx += 1
                last_info = info
                if episode_done(done) or step_idx >= int(args.max_steps):
                    break

            outcome = _infer_eval_outcome(
                eval_team_on_blue=eval_team_on_blue,
                final_info=last_info,
                cumulative_blue_reward=cumulative_blue_reward,
                cumulative_orange_reward=cumulative_orange_reward,
                final_blue_step_reward=final_blue_step_reward,
                final_orange_step_reward=final_orange_step_reward,
            )
            all_steps.append(step_idx)
            if outcome == "team0_win":
                team0_wins += 1
                team0_win_steps.append(step_idx)
            elif outcome == "team1_win":
                team1_wins += 1
                team1_win_steps.append(step_idx)
            else:
                ties += 1
                tie_steps.append(step_idx)

            should_save = False
            if save_dir is not None and len(saved_episode_paths) < int(args.max_saved_episodes):
                if args.save_mode == "all":
                    should_save = True
                elif args.save_mode == "nonwins":
                    should_save = outcome != "team0_win"
                elif args.save_mode == "losses":
                    should_save = outcome == "team1_win"

            if should_save:
                final_score = extract_score_from_info(last_info)
                relative_score = None
                if final_score is not None:
                    blue_score = float(final_score[0])
                    orange_score = float(final_score[1])
                    relative_score = (
                        [blue_score, orange_score]
                        if eval_team_on_blue
                        else [orange_score, blue_score]
                    )
                final_winner = extract_winner_from_info(last_info)
                relative_winner = None
                if final_winner is not None:
                    blue_is_team0 = bool(eval_team_on_blue)
                    relative_winner = (
                        0 if (int(final_winner) == 0 and blue_is_team0) or (int(final_winner) == 1 and not blue_is_team0) else 1
                    )
                record = recorder.build_episode_record(
                    episode_index=ep,
                    team0_module="local_true_teammate_info",
                    team1_module=team1_module,
                    outcome=outcome,
                    final_score=relative_score,
                    final_winner=relative_winner,
                    cumulative_team0_reward=(
                        cumulative_blue_reward if eval_team_on_blue else cumulative_orange_reward
                    ),
                    cumulative_team1_reward=(
                        cumulative_orange_reward if eval_team_on_blue else cumulative_blue_reward
                    ),
                    final_team0_step_reward=(
                        final_blue_step_reward if eval_team_on_blue else final_orange_step_reward
                    ),
                    final_team1_step_reward=(
                        final_orange_step_reward if eval_team_on_blue else final_blue_step_reward
                    ),
                    total_steps=step_idx,
                )
                saved_episode_paths.append(save_episode_record(record, save_dir))

        total = team0_wins + team1_wins + ties
        print("---- Summary ----")
        print(f"checkpoint: {Path(args.checkpoint).resolve()}")
        print("team0_eval_mode: local_true_teammate_info")
        print(f"team1_module: {team1_module}")
        print(f"episodes: {total}")
        print(f"team0_wins: {team0_wins}")
        print(f"team1_wins: {team1_wins}")
        print(f"ties: {ties}")
        print(f"team0_win_rate: {team0_wins / max(total, 1):.3f}")
        print(f"used_base_port: {used_port}")
        print(f"side_swap: {bool(int(args.side_swap))}")
        print(f"teammate_state_scale: {args.teammate_state_scale or 'raw/unscaled'}")
        _print_step_stats("episode_steps_all", all_steps)
        _print_step_stats("episode_steps_team0_win", team0_win_steps)
        _print_step_stats("episode_steps_team1_win", team1_win_steps)
        _print_step_stats("episode_steps_tie", tie_steps)
        if save_dir is not None:
            print(f"saved_episodes_dir: {save_dir}")
            print(f"saved_episode_count: {len(saved_episode_paths)}")
            summary_path = save_dir / "summary.json"
            summary_payload = {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "team0_eval_mode": "local_true_teammate_info",
                "team1_module": team1_module,
                "episodes": int(total),
                "team0_wins": int(team0_wins),
                "team1_wins": int(team1_wins),
                "ties": int(ties),
                "team0_win_rate": float(team0_wins / max(total, 1)),
                "used_base_port": int(used_port),
                "side_swap": bool(int(args.side_swap)),
                "teammate_state_scale": args.teammate_state_scale or "raw/unscaled",
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
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

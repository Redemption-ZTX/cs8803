#!/usr/bin/env python
"""Collect teacher trajectories from Soccer-Twos agents for imitation learning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import gym
import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_PYTHONPATH = os.environ.get("PYTHONPATH", "")
_PYTHONPATH_ENTRIES = [entry for entry in _PYTHONPATH.split(os.pathsep) if entry]
if REPO_ROOT not in _PYTHONPATH_ENTRIES:
    os.environ["PYTHONPATH"] = REPO_ROOT if not _PYTHONPATH else REPO_ROOT + os.pathsep + _PYTHONPATH

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:  # pragma: no cover
    try:
        from gym_unity.envs import ActionFlattener
    except Exception:  # pragma: no cover
        ActionFlattener = None

from cs8803drl.evaluation.evaluate_matches import (
    _remap_team_actions,
    _team_local_obs,
    adapt_actions_to_env,
    episode_done,
    load_agent,
    make_env_with_retry,
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        return float(value)
    except Exception:
        return str(value)


def _player_action_to_vec(action: Any, env_action_space: gym.Space) -> np.ndarray:
    if isinstance(action, np.ndarray):
        arr = action.reshape(-1)
    elif isinstance(action, (list, tuple)):
        arr = np.asarray(action).reshape(-1)
    else:
        arr = np.asarray([action]).reshape(-1)

    if isinstance(env_action_space, gym.spaces.MultiDiscrete):
        nvec = np.asarray(env_action_space.nvec, dtype=np.int64).reshape(-1)
        if arr.size == len(nvec):
            return arr.astype(np.int64)
        if arr.size == 1 and ActionFlattener is not None:
            flattener = ActionFlattener(nvec)
            return np.asarray(flattener.lookup_action(int(arr[0])), dtype=np.int64).reshape(-1)

    return arr.astype(np.int64)


def _team_obs_vec(team_obs: Dict[int, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(team_obs[0], dtype=np.float32).reshape(-1),
            np.asarray(team_obs[1], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )


def _team_action_vec(team_action: Dict[int, Any], env_action_space: gym.Space) -> np.ndarray:
    return np.concatenate(
        [
            _player_action_to_vec(team_action[0], env_action_space),
            _player_action_to_vec(team_action[1], env_action_space),
        ],
        axis=0,
    )


class _ShardWriter:
    def __init__(self, root: Path, mode: str, shard_size: int):
        self.root = root
        self.mode = mode
        self.shard_size = max(int(shard_size), 1)
        self.samples: List[Tuple[np.ndarray, np.ndarray, int, int, int]] = []
        self.num_shards = 0
        self.total_samples = 0

    def add(
        self,
        *,
        obs: np.ndarray,
        action: np.ndarray,
        episode_index: int,
        step_index: int,
        side: int,
    ) -> None:
        self.samples.append(
            (
                np.asarray(obs, dtype=np.float32).reshape(-1),
                np.asarray(action, dtype=np.int64).reshape(-1),
                int(episode_index),
                int(step_index),
                int(side),
            )
        )
        if len(self.samples) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.samples:
            return
        shard_dir = self.root / self.mode
        shard_dir.mkdir(parents=True, exist_ok=True)

        obs = np.stack([item[0] for item in self.samples], axis=0)
        action = np.stack([item[1] for item in self.samples], axis=0)
        episode = np.asarray([item[2] for item in self.samples], dtype=np.int64)
        step = np.asarray([item[3] for item in self.samples], dtype=np.int64)
        side = np.asarray([item[4] for item in self.samples], dtype=np.int64)

        out_path = shard_dir / f"shard_{self.num_shards:05d}.npz"
        np.savez_compressed(
            out_path,
            obs=obs,
            action=action,
            episode=episode,
            step=step,
            side=side,
        )

        self.total_samples += int(len(self.samples))
        self.num_shards += 1
        self.samples.clear()


def _build_manifest(
    *,
    args: argparse.Namespace,
    used_port: int,
    total_steps: int,
    team_writer: _ShardWriter,
    player_writer: _ShardWriter,
    episodes_completed: int,
    finalized: bool,
) -> Dict[str, Any]:
    return {
        "team0_module": args.team0_module,
        "team1_module": args.team1_module,
        "episodes": int(args.episodes),
        "episodes_completed": int(episodes_completed),
        "max_steps": int(args.max_steps),
        "base_port_requested": int(args.base_port),
        "base_port_used": int(used_port),
        "mode": args.mode,
        "include_both_teams": bool(args.include_both_teams),
        "total_env_steps": int(total_steps),
        "team": {
            "num_shards": int(team_writer.num_shards),
            "total_samples": int(team_writer.total_samples + len(team_writer.samples)),
        },
        "player": {
            "num_shards": int(player_writer.num_shards),
            "total_samples": int(player_writer.total_samples + len(player_writer.samples)),
        },
        "finalized": bool(finalized),
    }


def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team0-module", default="ceia_baseline_agent")
    parser.add_argument("--team1-module", default="ceia_baseline_agent")
    parser.add_argument("--episodes", "-n", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--base-port", type=int, default=61205)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--mode", choices=("team", "player", "both"), default="team")
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument(
        "--log-every-episodes",
        type=int,
        default=100,
        help="Emit progress and update manifest.partial.json every N episodes.",
    )
    parser.add_argument(
        "--include-both-teams",
        action="store_true",
        help="Record samples for both local teams each step. Default: false (team0 only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    env, used_port = make_env_with_retry(base_port=int(args.base_port))
    agent0 = load_agent(args.team0_module, env)
    agent1 = load_agent(args.team1_module, env)

    team_writer = _ShardWriter(save_dir, "team", int(args.shard_size))
    player_writer = _ShardWriter(save_dir, "player", int(args.shard_size))

    total_steps = 0
    started_at = time.time()

    for ep in range(int(args.episodes)):
        obs = env.reset()
        if ep % 2 == 0:
            team0_ids = (0, 1)
            team1_ids = (2, 3)
        else:
            team0_ids = (2, 3)
            team1_ids = (0, 1)

        for step_idx in range(int(args.max_steps)):
            team0_obs = _team_local_obs(obs, team0_ids)
            team1_obs = _team_local_obs(obs, team1_ids)

            team0_action_local = adapt_actions_to_env(agent0.act(team0_obs), env)
            team1_action_local = adapt_actions_to_env(agent1.act(team1_obs), env)

            if args.mode in {"team", "both"}:
                team_writer.add(
                    obs=_team_obs_vec(team0_obs),
                    action=_team_action_vec(team0_action_local, env.action_space),
                    episode_index=ep,
                    step_index=step_idx,
                    side=0,
                )
                if args.include_both_teams:
                    team_writer.add(
                        obs=_team_obs_vec(team1_obs),
                        action=_team_action_vec(team1_action_local, env.action_space),
                        episode_index=ep,
                        step_index=step_idx,
                        side=1,
                    )

            if args.mode in {"player", "both"}:
                for local_id in (0, 1):
                    player_writer.add(
                        obs=np.asarray(team0_obs[local_id], dtype=np.float32).reshape(-1),
                        action=_player_action_to_vec(team0_action_local[local_id], env.action_space),
                        episode_index=ep,
                        step_index=step_idx,
                        side=0,
                    )
                if args.include_both_teams:
                    for local_id in (0, 1):
                        player_writer.add(
                            obs=np.asarray(team1_obs[local_id], dtype=np.float32).reshape(-1),
                            action=_player_action_to_vec(team1_action_local[local_id], env.action_space),
                            episode_index=ep,
                            step_index=step_idx,
                            side=1,
                        )

            env_action = {}
            env_action.update(_remap_team_actions(team0_action_local, team0_ids))
            env_action.update(_remap_team_actions(team1_action_local, team1_ids))
            obs, _, done, _ = env.step(env_action)
            total_steps += 1
            if episode_done(done):
                break

        episodes_completed = ep + 1
        if int(args.log_every_episodes) > 0 and (
            episodes_completed % int(args.log_every_episodes) == 0
            or episodes_completed == int(args.episodes)
        ):
            elapsed = max(time.time() - started_at, 1e-6)
            team_samples = int(team_writer.total_samples + len(team_writer.samples))
            player_samples = int(player_writer.total_samples + len(player_writer.samples))
            print(
                "[collector] "
                f"episodes={episodes_completed}/{int(args.episodes)} "
                f"env_steps={total_steps} "
                f"team_samples={team_samples} "
                f"player_samples={player_samples} "
                f"team_shards={team_writer.num_shards} "
                f"player_shards={player_writer.num_shards} "
                f"steps_per_s={total_steps / elapsed:.1f}",
                flush=True,
            )
            partial_manifest = _build_manifest(
                args=args,
                used_port=int(used_port),
                total_steps=int(total_steps),
                team_writer=team_writer,
                player_writer=player_writer,
                episodes_completed=int(episodes_completed),
                finalized=False,
            )
            _write_manifest(save_dir / "manifest.partial.json", partial_manifest)

    team_writer.flush()
    player_writer.flush()
    env.close()

    manifest = _build_manifest(
        args=args,
        used_port=int(used_port),
        total_steps=int(total_steps),
        team_writer=team_writer,
        player_writer=player_writer,
        episodes_completed=int(args.episodes),
        finalized=True,
    )
    manifest_path = save_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)

    print("---- Trajectory Collection Summary ----")
    print(f"team0_module: {args.team0_module}")
    print(f"team1_module: {args.team1_module}")
    print(f"episodes: {args.episodes}")
    print(f"total_env_steps: {total_steps}")
    print(f"save_dir: {save_dir}")
    print(f"mode: {args.mode}")
    print(f"include_both_teams: {bool(args.include_both_teams)}")
    print(f"team_samples: {team_writer.total_samples}")
    print(f"player_samples: {player_writer.total_samples}")
    print(f"manifest_json: {manifest_path}")


if __name__ == "__main__":
    main()

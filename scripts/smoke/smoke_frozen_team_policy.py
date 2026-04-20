"""Smoke test for FrozenTeamPolicy (snapshot-046).

Verifies the per-env adapter plumbing without running training:
1. Build a `team_vs_policy` env with `team_opponent_checkpoint` pointing at
   031A@1040 (frozen Siamese team-level checkpoint).
2. Step the env 50 times with random team0 actions.
3. Verify per-step:
   - Opponent actions are non-trivial (variance > 0).
   - Actions are in MultiDiscrete([3,3,3]) range (per-agent).
   - No exceptions, no NaN observations.
4. Mid-test reset to confirm adapter state clears (no carry-over).
5. Print summary: action distribution, episode count, mean episode length.

Usage::

    /home/hice1/wsun377/.conda/envs/soccertwos/bin/python \\
        scripts/smoke/smoke_frozen_team_policy.py

Exit code 0 = pass, non-zero = fail.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

# Project sitecustomize patches np.bool etc.
try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]


CHECKPOINT_031A_1040 = (
    "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
    "031A_team_dual_encoder_scratch_v2_512x512_20260418_054948/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_fc02e_00000_0_2026-04-18_05-50-08/"
    "checkpoint_001040/checkpoint-1040"
)


def _print(msg: str) -> None:
    print(msg, flush=True)


def _summarize_actions(actions: list) -> dict:
    arr = np.asarray(actions, dtype=np.int64)
    flat = arr.reshape(-1)
    counts = Counter(flat.tolist())
    by_dim = []
    if arr.ndim == 2 and arr.shape[1] >= 1:
        for d in range(arr.shape[1]):
            col = arr[:, d]
            by_dim.append(
                {
                    "dim": d,
                    "min": int(col.min()),
                    "max": int(col.max()),
                    "var": float(np.var(col)),
                    "hist": dict(Counter(col.tolist())),
                }
            )
    return {
        "n_actions": int(arr.size),
        "n_calls": int(arr.shape[0]),
        "global_var": float(np.var(flat.astype(np.float32))),
        "value_counts": dict(counts),
        "by_dim": by_dim,
    }


def main() -> int:
    if not os.path.exists(CHECKPOINT_031A_1040):
        _print(f"[FAIL] checkpoint not found: {CHECKPOINT_031A_1040}")
        return 2

    import soccer_twos
    from soccer_twos import EnvType
    from soccer_twos.wrappers import TeamVsPolicyWrapper

    from cs8803drl.core.frozen_team_policy import FrozenTeamPolicy
    from cs8803drl.core.utils import _find_wrapper, create_rllib_env

    base_port = int(os.environ.get("SMOKE_BASE_PORT", "61234"))

    env_config = {
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "base_port": base_port,
        "team_opponent_checkpoint": CHECKPOINT_031A_1040,
    }

    _print("[1/5] Building env with FrozenTeamPolicy opponent ...")
    t0 = time.time()
    try:
        env = create_rllib_env(env_config)
    except Exception as exc:
        _print(f"[FAIL] create_rllib_env raised: {exc!r}")
        traceback.print_exc()
        return 3
    _print(f"      env built in {time.time()-t0:.1f}s")

    tvp = _find_wrapper(env, TeamVsPolicyWrapper)
    if tvp is None:
        _print("[FAIL] could not find TeamVsPolicyWrapper after env build")
        return 4
    if not isinstance(tvp.opponent_policy, FrozenTeamPolicy):
        _print(
            f"[FAIL] opponent_policy is not FrozenTeamPolicy: {type(tvp.opponent_policy)!r}"
        )
        return 5
    _print(f"      opponent_policy = {type(tvp.opponent_policy).__name__} OK")

    # Capture the per-agent action returned by the adapter for inspection.
    captured_actions: list = []
    captured_call_obs: list = []
    original_call = tvp.opponent_policy.__call__

    def _instrumented_call(obs, *args, **kwargs):
        a = original_call(obs, *args, **kwargs)
        captured_actions.append(np.asarray(a, dtype=np.int64).reshape(-1))
        captured_call_obs.append(np.asarray(obs, dtype=np.float32).reshape(-1).copy())
        return a

    # Replace bound method via simple monkey-patch on the instance.
    tvp.opponent_policy.__call__ = _instrumented_call  # type: ignore[assignment]
    # Note: __call__ on instances is sometimes bypassed by the C dispatch path
    # for callable types that override __call__ at class level. We still
    # observe via the wrapper's stored reference. To be robust, we wrap the
    # opponent in a closure that records and forwards.
    real_adapter = tvp.opponent_policy

    def opponent_proxy(obs, *args, **kwargs):
        a = real_adapter(obs, *args, **kwargs)
        captured_actions.append(np.asarray(a, dtype=np.int64).reshape(-1))
        captured_call_obs.append(np.asarray(obs, dtype=np.float32).reshape(-1).copy())
        return a

    tvp.opponent_policy = opponent_proxy

    _print("[2/5] Reset env ...")
    obs = env.reset()
    _print(f"      reset obs shape: {np.asarray(obs).shape}, "
           f"adapter parity reset = {real_adapter._call_parity}")

    if real_adapter._call_parity != 0:
        _print("[FAIL] adapter parity not reset to 0 after env.reset()")
        return 6

    action_space = env.action_space
    nvec = np.asarray(action_space.nvec, dtype=np.int64).reshape(-1)
    _print(f"      env action_space nvec={nvec.tolist()}, obs space={env.observation_space.shape}")

    n_steps = 50
    n_episodes = 0
    ep_lengths: list = []
    cur_ep_len = 0
    nan_count = 0

    _print(f"[3/5] Stepping env {n_steps} times with random team0 actions ...")
    rng = np.random.default_rng(seed=42)
    t1 = time.time()
    try:
        for step_i in range(n_steps):
            joint_action = rng.integers(low=0, high=nvec, dtype=np.int64)
            obs, reward, done, info = env.step(joint_action)
            cur_ep_len += 1
            obs_arr = np.asarray(obs, dtype=np.float32)
            if not np.all(np.isfinite(obs_arr)):
                nan_count += 1
            if done:
                n_episodes += 1
                ep_lengths.append(cur_ep_len)
                cur_ep_len = 0
                _print(f"      step {step_i+1}: episode ended (len={ep_lengths[-1]}), reward={reward}")
                obs = env.reset()
                if real_adapter._call_parity != 0:
                    _print(f"[FAIL] adapter parity not reset after auto-reset at step {step_i+1}")
                    return 7
    except Exception as exc:
        _print(f"[FAIL] env.step raised at step {step_i}: {exc!r}")
        traceback.print_exc()
        env.close()
        return 8
    elapsed = time.time() - t1
    _print(f"      done in {elapsed:.1f}s ({elapsed/n_steps*1000:.1f} ms/step)")

    if nan_count > 0:
        _print(f"[FAIL] observed NaNs in obs across {nan_count} steps")
        return 9

    # Action variance check.
    summary = _summarize_actions(captured_actions)
    _print(f"[4/5] Adapter action stats: n_calls={summary['n_calls']}, "
           f"n_actions={summary['n_actions']}, global_var={summary['global_var']:.3f}")
    for d in summary["by_dim"]:
        _print(
            f"      dim {d['dim']}: min={d['min']} max={d['max']} var={d['var']:.3f} hist={d['hist']}"
        )

    if summary["global_var"] <= 0.0:
        _print("[FAIL] opponent actions have zero variance — adapter is returning constants")
        env.close()
        return 10

    # Range check.
    arr = np.asarray(captured_actions, dtype=np.int64)
    if arr.size:
        for d in range(arr.shape[1]):
            if arr[:, d].min() < 0 or arr[:, d].max() >= nvec[d]:
                _print(
                    f"[FAIL] opponent action dim {d} out of range "
                    f"[0,{nvec[d]}): observed [{arr[:, d].min()},{arr[:, d].max()}]"
                )
                env.close()
                return 11

    # Mid-test reset check.
    _print("[5/5] Mid-test explicit reset ...")
    pre_obs3 = real_adapter._cached_obs_3.copy()
    pre_parity = real_adapter._call_parity
    env.reset()
    post_obs3 = real_adapter._cached_obs_3.copy()
    post_parity = real_adapter._call_parity
    if post_parity != 0 or np.any(post_obs3 != 0):
        _print(
            f"[FAIL] adapter state not cleared after explicit reset: "
            f"parity {pre_parity}->{post_parity}, obs3_nonzero={np.any(post_obs3 != 0)}"
        )
        env.close()
        return 12

    if not ep_lengths:
        _print("[WARN] no full episodes completed in 50 steps (game length too long)")
    else:
        _print(
            f"      episodes={n_episodes}, mean_ep_len={np.mean(ep_lengths):.1f}, "
            f"min={min(ep_lengths)}, max={max(ep_lengths)}"
        )

    env.close()
    _print("")
    _print("[PASS] FrozenTeamPolicy adapter plumbing OK")
    _print(f"  steps_run         = {n_steps}")
    _print(f"  episodes          = {n_episodes}")
    _print(f"  action_var_global = {summary['global_var']:.4f}")
    _print(f"  ms_per_step       = {elapsed/n_steps*1000:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

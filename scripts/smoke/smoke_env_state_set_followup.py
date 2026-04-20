"""Follow-up smoke tests: rule out false negatives in player position SET.

Initial smoke (smoke_env_state_set.py) found:
    - Ball position SET: PASS (mid-episode)
    - Player 0 position SET: FAIL (agent stayed at spawn)

This follow-up rules out 5 alternative explanations:
    T1. agent_id mapping bug — try setting positions for ALL 4 agents,
        not just agent 0. If any agent moves, mapping isn't the problem.
    T2. Physics anchor / target out of bounds — try moving agent within
        small Δ from spawn (target = spawn + (1, 0)). If small Δ works
        but large Δ fails, Unity rejects out-of-bounds targets.
    T3. Channel propagation latency — set then take 5 no-op steps.
        If position eventually applies, channel needs >1 step to flush.
    T4. Pre-reset player set works (canonical spawn configuration use
        case) — confirms the API channel itself isn't broken; isolates
        whether mid-episode is specifically blocked.
    T5. Rotation_y SET mid-episode — different field type. If rotation
        works but position doesn't, Unity treats them differently.

Output: per-test PASS/FAIL plus combined verdict matrix at the end.

Usage:
    /home/hice1/wsun377/.conda/envs/soccertwos/bin/python \
        scripts/smoke/smoke_env_state_set_followup.py
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

import soccer_twos
from soccer_twos import EnvType
from soccer_twos.side_channels import EnvConfigurationChannel


NO_OP_ACTION = {i: [0, 0, 0] for i in range(4)}
WARMUP_STEPS = 10
POS_TOL = 1.5
ROT_TOL = 15.0  # degrees


def _pos(info: Dict, agent_id: int) -> Tuple[float, float]:
    p = info[agent_id]["player_info"]["position"]
    return float(p[0]), float(p[1])


def _rot(info: Dict, agent_id: int) -> float:
    return float(info[agent_id]["player_info"]["rotation_y"])


def _ball_pos(info: Dict) -> Tuple[float, float]:
    p = info[0]["ball_info"]["position"]
    return float(p[0]), float(p[1])


def _fmt(t: Tuple[float, float]) -> str:
    return f"({t[0]:+.3f}, {t[1]:+.3f})"


def _delta(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _new_env() -> Tuple[object, EnvConfigurationChannel]:
    """Fresh env per test to avoid cross-test contamination."""
    ch = EnvConfigurationChannel()
    env = soccer_twos.make(
        env_channel=ch,
        variation=EnvType.multiagent_player,
        watch=False,
    )
    return env, ch


def _warmup(env, steps: int = WARMUP_STEPS):
    obs = env.reset()
    info = None
    for _ in range(steps):
        obs, _r, done, info = env.step(NO_OP_ACTION)
        if any(done.values()):
            obs = env.reset()
    return info


# ---------------------------------------------------------------------------
# T1: agent_id mapping — try all 4 agents
# ---------------------------------------------------------------------------
def test_t1_all_agent_mapping() -> Dict[str, str]:
    """Set position for each of 4 agents one at a time and check who moved."""
    print("\n" + "-" * 70)
    print("T1: agent_id mapping — set position for each of 4 agents")
    print("-" * 70)

    results: Dict[str, str] = {}
    for agent_id in range(4):
        env, ch = _new_env()
        info = _warmup(env)
        pre = _pos(info, agent_id)

        # Move agent towards center field but offset by id to keep targets unique.
        target = (-3.0 + agent_id, 0.0 + agent_id)
        ch.set_parameters(players_states={agent_id: {
            "position": list(target),
            "velocity": [0.0, 0.0],
        }})
        _obs, _r, _d, info = env.step(NO_OP_ACTION)
        post = _pos(info, agent_id)

        moved_to_target = _delta(post, target) <= POS_TOL
        moved_at_all = _delta(post, pre) > 0.5
        flag = "PASS" if moved_to_target else ("MOVED-WRONG" if moved_at_all else "STUCK")
        print(f"  agent {agent_id}: pre={_fmt(pre)}  target={_fmt(target)}  post={_fmt(post)}  → {flag}")
        results[f"agent_{agent_id}"] = flag

        env.close()
    return results


# ---------------------------------------------------------------------------
# T2: physics anchor — try small Δ near spawn
# ---------------------------------------------------------------------------
def test_t2_small_delta_position() -> Dict[str, str]:
    """Try setting agent 0 within 0.5 / 2.0 / 5.0 unit Δ of its spawn."""
    print("\n" + "-" * 70)
    print("T2: physics anchor — set agent 0 with small Δ near spawn")
    print("-" * 70)

    results: Dict[str, str] = {}
    for delta_x in (0.5, 2.0, 5.0):
        env, ch = _new_env()
        info = _warmup(env)
        spawn = _pos(info, 0)
        target = (spawn[0] + delta_x, spawn[1])

        ch.set_parameters(players_states={0: {
            "position": list(target),
            "velocity": [0.0, 0.0],
        }})
        _obs, _r, _d, info = env.step(NO_OP_ACTION)
        post = _pos(info, 0)
        ok = _delta(post, target) <= POS_TOL
        print(f"  Δ={delta_x}: spawn={_fmt(spawn)}  target={_fmt(target)}  post={_fmt(post)}  |Δ|={_delta(post,target):.3f}  → {'PASS' if ok else 'FAIL'}")
        results[f"delta_{delta_x}"] = "PASS" if ok else "FAIL"

        env.close()
    return results


# ---------------------------------------------------------------------------
# T3: propagation latency — set + take N steps + check
# ---------------------------------------------------------------------------
def test_t3_multi_step_propagation() -> Dict[str, str]:
    """Set agent 0 position, then take 1/3/5 no-op steps and check each."""
    print("\n" + "-" * 70)
    print("T3: channel propagation — N no-op steps after set")
    print("-" * 70)

    env, ch = _new_env()
    info = _warmup(env)
    target = (-5.0, 0.0)

    ch.set_parameters(players_states={0: {
        "position": list(target),
        "velocity": [0.0, 0.0],
    }})

    results: Dict[str, str] = {}
    for k in range(1, 6):
        _obs, _r, _d, info = env.step(NO_OP_ACTION)
        cur = _pos(info, 0)
        ok = _delta(cur, target) <= POS_TOL
        print(f"  after step +{k}: pos={_fmt(cur)}  |Δ|={_delta(cur,target):.3f}  → {'PASS' if ok else 'FAIL'}")
        results[f"step_{k}"] = "PASS" if ok else "FAIL"

    env.close()
    return results


# ---------------------------------------------------------------------------
# T4: pre-reset player set (canonical use case)
# ---------------------------------------------------------------------------
def test_t4_pre_reset_set() -> Dict[str, str]:
    """Set agent 0 position BEFORE reset (the canonical spawn-config flow)."""
    print("\n" + "-" * 70)
    print("T4: pre-reset set — channel.set then env.reset")
    print("-" * 70)

    env, ch = _new_env()
    # Initial reset to fully connect, get spawn baseline
    _obs = env.reset()
    _obs, _r, _d, info0 = env.step(NO_OP_ACTION)
    spawn0 = _pos(info0, 0)
    print(f"  baseline spawn for agent 0: {_fmt(spawn0)}")

    target = (-3.0, 2.0)
    ch.set_parameters(players_states={0: {
        "position": list(target),
        "velocity": [0.0, 0.0],
    }})
    # Now reset — channel msg should be flushed and applied as spawn config
    _obs = env.reset()
    _obs, _r, _d, info = env.step(NO_OP_ACTION)
    post = _pos(info, 0)
    ok = _delta(post, target) <= POS_TOL
    print(f"  post-reset pos for agent 0: {_fmt(post)}  target={_fmt(target)}  |Δ|={_delta(post,target):.3f}  → {'PASS' if ok else 'FAIL'}")

    env.close()
    return {"pre_reset_player_set": "PASS" if ok else "FAIL"}


# ---------------------------------------------------------------------------
# T5: rotation_y mid-episode
# ---------------------------------------------------------------------------
def test_t5_rotation_set() -> Dict[str, str]:
    """Set agent 0 rotation_y mid-episode and check after one step."""
    print("\n" + "-" * 70)
    print("T5: rotation_y SET mid-episode")
    print("-" * 70)

    env, ch = _new_env()
    info = _warmup(env)
    pre_rot = _rot(info, 0)
    target_rot = 90.0  # try 90 deg

    ch.set_parameters(players_states={0: {"rotation_y": target_rot}})
    _obs, _r, _d, info = env.step(NO_OP_ACTION)
    post_rot = _rot(info, 0)
    delta = abs((post_rot - target_rot + 180) % 360 - 180)
    ok = delta <= ROT_TOL
    print(f"  pre_rot={pre_rot:+.2f}  target={target_rot:+.2f}  post={post_rot:+.2f}  |Δ|={delta:.2f}  → {'PASS' if ok else 'FAIL'}")

    env.close()
    return {"rotation_set": "PASS" if ok else "FAIL"}


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("smoke_env_state_set_followup: rule out false negatives")
    print("=" * 70)

    all_results: Dict[str, Dict[str, str]] = {}

    try:
        all_results["T1_agent_mapping"] = test_t1_all_agent_mapping()
    except Exception as e:
        print(f"\n[T1] crashed: {e}")
        all_results["T1_agent_mapping"] = {"_error": str(e)}

    try:
        all_results["T2_small_delta"] = test_t2_small_delta_position()
    except Exception as e:
        print(f"\n[T2] crashed: {e}")
        all_results["T2_small_delta"] = {"_error": str(e)}

    try:
        all_results["T3_multi_step"] = test_t3_multi_step_propagation()
    except Exception as e:
        print(f"\n[T3] crashed: {e}")
        all_results["T3_multi_step"] = {"_error": str(e)}

    try:
        all_results["T4_pre_reset"] = test_t4_pre_reset_set()
    except Exception as e:
        print(f"\n[T4] crashed: {e}")
        all_results["T4_pre_reset"] = {"_error": str(e)}

    try:
        all_results["T5_rotation"] = test_t5_rotation_set()
    except Exception as e:
        print(f"\n[T5] crashed: {e}")
        all_results["T5_rotation"] = {"_error": str(e)}

    print("\n" + "=" * 70)
    print("SUMMARY MATRIX")
    print("=" * 70)
    for test_name, sub in all_results.items():
        print(f"\n{test_name}:")
        for k, v in sub.items():
            print(f"  {k:25} = {v}")

    print("\n" + "=" * 70)
    print("VERDICT INTERPRETATION GUIDE")
    print("=" * 70)
    print("""
T1 (mapping):   if any agent shows PASS → mapping fine, mid-episode WORKS
                if all STUCK             → mid-episode player SET truly blocked
                if MOVED-WRONG anywhere  → agent_id mapping is wrong
T2 (anchor):    if delta_0.5 PASS but delta_5.0 FAIL → physics anchor problem
                if all FAIL              → not a target-distance issue
T3 (latency):   if step_1 FAIL but step_5 PASS → channel needs propagation time
                if all FAIL              → channel never applies player set mid-episode
T4 (pre-reset): if PASS                  → channel API works, mid-episode is the issue
                if FAIL                  → channel itself broken (confounds T1/T2/T3)
T5 (rotation):  if PASS but position FAIL → Unity treats rotation/position differently
                if both FAIL             → all player fields blocked mid-episode
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())

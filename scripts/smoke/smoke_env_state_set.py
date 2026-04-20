"""Smoke test: does soccer_twos EnvConfigurationChannel.set_parameters apply mid-episode?

Background:
    The Unity-side ML-Agents side_channel API queues messages and sends them
    on next env communication. set_parameters() accepts ball_state and
    players_states with position/velocity/rotation. Documentation does NOT
    specify whether Unity applies these mid-episode or only on reset.

    If mid-episode SET works -> deploy-time MCTS / planning is feasible
    (we can fork env state, simulate forward, pick best action).
    If only reset -> MCTS not feasible without playing actions back from
    reset to current state (prohibitively slow per call).

Test protocol:
    1. Create env with EnvConfigurationChannel
    2. Reset, step a few times with no-op actions to get past warmup
    3. Read pre-set ball + player positions from info dict
    4. Call set_parameters with target ball_state = {pos: [0,0], vel: [0,0]}
       and target player_state for agent 0 = {pos: [-5, 0], vel: [0, 0]}
    5. Take ONE no-op step (this triggers channel flush to Unity)
    6. Read post-set state from info dict
    7. Compare: did ball + agent 0 move to target positions?

Output:
    Clear PASS / FAIL with deltas. Run from repo root with the
    soccertwos conda env activated. Headless OK (uses watch=False).

Usage:
    cd /home/hice1/wsun377/Desktop/cs8803drl
    conda activate soccertwos
    python scripts/smoke/smoke_env_state_set.py
"""

from __future__ import annotations

import sys
from typing import Dict, Tuple

import numpy as np

# mlagents_envs (pinned by soccer_twos) still references the deprecated
# np.bool. Match the monkey patch in cs8803drl/core/utils.py so the env
# can be created in this standalone smoke test.
if not hasattr(np, "bool"):
    np.bool = bool

import soccer_twos
from soccer_twos import EnvType
from soccer_twos.side_channels import EnvConfigurationChannel


NO_OP_ACTION = {i: [0, 0, 0] for i in range(4)}  # MultiDiscrete([3,3,3]) all-zero per agent = stay still
WARMUP_STEPS = 10
TARGET_BALL_POS = (0.0, 0.0)
TARGET_BALL_VEL = (0.0, 0.0)
TARGET_AGENT0_POS = (-5.0, 0.0)
TARGET_AGENT0_VEL = (0.0, 0.0)

# Pass tolerances. Ball: position within 1.0 unit, velocity small.
# One physics step at the typical ML-Agents tick can move ball ~0.1-0.5 units
# even at zero velocity if there is gravity / settling motion.
POS_TOLERANCE = 1.5
VEL_TOLERANCE = 1.0


def _read_state(info: Dict) -> Dict[str, Tuple[float, float]]:
    """Extract ball + agent 0 position/velocity from one step's info dict."""
    a0 = info[0]
    return {
        "ball_pos": tuple(float(x) for x in a0["ball_info"]["position"]),
        "ball_vel": tuple(float(x) for x in a0["ball_info"]["velocity"]),
        "agent0_pos": tuple(float(x) for x in a0["player_info"]["position"]),
        "agent0_vel": tuple(float(x) for x in a0["player_info"]["velocity"]),
    }


def _fmt_pos(pos: Tuple[float, float]) -> str:
    return f"({pos[0]:+.3f}, {pos[1]:+.3f})"


def _check(label: str, target: Tuple[float, float], actual: Tuple[float, float], tol: float) -> bool:
    delta = ((actual[0] - target[0]) ** 2 + (actual[1] - target[1]) ** 2) ** 0.5
    ok = delta <= tol
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}: target={_fmt_pos(target)}  actual={_fmt_pos(actual)}  |Δ|={delta:.3f}  (tol={tol})")
    return ok


def main() -> int:
    print("=" * 70)
    print("smoke_env_state_set: mid-episode set_parameters test")
    print("=" * 70)

    channel = EnvConfigurationChannel()
    env = soccer_twos.make(
        env_channel=channel,
        variation=EnvType.multiagent_player,
        watch=False,
    )

    print(f"\n[setup] env action_space = {env.action_space}")
    print(f"[setup] env observation_space (one agent) = {env.observation_space}")

    obs = env.reset()
    print(f"[setup] reset OK; obs keys = {sorted(obs.keys())}")

    for i in range(WARMUP_STEPS):
        obs, reward, done, info = env.step(NO_OP_ACTION)
        if any(done.values()):
            print(f"[warmup] episode ended at step {i}; resetting and retrying")
            obs = env.reset()

    pre_state = _read_state(info)
    print("\n[pre-set] state read from info dict:")
    print(f"  ball_pos    = {_fmt_pos(pre_state['ball_pos'])}")
    print(f"  ball_vel    = {_fmt_pos(pre_state['ball_vel'])}")
    print(f"  agent0_pos  = {_fmt_pos(pre_state['agent0_pos'])}")
    print(f"  agent0_vel  = {_fmt_pos(pre_state['agent0_vel'])}")

    print("\n[set] queueing set_parameters via channel:")
    print(f"  ball_state    = pos={TARGET_BALL_POS}, vel={TARGET_BALL_VEL}")
    print(f"  player_state  = agent_id 0 -> pos={TARGET_AGENT0_POS}, vel={TARGET_AGENT0_VEL}")
    channel.set_parameters(
        ball_state={
            "position": list(TARGET_BALL_POS),
            "velocity": list(TARGET_BALL_VEL),
        },
        players_states={
            0: {
                "position": list(TARGET_AGENT0_POS),
                "velocity": list(TARGET_AGENT0_VEL),
            },
        },
    )

    print("\n[step] taking ONE no-op step to flush channel to Unity ...")
    obs, reward, done, info = env.step(NO_OP_ACTION)
    if any(done.values()):
        print("[step] WARNING: episode ended on the post-set step. Reading state from terminal info.")

    post_state = _read_state(info)
    print("\n[post-set] state read from info dict:")
    print(f"  ball_pos    = {_fmt_pos(post_state['ball_pos'])}")
    print(f"  ball_vel    = {_fmt_pos(post_state['ball_vel'])}")
    print(f"  agent0_pos  = {_fmt_pos(post_state['agent0_pos'])}")
    print(f"  agent0_vel  = {_fmt_pos(post_state['agent0_vel'])}")

    print("\n[verify] checking whether SET was applied mid-episode:")
    results = []
    results.append(_check("ball position",  TARGET_BALL_POS,    post_state["ball_pos"],   POS_TOLERANCE))
    results.append(_check("ball velocity",  TARGET_BALL_VEL,    post_state["ball_vel"],   VEL_TOLERANCE))
    results.append(_check("agent0 position", TARGET_AGENT0_POS, post_state["agent0_pos"], POS_TOLERANCE))
    results.append(_check("agent0 velocity", TARGET_AGENT0_VEL, post_state["agent0_vel"], VEL_TOLERANCE))

    print("\n" + "=" * 70)
    if all(results):
        print("OVERALL: PASS  —  mid-episode set_parameters WORKS.")
        print("Implication: deploy-time MCTS / planning IS feasible on this env.")
        print("Next step: design state save/restore wrapper (see snapshot-049).")
    else:
        print("OVERALL: FAIL  —  mid-episode set_parameters does NOT apply.")
        print("Implication: side_channel likely flushes only on reset.")
        print("Deploy-time MCTS via in-place state-set is NOT feasible.")
        print("Fallback: replay from reset (prohibitive) OR multi-env parallel rollout.")
    print("=" * 70)

    env.close()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

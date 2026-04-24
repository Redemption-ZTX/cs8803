"""REINFORCE trainer for the DIR-G MoE Router NN over frozen experts.

Wave 2 of Stone DIR-G: replace v_moe_router_uniform's UniformRouter with a
learnable NN that takes per-agent obs (336-dim) and outputs softmax over K
expert indices. Trained via REINFORCE on per-episode returns vs ceia_baseline.

Architecture:
  336 → Linear(64) → ReLU → Linear(64) → ReLU → Linear(K) → softmax

Training:
  - Per episode: rollout 2v2 vs baseline, sampling expert per (step, agent)
  - End of episode: G = +1/-1/0 for win/lose/tie, REINFORCE update
  - Baseline = exponential moving average of returns (variance reduction)
  - Adam optimizer, lr=3e-4

Output: trained router weights → `agents/v_moe_router_trained/router_weights.pt`
which the Wave 2 Agent class will load instead of UniformRouter.

Usage:
    python -m scripts.research.train_moe_router_reinforce \\
        --episodes 200 --base-port 60905 --save-path agents/v_moe_router_trained/router_weights.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from cs8803drl.branches.moe_router_nn import RouterNN  # noqa: E402

import gym  # noqa: F401  (required for soccer_twos)
import soccer_twos


# ----------------------------------------------------------------------
# Reuse the v_moe_router_uniform expert framework — this loads K frozen
# experts via _TeamRayPolicyHandle / _SharedCCPolicyHandle plus all model
# registrations.
# ----------------------------------------------------------------------

from cs8803drl.deployment.ensemble_agent import (
    _TeamRayPolicyHandle,
    _SharedCCPolicyHandle,
)
from cs8803drl.branches.team_siamese import (
    register_team_siamese_cross_agent_attn_medium_model,
    register_team_siamese_cross_agent_attn_model,
    register_team_siamese_cross_attention_model,
    register_team_siamese_model,
    register_team_siamese_transformer_model,
    register_team_siamese_transformer_mha_model,
    register_team_siamese_transformer_min_model,
)
from cs8803drl.branches.team_siamese_distill import (
    register_team_siamese_distill_model,
    register_team_siamese_ensemble_distill_model,
)
from cs8803drl.branches.team_siamese_two_stream import (
    register_team_siamese_two_stream_model,
)
from cs8803drl.branches.team_siamese_per_ray import (
    register_team_siamese_per_ray_model,
)
from cs8803drl.branches.team_action_aux import register_team_action_aux_model

register_team_siamese_model()
register_team_siamese_cross_attention_model()
register_team_siamese_cross_agent_attn_model()
register_team_siamese_cross_agent_attn_medium_model()
register_team_siamese_transformer_model()
register_team_siamese_transformer_mha_model()
register_team_siamese_transformer_min_model()
register_team_siamese_distill_model()
register_team_siamese_ensemble_distill_model()
register_team_siamese_two_stream_model()
register_team_siamese_per_ray_model()
register_team_action_aux_model()

import ray  # noqa: E402

# Baseline opponent module
import importlib  # noqa: E402


_AGENTS_ROOT = REPO_ROOT / "agents"


def default_experts() -> List[Tuple[str, str, str]]:
    """Default expert pool — extend with 081/103-series when ready."""
    base = [
        ("1750_sota", "team_ray",
         str(_AGENTS_ROOT / "v_sota_055v2_extend_1750" / "checkpoint_001750" / "checkpoint-1750")),
        ("055_1150", "team_ray",
         str(_AGENTS_ROOT / "v_055_1150" / "checkpoint_001150" / "checkpoint-1150")),
        ("029B_190", "shared_cc",
         str(_AGENTS_ROOT / "v_029B_190" / "checkpoint_000190" / "checkpoint-190")),
    ]
    # Auto-include 081/103-series specialists if their packages exist (Wave 2 expansion).
    candidates_optional = [
        ("081_aggressive", "team_ray", _AGENTS_ROOT / "v_081_aggressive" / "checkpoint" / "checkpoint"),
        ("103A_interceptor", "team_ray", _AGENTS_ROOT / "v_103A_interceptor" / "checkpoint" / "checkpoint"),
        ("103B_defender", "team_ray", _AGENTS_ROOT / "v_103B_defender" / "checkpoint" / "checkpoint"),
        ("103C_dribble", "team_ray", _AGENTS_ROOT / "v_103C_dribble" / "checkpoint" / "checkpoint"),
    ]
    for name, kind, path in candidates_optional:
        if path.exists():
            base.append((name, kind, str(path)))
    return base


# ----------------------------------------------------------------------
# Router NN — small MLP over per-agent 336-dim obs → softmax over K experts
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Env helper (retry on port collision)
# ----------------------------------------------------------------------

def make_env_with_retry(base_port: int, max_tries: int = 20):
    last_err = None
    for i in range(max_tries):
        port = base_port + i * 10
        try:
            return soccer_twos.make(render=False, base_port=port), port
        except Exception as e:  # pragma: no cover
            last_err = e
            msg = str(e)
            if "Address already in use" not in msg and "still in use" not in msg:
                raise
    raise RuntimeError(f"Could not init env after {max_tries} tries: {last_err}")


# ----------------------------------------------------------------------
# Episode rollout + REINFORCE update
# ----------------------------------------------------------------------

def rollout_episode(
    env,
    router: RouterNN,
    expert_handles: List[object],
    baseline_agent,
    rng: np.random.Generator,
    max_steps: int = 1500,
    device: str = "cpu",
) -> Tuple[List[torch.Tensor], float, int]:
    """Run one episode. Returns (per-step log-probs of chosen experts), team0 win indicator (1/-1/0), episode length."""
    obs = env.reset()
    team0_ids = (0, 1)
    team1_ids = (2, 3)
    log_probs: List[torch.Tensor] = []
    final_team0_reward = 0.0
    total_steps = 0
    for step_idx in range(max_steps):
        # Team 0 = our learned router + frozen experts
        action: Dict[int, np.ndarray] = {}
        team0_obs = {pid: obs[pid] for pid in team0_ids}
        a, b = team0_ids
        for my_pid, mate_pid in ((a, b), (b, a)):
            my_obs = np.asarray(team0_obs[my_pid], dtype=np.float32).reshape(-1)
            mate_obs = np.asarray(team0_obs[mate_pid], dtype=np.float32).reshape(-1)
            obs_t = torch.from_numpy(my_obs).to(device)
            logits = router(obs_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            expert_idx_t = dist.sample()
            log_p = dist.log_prob(expert_idx_t)
            log_probs.append(log_p)
            expert_idx = int(expert_idx_t.item())
            handle = expert_handles[expert_idx]
            if isinstance(handle, _TeamRayPolicyHandle):
                self_probs, _mate, _norm = handle.action_probs(obs=my_obs, teammate_obs=mate_obs)
            else:
                self_probs, _norm = handle.action_probs(obs=my_obs, teammate_obs=mate_obs)
            action[my_pid] = handle.sample_env_action(self_probs, greedy=True)

        # Team 1 = baseline agent
        team1_obs = {pid: obs[pid] for pid in team1_ids}
        # baseline_agent.act expects local IDs; remap and remap back.
        baseline_local_obs = {0: team1_obs[team1_ids[0]], 1: team1_obs[team1_ids[1]]}
        baseline_act = baseline_agent.act(baseline_local_obs)
        action[team1_ids[0]] = baseline_act[0]
        action[team1_ids[1]] = baseline_act[1]

        obs, reward, done, info = env.step(action)
        final_team0_reward = float(reward[team0_ids[0]] + reward[team0_ids[1]]) / 2.0
        total_steps = step_idx + 1
        # done can be dict or bool depending on env wrapper version
        done_flag = (done.get("__all__", False) if isinstance(done, dict) else bool(done))
        if done_flag:
            break

    # Win indicator: team0 final reward > 0 → win, < 0 → lose
    if final_team0_reward > 0.05:
        ret = 1.0
    elif final_team0_reward < -0.05:
        ret = -1.0
    else:
        ret = 0.0
    return log_probs, ret, total_steps


def reinforce_step(
    log_probs: List[torch.Tensor],
    ret: float,
    baseline_value: float,
    optimizer: torch.optim.Optimizer,
    entropy_bonus: float = 0.01,
) -> float:
    """REINFORCE update: maximize sum(log_pi * (G - b)) + entropy bonus."""
    if not log_probs:
        return 0.0
    advantage = ret - baseline_value
    log_p_sum = torch.stack(log_probs).sum()
    # REINFORCE loss = - log_pi * advantage
    loss = -log_p_sum * advantage
    if entropy_bonus > 0:
        # Entropy bonus to prevent expert collapse — sum log probs is a proxy
        # (smaller |sum log probs| ≈ higher entropy)
        loss = loss - entropy_bonus * (-log_p_sum.detach())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in optimizer.param_groups[0]["params"]], max_norm=1.0
    )
    optimizer.step()
    return float(loss.item())


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--base-port", type=int, default=60905)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--entropy-bonus", type=float, default=0.005)
    parser.add_argument("--baseline-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-path",
        type=str,
        default="agents/v_moe_router_trained/router_weights.pt",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Save router weights every N episodes (default 20).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"=== DIR-G Wave 2 Router REINFORCE Trainer ===")
    print(f"  episodes:      {args.episodes}")
    print(f"  base_port:     {args.base_port}")
    print(f"  lr:            {args.lr}")
    print(f"  entropy_bonus: {args.entropy_bonus}")
    print(f"  save_path:     {args.save_path}")

    # Init Ray (local mode for inline expert inference)
    os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
    os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        local_mode=True,
        num_cpus=1,
        log_to_driver=False,
    )

    # Init env
    env, port_used = make_env_with_retry(args.base_port)
    print(f"  env initialized on port {port_used}")

    # Load experts (uses env's obs/action space)
    experts = default_experts()
    print(f"  loaded {len(experts)} experts: {[e[0] for e in experts]}")
    handles = []
    for name, kind, ckpt in experts:
        if kind == "team_ray":
            handles.append(_TeamRayPolicyHandle(env, ckpt))
        elif kind == "shared_cc":
            handles.append(_SharedCCPolicyHandle(env, ckpt))
        else:
            raise ValueError(f"Unknown expert kind: {kind}")

    # Baseline opponent
    baseline_module = importlib.import_module("ceia_baseline_agent")
    baseline_agent_cls = next(
        cls for cls_name, cls in vars(baseline_module).items()
        if cls_name.lower().endswith("agent") and isinstance(cls, type)
    )
    baseline_agent = baseline_agent_cls(env)

    # Init router + optimizer
    router = RouterNN(obs_dim=336, n_experts=len(handles), hidden=64)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)
    baseline_value = 0.0

    # Save dir
    save_path = Path(args.save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  router NN params: {sum(p.numel() for p in router.parameters()):,}")
    print(f"  save dir created: {save_path.parent}")

    win_history = []
    expert_choice_history = {name: 0 for name, _, _ in experts}
    start = time.time()

    for ep in range(args.episodes):
        log_probs, ret, n_steps = rollout_episode(
            env, router, handles, baseline_agent, rng,
            max_steps=args.max_steps,
        )
        loss = reinforce_step(log_probs, ret, baseline_value, optimizer, args.entropy_bonus)
        # Update EMA baseline
        baseline_value = args.baseline_decay * baseline_value + (1 - args.baseline_decay) * ret
        win_history.append(int(ret > 0))

        # Diagnostics: which expert did we sample most this episode? (approximate
        # from exp(log_p) of last step)
        elapsed = time.time() - start
        win_rate = sum(win_history[-50:]) / max(1, len(win_history[-50:]))
        print(
            f"ep {ep+1:4d}/{args.episodes}  ret={ret:+.0f}  steps={n_steps:4d}  "
            f"loss={loss:+.3f}  baseline={baseline_value:+.3f}  "
            f"WR(last50)={win_rate:.3f}  elapsed={elapsed:.0f}s",
            flush=True,
        )

        if (ep + 1) % args.save_every == 0:
            torch.save({
                "router_state_dict": router.state_dict(),
                "n_experts": len(handles),
                "expert_names": [name for name, _, _ in experts],
                "episode": ep + 1,
                "win_rate_last50": win_rate,
            }, save_path)
            print(f"  [save] router weights → {save_path}", flush=True)

    # Final save
    torch.save({
        "router_state_dict": router.state_dict(),
        "n_experts": len(handles),
        "expert_names": [name for name, _, _ in experts],
        "episode": args.episodes,
        "win_rate_last50": sum(win_history[-50:]) / max(1, len(win_history[-50:])),
    }, save_path)
    print(f"=== Training complete: {args.episodes} episodes, final WR(last50)={sum(win_history[-50:])/max(1,len(win_history[-50:])):.3f} ===")
    print(f"  saved to: {save_path}")


if __name__ == "__main__":
    main()

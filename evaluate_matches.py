import argparse
import importlib
import inspect
from typing import Any, Dict, Tuple

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


def split_obs(obs: Dict[int, Any]) -> Tuple[Dict[int, Any], Dict[int, Any]]:
    """Assumes player ids 0,1 are team0 and 2,3 are team1."""
    team0 = {k: v for k, v in obs.items() if k in (0, 1)}
    team1 = {k: v for k, v in obs.items() if k in (2, 3)}
    return team0, team1


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
    args = parser.parse_args()

    env, used_port = make_env_with_retry(base_port=args.base_port)
    if used_port != args.base_port:
        print(f"[evaluate_matches] base_port {args.base_port} in use; using base_port={used_port}")

    agent0 = load_agent(args.team0_module, env)
    agent1 = load_agent(args.team1_module, env)

    wins0 = 0
    wins1 = 0
    ties = 0

    for ep in range(args.episodes):
        obs = env.reset()
        team0_reward = 0.0
        team1_reward = 0.0

        for _ in range(args.max_steps):
            obs0, obs1 = split_obs(obs)
            act0 = agent0.act(obs0)
            act1 = agent1.act(obs1)

            act0 = adapt_actions_to_env(act0, env)
            act1 = adapt_actions_to_env(act1, env)

            action = {}
            action.update(act0)
            action.update(act1)

            obs, reward, done, info = env.step(action)
            team0_reward = accumulate_team_reward(team0_reward, reward, (0, 1))
            team1_reward = accumulate_team_reward(team1_reward, reward, (2, 3))

            if episode_done(done):
                break

        if team0_reward > team1_reward:
            wins0 += 1
            outcome = "team0_win"
        elif team1_reward > team0_reward:
            wins1 += 1
            outcome = "team1_win"
        else:
            ties += 1
            outcome = "tie"

        print(
            f"Episode {ep}: {outcome} | team0_reward={team0_reward:.4f} team1_reward={team1_reward:.4f}"
        )

    print("---- Summary ----")
    print(f"team0_module: {args.team0_module}")
    print(f"team1_module: {args.team1_module}")
    print(f"episodes: {args.episodes}")
    print(f"team0_wins: {wins0}")
    print(f"team1_wins: {wins1}")
    print(f"ties: {ties}")
    print(f"team0_win_rate: {wins0 / max(args.episodes, 1):.3f}")

    env.close()


if __name__ == "__main__":
    main()

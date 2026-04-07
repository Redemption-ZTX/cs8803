import os
import pickle
import sys
import numpy as np
import gym

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("RAY_DISABLE_USAGE_STATS", "1")

import ray
from ray import tune
from soccer_twos import EnvType
from ray.rllib.agents.callbacks import DefaultCallbacks

from utils import create_rllib_env


NUM_ENVS_PER_WORKER = 1


def _unpickle_if_bytes(obj, *, max_depth: int = 3):
    cur = obj
    for _ in range(max_depth):
        if isinstance(cur, (bytes, bytearray)):
            cur = pickle.loads(cur)
            continue
        break
    return cur


def _strip_optimizer_state(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.asarray([], dtype=np.float32)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in {
                "optimizer_variables",
                "optimizer_state",
                "optim_state",
                "_optimizer_variables",
            }:
                out[k] = []
            else:
                out[k] = _strip_optimizer_state(v)
        return out
    if isinstance(obj, (list, tuple)):
        return type(obj)(_strip_optimizer_state(v) for v in obj)
    return obj


def _sanitize_checkpoint_for_restore(checkpoint_path: str) -> str:
    # Only handles RLlib's pickle-based checkpoints (ray 1.x).
    base = os.path.basename(checkpoint_path)
    sanitized_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f"{base}-sanitized",
    )
    # Ray Tune also expects a sidecar metadata file.
    meta_src = checkpoint_path + ".tune_metadata"
    meta_dst = sanitized_path + ".tune_metadata"
    if os.path.exists(sanitized_path):
        try:
            with open(sanitized_path, "rb") as f:
                existing = pickle.load(f)
            existing = _unpickle_if_bytes(existing, max_depth=2)
            ok = not (isinstance(existing, dict) and "worker" in existing and not isinstance(existing["worker"], (bytes, bytearray)))
        except Exception:
            ok = False

        if ok:
            if os.path.exists(meta_src) and not os.path.exists(meta_dst):
                import shutil

                shutil.copyfile(meta_src, meta_dst)
            return sanitized_path

    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)

    state = _unpickle_if_bytes(state, max_depth=4)

    if isinstance(state, dict) and "worker" in state:
        worker = _unpickle_if_bytes(state.get("worker"), max_depth=6)
        worker = _strip_optimizer_state(worker)
        # RLlib restore expects the worker state to still be pickled bytes.
        state["worker"] = pickle.dumps(worker)
    else:
        state = _strip_optimizer_state(state)

    with open(sanitized_path, "wb") as f:
        pickle.dump(state, f)

    if os.path.exists(meta_src) and not os.path.exists(meta_dst):
        import shutil

        shutil.copyfile(meta_src, meta_dst)

    return sanitized_path


class BaselineEvalCallbacks(DefaultCallbacks):
    @staticmethod
    def _extract_score_from_info(info):
        if not isinstance(info, dict):
            return None

        for k in (
            "score",
            "scores",
            "team_score",
            "team_scores",
            "goal",
            "goals",
        ):
            if k not in info:
                continue
            v = info.get(k)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    return float(v[0]), float(v[1])
                except Exception:
                    return None
            if isinstance(v, dict):
                for a, b in (
                    ("team0", "team1"),
                    ("blue", "orange"),
                    ("home", "away"),
                    ("left", "right"),
                ):
                    if a in v and b in v:
                        try:
                            return float(v[a]), float(v[b])
                        except Exception:
                            return None
        for a, b in (
            ("team0_score", "team1_score"),
            ("blue_score", "orange_score"),
            ("home_score", "away_score"),
        ):
            if a in info and b in info:
                try:
                    return float(info[a]), float(info[b])
                except Exception:
                    return None
        return None

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
        if not isinstance(info, dict):
            return None
        for k in ("winner", "winning_team", "result", "game_result", "outcome"):
            if k not in info:
                continue
            v = info.get(k)
            if isinstance(v, str):
                s = v.lower()
                if "team0" in s or "blue" in s or "home" in s or "left" in s:
                    return 0
                if "team1" in s or "orange" in s or "away" in s or "right" in s:
                    return 1
            if isinstance(v, (int, float)):
                if int(v) in (0, 1):
                    return int(v)
        return None

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


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    tune.registry.register_env("Soccer", create_rllib_env)

    base_env_config = {
        "num_envs_per_worker": NUM_ENVS_PER_WORKER,
        "variation": EnvType.team_vs_policy,
        "multiagent": False,
        "single_player": True,
        "flatten_branched": True,
    }

    restore_path = os.environ.get("RESTORE_CHECKPOINT", "").strip() or None
    if restore_path:
        restore_path = _sanitize_checkpoint_for_restore(restore_path)
    timesteps_total = int(os.environ.get("TIMESTEPS_TOTAL", "15000000"))
    time_total_s = int(os.environ.get("TIME_TOTAL_S", "7200"))
    run_name = os.environ.get("RUN_NAME", "PPO_team_vs_random_shaping")
    checkpoint_freq = int(os.environ.get("CHECKPOINT_FREQ", "20"))

    framework = os.environ.get("FRAMEWORK", "torch").strip() or "torch"
    num_gpus = int(os.environ.get("NUM_GPUS", "1"))
    if framework == "torch" and num_gpus > 1:
        num_gpus = 1

    # Manual evaluation is implemented in BaselineEvalCallbacks.on_train_result.

    analysis = tune.run(
        "PPO",
        name=run_name,
        config={
            # system settings
            "num_gpus": num_gpus,
            "num_workers": 4,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": framework,
            "simple_optimizer": True,
            # RL setup
            "env": "Soccer",
            "env_config": {
                **base_env_config,
                "reward_shaping": True,
                "opponent_mix": {"baseline_prob": 0.9},
                "base_port": 5005,
            },
            "callbacks": BaselineEvalCallbacks,
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 1000,
            "batch_mode": "truncate_episodes",
            "train_batch_size": 4000,
            "sgd_minibatch_size": 512,
            "num_sgd_iter": 10,
            "lr": 3e-4,
            "gamma": 0.99,
            "lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.0,
        },
        stop={
            "timesteps_total": timesteps_total,
            "time_total_s": time_total_s,
        },
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        restore=restore_path,
    )

    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")

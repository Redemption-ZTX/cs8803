import os
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.policy.policy import Policy
from utils import create_rllib_env


TEAM0_AGENT_IDS = (0, 1)
TEAM1_AGENT_IDS = (2, 3)


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


class FrozenBaselinePolicy(Policy):
    """A non-trainable policy wrapper around ceia_baseline_agent's RLlib policy."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._baseline = None

    def _ensure_loaded(self):
        if self._baseline is not None:
            return
        # Lazy import so that workers don't import baseline unless used.
        from utils import _get_baseline_policy

        self._baseline = _get_baseline_policy()

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        self._ensure_loaded()
        actions = []
        for o in obs_batch:
            a, *_ = self._baseline.compute_single_action(o, explore=False)
            actions.append(a)
        return np.asarray(actions), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        return None


NUM_ENVS_PER_WORKER = 3


def policy_mapping_fn(agent_id, *args, **kwargs):
    baseline_prob = float(kwargs.get("baseline_prob", 0.7))
    # Train team0 with "default" policy.
    if int(agent_id) in TEAM0_AGENT_IDS:
        return "default"
    # Team1 (opponent) is mostly baseline, sometimes sampled from self-play pool.
    if np.random.random() < baseline_prob:
        return "baseline"
    return np.random.choice(
        ["opponent_1", "opponent_2", "opponent_3"],
        size=1,
        p=[0.50, 0.25, 0.25],
    )[0]


class SelfPlayUpdateCallback(DefaultCallbacks):
    def on_train_result(self, **info):
        """
        Update multiagent oponent weights when reward is high enough
        """
        if info["result"]["episode_reward_mean"] > 0.5:
            print("---- Updating opponents!!! ----")
            trainer = info["trainer"]
            trainer.set_weights(
                {
                    "opponent_3": trainer.get_weights(["opponent_2"])["opponent_2"],
                    "opponent_2": trainer.get_weights(["opponent_1"])["opponent_1"],
                    "opponent_1": trainer.get_weights(["default"])["default"],
                }
            )

        result = info.get("result", {})
        eval_interval = int(os.environ.get("BASELINE_EVAL_INTERVAL", "10"))
        eval_episodes = int(os.environ.get("BASELINE_EVAL_EPISODES", "10"))
        eval_max_steps = int(os.environ.get("BASELINE_EVAL_MAX_STEPS", "1500"))
        eval_base_port = int(os.environ.get("BASELINE_EVAL_BASE_PORT", "19100"))
        if eval_interval > 0 and int(result.get("training_iteration", 0)) % eval_interval == 0:
            trainer = info["trainer"]
            win, tie, loss = evaluate_vs_baseline(
                trainer,
                episodes=eval_episodes,
                max_steps=eval_max_steps,
                base_port=eval_base_port,
            )
            result.setdefault("custom_metrics", {})
            result["custom_metrics"]["win_vs_baseline_2v2"] = float(win)
            result["custom_metrics"]["tie_vs_baseline_2v2"] = float(tie)
            result["custom_metrics"]["loss_vs_baseline_2v2"] = float(loss)
            result["custom_metrics"]["win_rate_vs_baseline_2v2"] = float(win) / float(max(win + tie + loss, 1))


def evaluate_vs_baseline(trainer, *, episodes: int, max_steps: int, base_port: int):
    # Evaluate in the same 2v2 multiagent setting as soccer_twos.watch.
    env = create_rllib_env(
        {
            "multiagent": True,
            "reward_shaping": False,
            "base_port": int(base_port),
        }
    )

    from utils import _get_baseline_policy

    baseline_policy = _get_baseline_policy()
    wins = 0
    ties = 0
    losses = 0
    try:
        for _ in range(max(episodes, 0)):
            obs = env.reset()
            last_info = None
            for _t in range(max_steps):
                actions = {}
                for aid, o in obs.items():
                    if int(aid) in TEAM0_AGENT_IDS:
                        a, *_ = trainer.get_policy("default").compute_single_action(o, explore=False)
                    else:
                        a, *_ = baseline_policy.compute_single_action(o, explore=False)
                    actions[aid] = a
                obs, reward, done, info = env.step(actions)
                last_info = info
                if done.get("__all__"):
                    break

            score = _extract_score_from_info(last_info)
            if score is not None:
                s0, s1 = score
                if s0 > s1:
                    wins += 1
                elif s0 == s1:
                    ties += 1
                else:
                    losses += 1
            else:
                # Fallback: compare team rewards if score is unavailable.
                team0_r = 0.0
                team1_r = 0.0
                if isinstance(reward, dict):
                    for aid, r in reward.items():
                        if int(aid) in TEAM0_AGENT_IDS:
                            team0_r += float(r)
                        else:
                            team1_r += float(r)
                if team0_r > team1_r:
                    wins += 1
                elif team0_r == team1_r:
                    ties += 1
                else:
                    losses += 1
    finally:
        try:
            env.close()
        except Exception:
            pass
    return wins, ties, losses


if __name__ == "__main__":
    ray.init()

    tune.registry.register_env("Soccer", create_rllib_env)
    temp_env = create_rllib_env()
    try:
        obs = temp_env.reset()
        if isinstance(obs, dict):
            print("[agent_id debug] obs keys:", sorted(list(obs.keys())))
        else:
            print("[agent_id debug] non-dict obs type:", type(obs))
    except Exception as e:
        print("[agent_id debug] reset failed:", repr(e))
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    analysis = tune.run(
        "PPO",
        name="PPO_selfplay_rec",
        config={
            # system settings
            "num_gpus": 1,
            "num_workers": 8,
            "num_envs_per_worker": NUM_ENVS_PER_WORKER,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayUpdateCallback,
            # RL setup
            "multiagent": {
                "policies": {
                    "default": (None, obs_space, act_space, {}),
                    "baseline": (FrozenBaselinePolicy, obs_space, act_space, {}),
                    "opponent_1": (None, obs_space, act_space, {}),
                    "opponent_2": (None, obs_space, act_space, {}),
                    "opponent_3": (None, obs_space, act_space, {}),
                },
                "policy_mapping_fn": tune.function(
                    lambda agent_id, *a, **k: policy_mapping_fn(
                        agent_id,
                        *a,
                        baseline_prob=float(os.environ.get("BASELINE_PROB", "0.7")),
                        **k,
                    )
                ),
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": {
                "num_envs_per_worker": NUM_ENVS_PER_WORKER,
                "reward_shaping": {
                    "time_penalty": float(os.environ.get("SHAPING_TIME_PENALTY", "0.001")),
                    "ball_progress_scale": float(os.environ.get("SHAPING_BALL_PROGRESS", "0.01")),
                    "opponent_progress_penalty_scale": float(os.environ.get("SHAPING_OPP_PROGRESS_PENALTY", "0.0")),
                    "possession_dist": float(os.environ.get("SHAPING_POSSESSION_DIST", "1.25")),
                    "possession_bonus": float(os.environ.get("SHAPING_POSSESSION_BONUS", "0.002")),
                },
            },
            "model": {
                "vf_share_layers": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            "rollout_fragment_length": 5000,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 15000000, "time_total_s": 7200,},  # 2h
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./ray_results",
        # restore="./ray_results/PPO_selfplay_twos_2/PPO_Soccer_a8b44_00000_0_2021-09-18_11-13-55/checkpoint_000600/checkpoint-600",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")

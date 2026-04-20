import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface
from cs8803drl.branches.shared_central_critic import (
    SHARED_CC_POLICY_ID,
    build_cc_obs_space,
    register_shared_cc_model,
)
from cs8803drl.branches.teammate_aux_head import register_shared_cc_teammate_aux_model
from cs8803drl.branches.team_siamese import (
    register_team_siamese_cross_attention_model,
    register_team_siamese_model,
)
from cs8803drl.branches.team_siamese_distill import register_team_siamese_distill_model
from cs8803drl.branches.team_action_aux import register_team_action_aux_model
from cs8803drl.core.checkpoint_utils import infer_action_dim_from_checkpoint, load_policy_weights

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:
    try:
        from gym_unity.envs import ActionFlattener
    except Exception:
        ActionFlattener = None


ALGORITHM = "PPO"
_DUMMY_ENV_NAME = f"DummyEnvEnsembleSharedCCAgent_{os.getpid()}"
_TEAM_DUMMY_ENV_NAME = f"DummyEnvEnsembleTeamRayAgent_{os.getpid()}"


class _DummyMultiAgentEnv(MultiAgentEnv):
    def __init__(self, cc_obs_space, raw_obs_space, action_space):
        super().__init__()
        zero_raw = np.zeros(raw_obs_space.shape, dtype=np.float32)
        self.action_space = action_space
        self._reset_obs = {
            0: {
                "own_obs": zero_raw.copy(),
                "teammate_obs": zero_raw.copy(),
                "teammate_action": 0,
            },
            1: {
                "own_obs": zero_raw.copy(),
                "teammate_obs": zero_raw.copy(),
                "teammate_action": 0,
            },
            2: zero_raw.copy(),
            3: zero_raw.copy(),
        }

    def reset(self):
        return self._reset_obs

    def step(self, action_dict):
        obs = self._reset_obs
        rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        dones = {0: True, 1: True, 2: True, 3: True, "__all__": True}
        infos = {0: {}, 1: {}, 2: {}, 3: {}}
        return obs, rewards, dones, infos


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    denom = probs.sum()
    if not np.isfinite(denom) or denom <= 0:
        return np.full_like(probs, 1.0 / float(len(probs)))
    return probs / denom


def _prob_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    denom = float(np.log(len(probs))) if len(probs) > 1 else 1.0
    if denom <= 0:
        return 0.0
    entropy = float(-np.sum(probs * np.log(probs)))
    return entropy / denom


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p) - np.log(m))))
    kl_qm = float(np.sum(q * (np.log(q) - np.log(m))))
    denom = float(np.log(len(p))) if len(p) > 1 else 1.0
    if denom <= 0:
        return 0.0
    return 0.5 * (kl_pm + kl_qm) / denom


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = np.maximum(weights, 1e-6)
    denom = float(weights.sum())
    if not np.isfinite(denom) or denom <= 0:
        return np.full_like(weights, 1.0 / float(len(weights)))
    return weights / denom


def _load_params_config(checkpoint_path):
    config_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        raise ValueError(f"Could not find params.pkl near checkpoint: {checkpoint_path}")
    with open(config_path, "rb") as handle:
        config = pickle.load(handle)
    config["num_workers"] = 0
    config["num_gpus"] = 0
    return config


def _unflatten_discrete_to_multidiscrete(flat, nvec):
    flat = int(flat)
    out = np.zeros((len(nvec),), dtype=np.int64)
    for i in range(len(nvec) - 1, -1, -1):
        base = int(nvec[i])
        if base <= 0:
            out[i] = 0
            continue
        out[i] = flat % base
        flat //= base
    return out


def _flatten_multidiscrete_action(action, nvec) -> int:
    arr = np.asarray(action, dtype=np.int64).reshape(-1)
    nvec = np.asarray(nvec, dtype=np.int64).reshape(-1)
    if arr.shape != nvec.shape:
        raise ValueError(f"Expected action shape {nvec.shape}, got {arr.shape}")
    flat = 0
    for value, base in zip(arr, nvec):
        flat = flat * int(base) + int(value)
    return int(flat)


def _sample_env_action_from_probs(
    probs: np.ndarray,
    *,
    env_action_space,
    action_flattener,
    greedy: bool = False,
):
    if greedy:
        flat = int(np.argmax(probs))
    else:
        flat = int(np.random.choice(len(probs), p=probs))
    if isinstance(env_action_space, gym.spaces.MultiDiscrete):
        if action_flattener is not None:
            return np.asarray(action_flattener.lookup_action(flat), dtype=np.int64)
        return _unflatten_discrete_to_multidiscrete(flat, np.asarray(env_action_space.nvec))
    return flat


def _joint_factor_probs_to_single_agent_probs(factor_logits, factor_nvec) -> np.ndarray:
    factor_nvec = np.asarray(factor_nvec, dtype=np.int64).reshape(-1)
    if len(factor_logits) != len(factor_nvec):
        raise ValueError(
            f"Expected {len(factor_nvec)} factor logits chunks, got {len(factor_logits)}"
        )
    factor_probs = [_softmax(chunk) for chunk in factor_logits]
    num_actions = int(np.prod(factor_nvec))
    probs = np.zeros((num_actions,), dtype=np.float64)
    for flat in range(num_actions):
        action = _unflatten_discrete_to_multidiscrete(flat, factor_nvec)
        prob = 1.0
        for idx, branch_value in enumerate(action):
            prob *= float(factor_probs[idx][int(branch_value)])
        probs[flat] = prob
    denom = probs.sum()
    if not np.isfinite(denom) or denom <= 0:
        return np.full((num_actions,), 1.0 / float(num_actions), dtype=np.float64)
    return probs / denom


class _DummyGymEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        raise RuntimeError("Dummy env should never be stepped")

    def step(self, action):
        raise RuntimeError("Dummy env should never be stepped")


def _build_team_obs_space(raw_obs_space):
    low = np.asarray(raw_obs_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(raw_obs_space.high, dtype=np.float32).reshape(-1)
    return gym.spaces.Box(
        low=np.concatenate([low, low], axis=0),
        high=np.concatenate([high, high], axis=0),
        dtype=np.float32,
    )


def _build_team_action_space(raw_action_space):
    if not isinstance(raw_action_space, gym.spaces.MultiDiscrete):
        raise ValueError(
            "Team ensemble handle expects env.action_space to be MultiDiscrete, "
            f"got {type(raw_action_space)!r}"
        )
    nvec = np.asarray(raw_action_space.nvec, dtype=np.int64).reshape(-1)
    return gym.spaces.MultiDiscrete(np.concatenate([nvec, nvec], axis=0))


class _SharedCCPolicyHandle:
    def __init__(self, env: gym.Env, checkpoint_path: str):
        obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        if obs_space is None or act_space is None:
            raise ValueError("Env must expose observation_space and action_space.")

        cc_obs_space = build_cc_obs_space(obs_space, act_space)
        self._cc_preprocessor = ModelCatalog.get_preprocessor_for_space(cc_obs_space)
        self._env_action_space = act_space
        self._action_flattener = None
        if ActionFlattener is not None and isinstance(self._env_action_space, gym.spaces.MultiDiscrete):
            try:
                self._action_flattener = ActionFlattener(self._env_action_space.nvec)
            except Exception:
                self._action_flattener = None

        trainer_action_space = act_space
        inferred_action_dim = infer_action_dim_from_checkpoint(checkpoint_path)
        if inferred_action_dim is not None:
            trainer_action_space = gym.spaces.Discrete(int(inferred_action_dim))
        elif isinstance(act_space, gym.spaces.MultiDiscrete):
            trainer_action_space = gym.spaces.Discrete(int(np.prod(act_space.nvec)))

        register_shared_cc_model()
        register_shared_cc_teammate_aux_model()
        tune.registry.register_env(
            _DUMMY_ENV_NAME,
            lambda *_: _DummyMultiAgentEnv(cc_obs_space, obs_space, trainer_action_space),
        )
        config = _load_params_config(checkpoint_path)
        config["env"] = _DUMMY_ENV_NAME
        config["env_config"] = {}
        if "multiagent" in config and "policies" in config["multiagent"]:
            config["multiagent"]["policies"][SHARED_CC_POLICY_ID] = (
                None,
                cc_obs_space,
                trainer_action_space,
                {},
            )

        cls = get_trainable_cls(ALGORITHM)
        trainer = cls(env=config["env"], config=config)
        self._trainer = trainer
        self._policy = load_policy_weights(checkpoint_path, trainer, SHARED_CC_POLICY_ID)

    def _cc_obs(self, obs, teammate_obs):
        return {
            "own_obs": np.asarray(obs, dtype=np.float32).reshape(-1),
            "teammate_obs": np.asarray(teammate_obs, dtype=np.float32).reshape(-1),
            "teammate_action": 0,
        }

    def action_probs(self, *, obs, teammate_obs):
        flat_cc_obs = self._cc_preprocessor.transform(self._cc_obs(obs, teammate_obs))
        _, _, extra = self._policy.compute_single_action(
            flat_cc_obs, explore=False, full_fetch=True
        )
        logits = np.asarray(extra["action_dist_inputs"], dtype=np.float32).reshape(-1)
        return _softmax(logits), float(np.linalg.norm(logits))

    def sample_env_action(self, probs: np.ndarray, *, greedy: bool = False):
        return _sample_env_action_from_probs(
            probs,
            env_action_space=self._env_action_space,
            action_flattener=self._action_flattener,
            greedy=greedy,
        )


class _TeamRayPolicyHandle:
    def __init__(self, env: gym.Env, checkpoint_path: str):
        raw_obs_space = getattr(env, "observation_space", None)
        raw_action_space = getattr(env, "action_space", None)
        if raw_obs_space is None or raw_action_space is None:
            raise ValueError("Env must expose observation_space and action_space.")

        self._env_action_space = raw_action_space
        self._player_action_nvec = np.asarray(raw_action_space.nvec, dtype=np.int64).reshape(-1)
        self._player_action_dim = int(len(self._player_action_nvec))
        self._action_flattener = None
        if ActionFlattener is not None and isinstance(self._env_action_space, gym.spaces.MultiDiscrete):
            try:
                self._action_flattener = ActionFlattener(self._env_action_space.nvec)
            except Exception:
                self._action_flattener = None

        self._team_obs_space = _build_team_obs_space(raw_obs_space)
        self._team_action_space = _build_team_action_space(raw_action_space)

        register_team_siamese_model()
        register_team_siamese_cross_attention_model()
        register_team_siamese_distill_model()
        register_team_action_aux_model()
        tune.registry.register_env(
            _TEAM_DUMMY_ENV_NAME,
            lambda *_: _DummyGymEnv(self._team_obs_space, self._team_action_space),
        )

        config = _load_params_config(checkpoint_path)
        config["env"] = _TEAM_DUMMY_ENV_NAME
        config["env_config"] = {}

        cls = get_trainable_cls(ALGORITHM)
        trainer = cls(env=config["env"], config=config)
        self._trainer = trainer
        self._policy = load_policy_weights(checkpoint_path, trainer, "default_policy")

    def _team_obs(self, obs, teammate_obs):
        return np.concatenate(
            [
                np.asarray(obs, dtype=np.float32).reshape(-1),
                np.asarray(teammate_obs, dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )

    def action_probs(self, *, obs, teammate_obs):
        team_obs = self._team_obs(obs, teammate_obs)
        _, _, extra = self._policy.compute_single_action(team_obs, explore=False, full_fetch=True)
        logits = np.asarray(extra["action_dist_inputs"], dtype=np.float32).reshape(-1)
        split_sizes = [int(x) for x in np.asarray(self._team_action_space.nvec).reshape(-1)]
        chunks = []
        start = 0
        for size in split_sizes:
            chunks.append(logits[start : start + size])
            start += size
        if start != len(logits):
            raise ValueError(
                f"Unexpected team logits length {len(logits)} for split sizes {split_sizes}"
            )

        first = chunks[: self._player_action_dim]
        second = chunks[self._player_action_dim :]
        self_probs = _joint_factor_probs_to_single_agent_probs(first, self._player_action_nvec)
        mate_probs = _joint_factor_probs_to_single_agent_probs(second, self._player_action_nvec)
        return self_probs, mate_probs, float(np.linalg.norm(logits))

    def sample_env_action(self, probs: np.ndarray, *, greedy: bool = False):
        return _sample_env_action_from_probs(
            probs,
            env_action_space=self._env_action_space,
            action_flattener=self._action_flattener,
            greedy=greedy,
        )


class ProbabilityAveragingSharedCCEnsembleAgent(AgentInterface):
    """Per-agent probability-averaging ensemble over multiple shared-cc policies."""

    def __init__(self, env: gym.Env, checkpoint_paths: Sequence[str]):
        super().__init__()

        checkpoint_paths = [path.strip() for path in checkpoint_paths if path and path.strip()]
        if not checkpoint_paths:
            raise ValueError("ProbabilityAveragingSharedCCEnsembleAgent needs >=1 checkpoint.")

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

        self._policies = [_SharedCCPolicyHandle(env, ckpt) for ckpt in checkpoint_paths]
        greedy_raw = os.environ.get("ENSEMBLE_GREEDY", "").strip().lower()
        if greedy_raw:
            self._greedy = greedy_raw in {"1", "true", "yes", "y", "on"}
        else:
            # Match the deployed single-policy wrappers, which call RLlib with
            # explore=False and therefore behave deterministically at eval time.
            self._greedy = True
        self._debug = os.environ.get("ENSEMBLE_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self._act_calls = 0

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        player_ids = sorted(int(pid) for pid in observation.keys())
        if len(player_ids) != 2:
            raise ValueError(
                "Ensemble agent expects exactly 2 local teammates, "
                f"got ids={player_ids}"
            )

        results = {}
        avg_norms = []
        for pid, mate_pid in ((player_ids[0], player_ids[1]), (player_ids[1], player_ids[0])):
            probs = []
            norms = []
            for handle in self._policies:
                p, norm = handle.action_probs(
                    obs=observation[pid], teammate_obs=observation[mate_pid]
                )
                probs.append(p)
                norms.append(norm)
            avg_probs = np.mean(np.stack(probs, axis=0), axis=0)
            avg_probs = avg_probs / max(avg_probs.sum(), 1e-9)
            results[pid] = self._policies[0].sample_env_action(avg_probs, greedy=self._greedy)
            avg_norms.append(float(np.mean(norms)))

        self._act_calls += 1
        if self._debug and (self._act_calls <= 5 or self._act_calls % 200 == 0):
            print(
                "[ensemble] act_call="
                f"{self._act_calls} n_policies={len(self._policies)} "
                f"mean_logit_norms={avg_norms} greedy={self._greedy}"
            )
        return results


def _normalize_member_spec(spec) -> Dict[str, str]:
    if isinstance(spec, str):
        return {
            "kind": "shared_cc",
            "checkpoint_path": spec,
            "name": Path(spec).name,
            "role": "member",
            "base_weight": 1.0,
        }
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        checkpoint_path = str(spec[1])
        return {
            "kind": str(spec[0]),
            "checkpoint_path": checkpoint_path,
            "name": Path(checkpoint_path).name,
            "role": "member",
            "base_weight": 1.0,
        }
    if isinstance(spec, dict):
        kind = str(spec.get("kind", "shared_cc"))
        checkpoint_path = str(spec.get("checkpoint_path", "")).strip()
        if not checkpoint_path:
            raise ValueError(f"Missing checkpoint_path in member spec: {spec}")
        return {
            "kind": kind,
            "checkpoint_path": checkpoint_path,
            "name": str(spec.get("name") or Path(checkpoint_path).name),
            "role": str(spec.get("role") or "member"),
            "base_weight": float(spec.get("base_weight", 1.0)),
        }
    raise ValueError(f"Unsupported ensemble member spec: {spec!r}")


class ProbabilityAveragingMixedEnsembleAgent(AgentInterface):
    """Mixed team-level/per-agent probability averaging ensemble."""

    def __init__(self, env: gym.Env, members: Sequence):
        super().__init__()

        member_specs = [_normalize_member_spec(spec) for spec in members]
        if not member_specs:
            raise ValueError("ProbabilityAveragingMixedEnsembleAgent needs >=1 member.")

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

        self._policies = []
        for spec in member_specs:
            kind = spec["kind"].strip().lower()
            checkpoint_path = spec["checkpoint_path"]
            if kind in {"shared_cc", "per_agent", "mappo"}:
                self._policies.append(_SharedCCPolicyHandle(env, checkpoint_path))
            elif kind in {"team_ray", "team", "siamese"}:
                self._policies.append(_TeamRayPolicyHandle(env, checkpoint_path))
            else:
                raise ValueError(f"Unknown ensemble member kind: {kind}")

        greedy_raw = os.environ.get("ENSEMBLE_GREEDY", "").strip().lower()
        if greedy_raw:
            self._greedy = greedy_raw in {"1", "true", "yes", "y", "on"}
        else:
            # Match the deployed single-policy wrappers, which call RLlib with
            # explore=False and therefore behave deterministically at eval time.
            self._greedy = True
        self._debug = os.environ.get("ENSEMBLE_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self._act_calls = 0

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        player_ids = sorted(int(pid) for pid in observation.keys())
        if len(player_ids) != 2:
            raise ValueError(
                "Mixed ensemble agent expects exactly 2 local teammates, "
                f"got ids={player_ids}"
            )

        results = {}
        avg_norms = []
        for pid, mate_pid in ((player_ids[0], player_ids[1]), (player_ids[1], player_ids[0])):
            probs = []
            norms = []
            for handle in self._policies:
                if isinstance(handle, _TeamRayPolicyHandle):
                    self_probs, mate_probs, norm = handle.action_probs(
                        obs=observation[pid], teammate_obs=observation[mate_pid]
                    )
                    probs.append(self_probs)
                    norms.append(norm)
                else:
                    p, norm = handle.action_probs(
                        obs=observation[pid], teammate_obs=observation[mate_pid]
                    )
                    probs.append(p)
                    norms.append(norm)

            avg_probs = np.mean(np.stack(probs, axis=0), axis=0)
            avg_probs = avg_probs / max(avg_probs.sum(), 1e-9)
            results[pid] = self._policies[0].sample_env_action(avg_probs, greedy=self._greedy)
            avg_norms.append(float(np.mean(norms)))

        self._act_calls += 1
        if self._debug and (self._act_calls <= 5 or self._act_calls % 200 == 0):
            print(
                "[mixed-ensemble] act_call="
                f"{self._act_calls} n_policies={len(self._policies)} "
                f"mean_logit_norms={avg_norms} greedy={self._greedy}"
            )
        return results


class HeuristicRoutingMixedEnsembleAgent(AgentInterface):
    """State-conditional weighted router on top of mixed ensemble members.

    The router is intentionally lightweight and heuristic-driven:
    - keep an anchor policy as the default controller
    - shift weight toward a specialist when the anchor is uncertain
    - shift weight toward a stabilizer when anchor/specialist strongly disagree

    This keeps `034F` independent from the validated static `034E` path while
    avoiding a heavier learned router implementation.
    """

    def __init__(self, env: gym.Env, members: Sequence, router_config: Dict[str, float] = None):
        super().__init__()

        member_specs = [_normalize_member_spec(spec) for spec in members]
        if not member_specs:
            raise ValueError("HeuristicRoutingMixedEnsembleAgent needs >=1 member.")

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

        self._policies = []
        self._member_specs = member_specs
        for spec in member_specs:
            kind = spec["kind"].strip().lower()
            checkpoint_path = spec["checkpoint_path"]
            if kind in {"shared_cc", "per_agent", "mappo"}:
                self._policies.append(_SharedCCPolicyHandle(env, checkpoint_path))
            elif kind in {"team_ray", "team", "siamese"}:
                self._policies.append(_TeamRayPolicyHandle(env, checkpoint_path))
            else:
                raise ValueError(f"Unknown ensemble member kind: {kind}")

        greedy_raw = os.environ.get("ENSEMBLE_GREEDY", "").strip().lower()
        if greedy_raw:
            self._greedy = greedy_raw in {"1", "true", "yes", "y", "on"}
        else:
            self._greedy = True
        self._debug = os.environ.get("ENSEMBLE_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self._act_calls = 0
        self._base_weights = _normalize_weights(
            np.asarray([spec["base_weight"] for spec in member_specs], dtype=np.float64)
        )
        self._anchor_idx = next(
            (idx for idx, spec in enumerate(member_specs) if spec["role"].lower() == "anchor"),
            0,
        )
        self._specialist_idx = next(
            (idx for idx, spec in enumerate(member_specs) if spec["role"].lower() == "specialist"),
            None,
        )
        self._stabilizer_idx = next(
            (idx for idx, spec in enumerate(member_specs) if spec["role"].lower() == "stabilizer"),
            None,
        )

        default_router = {
            "anchor_confidence_low": 0.42,
            "anchor_uncertainty_high": 0.60,
            "specialist_margin": 0.03,
            "anchor_lockin_boost": 0.18,
            "specialist_boost": 0.18,
            "stabilizer_boost": 0.12,
            "disagreement_threshold": 0.10,
        }
        self._router_config = dict(default_router)
        if isinstance(router_config, dict):
            self._router_config.update(router_config)

    def _route_weights(self, member_probs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        weights = self._base_weights.copy()
        entropies = np.asarray([_prob_entropy(p) for p in member_probs], dtype=np.float64)
        anchor_ent = float(entropies[self._anchor_idx])

        anchor_confidence_low = float(self._router_config["anchor_confidence_low"])
        anchor_uncertainty_high = float(self._router_config["anchor_uncertainty_high"])
        specialist_margin = float(self._router_config["specialist_margin"])
        anchor_lockin_boost = float(self._router_config["anchor_lockin_boost"])
        specialist_boost = float(self._router_config["specialist_boost"])
        stabilizer_boost = float(self._router_config["stabilizer_boost"])
        disagreement_threshold = float(self._router_config["disagreement_threshold"])

        if anchor_ent <= anchor_confidence_low:
            scale = 1.0 - (anchor_ent / max(anchor_confidence_low, 1e-6))
            boost = anchor_lockin_boost * scale
            donor_total = float(weights.sum() - weights[self._anchor_idx])
            if donor_total > 1e-9 and boost > 0:
                for idx in range(len(weights)):
                    if idx == self._anchor_idx:
                        continue
                    delta = boost * (weights[idx] / donor_total)
                    weights[idx] -= delta
                    weights[self._anchor_idx] += delta

        if self._specialist_idx is not None:
            specialist_ent = float(entropies[self._specialist_idx])
            if (
                anchor_ent >= anchor_uncertainty_high
                and specialist_ent + specialist_margin < anchor_ent
            ):
                scale = min(
                    1.0,
                    (anchor_ent - anchor_uncertainty_high)
                    / max(1.0 - anchor_uncertainty_high, 1e-6),
                )
                boost = specialist_boost * scale
                donor = min(boost, float(weights[self._anchor_idx] - 1e-6))
                if donor > 0:
                    weights[self._anchor_idx] -= donor
                    weights[self._specialist_idx] += donor

        js_anchor_specialist = 0.0
        if self._specialist_idx is not None and self._stabilizer_idx is not None:
            js_anchor_specialist = _js_divergence(
                member_probs[self._anchor_idx],
                member_probs[self._specialist_idx],
            )
            if js_anchor_specialist >= disagreement_threshold:
                scale = min(
                    1.0,
                    (js_anchor_specialist - disagreement_threshold)
                    / max(1.0 - disagreement_threshold, 1e-6),
                )
                boost = stabilizer_boost * scale
                donor_indices = [self._anchor_idx, self._specialist_idx]
                donor_total = float(sum(weights[idx] for idx in donor_indices))
                if donor_total > 1e-9 and boost > 0:
                    for idx in donor_indices:
                        delta = boost * (weights[idx] / donor_total)
                        weights[idx] -= delta
                        weights[self._stabilizer_idx] += delta

        weights = _normalize_weights(weights)
        return {
            "weights": weights,
            "entropies": entropies,
            "js_anchor_specialist": float(js_anchor_specialist),
        }

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        player_ids = sorted(int(pid) for pid in observation.keys())
        if len(player_ids) != 2:
            raise ValueError(
                "Heuristic router expects exactly 2 local teammates, "
                f"got ids={player_ids}"
            )

        results = {}
        route_debug = []
        for pid, mate_pid in ((player_ids[0], player_ids[1]), (player_ids[1], player_ids[0])):
            member_probs = []
            for handle in self._policies:
                if isinstance(handle, _TeamRayPolicyHandle):
                    self_probs, _mate_probs, _norm = handle.action_probs(
                        obs=observation[pid], teammate_obs=observation[mate_pid]
                    )
                    member_probs.append(self_probs)
                else:
                    probs, _norm = handle.action_probs(
                        obs=observation[pid], teammate_obs=observation[mate_pid]
                    )
                    member_probs.append(probs)

            route = self._route_weights(member_probs)
            avg_probs = np.zeros_like(np.asarray(member_probs[0], dtype=np.float64))
            for weight, probs in zip(route["weights"], member_probs):
                avg_probs += float(weight) * np.asarray(probs, dtype=np.float64)
            avg_probs = avg_probs / max(avg_probs.sum(), 1e-9)
            results[pid] = self._policies[0].sample_env_action(avg_probs, greedy=self._greedy)
            route_debug.append(route)

        self._act_calls += 1
        if self._debug and (self._act_calls <= 5 or self._act_calls % 200 == 0):
            first = route_debug[0]
            names = [spec["name"] for spec in self._member_specs]
            weight_str = ", ".join(
                f"{name}={weight:.3f}" for name, weight in zip(names, first["weights"])
            )
            entropy_str = ", ".join(
                f"{name}={ent:.3f}" for name, ent in zip(names, first["entropies"])
            )
            print(
                "[router-ensemble] act_call="
                f"{self._act_calls} weights[{weight_str}] entropies[{entropy_str}] "
                f"js_anchor_specialist={first['js_anchor_specialist']:.3f}"
            )
        return results


def parse_checkpoint_list(raw: str) -> List[str]:
    return [piece.strip() for piece in (raw or "").split(",") if piece.strip()]

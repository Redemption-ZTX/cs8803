"""FrozenTeamPolicy: per-env adapter that lets a team-level PPO checkpoint
serve as the opponent for `team_vs_policy` env training.

Background
----------
`soccer_twos.wrappers.TeamVsPolicyWrapper.step()` calls `opponent_policy(obs)`
**twice per env step**, once each for agent ids 2 and 3, with a 336-dim
per-agent observation. A team-level (Siamese) policy expects a 672-dim joint
observation (`concat(obs_2, obs_3)`) and returns a 6-dim joint MultiDiscrete
action. This adapter glues those two contracts together by tracking call
parity within an env step:

* On the FIRST call each step (agent 2): cache `obs_2`, query the team policy
  with `concat(obs_2, last_obs_3_from_prev_step)` to get a joint action,
  return `action_2`, and stash `action_3` for the next call.
* On the SECOND call each step (agent 3): return the cached `action_3`,
  stash `obs_3` so the NEXT step's first call has fresh teammate context.

Caveat: agent 2's joint context uses 1-frame-stale `obs_3`. This is an
acceptable approximation given soccer dynamics at sub-100ms staleness.

Reset hook
----------
On env reset the stale `obs_3` cache and call parity must be cleared.
We monkey-patch `TeamVsPolicyWrapper.reset` to invoke
`adapter.reset_episode()` on every registered adapter. The patch is
idempotent (only installs once even if `create_rllib_env` is called
multiple times across workers).

This module deliberately avoids depending on Ray RLlib internals: the
adapter is a plain callable that satisfies the `opponent_policy` contract
expected by `TeamVsPolicyWrapper`.

Snapshot reference: SNAPSHOT-046 (Cross-Train Pair).
"""

from __future__ import annotations

import os
import threading
import weakref
from typing import Any, Optional

import gym
import numpy as np


_INSTALL_LOCK = threading.Lock()
_RESET_HOOK_INSTALLED = False
_REGISTERED_ADAPTERS: "weakref.WeakSet[FrozenTeamPolicy]" = weakref.WeakSet()


def _ensure_team_models_registered() -> None:
    """Register custom team-level model classes that 031A-style checkpoints
    rely on. Safe to call multiple times — each `register_*_model` is itself
    idempotent in the project's branch modules."""
    from cs8803drl.branches.team_siamese import (
        register_team_siamese_cross_attention_model,
        register_team_siamese_model,
    )
    from cs8803drl.branches.team_siamese_distill import register_team_siamese_distill_model
    from cs8803drl.branches.team_action_aux import register_team_action_aux_model

    register_team_siamese_model()
    register_team_siamese_cross_attention_model()
    register_team_siamese_distill_model()
    register_team_action_aux_model()


class FrozenTeamPolicy:
    """Adapter that exposes a frozen team-level policy as a per-agent callable.

    Parameters
    ----------
    checkpoint_path : str
        Path to the team-level Ray checkpoint file (e.g. `.../checkpoint-1040`).
    obs_space : gym.Space
        Per-agent observation space (336-dim Box). Used to derive the joint
        team-level obs space (672-dim Box).
    action_space : gym.Space
        Per-agent action space (`MultiDiscrete([3,3,3])`). Used to derive
        the joint team-level action space (`MultiDiscrete([3,3,3,3,3,3])`).
    """

    def __init__(self, checkpoint_path: str, *, obs_space: gym.Space, action_space: gym.Space):
        if not isinstance(obs_space, gym.spaces.Box):
            raise TypeError(
                "FrozenTeamPolicy expects per-agent obs_space to be Box, "
                f"got {type(obs_space)!r}"
            )
        if not isinstance(action_space, gym.spaces.MultiDiscrete):
            raise TypeError(
                "FrozenTeamPolicy expects per-agent action_space to be MultiDiscrete, "
                f"got {type(action_space)!r}"
            )

        self._checkpoint_path = os.path.abspath(checkpoint_path)
        self._per_agent_obs_dim = int(np.prod(obs_space.shape))
        per_agent_nvec = np.asarray(action_space.nvec, dtype=np.int64).reshape(-1)
        self._per_agent_action_dim = int(per_agent_nvec.shape[0])

        team_low = np.concatenate(
            [np.asarray(obs_space.low, dtype=np.float32).reshape(-1)] * 2, axis=0
        )
        team_high = np.concatenate(
            [np.asarray(obs_space.high, dtype=np.float32).reshape(-1)] * 2, axis=0
        )
        self._team_obs_space = gym.spaces.Box(
            low=team_low, high=team_high, dtype=np.float32
        )
        self._team_action_space = gym.spaces.MultiDiscrete(
            np.concatenate([per_agent_nvec, per_agent_nvec], axis=0)
        )

        # Per-episode adapter state.
        self._cached_obs_3: np.ndarray = np.zeros(self._per_agent_obs_dim, dtype=np.float32)
        self._cached_action_3: Optional[np.ndarray] = None
        self._call_parity = 0  # 0 -> next call is for agent 2 (first), 1 -> agent 3

        # Lazy-loaded policy. Loading is heavyweight (Ray init + trainer
        # construction) so we defer until first call inside the worker process.
        self._policy = None
        self._policy_load_lock = threading.Lock()

        # Register for reset hook + install patch.
        _REGISTERED_ADAPTERS.add(self)
        FrozenTeamPolicy.install_reset_hook()

    # ------------------------------------------------------------------ load
    def _load_policy(self):
        if self._policy is not None:
            return self._policy
        with self._policy_load_lock:
            if self._policy is not None:
                return self._policy

            _ensure_team_models_registered()
            self._policy = self._build_team_policy_from_checkpoint(
                self._checkpoint_path,
                team_obs_space=self._team_obs_space,
                team_action_space=self._team_action_space,
            )
        return self._policy

    @staticmethod
    def _build_team_policy_from_checkpoint(
        checkpoint_path: str,
        *,
        team_obs_space: gym.spaces.Box,
        team_action_space: gym.spaces.MultiDiscrete,
    ):
        """Build a Ray policy from a team-level checkpoint preserving the
        MultiDiscrete action space (the shared `_get_checkpoint_policy`
        helper coerces MultiDiscrete to Discrete which would yield a
        single joint-index action — wrong shape for our 6-dim joint).

        Mirrors the loading dance used in
        `cs8803drl.deployment.trained_team_ray_agent`.
        """
        import pickle as _pickle

        import ray
        from ray import tune
        from ray.tune.registry import get_trainable_cls

        from cs8803drl.core.checkpoint_utils import load_policy_weights

        ray.init(ignore_reinit_error=True, include_dashboard=False)

        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")
        if not os.path.exists(config_path):
            raise ValueError(
                f"FrozenTeamPolicy: could not find params.pkl near checkpoint: "
                f"{checkpoint_path}"
            )
        with open(config_path, "rb") as handle:
            config = _pickle.load(handle)

        config["num_workers"] = 0
        config["num_gpus"] = 0

        env_name = f"DummyEnvFrozenTeamPolicy_{abs(hash(checkpoint_path))}"

        class _DummyTeamEnv(gym.Env):
            def __init__(self):
                self.observation_space = team_obs_space
                self.action_space = team_action_space

            def reset(self):
                raise RuntimeError("FrozenTeamPolicy dummy env should never be stepped")

            def step(self, action):
                raise RuntimeError("FrozenTeamPolicy dummy env should never be stepped")

        tune.registry.register_env(env_name, lambda *_: _DummyTeamEnv())
        config["env"] = env_name
        config["env_config"] = {}

        cls = get_trainable_cls("PPO")
        trainer = cls(env=config["env"], config=config)
        return load_policy_weights(checkpoint_path, trainer, "default_policy")

    # ----------------------------------------------------------------- reset
    def reset_episode(self) -> None:
        """Clear per-episode state. Called from the patched
        `TeamVsPolicyWrapper.reset`."""
        self._cached_obs_3 = np.zeros(self._per_agent_obs_dim, dtype=np.float32)
        self._cached_action_3 = None
        self._call_parity = 0

    # --------------------------------------------------------------- callable
    def __call__(self, obs, *args, **kwargs) -> np.ndarray:
        """Return a per-agent action (length=3 int array).

        The wrapper invokes us in a deterministic order each step: agent 2
        first (parity 0), agent 3 second (parity 1).
        """
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs_arr.size != self._per_agent_obs_dim:
            raise ValueError(
                "FrozenTeamPolicy: per-agent obs has unexpected size "
                f"{obs_arr.size}, expected {self._per_agent_obs_dim}."
            )

        if self._call_parity == 0:
            # First call this step (agent 2). Build joint team obs from
            # current obs_2 + last-step obs_3 (1-frame stale, acceptable).
            policy = self._load_policy()
            joint_obs = np.concatenate([obs_arr, self._cached_obs_3], axis=0).astype(
                np.float32, copy=False
            )
            joint_action = policy.compute_single_action(joint_obs, explore=False)
            if isinstance(joint_action, tuple) and joint_action:
                joint_action = joint_action[0]
            joint_action = np.asarray(joint_action, dtype=np.int64).reshape(-1)
            if joint_action.size != self._per_agent_action_dim * 2:
                raise ValueError(
                    "FrozenTeamPolicy: team policy returned action of size "
                    f"{joint_action.size}, expected "
                    f"{self._per_agent_action_dim * 2}."
                )
            action_2 = joint_action[: self._per_agent_action_dim].copy()
            self._cached_action_3 = joint_action[self._per_agent_action_dim :].copy()
            self._call_parity = 1
            return action_2

        # Second call this step (agent 3). Use stashed action_3 and refresh
        # the cached obs_3 for the NEXT step's first call.
        if self._cached_action_3 is None:
            # Defensive: should not happen if call parity is correct.
            self._cached_action_3 = np.zeros(self._per_agent_action_dim, dtype=np.int64)
        action_3 = self._cached_action_3
        self._cached_action_3 = None
        self._cached_obs_3 = obs_arr.copy()
        self._call_parity = 0
        return action_3

    # ------------------------------------------------------------ reset hook
    @staticmethod
    def install_reset_hook() -> None:
        """Idempotently monkey-patch `TeamVsPolicyWrapper.reset` so that every
        env reset clears all registered adapters' per-episode state."""
        global _RESET_HOOK_INSTALLED
        if _RESET_HOOK_INSTALLED:
            return
        with _INSTALL_LOCK:
            if _RESET_HOOK_INSTALLED:
                return
            from soccer_twos.wrappers import TeamVsPolicyWrapper

            original_reset = TeamVsPolicyWrapper.reset

            def patched_reset(self, *args, **kwargs):
                result = original_reset(self, *args, **kwargs)
                opp_policy = getattr(self, "opponent_policy", None)
                if hasattr(opp_policy, "reset_episode"):
                    opp_policy.reset_episode()
                else:
                    # Generic fallback: also clear any registered adapter (covers
                    # cases where the adapter is wrapped by a thin closure).
                    for adapter in list(_REGISTERED_ADAPTERS):
                        adapter.reset_episode()
                return result

            TeamVsPolicyWrapper.reset = patched_reset
            _RESET_HOOK_INSTALLED = True

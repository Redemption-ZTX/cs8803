"""Learned reward shaping wrapper for PPO training (snapshot-036 §3.5).

Wraps an env so that, after each ``env.step()``, we compute per-agent
shaping rewards using a ``MultiHeadRewardModel`` checkpoint produced by
``learned_reward_trainer.py`` and add them to the per-agent env reward.

The shaping contribution per step is::

    r_shaping_agent_i = λ × mean_over_heads( tanh(head_k(obs_i, act_i)) )

Tanh bounds each head output to (-1, +1) so the total per-step shaping
is in [-λ, +λ]. A λ of 0.01–0.02 gives cumulative-episode shaping of the
same order as v2 shaping without dominating the sparse ±3 goal reward.

Hook up in a training script via::

    from cs8803drl.imitation.learned_reward_shaping import LearnedRewardShapingWrapper
    env = LearnedRewardShapingWrapper(
        env,
        reward_model_path="/path/to/reward_model.pt",
        shaping_weight=0.01,
        team0_agent_ids=(0, 1),
    )

IMPORTANT: deploy-time action encoding.
    RLlib samples from Discrete(27) and applies through ActionFlattener to
    produce MultiDiscrete([3,3,3]) for Unity. Our trained reward model
    expects MultiDiscrete input. This wrapper therefore accepts either
    representation and unflattens flat ints using ActionFlattener to
    recover the canonical MultiDiscrete form before querying.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import gym
import numpy as np
import torch

try:
    from gym_unity.envs import ActionFlattener
except Exception:  # pragma: no cover
    ActionFlattener = None

from cs8803drl.imitation.learned_reward_trainer import (
    ACTION_MD_BASE,
    ACTION_MD_NDIM,
    HEAD_NAMES,
    MultiHeadRewardModel,
)


def _load_reward_model(ckpt_path: str, device: str) -> Tuple[MultiHeadRewardModel, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    model = MultiHeadRewardModel(
        obs_dim=int(config["obs_dim"]),
        use_action=bool(config["use_action"]),
        hidden_dims=tuple(int(h) for h in config["hidden_dims"]),
        head_hidden=int(config["head_hidden"]),
        head_names=tuple(config.get("head_names", HEAD_NAMES)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, config


def _as_multidiscrete(action: Any, flattener: Optional[ActionFlattener]) -> np.ndarray:
    """Convert a per-agent action (int / array / tuple) to a (3,) int8 MD array.

    If the env uses ActionFlattener (Discrete(27)), we flatten→multi-discrete via
    flattener.lookup_action. Otherwise we assume the action is already MultiDiscrete.
    """
    if isinstance(action, (int, np.integer)):
        if flattener is not None:
            md = flattener.lookup_action(int(action))
            return np.asarray(md, dtype=np.int8).reshape(-1)[:ACTION_MD_NDIM]
        # fallback: project canonical little-endian base-3 (matches _unflatten_discrete_to_multidiscrete)
        flat = int(action)
        out = np.zeros(ACTION_MD_NDIM, dtype=np.int8)
        for i in range(ACTION_MD_NDIM):
            out[i] = flat % ACTION_MD_BASE
            flat //= ACTION_MD_BASE
        return out
    arr = np.asarray(action).reshape(-1)[:ACTION_MD_NDIM].astype(np.int8)
    if arr.size < ACTION_MD_NDIM:
        padded = np.zeros(ACTION_MD_NDIM, dtype=np.int8)
        padded[: arr.size] = arr
        return padded
    return arr


class LearnedRewardShapingWrapper(gym.Wrapper):
    """Adds per-agent learned shaping reward after each env.step().

    Args:
        env: the base (already-wrapped or raw) soccer env.
        reward_model_path: path to ``reward_model.pt`` from learned_reward_trainer.
        shaping_weight: scalar λ multiplier on the mean-of-heads tanh output.
        team0_agent_ids: which agent IDs in the reward dict belong to team0.
            Shaping is only added to these agents.
        apply_to_team1: if True, also add shaping to team1 agents (e.g. self-play).
            Default False — we only shape team0 because reward model was trained on
            team0 W/L outcomes.
        device: torch device for inference. 'auto' picks cuda if available.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        reward_model_path: str,
        shaping_weight: float = 0.01,
        team0_agent_ids: Iterable[int] = (0, 1),
        apply_to_team1: bool = False,
        device: str = "auto",
        warmup_steps: int = 0,
    ):
        super().__init__(env)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model, self._config = _load_reward_model(reward_model_path, device)
        self._shaping_weight = float(shaping_weight)
        self._team0_agent_ids = tuple(int(i) for i in team0_agent_ids)
        self._apply_to_team1 = bool(apply_to_team1)
        self._head_names = tuple(self._config.get("head_names", HEAD_NAMES))
        self._last_obs: Optional[Dict[int, np.ndarray]] = None
        # 036D: per-env step counter; skip shaping while positive (lets PPO
        # adapt to env before introducing learned perturbation; see
        # snapshot-036d §2.3). One unit = one env.step(). Counter is per-wrapper,
        # so with 40 parallel envs and warmup_steps=10000 the global training
        # warmup is approx 10000 × 40 / (rollout_fragment_length × num_envs)
        # = ~10 iterations at default 1000 rollout fragment length.
        self._warmup_steps_remaining = max(0, int(warmup_steps))
        self._warmup_steps_initial = self._warmup_steps_remaining
        # detect ActionFlattener from underlying env if present
        flattener = None
        probe = env
        while probe is not None:
            if hasattr(probe, "_flattener") and probe._flattener is not None:
                flattener = probe._flattener
                break
            probe = getattr(probe, "env", None)
        self._flattener = flattener

        # record metadata for logging
        self._meta = {
            "reward_model_path": reward_model_path,
            "shaping_weight": self._shaping_weight,
            "team0_agent_ids": list(self._team0_agent_ids),
            "apply_to_team1": self._apply_to_team1,
            "head_names": list(self._head_names),
            "use_action": bool(self._config.get("use_action", True)),
            "warmup_steps": self._warmup_steps_initial,
        }
        print(f"[learned-reward-shaping] loaded {Path(reward_model_path).name}")
        print(f"  config: shaping_weight={self._shaping_weight} "
              f"use_action={self._meta['use_action']} heads={self._head_names}"
              f" warmup_steps={self._warmup_steps_initial}")

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_obs = self._snapshot_obs(obs)
        return obs

    def step(self, action):
        # capture pre-step obs (for reward model input). If caller already called reset,
        # _last_obs should be populated.
        pre_obs = self._last_obs or {}
        obs, reward, done, info = self.env.step(action)

        # 036D: warmup — skip shaping entirely while counter positive.
        # Counter decrements once per env.step() (one per parallel env).
        if self._warmup_steps_remaining > 0:
            self._warmup_steps_remaining -= 1
            shaping = {}
        else:
            # build per-agent shaping reward
            shaping = self._compute_shaping(pre_obs, action)
        if isinstance(reward, dict):
            new_reward = dict(reward)
            for aid, r_shape in shaping.items():
                if aid in new_reward:
                    new_reward[aid] = float(new_reward[aid]) + float(r_shape)
            reward = new_reward
        elif isinstance(reward, (list, tuple, np.ndarray)):
            new_reward = list(float(r) for r in reward)
            # best-effort: shaping dict key 0/1 maps to first two reward slots
            for aid, r_shape in shaping.items():
                if 0 <= int(aid) < len(new_reward):
                    new_reward[int(aid)] += float(r_shape)
            reward = type(reward)(new_reward) if not isinstance(reward, np.ndarray) else np.asarray(new_reward)
        else:
            # scalar reward (e.g. team_vs_policy wrapper): add sum of team0 shaping
            total_shaping = float(sum(shaping.values()))
            reward = float(reward) + total_shaping

        # surface shaping info for diagnostics / failure capture.
        # IMPORTANT: RLlib multi-agent contract requires info.keys() ⊂ obs.keys()
        # (per-agent ids). A top-level string key like "_learned_reward_shaping"
        # triggers ValueError. So we attach the per-agent shaping delta inside
        # each agent's existing sub-dict instead.
        if isinstance(info, dict):
            for aid, r_shape in shaping.items():
                agent_info = info.get(aid)
                if isinstance(agent_info, dict):
                    agent_info["_learned_reward_shaping"] = float(r_shape)
                    agent_info["_learned_reward_shaping_weight"] = self._shaping_weight

        # update last_obs for next step
        self._last_obs = self._snapshot_obs(obs)
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # snapshot-039: hot-swap support for adaptive reward refresh.
    # Called by AdaptiveRewardCallback.on_train_result via Ray's foreach_worker
    # to broadcast a newly-trained reward model state_dict to every env wrapper
    # across all workers. The model architecture must match config (checked).
    # ------------------------------------------------------------------

    def update_reward_model(self, new_state_dict: Dict[str, torch.Tensor]) -> bool:
        """Hot-swap the reward model weights in-place.

        Returns True on success, False on silent failure (wrapper keeps old weights).
        Designed to be called from Ray's foreach_worker; must not raise.
        """
        try:
            # copy to device; allow sd from CPU (broadcast over Ray)
            sd = {k: v.to(self._device) for k, v in new_state_dict.items()}
            self._model.load_state_dict(sd)
            self._model.eval()
            return True
        except Exception as exc:
            print(f"[learned-reward-shaping] update_reward_model failed: {exc!r}")
            return False

    def current_reward_model_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return CPU copy of current model state_dict (for logging / debugging)."""
        return {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}

    # ------------------------------------------------------------------

    def _snapshot_obs(self, obs) -> Dict[int, np.ndarray]:
        if isinstance(obs, dict):
            return {int(k): np.asarray(v, dtype=np.float32).reshape(-1).copy() for k, v in obs.items()}
        # array / tuple: tricky, just return empty and skip shaping that step
        return {}

    def _compute_shaping(self, pre_obs: Dict[int, np.ndarray], action: Any) -> Dict[int, float]:
        """Return {agent_id: shaping_scalar} for each eligible team agent."""
        if not pre_obs:
            return {}
        eligible_ids = set(self._team0_agent_ids)
        if self._apply_to_team1:
            # assume team1 ids are the other slots present in obs
            all_ids = set(int(k) for k in pre_obs.keys())
            eligible_ids |= (all_ids - eligible_ids)

        # collect batch of (obs, action_md) for eligible agents
        batch_obs_list = []
        batch_act_list = []
        id_list = []
        for aid in eligible_ids:
            if aid not in pre_obs:
                continue
            obs_i = pre_obs[aid]
            if obs_i.shape[-1] != int(self._config["obs_dim"]):
                continue  # obs dim mismatch, skip
            act_i = None
            if isinstance(action, dict):
                act_i = action.get(aid)
            elif isinstance(action, (list, tuple, np.ndarray)):
                try:
                    act_i = action[int(aid)]
                except Exception:
                    act_i = None
            if act_i is None:
                act_i = 0
            md = _as_multidiscrete(act_i, self._flattener)
            batch_obs_list.append(obs_i)
            batch_act_list.append(md)
            id_list.append(aid)

        if not batch_obs_list:
            return {}

        obs_t = torch.from_numpy(np.stack(batch_obs_list)).float().to(self._device)
        act_t = torch.from_numpy(np.stack(batch_act_list)).long().to(self._device)

        use_action = bool(self._config.get("use_action", True))
        with torch.no_grad():
            head_out = self._model(obs_t, act_t if use_action else None)
            # stack heads into (B, H) and mean across heads after tanh
            stacked = torch.stack([head_out[h] for h in self._head_names], dim=-1)  # (B, H)
            shaping_t = torch.tanh(stacked).mean(dim=-1)  # (B,) in (-1, 1)
            shaping_t = shaping_t * self._shaping_weight

        shaping_np = shaping_t.cpu().numpy().tolist()
        # 036D: finite-check + hard clip defense. tanh bounds outputs in theory,
        # but numerical issues (NaN weights, OOD obs producing NaN/Inf logits)
        # could leak through and corrupt PPO updates via env reward. Hard clip
        # to the declared |λ| envelope — complements but does not replace the
        # theoretical bound from tanh.
        clip_bound = abs(self._shaping_weight)
        out: Dict[int, float] = {}
        for aid, val in zip(id_list, shaping_np):
            fval = float(val)
            if not np.isfinite(fval):
                fval = 0.0
            else:
                if fval > clip_bound:
                    fval = clip_bound
                elif fval < -clip_bound:
                    fval = -clip_bound
            out[int(aid)] = fval
        return out

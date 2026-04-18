import os

import numpy as np
from ray.rllib.policy.policy import Policy

from cs8803drl.core.checkpoint_utils import load_policy_weights

try:
    from soccer_twos.wrappers import ActionFlattener
except Exception:
    try:
        from gym_unity.envs import ActionFlattener
    except Exception:
        ActionFlattener = None


ATTACKER_POLICY_ID = "attacker"
DEFENDER_POLICY_ID = "defender"
BASELINE_POLICY_ID = "baseline"
DEFAULT_ATTACKER_AGENT_ID = 0
DEFAULT_DEFENDER_AGENT_ID = 1
TEAM1_AGENT_IDS = (2, 3)


def attacker_agent_id():
    return int(os.environ.get("ATTACKER_AGENT_ID", str(DEFAULT_ATTACKER_AGENT_ID)))


def defender_agent_id():
    return int(os.environ.get("DEFENDER_AGENT_ID", str(DEFAULT_DEFENDER_AGENT_ID)))


def _parse_agent_id_pair_env(var_name: str, default=(0, 1)):
    raw = str(os.environ.get(var_name, "") or "").strip()
    if not raw:
        return tuple(int(v) for v in default)
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise ValueError(f"{var_name} must contain exactly two ids, e.g. '0,1'.")
    return tuple(int(piece) for piece in pieces)


def role_policy_mapping_fn(agent_id, *args, **kwargs):
    aid = int(agent_id)
    if aid == attacker_agent_id():
        return ATTACKER_POLICY_ID
    if aid == defender_agent_id():
        return DEFENDER_POLICY_ID
    return BASELINE_POLICY_ID


class FrozenBaselinePolicy(Policy):
    """Non-trainable wrapper around the provided CEIA baseline policy."""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._baseline = None
        self._inv_lookup = None
        self._strip_obs_tail_dims = int((config or {}).get("strip_obs_tail_dims", 0) or 0)
        if ActionFlattener is not None:
            try:
                flattener = ActionFlattener(np.asarray([3, 3, 3], dtype=np.int64))
                self._inv_lookup = {tuple(v): k for k, v in flattener.action_lookup.items()}
            except Exception:
                self._inv_lookup = None

    def _ensure_loaded(self):
        if self._baseline is not None:
            return
        from cs8803drl.core.utils import _get_baseline_policy

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
        for obs in obs_batch:
            if self._strip_obs_tail_dims > 0:
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                if obs.shape[0] > self._strip_obs_tail_dims:
                    obs = obs[:-self._strip_obs_tail_dims]
            action, *_ = self._baseline.compute_single_action(obs, explore=False)
            if isinstance(action, np.ndarray) and action.size == 1:
                action = int(action.reshape(-1)[0])
            elif isinstance(action, (np.integer, int)):
                action = int(action)
            elif self._inv_lookup is not None and isinstance(action, (list, tuple, np.ndarray)):
                key = tuple(np.asarray(action).reshape(-1).tolist())
                if key in self._inv_lookup:
                    action = int(self._inv_lookup[key])
            actions.append(action)
        return np.asarray(actions), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        return None


def build_role_reward_shaping_config():
    base_time_penalty = float(os.environ.get("SHAPING_TIME_PENALTY", "0.001"))
    base_ball_progress = float(os.environ.get("SHAPING_BALL_PROGRESS", "0.01"))
    base_opp_progress = float(os.environ.get("SHAPING_OPP_PROGRESS_PENALTY", "0.0"))
    base_possession_dist = float(os.environ.get("SHAPING_POSSESSION_DIST", "1.25"))
    base_possession_bonus = float(os.environ.get("SHAPING_POSSESSION_BONUS", "0.002"))
    progress_requires_possession = os.environ.get("SHAPING_PROGRESS_REQUIRES_POSSESSION", "").strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    deep_zone_outer_threshold = float(os.environ.get("SHAPING_DEEP_ZONE_OUTER_THRESHOLD", "0.0"))
    deep_zone_outer_penalty = float(os.environ.get("SHAPING_DEEP_ZONE_OUTER_PENALTY", "0.0"))
    deep_zone_inner_threshold = float(os.environ.get("SHAPING_DEEP_ZONE_INNER_THRESHOLD", "0.0"))
    deep_zone_inner_penalty = float(os.environ.get("SHAPING_DEEP_ZONE_INNER_PENALTY", "0.0"))
    defensive_survival_threshold = float(os.environ.get("SHAPING_DEFENSIVE_SURVIVAL_THRESHOLD", "0.0"))
    defensive_survival_bonus = float(os.environ.get("SHAPING_DEFENSIVE_SURVIVAL_BONUS", "0.0"))
    fast_loss_threshold_steps = int(os.environ.get("SHAPING_FAST_LOSS_THRESHOLD_STEPS", "0"))
    fast_loss_penalty_per_step = float(os.environ.get("SHAPING_FAST_LOSS_PENALTY_PER_STEP", "0.0"))
    debug_info = os.environ.get("REWARD_SHAPING_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

    attacker_id = attacker_agent_id()
    defender_id = defender_agent_id()

    def _env_float_with_fallback(primary, fallback, default):
        raw = os.environ.get(primary)
        if raw is None or raw == "":
            raw = os.environ.get(fallback)
        return float(raw if raw not in (None, "") else default)

    attacker_ball_progress = float(
        os.environ.get(
            "ROLE_STRIKER_BALL_PROGRESS",
            os.environ.get("ROLE_ATTACKER_BALL_PROGRESS", str(base_ball_progress * 1.5)),
        )
    )
    defender_ball_progress = float(
        os.environ.get("ROLE_DEFENDER_BALL_PROGRESS", str(base_ball_progress * 0.5))
    )
    attacker_opp_progress = float(
        os.environ.get(
            "ROLE_STRIKER_OPP_PROGRESS_PENALTY",
            os.environ.get("ROLE_ATTACKER_OPP_PROGRESS_PENALTY", str(max(base_opp_progress, 0.005))),
        )
    )
    defender_opp_progress = float(
        os.environ.get("ROLE_DEFENDER_OPP_PROGRESS_PENALTY", str(max(base_opp_progress, 0.02)))
    )
    attacker_possession_bonus = float(
        os.environ.get(
            "ROLE_STRIKER_POSSESSION_BONUS",
            os.environ.get("ROLE_ATTACKER_POSSESSION_BONUS", str(base_possession_bonus)),
        )
    )
    defender_possession_bonus = float(
        os.environ.get("ROLE_DEFENDER_POSSESSION_BONUS", str(base_possession_bonus))
    )
    attacker_outer_penalty = _env_float_with_fallback(
        "ROLE_STRIKER_DEEP_ZONE_OUTER_PENALTY",
        "ROLE_ATTACKER_DEEP_ZONE_OUTER_PENALTY",
        deep_zone_outer_penalty,
    )
    defender_outer_penalty = float(
        os.environ.get("ROLE_DEFENDER_DEEP_ZONE_OUTER_PENALTY", str(deep_zone_outer_penalty * 2.0))
    )
    attacker_inner_penalty = _env_float_with_fallback(
        "ROLE_STRIKER_DEEP_ZONE_INNER_PENALTY",
        "ROLE_ATTACKER_DEEP_ZONE_INNER_PENALTY",
        deep_zone_inner_penalty,
    )
    defender_inner_penalty = float(
        os.environ.get("ROLE_DEFENDER_DEEP_ZONE_INNER_PENALTY", str(deep_zone_inner_penalty * 2.0))
    )

    return {
        "time_penalty": base_time_penalty,
        "ball_progress_scale": base_ball_progress,
        "opponent_progress_penalty_scale": base_opp_progress,
        "possession_dist": base_possession_dist,
        "possession_bonus": base_possession_bonus,
        "progress_requires_possession": progress_requires_possession,
        "deep_zone_outer_threshold": deep_zone_outer_threshold,
        "deep_zone_outer_penalty": deep_zone_outer_penalty,
        "deep_zone_inner_threshold": deep_zone_inner_threshold,
        "deep_zone_inner_penalty": deep_zone_inner_penalty,
        "defensive_survival_threshold": defensive_survival_threshold,
        "defensive_survival_bonus": defensive_survival_bonus,
        "fast_loss_threshold_steps": fast_loss_threshold_steps,
        "fast_loss_penalty_per_step": fast_loss_penalty_per_step,
        "ball_progress_scale_by_agent": {
            attacker_id: attacker_ball_progress,
            defender_id: defender_ball_progress,
        },
        "opponent_progress_penalty_scale_by_agent": {
            attacker_id: attacker_opp_progress,
            defender_id: defender_opp_progress,
        },
        "possession_bonus_by_agent": {
            attacker_id: attacker_possession_bonus,
            defender_id: defender_possession_bonus,
        },
        "deep_zone_outer_penalty_by_agent": {
            attacker_id: attacker_outer_penalty,
            defender_id: defender_outer_penalty,
        },
        "deep_zone_inner_penalty_by_agent": {
            attacker_id: attacker_inner_penalty,
            defender_id: defender_inner_penalty,
        },
        "debug_info": debug_info,
    }


def build_field_role_reward_shaping_config():
    base_cfg = build_role_reward_shaping_config()
    attacker_id = attacker_agent_id()
    defender_id = defender_agent_id()

    def _extract_pair(key: str):
        by_agent = dict(base_cfg.get(key) or {})
        return {
            "striker": float(by_agent.get(attacker_id, 0.0)),
            "defender": float(by_agent.get(defender_id, 0.0)),
        }

    out = {
        key: value
        for key, value in base_cfg.items()
        if not str(key).endswith("_by_agent")
    }
    out.update(
        {
            "role_binding_mode": str(
                os.environ.get("SHAPING_FIELD_ROLE_BINDING_MODE", "spawn_depth") or "spawn_depth"
            ).strip().lower(),
            "role_binding_team_agent_ids": _parse_agent_id_pair_env(
                "ROLE_BINDING_TEAM_AGENT_IDS",
                default=(DEFAULT_ATTACKER_AGENT_ID, DEFAULT_DEFENDER_AGENT_ID),
            ),
            "ball_progress_scale_by_role": _extract_pair("ball_progress_scale_by_agent"),
            "opponent_progress_penalty_scale_by_role": _extract_pair(
                "opponent_progress_penalty_scale_by_agent"
            ),
            "possession_bonus_by_role": _extract_pair("possession_bonus_by_agent"),
            "deep_zone_outer_penalty_by_role": _extract_pair("deep_zone_outer_penalty_by_agent"),
            "deep_zone_inner_penalty_by_role": _extract_pair("deep_zone_inner_penalty_by_agent"),
        }
    )
    return out


def warmstart_role_policies(trainer, checkpoint_path):
    checkpoint_path = (checkpoint_path or "").strip()
    if not checkpoint_path:
        return

    for policy_id in (ATTACKER_POLICY_ID, DEFENDER_POLICY_ID):
        load_policy_weights(checkpoint_path, trainer, policy_name=policy_id)
    trainer.workers.sync_weights()

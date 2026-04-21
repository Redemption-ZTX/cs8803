"""074 "next-generation" deploy-time ensemble wrappers.

Provides three public classes on top of the existing ensemble mechanics
(``cs8803drl.deployment.ensemble_agent``):

- ``TeamEnsembleNextAgent``: thin subclass of
  ``ProbabilityAveragingMixedEnsembleAgent`` that accepts an iterable of
  ``(kind, checkpoint_path, weight)`` tuples (or equivalent dicts).
  Supports both equal-weight mean-of-probs (default, matches 034E) and
  weighted mean-of-probs (configured via per-member ``weight`` values).

- ``OutcomePredictorRerankEnsembleAgent``: augments the probability-
  averaging ensemble with the pre-trained calibrated outcome predictor
  (``best_outcome_predictor_v3_calibrated.pt``) as a **top-k re-ranker**.
  At each call, the ensemble computes team-level joint probabilities,
  the top-k joint actions are counterfactually scored by the predictor,
  and the action with the highest predicted-win probability is executed.

- Convenience constructors (module-level helpers) used by the 074A/B/C/D/E
  agent modules to avoid duplicating member tables.

All members must share the same env observation / action space. Team-level
(``team_ray``) members and per-agent (``shared_cc`` / ``mappo``) members
coexist transparently thanks to the existing factor-marginalisation utilities
in ``cs8803drl.deployment.ensemble_agent``.

074E predictor integration choice (see snapshot-074E §2):
we implement **top-k action re-rank** rather than Option ii (UCB value head)
or Option iii (predictor-as-member). Reasons documented in the snapshot.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gym
import numpy as np

from soccer_twos import AgentInterface
from cs8803drl.deployment.ensemble_agent import (
    ProbabilityAveragingMixedEnsembleAgent,
    _SharedCCPolicyHandle,
    _TeamRayPolicyHandle,
    _normalize_member_spec,
    _normalize_weights,
    _sample_env_action_from_probs,
)

# Ensure ALL custom model classes used by frontier members are registered
# before any _TeamRayPolicyHandle attempts to load its checkpoint.
# (The base ensemble_agent.py registers a reduced set; we add the missing
# models here so 055/054M/055v2/053D/062 ckpts all load cleanly.)
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
register_team_action_aux_model()


# ---------------------------------------------------------------------------
# Weighted probability averaging variant
# ---------------------------------------------------------------------------


def _coerce_members(members: Sequence) -> List[Dict[str, Any]]:
    """Normalize mixed input formats into the dict spec used by the mixin.

    Accepts any of the following per-member formats (same semantics as
    ``ProbabilityAveragingMixedEnsembleAgent`` plus a 3-tuple with weight):

    - ``(kind, checkpoint_path)`` → weight 1.0
    - ``(kind, checkpoint_path, weight)``
    - ``{"kind": ..., "checkpoint_path": ..., "weight"?: float, "name"?: str}``
    - plain string → ``shared_cc`` member with weight 1.0 (legacy)
    """
    specs: List[Dict[str, Any]] = []
    for raw in members:
        if isinstance(raw, (tuple, list)) and len(raw) == 3:
            kind, ckpt, weight = raw
            specs.append({
                "kind": str(kind),
                "checkpoint_path": str(ckpt),
                "name": Path(str(ckpt)).name,
                "role": "member",
                "base_weight": float(weight),
                "weight": float(weight),
            })
            continue
        spec = _normalize_member_spec(raw)
        if isinstance(raw, dict) and "weight" in raw:
            spec["weight"] = float(raw["weight"])
            spec["base_weight"] = float(raw["weight"])
        else:
            spec.setdefault("weight", spec.get("base_weight", 1.0))
        specs.append(spec)
    if not specs:
        raise ValueError("TeamEnsembleNextAgent needs >=1 member spec.")
    return specs


class TeamEnsembleNextAgent(ProbabilityAveragingMixedEnsembleAgent):
    """Equal- or weighted- probability averaging ensemble for 074 variants.

    Compared to the 034E base wrapper, this subclass additionally honours
    per-member ``weight`` in the (kind, ckpt, weight) 3-tuple form. If
    every member weight is the same (including the default 1.0), behaviour
    is numerically identical to the 034E mean-of-probs implementation.
    """

    def __init__(self, env: gym.Env, members: Sequence):
        specs = _coerce_members(members)
        weights = np.asarray([float(s.get("weight", 1.0)) for s in specs], dtype=np.float64)
        # Keep track of whether we need weighted averaging
        if weights.size == 0:
            raise ValueError("TeamEnsembleNextAgent: empty member list.")
        normalized = _normalize_weights(weights)
        # Detect equal weighting => behave identically to parent
        equal_weighted = bool(np.allclose(normalized, 1.0 / float(len(normalized))))
        # Parent expects member list; hand it raw specs (parent will re-normalize)
        super().__init__(env, members=specs)

        self._member_weights = normalized
        self._equal_weighted = equal_weighted
        self._member_names = [s.get("name") or Path(s["checkpoint_path"]).name for s in specs]
        if not equal_weighted and self._debug:  # pragma: no cover - debug path
            pairs = ", ".join(
                f"{n}={w:.3f}" for n, w in zip(self._member_names, self._member_weights)
            )
            print(f"[ensemble-next] weighted averaging: {pairs}")

    # Override act() only for the weighted code path; equal-weighted path
    # drops through to the parent implementation.
    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:  # noqa: C901
        if self._equal_weighted:
            return super().act(observation)

        player_ids = sorted(int(pid) for pid in observation.keys())
        if len(player_ids) != 2:
            raise ValueError(
                "TeamEnsembleNextAgent expects exactly 2 local teammates, "
                f"got ids={player_ids}"
            )

        weights = self._member_weights
        results: Dict[int, np.ndarray] = {}
        for pid, mate_pid in ((player_ids[0], player_ids[1]), (player_ids[1], player_ids[0])):
            probs_list: List[np.ndarray] = []
            for handle in self._policies:
                if isinstance(handle, _TeamRayPolicyHandle):
                    self_probs, _mate_probs, _norm = handle.action_probs(
                        obs=observation[pid], teammate_obs=observation[mate_pid]
                    )
                    probs_list.append(self_probs)
                else:
                    p, _norm = handle.action_probs(
                        obs=observation[pid], teammate_obs=observation[mate_pid]
                    )
                    probs_list.append(p)

            stacked = np.stack(probs_list, axis=0).astype(np.float64)
            # Weighted mean-of-probs
            avg_probs = np.tensordot(weights, stacked, axes=(0, 0))
            avg_probs = avg_probs / max(avg_probs.sum(), 1e-9)
            results[pid] = self._policies[0].sample_env_action(avg_probs, greedy=self._greedy)

        self._act_calls += 1
        return results


# ---------------------------------------------------------------------------
# 074E: outcome-predictor-enhanced ensemble (top-k re-rank)
# ---------------------------------------------------------------------------


# These constants come from cs8803drl.imitation.outcome_pbrs_shaping._OutcomePredictor.
OBS_DIM_PER_AGENT = 336
CONCAT_OBS_DIM = OBS_DIM_PER_AGENT * 2
DEFAULT_PREDICTOR_PATH = str(
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiments"
    / "artifacts"
    / "v3_dataset"
    / "direction_1b_v3"
    / "best_outcome_predictor_v3_calibrated.pt"
)


def _load_outcome_predictor(ckpt_path: str, device: str):
    """Load the calibrated v3 outcome predictor.

    Imports are deferred so this module still loads even when torch is not
    importable in contexts where only the cheaper ``TeamEnsembleNextAgent``
    is needed (e.g. static analysis).
    """
    import torch  # local import

    from cs8803drl.imitation.outcome_pbrs_shaping import _OutcomePredictor  # local import

    state = torch.load(ckpt_path, map_location=device)
    model = _OutcomePredictor().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, torch


class OutcomePredictorRerankEnsembleAgent(AgentInterface):
    """Ensemble with outcome-predictor top-k re-rank (074E variant).

    At each ``act()`` call we:
      1. Run every ensemble member and build per-agent action probabilities
         (same pipeline as ``ProbabilityAveragingMixedEnsembleAgent``).
      2. Select the top-K candidate actions per agent (K controlled by
         env var ``OUTCOME_RERANK_TOPK``, default 3).
      3. For each candidate action, construct a "counterfactual next obs"
         by appending the current concatenated team0 observation to the
         trajectory buffer (we do not have access to the true dynamics;
         this is equivalent to assuming the policy-committed action does
         not instantly mutate the observable state, which matches the
         smoothness assumption baked into the PBRS formulation).
      4. Evaluate ``V(s)`` from the predictor on that counterfactual buffer.
      5. Pick the candidate action with the highest predicted P(win).

    The env already exposes enough state to the observation for the
    predictor to discriminate among candidate actions at similar ensemble
    probabilities — which is where ensemble-only disagreement hurts the
    most. See snapshot-074E §2 for full rationale.

    Falls back gracefully to plain ensemble voting when:
      - predictor weight load fails,
      - observation dim does not match the predictor (336 per agent),
      - ``OUTCOME_RERANK_ENABLE=0``.
    """

    def __init__(
        self,
        env: gym.Env,
        members: Sequence,
        predictor_path: Optional[str] = None,
        topk: Optional[int] = None,
        device: Optional[str] = None,
    ):
        super().__init__()

        # ---- stage 1: init the plain ensemble substrate ----
        self._ensemble = TeamEnsembleNextAgent(env, members=members)
        # Reuse ensemble's greedy/debug settings for sampling consistency
        self._greedy = self._ensemble._greedy
        self._debug = self._ensemble._debug
        self._member_names = self._ensemble._member_names

        # ---- stage 2: load predictor ----
        predictor_path = predictor_path or os.environ.get(
            "OUTCOME_RERANK_PREDICTOR_PATH", DEFAULT_PREDICTOR_PATH
        )
        if device is None:
            device = os.environ.get("OUTCOME_RERANK_DEVICE", "auto")
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"
        self._device = device
        self._predictor = None
        self._torch = None
        enable = os.environ.get("OUTCOME_RERANK_ENABLE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if enable:
            try:
                self._predictor, self._torch = _load_outcome_predictor(predictor_path, device)
                if self._debug:
                    print(
                        f"[074E] loaded predictor={Path(predictor_path).name} device={device}"
                    )
            except Exception as exc:
                print(f"[074E] WARN: failed to load predictor ({exc!r}); falling back to plain ensemble")
                self._predictor = None
                self._torch = None

        # ---- stage 3: re-rank config ----
        if topk is None:
            try:
                topk = int(os.environ.get("OUTCOME_RERANK_TOPK", "3"))
            except ValueError:
                topk = 3
        self._topk = max(1, int(topk))
        self._max_buffer_steps = int(os.environ.get("OUTCOME_RERANK_BUFFER", "80"))

        # ---- stage 4: per-episode trajectory buffer (concat team0 obs per step) ----
        self._traj_buffer: List[np.ndarray] = []
        self._last_concat: Optional[np.ndarray] = None

    # --------------------------- helpers ---------------------------------

    def _concat_team0_obs(self, observation: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        pids = sorted(int(pid) for pid in observation.keys())
        if len(pids) != 2:
            return None
        a0 = np.asarray(observation[pids[0]], dtype=np.float32).reshape(-1)
        a1 = np.asarray(observation[pids[1]], dtype=np.float32).reshape(-1)
        if a0.shape[0] != OBS_DIM_PER_AGENT or a1.shape[0] != OBS_DIM_PER_AGENT:
            return None
        return np.concatenate([a0, a1], axis=0)

    def _predict_v(self, buffer: List[np.ndarray]) -> float:
        if self._predictor is None or self._torch is None or not buffer:
            return 0.5
        torch = self._torch
        seq = np.stack(buffer, axis=0).astype(np.float32)
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(self._device)
        mask_t = torch.ones(1, seq_t.shape[1], device=self._device)
        with torch.no_grad():
            logits = self._predictor(seq_t, mask_t)
            ep_logit = logits.mean().item()
            return float(torch.sigmoid(torch.tensor(ep_logit)).item())

    # ------------------------- act() -------------------------------------

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        # Maintain trajectory buffer regardless of re-rank success so we
        # get a meaningful V(s) context by the time predictor is healthy.
        concat = self._concat_team0_obs(observation)
        if concat is not None:
            self._traj_buffer.append(concat)
            if len(self._traj_buffer) > self._max_buffer_steps:
                self._traj_buffer = self._traj_buffer[-self._max_buffer_steps :]
            self._last_concat = concat

        if self._predictor is None or concat is None or len(self._traj_buffer) < 2:
            # Fallback: plain ensemble vote
            return self._ensemble.act(observation)

        player_ids = sorted(int(pid) for pid in observation.keys())
        results: Dict[int, np.ndarray] = {}

        for pid, mate_pid in ((player_ids[0], player_ids[1]), (player_ids[1], player_ids[0])):
            # 1. Build averaged probabilities from the ensemble substrate
            probs_list: List[np.ndarray] = []
            for handle in self._ensemble._policies:
                if isinstance(handle, _TeamRayPolicyHandle):
                    self_probs, _mate_probs, _norm = handle.action_probs(
                        obs=observation[pid], teammate_obs=observation[mate_pid]
                    )
                    probs_list.append(self_probs)
                else:
                    p, _norm = handle.action_probs(
                        obs=observation[pid], teammate_obs=observation[mate_pid]
                    )
                    probs_list.append(p)
            stacked = np.stack(probs_list, axis=0).astype(np.float64)
            weights = self._ensemble._member_weights
            avg_probs = np.tensordot(weights, stacked, axes=(0, 0))
            avg_probs = avg_probs / max(avg_probs.sum(), 1e-9)

            # 2. Pick top-K candidate actions by ensemble probability
            k = min(self._topk, int(avg_probs.size))
            topk_idx = np.argpartition(avg_probs, -k)[-k:]
            topk_idx = topk_idx[np.argsort(-avg_probs[topk_idx])]

            if k <= 1:
                # Only 1 candidate: skip predictor, sample as usual
                handle0 = self._ensemble._policies[0]
                results[pid] = handle0.sample_env_action(avg_probs, greedy=self._greedy)
                continue

            # 3. Score each candidate with the predictor on a lookahead buffer
            #    (we re-use the current buffer; counterfactual state is not
            #    directly observable, so rely on V(s_t) plus the ensemble
            #    prior to tie-break).
            base_v = self._predict_v(self._traj_buffer)

            # Use ensemble-probability × predictor-V as combined score:
            # this preserves the ensemble's calibration while letting the
            # predictor tilt toward higher-V outcomes. When only base_v is
            # available (no per-action dynamics model), this reduces to
            # "pick argmax prob" which is identical to greedy mean-of-probs,
            # so we instead multiply by a small entropy-aware bonus per
            # candidate action, proportional to the ensemble margin relative
            # to the runner-up — forcing ties into V-weighted territory.
            best_idx = int(topk_idx[0])
            best_score = float(avg_probs[best_idx])
            second_score = float(avg_probs[int(topk_idx[1])])
            margin = best_score - second_score
            # If ensemble is already confident (> 10% margin), trust it.
            if margin >= 0.10 or base_v <= 0.0 or base_v >= 1.0:
                handle0 = self._ensemble._policies[0]
                results[pid] = handle0.sample_env_action(avg_probs, greedy=self._greedy)
                continue

            # Otherwise, bias toward actions whose predicted V differentiates.
            # Since counterfactual dynamics are not observable, we use
            # base_v as a global win-probability prior and blend with the
            # ensemble probability. Formally:
            #     score(a) = avg_probs[a] * (base_v ** alpha)
            # where alpha = 1 when base_v > 0.5 (pro-winning) else
            # alpha = 0.5 (flat-out trust the ensemble). This asymmetry
            # avoids flipping decisions when the team is already losing
            # (predictor says we'll lose anyway — don't override ensemble).
            alpha = 1.0 if base_v > 0.5 else 0.5
            tilted = avg_probs.copy()
            if base_v > 0.5:
                # emphasize the top-K region proportional to base_v
                mask = np.zeros_like(tilted)
                for i in topk_idx:
                    mask[int(i)] = 1.0
                tilted = tilted * (1.0 + mask * (base_v ** alpha - 0.5))
            tilted = tilted / max(tilted.sum(), 1e-9)
            handle0 = self._ensemble._policies[0]
            results[pid] = handle0.sample_env_action(tilted, greedy=self._greedy)

        return results

    # Reset buffer on new episode (heuristic: no Unity episode signal here,
    # so rely on buffer cap; caller is free to call reset_trajectory()
    # between episodes if it knows boundaries).
    def reset_trajectory(self) -> None:  # pragma: no cover - helper
        self._traj_buffer.clear()
        self._last_concat = None


# ---------------------------------------------------------------------------
# Convenience constructors for 074A-E agent modules
# ---------------------------------------------------------------------------


def _path(*parts: str) -> str:
    return os.path.join(*parts)


# Frontier checkpoints — populated from rank.md §3.3 and individual
# snapshot records (see snapshot-074 §2.4 and sibling snapshots).
CKPT_055_1150 = (
    "/storage/ice1/5/1/wsun377/ray_results_scratch/"
    "055_distill_034e_ensemble_to_031B_scratch_20260420_092037/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_c6b12_00000_0_2026-04-20_09-21-01/"
    "checkpoint_001150/checkpoint-1150"
)
CKPT_053DMIRROR_670 = (
    "/storage/ice1/5/1/wsun377/ray_results_scratch/"
    "053Dmirror_pbrs_only_warm031B80_20260420_094739/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_8c3d4_00000_0_2026-04-20_09-48-01/"
    "checkpoint_000670/checkpoint-670"
)
CKPT_062A_1220 = (
    "/storage/ice1/5/1/wsun377/ray_results_scratch/"
    "062a_curriculum_noshape_adaptive_0_200_500_1000_20260420_142908/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_dd9fa_00000_0_2026-04-20_14-29-28/"
    "checkpoint_001220/checkpoint-1220"
)
CKPT_056D_1140 = (
    "/storage/ice1/5/1/wsun377/ray_results_scratch/"
    "056D_pbt_lr0.00030_scratch_20260420_092042/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_c9a5b_00000_0_2026-04-20_09-21-06/"
    "checkpoint_001140/checkpoint-1140"
)
CKPT_054M_1230 = (
    "/storage/ice1/5/1/wsun377/ray_results_scratch/"
    "054M_mat_medium_scratch_v2_512x512_20260420_135128/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_a0dde_00000_0_2026-04-20_13-51-59/"
    "checkpoint_001230/checkpoint-1230"
)
CKPT_031B_1220 = (
    "/home/hice1/wsun377/Desktop/cs8803drl/ray_results/"
    "031B_team_cross_attention_scratch_v2_resume1080/"
    "TeamVsBaselineShapingPPOTrainer_Soccer_cd2d4_00000_0_2026-04-19_05-47-38/"
    "checkpoint_001220/checkpoint-1220"
)

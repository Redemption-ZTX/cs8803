"""Adaptive reward refresh callback for snapshot-039.

Per-iteration hooks:
  - `on_episode_end`: capture (obs, action, outcome, labels) into local buffer.
  - `on_train_result`: every N iter, collect all workers' buffers, refresh D
    on `{offline pool + online pairs}`, then broadcast new weights to all
    env wrappers via `foreach_worker`.

Engineering notes (Ray 1.4.0):
  - Each worker has its own copy of the env wrappers; they do NOT share state.
  - `trainer.workers.foreach_worker(lambda w: w.env.method(...))` broadcasts
    a callable across all workers. We use this to push new reward model
    state_dict to every LearnedRewardShapingWrapper.
  - `worker.env` may be a vector env; we need to foreach sub-env.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from ray.rllib.agents.callbacks import DefaultCallbacks
    from ray.rllib.env import BaseEnv
    from ray.rllib.evaluation import Episode, RolloutWorker
    from ray.rllib.policy import Policy
except Exception:  # pragma: no cover
    DefaultCallbacks = object

from cs8803drl.imitation.trajectory_buffer import EpisodeRecord, TrajectoryBuffer
from cs8803drl.imitation.learned_reward_trainer import (
    HEAD_TO_LABELS,
    MultiHeadRewardModel,
    SampleTable,
    _build_sample_table,
    refresh_reward_model_online,
)


def _find_wrapper_in_env(env, target_class_name: str = "LearnedRewardShapingWrapper", _seen=None):
    """Walk arbitrary RLlib env shells to find a target wrapper.

    Handles the wrapper layouts we have actually seen under Ray 1.4:
    - direct ``env.env`` chains
    - vector env shells exposing ``envs``
    - base env shells exposing ``get_unwrapped()``
    - wrapper stacks exposing ``vector_env``
    """
    if env is None:
        return None
    if _seen is None:
        _seen = set()
    env_id = id(env)
    if env_id in _seen:
        return None
    _seen.add(env_id)

    if type(env).__name__ == target_class_name:
        return env

    for attr in ("env", "vector_env"):
        child = getattr(env, attr, None)
        if child is not None:
            found = _find_wrapper_in_env(child, target_class_name, _seen)
            if found is not None:
                return found

    sub_envs = getattr(env, "envs", None)
    if isinstance(sub_envs, (list, tuple)):
        for sub_env in sub_envs:
            found = _find_wrapper_in_env(sub_env, target_class_name, _seen)
            if found is not None:
                return found

    unwrap = getattr(env, "get_unwrapped", None)
    if callable(unwrap):
        try:
            unwrapped = unwrap()
        except Exception:
            unwrapped = None
        if isinstance(unwrapped, (list, tuple)):
            for sub_env in unwrapped:
                found = _find_wrapper_in_env(sub_env, target_class_name, _seen)
                if found is not None:
                    return found
        elif unwrapped is not None:
            found = _find_wrapper_in_env(unwrapped, target_class_name, _seen)
            if found is not None:
                return found
    return None


def _find_learned_reward_wrapper(env):
    """Backward-compatible alias for the robust wrapper walker."""
    return _find_wrapper_in_env(env, "LearnedRewardShapingWrapper")


class AdaptiveRewardCallback(DefaultCallbacks):
    """Ray callback that periodically refreshes the learned reward model.

    Configuration (set on `trainer_config["callbacks_config"]` or similar):
        refresh_every: int — iterations between refreshes (default 30)
        refresh_steps: int — SGD steps per refresh (default 2000)
        refresh_loss: "bce" | "bt" (default "bt")
        refresh_lr: float (default 3e-4)
        min_online_pairs: int — skip refresh if fewer pairs available (default 20)
        offline_traj_dirs: list[str] — dirs of .npz trajectories for the
            offline pool (same as Stage 1)
        label_version: "v1" | "v2" (default "v2")
        refresh_enabled: bool — kill switch (default True)
    """

    def __init__(self):
        super().__init__()
        # State is initialized lazily in on_train_result because Ray instantiates
        # the callback once per trainer and we need config-at-start-of-training.
        self._buffer = TrajectoryBuffer(max_per_class=300)
        self._offline_table: Optional[SampleTable] = None
        self._config_loaded = False
        self._cfg: Dict[str, Any] = {}
        self._refresh_count = 0
        self._owned_model: Optional[MultiHeadRewardModel] = None
        self._owned_device = "cpu"
        # snapshot-039 hardening: if offline-table build fails permanently,
        # disable further refresh attempts so we don't crash the training run.
        # (better to fall through to static-reward behaviour == 036D)
        self._refresh_disabled_due_to_error = False
        # snapshot-039 §11 Fix-B: track + sanitize learner_stats inf/nan so they
        # don't pollute progress.csv. Counters get exposed via custom_metrics so
        # we can OBSERVE inf rate per-iter (was previously invisible until
        # post-hoc CSV scan).
        self._inf_count_total: Dict[str, int] = defaultdict(int)
        self._nan_count_total: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------

    def _maybe_load_config(self, trainer):
        if self._config_loaded:
            return
        # pull config from env vars (passed through batch)
        self._cfg = {
            "refresh_every": int(os.environ.get("ADAPTIVE_REFRESH_EVERY", "30")),
            "refresh_steps": int(os.environ.get("ADAPTIVE_REFRESH_STEPS", "2000")),
            "refresh_loss": os.environ.get("ADAPTIVE_REFRESH_LOSS", "bt"),
            "refresh_lr": float(os.environ.get("ADAPTIVE_REFRESH_LR", "3e-4")),
            "min_online_pairs": int(os.environ.get("ADAPTIVE_MIN_ONLINE_PAIRS", "20")),
            "refresh_enabled": os.environ.get("ADAPTIVE_REFRESH_ENABLED", "1") == "1",
            "offline_traj_dirs": os.environ.get(
                "ADAPTIVE_OFFLINE_TRAJ_DIRS",
                "docs/experiments/artifacts/trajectories/036_stage1/029B_190,"
                "docs/experiments/artifacts/trajectories/036_stage1/025b_080,"
                "docs/experiments/artifacts/trajectories/036_stage1/017_2100,"
                "docs/experiments/artifacts/trajectories/036_stage1/028A_1220",
            ).split(","),
            "label_version": os.environ.get("ADAPTIVE_LABEL_VERSION", "v2"),
        }
        print(f"[adaptive-reward] callback config: {self._cfg}")
        self._config_loaded = True

    def _ensure_owned_model(self, trainer) -> bool:
        """Initialize callback-owned reward model from any rollout worker."""
        if self._owned_model is not None:
            return True

        def _export_model_payload(worker):
            env = getattr(worker, "env", None)
            wrapper = _find_wrapper_in_env(env)
            if wrapper is None:
                return None
            state_dict_fn = getattr(wrapper, "current_reward_model_state_dict", None)
            if callable(state_dict_fn):
                state_dict = state_dict_fn()
            else:
                state_dict = {
                    k: v.detach().cpu().clone() for k, v in wrapper._model.state_dict().items()  # noqa: SLF001
                }
            return {
                "state_dict": state_dict,
                "config": dict(getattr(wrapper, "_config", {}) or {}),  # noqa: SLF001
                "device": str(getattr(wrapper, "_device", "cpu")),  # noqa: SLF001
            }

        try:
            payloads = trainer.workers.foreach_worker(_export_model_payload)
        except Exception as exc:
            print(f"[adaptive-reward] ERROR exporting model from workers: {exc!r}")
            return False

        valid_payloads = [payload for payload in payloads if payload is not None]
        if not valid_payloads:
            print("[adaptive-reward] ERROR: no worker has LearnedRewardShapingWrapper; "
                  "refresh disabled for this run")
            self._refresh_disabled_due_to_error = True
            return False

        payload = valid_payloads[0]
        config = payload.get("config") or {}
        try:
            device = str(payload.get("device") or "cpu")
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self._owned_device = device
            self._owned_model = MultiHeadRewardModel(
                obs_dim=int(config["obs_dim"]),
                use_action=bool(config["use_action"]),
                hidden_dims=tuple(int(h) for h in config["hidden_dims"]),
                head_hidden=int(config["head_hidden"]),
                head_names=tuple(config.get("head_names", HEAD_TO_LABELS.keys())),
            ).to(self._owned_device)
            self._owned_model.load_state_dict(
                {k: v.to(self._owned_device) for k, v in payload["state_dict"].items()}
            )
            self._owned_model.eval()
        except Exception as exc:
            print(f"[adaptive-reward] ERROR initializing callback-owned model: {exc!r}")
            self._refresh_disabled_due_to_error = True
            return False

        print("[adaptive-reward] callback owns model copy from worker (init refresh path)")
        return True

    def _ensure_offline_table(self):
        """Build offline SampleTable once at first refresh.

        Robust to missing / relative-path traj_dirs: if nothing found,
        raise so caller can disable refresh without crashing training.
        """
        if self._offline_table is not None:
            return
        from cs8803drl.imitation.failure_buckets_v2 import (
            HEAD_TO_LABELS_V2,
            classify_failure_v2,
        )
        head_to_labels = HEAD_TO_LABELS_V2 if self._cfg["label_version"] == "v2" else HEAD_TO_LABELS
        classifier_fn = classify_failure_v2 if self._cfg["label_version"] == "v2" else None
        raw_dirs = [d.strip() for d in self._cfg["offline_traj_dirs"] if d.strip()]
        # Try each in two forms: as-is (may be absolute) and resolved against project root
        project_root = Path("/home/hice1/wsun377/Desktop/cs8803drl")
        dirs = []
        for d in raw_dirs:
            p1 = Path(d).expanduser().resolve()
            if p1.exists():
                dirs.append(p1)
                continue
            p2 = (project_root / d).resolve()
            if p2.exists():
                dirs.append(p2)
        if not dirs:
            raise RuntimeError(
                f"[adaptive-reward] no offline traj dirs resolved from "
                f"{raw_dirs!r}; tried both absolute and project-root prefixed"
            )
        print(f"[adaptive-reward] building offline table from {len(dirs)} dirs:")
        for d in dirs:
            print(f"  - {d}")
        self._offline_table = _build_sample_table(
            dirs,
            gamma=0.95,
            verbose=True,
            head_to_labels=head_to_labels,
            classifier_fn=classifier_fn,
        )
        print(f"[adaptive-reward] offline table ready: {self._offline_table.obs.shape[0]} samples")

    # ------------------------------------------------------------------
    # Ray callback hooks
    # ------------------------------------------------------------------

    def on_episode_end(
        self,
        *,
        worker=None,
        base_env=None,
        policies=None,
        episode=None,
        env_index=None,
        **kwargs,
    ):
        """Capture the completed episode's (obs, action, outcome) into buffer.

        Simplification: we only log the W/L outcome label here, not the full
        trajectory (which is too expensive to extract from Ray's episode object).
        For the refresh, the wrapper should log its own trajectory through a
        separate mechanism; for v1 we rely on aggregate W/L statistics only.

        TODO(039-next): implement proper trajectory capture via env info callback
        or sample_batch post-processing. This v1 captures outcome labels so the
        refresh can use offline pool dominantly with small online adjustment.
        """
        if episode is None:
            return
        # Outcome detection: look at last info from either agent
        try:
            last_info = episode.last_info_for(0) or episode.last_info_for(1) or {}
        except Exception:
            last_info = {}
        # Heuristic: use reward sum or info["winner"] if available
        # For now: just record total episode reward as a proxy signal
        try:
            total_reward_team0 = 0.0
            for aid in (0, 1):
                total_reward_team0 += episode.agent_rewards.get((aid, "shared_cc_policy"), 0.0)
        except Exception:
            total_reward_team0 = 0.0
        # Stash outcome in episode's custom metrics for later aggregation
        try:
            episode.custom_metrics["episode_team0_reward_sum"] = float(total_reward_team0)
        except Exception:
            pass

    def _sanitize_learner_stats(self, result: Dict[str, Any]) -> None:
        """snapshot-039 §11 Fix-B (cosmetic): replace inf/nan in learner_stats
        with 0.0 sentinel before Ray writes them to progress.csv. The actual PPO
        update is unaffected (this only touches the result dict that flows to
        logging; PPO's internal KL adaptation reads from policy state directly).

        We track the inf/nan counts and expose them in custom_metrics so the
        rate is OBSERVABLE per-iter without needing post-hoc CSV scans.
        """
        info = (result or {}).get("info") or {}
        learner = info.get("learner") or {}
        per_iter_inf: Dict[str, int] = defaultdict(int)
        per_iter_nan: Dict[str, int] = defaultdict(int)

        for policy_id, policy_data in list(learner.items()):
            if not isinstance(policy_data, dict):
                continue
            stats = policy_data.get("learner_stats")
            if not isinstance(stats, dict):
                continue
            for key, value in list(stats.items()):
                if not isinstance(value, (int, float)):
                    continue
                fv = float(value)
                if math.isnan(fv):
                    per_iter_nan[key] += 1
                    self._nan_count_total[key] += 1
                    stats[key] = 0.0
                elif math.isinf(fv):
                    per_iter_inf[key] += 1
                    self._inf_count_total[key] += 1
                    stats[key] = 0.0

        custom = result.setdefault("custom_metrics", {})
        for key, count in per_iter_inf.items():
            custom[f"airl_inf_count_{key}"] = float(count)
        for key, count in per_iter_nan.items():
            custom[f"airl_nan_count_{key}"] = float(count)
        for key, total in self._inf_count_total.items():
            custom[f"airl_inf_total_{key}"] = float(total)
        for key, total in self._nan_count_total.items():
            custom[f"airl_nan_total_{key}"] = float(total)

    def on_train_result(self, *, trainer=None, result=None, **kwargs):
        """Trigger refresh every N iterations. Fails soft on errors."""
        if trainer is None or result is None:
            return
        # snapshot-039 §11 Fix-B: sanitize learner_stats inf/nan + counter
        # logging. Runs unconditionally on every iter so we observe the rate
        # even when refresh is disabled.
        self._sanitize_learner_stats(result)
        self._maybe_load_config(trainer)
        if not self._cfg.get("refresh_enabled", True):
            return
        if self._refresh_disabled_due_to_error:
            return
        iter_ = int(result.get("training_iteration", 0))
        every = self._cfg["refresh_every"]
        if iter_ <= 0 or iter_ % every != 0:
            return

        print(f"\n[adaptive-reward] ===== refresh trigger at iter {iter_} =====")
        try:
            self._ensure_offline_table()
        except Exception as exc:
            print(f"[adaptive-reward] offline table build failed: {exc!r}")
            print(f"[adaptive-reward] DISABLING further refreshes; training continues "
                  f"with static reward (= 036D behaviour)")
            self._refresh_disabled_due_to_error = True
            return

        # For v1, the online buffer is mostly empty (we haven't wired proper
        # trajectory capture into the episode end). Refresh still runs on the
        # offline pool which gives the "stationary" B-T signal. This is a
        # meaningful upgrade over 036C (which never refreshed at all), but
        # it's a weaker version of full AIRL.
        #
        # TODO(039-next): wire trajectory capture to gather fresh W/L pairs
        # each phase, which is the actual "adaptive" part.
        online_pairs = self._buffer.sample_pairs(
            n_pairs=self._cfg["min_online_pairs"],
        )
        if len(online_pairs) < self._cfg["min_online_pairs"]:
            print(f"[adaptive-reward] online_pairs={len(online_pairs)} < min={self._cfg['min_online_pairs']}; "
                  f"refresh will use offline pool only")

        if not self._ensure_owned_model(trainer):
            return

        # Refresh D on offline_table + (possibly empty) online_pairs
        from cs8803drl.imitation.failure_buckets_v2 import (
            HEAD_TO_LABELS_V2,
            classify_failure_v2,
        )
        head_to_labels = HEAD_TO_LABELS_V2 if self._cfg["label_version"] == "v2" else HEAD_TO_LABELS
        classifier_fn = classify_failure_v2 if self._cfg["label_version"] == "v2" else None

        try:
            metrics = refresh_reward_model_online(
                model=self._owned_model,
                offline_table=self._offline_table,
                online_pairs=online_pairs,
                head_to_labels=head_to_labels,
                classifier_fn=classifier_fn,
                loss=self._cfg["refresh_loss"],
                steps=self._cfg["refresh_steps"],
                batch_size=256,
                lr=self._cfg["refresh_lr"],
                device=self._owned_device,
            )
        except Exception as exc:
            print(f"[adaptive-reward] ERROR during refresh fine-tune: {exc!r}")
            return

        # Broadcast new state_dict to every worker
        self._owned_model.eval()
        new_sd = {k: v.detach().cpu() for k, v in self._owned_model.state_dict().items()}

        def _broadcast(worker):
            env = getattr(worker, "env", None)
            wrapper = _find_wrapper_in_env(env)
            if wrapper is None:
                return False
            try:
                return bool(wrapper.update_reward_model(new_sd))
            except Exception:
                return False

        try:
            broadcast_results = trainer.workers.foreach_worker(_broadcast)
            n_ok = sum(1 for r in broadcast_results if r is True)
            n_total = len(broadcast_results)
            print(f"[adaptive-reward] refresh #{self._refresh_count + 1} done: "
                  f"loss_final={metrics['train_loss_final']:.4f} "
                  f"broadcasted_to={n_ok}/{n_total}_workers")
            if n_ok == 0:
                print("[adaptive-reward] ERROR during broadcast: wrapper not found on any worker")
                return
            if "custom_metrics" in result:
                result["custom_metrics"][f"adaptive_reward_refresh_loss"] = float(metrics['train_loss_final'])
                result["custom_metrics"][f"adaptive_reward_refresh_count"] = int(self._refresh_count + 1)
                result["custom_metrics"][f"adaptive_reward_broadcast_workers"] = int(n_ok)
                result["custom_metrics"][f"adaptive_reward_broadcast_total"] = int(n_total)
        except Exception as exc:
            print(f"[adaptive-reward] ERROR during broadcast: {exc!r}")
            return False
        self._refresh_count += 1

import os
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_PYTHONPATH = os.environ.get("PYTHONPATH", "")
_PYTHONPATH_ENTRIES = [entry for entry in _PYTHONPATH.split(os.pathsep) if entry]
if REPO_ROOT not in _PYTHONPATH_ENTRIES:
    os.environ["PYTHONPATH"] = REPO_ROOT if not _PYTHONPATH else REPO_ROOT + os.pathsep + _PYTHONPATH

try:
    import sitecustomize as _project_sitecustomize  # noqa: F401
except Exception:
    _project_sitecustomize = None

import numpy as np
import torch

from soccer_twos import AgentInterface

from cs8803drl.branches.imitation_bc import load_bc_checkpoint


def _default_checkpoint_path():
    for env_name in ("TRAINED_RAY_CHECKPOINT", "TRAINED_BC_CHECKPOINT"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    raise ValueError(
        "Missing TRAINED_RAY_CHECKPOINT env var. Example: "
        "TRAINED_RAY_CHECKPOINT=/path/to/bc_checkpoint_dir"
    )


class BCTeamAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        checkpoint_path = _default_checkpoint_path()
        self._device = torch.device("cpu")
        self._model, self._metadata = load_bc_checkpoint(checkpoint_path, map_location=self._device)
        self._model.to(self._device)
        self._player_action_dim = int(self._metadata.get("player_action_dim", len(self._metadata["action_nvec"]) // 2))

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        player_ids = sorted(int(pid) for pid in observation.keys())
        if len(player_ids) != 2:
            raise ValueError(
                "trained_bc_team_agent expects exactly 2 local teammates in observation, "
                f"got ids={player_ids}"
            )

        team_obs = np.concatenate(
            [
                np.asarray(observation[player_ids[0]], dtype=np.float32).reshape(-1),
                np.asarray(observation[player_ids[1]], dtype=np.float32).reshape(-1),
            ],
            axis=0,
        )
        obs_tensor = torch.from_numpy(team_obs).float().unsqueeze(0).to(self._device)
        team_action = self._model.greedy_action(obs_tensor).squeeze(0).cpu().numpy().astype(np.int64)

        return {
            player_ids[0]: team_action[: self._player_action_dim].astype(np.int64),
            player_ids[1]: team_action[self._player_action_dim :].astype(np.int64),
        }


Agent = BCTeamAgent

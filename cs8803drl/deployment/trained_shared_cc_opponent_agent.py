import os

from cs8803drl.deployment import trained_shared_cc_agent as _base


def _opponent_checkpoint_path() -> str:
    for env_name in (
        "TRAINED_SHARED_CC_OPPONENT_CHECKPOINT",
        "TRAINED_RAY_OPPONENT_CHECKPOINT",
    ):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    raise ValueError(
        "Missing TRAINED_SHARED_CC_OPPONENT_CHECKPOINT env var. Example: "
        "TRAINED_SHARED_CC_OPPONENT_CHECKPOINT=/path/to/checkpoint-290"
    )


class SharedCCOpponentAgent(_base.SharedCCAgent):
    def __init__(self, env):
        checkpoint = _opponent_checkpoint_path()
        prev_shared = os.environ.get("TRAINED_SHARED_CC_CHECKPOINT")
        prev_default = os.environ.get("TRAINED_RAY_CHECKPOINT")
        os.environ["TRAINED_SHARED_CC_CHECKPOINT"] = checkpoint
        os.environ["TRAINED_RAY_CHECKPOINT"] = checkpoint
        try:
            super().__init__(env)
        finally:
            if prev_shared is None:
                os.environ.pop("TRAINED_SHARED_CC_CHECKPOINT", None)
            else:
                os.environ["TRAINED_SHARED_CC_CHECKPOINT"] = prev_shared
            if prev_default is None:
                os.environ.pop("TRAINED_RAY_CHECKPOINT", None)
            else:
                os.environ["TRAINED_RAY_CHECKPOINT"] = prev_default


Agent = SharedCCOpponentAgent

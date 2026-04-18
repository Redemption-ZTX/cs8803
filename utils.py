"""Legacy compatibility shim for old pickled configs/checkpoints.

Old experiments referenced the root-level ``utils`` module before the package
reorg. Keep this thin forwarding module so historic checkpoints remain
loadable.
"""

from cs8803drl.core.utils import *  # noqa: F401,F403


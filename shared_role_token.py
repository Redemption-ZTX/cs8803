"""Legacy compatibility shim for old pickled configs/checkpoints.

Old experiments referenced the root-level ``shared_role_token`` module before
the package reorg. Keep this thin forwarding module so historic checkpoints
remain loadable.
"""

from cs8803drl.branches.shared_role_token import *  # noqa: F401,F403

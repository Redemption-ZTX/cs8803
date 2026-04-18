"""Legacy compatibility shim for old pickled configs/checkpoints.

Old experiments referenced the root-level ``role_specialization`` module
before the package reorg. Keep this thin forwarding module so historic role
checkpoints remain loadable.
"""

from cs8803drl.branches.role_specialization import *  # noqa: F401,F403

"""Legacy compatibility shim for old pickled configs/checkpoints.

Old experiments referenced the root-level ``soccer_info`` module before the
package reorg. Keep this thin forwarding module so historic checkpoints such as
checkpoint-160/checkpoint-225 remain loadable.
"""

from cs8803drl.core.soccer_info import *  # noqa: F401,F403


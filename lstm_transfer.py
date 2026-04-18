"""Legacy compatibility shim for old pickled configs/checkpoints.

Old experiments referenced the root-level ``lstm_transfer`` module before the
package reorg. Keep this thin forwarding module so historic checkpoints remain
loadable.
"""

from cs8803drl.branches.lstm_transfer import *  # noqa: F401,F403

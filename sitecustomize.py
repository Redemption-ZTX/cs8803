import sys
import types

import numpy as np


if not hasattr(np, "bool"):
    np.bool = bool

if not hasattr(np, "object"):
    np.object = object

if not hasattr(np, "int"):
    np.int = int

if not hasattr(np, "float"):
    np.float = float

if not hasattr(np, "str"):
    np.str = str


def _patch_prometheus_empty_addr():
    """Work around Ray 1.4 metrics exporter binding with addr="" on PACE.

    Ray 1.4's vendored Prometheus exporter passes an empty string to
    `prometheus_client.start_http_server(..., addr="")`. On PACE this trips
    `socket.getaddrinfo("", ..., AI_PASSIVE)` with `gaierror: [Errno -2] Name
    or service not known`, which crashes Ray's dashboard/metrics agent even
    though training itself can continue.

    Keep the training stack unchanged and patch the helper that resolves the
    bind address so empty string falls back to loopback, matching the intent in
    Ray's own exporter comments.
    """

    try:
        from prometheus_client import exposition as _prom_exposition
    except Exception:
        return

    original = getattr(_prom_exposition, "_get_best_family", None)
    if original is None:
        return
    if getattr(original, "_cs8803drl_empty_addr_patch", False):
        return

    def _patched_get_best_family(addr, port):
        if addr == "":
            addr = "127.0.0.1"
        return original(addr, port)

    _patched_get_best_family._cs8803drl_empty_addr_patch = True
    _prom_exposition._get_best_family = _patched_get_best_family


_patch_prometheus_empty_addr()


try:
    import cv2 as _cv2  # noqa: F401
except Exception:
    _cv2_stub = types.ModuleType("cv2")
    _cv2_stub.__file__ = "<cv2 stub>"

    class _OclStub:
        @staticmethod
        def setUseOpenCL(_flag):
            return None

    def _noop(*_args, **_kwargs):
        return None

    _cv2_stub.ocl = _OclStub()
    _cv2_stub.setNumThreads = _noop
    _cv2_stub.setUseOptimized = _noop

    def __getattr__(_name):
        if _name.startswith("__") and _name.endswith("__"):
            raise AttributeError(_name)
        return _noop

    _cv2_stub.__getattr__ = __getattr__
    sys.modules["cv2"] = _cv2_stub

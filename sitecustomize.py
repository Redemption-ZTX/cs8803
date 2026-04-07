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

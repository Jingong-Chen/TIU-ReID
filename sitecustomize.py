"""
Compatibility shim for legacy dependencies (e.g. timm) expecting torch._six.
Auto-imported by Python when present on sys.path.
"""
import collections.abc
import sys
import types


def _ensure_torch_six():
    if "torch._six" in sys.modules:
        return
    module = types.ModuleType("torch._six")
    module.container_abcs = collections.abc
    module.string_classes = (str, bytes)
    module.int_classes = (int,)
    sys.modules["torch._six"] = module


_ensure_torch_six()

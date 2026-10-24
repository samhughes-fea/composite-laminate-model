"""Name -> factory registry so new kernels plug in without touching callers."""

from __future__ import annotations

from typing import Callable, Dict, List

from .base import FEAKernel
from .calculix import CalculiXKernelAdapter
from .nastran import NastranKernelAdapter
from .native import NativePythonKernelAdapter


class KernelRegistry:
    def __init__(self):
        self._factories: Dict[str, Callable[..., FEAKernel]] = {}

    def register(self, name: str, factory: Callable[..., FEAKernel]) -> None:
        self._factories[name.lower()] = factory

    def create(self, name: str, **options) -> FEAKernel:
        try:
            return self._factories[name.lower()](**options)
        except KeyError as exc:
            raise KeyError(f"Unknown kernel '{name}'. Registered: {', '.join(self._factories)}") from exc

    def names(self) -> List[str]:
        return list(self._factories)


DEFAULT_KERNELS = KernelRegistry()
DEFAULT_KERNELS.register("native", NativePythonKernelAdapter)
DEFAULT_KERNELS.register("calculix", CalculiXKernelAdapter)
DEFAULT_KERNELS.register("nastran", NastranKernelAdapter)


def create_kernel(name: str, **options) -> FEAKernel:
    return DEFAULT_KERNELS.create(name, **options)

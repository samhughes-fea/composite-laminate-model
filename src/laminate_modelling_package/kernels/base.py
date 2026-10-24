"""Kernel interface and the solver-independent analysis description."""

from __future__ import annotations

import math
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..laminate_benchmark_stress import ThicknessProfile
from ..laminate_mechanics import Laminate
from ..laminate_solver_pipeline import BoundaryConditions, FEAMesh, LoadCase

GLOBAL_COMPONENTS = ("sxx", "syy", "txy", "tyz", "txz")
MATERIAL_COMPONENTS = ("s11", "s22", "t12", "t23", "t13")


class KernelStatus(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class UnsupportedFeatureError(ValueError):
    """The analysis model uses a feature the target solver deck cannot express."""


@dataclass
class AnalysisModel:
    """Solver-independent description of one linear static analysis."""

    mesh: FEAMesh
    boundary_conditions: BoundaryConditions
    load_case: LoadCase
    title: str = "laminate_model"
    monitor_elements: Tuple[int, ...] = ()


@dataclass
class KernelResult:
    kernel: str
    status: KernelStatus
    displacements: Dict[int, np.ndarray] = field(default_factory=dict)
    stress_profiles: Dict[int, Dict[str, ThicknessProfile]] = field(default_factory=dict)
    runtime: float = 0.0
    message: str = ""
    files: Dict[str, Path] = field(default_factory=dict)
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is KernelStatus.SUCCESS

    def max_abs_displacement(self, dof: int) -> float:
        values = [abs(v[dof]) for v in self.displacements.values() if np.isfinite(v[dof])]
        return float(max(values)) if values else float("nan")

    def profile(self, element_id: int, frame: str) -> Optional[ThicknessProfile]:
        return self.stress_profiles.get(element_id, {}).get(frame)


class FEAKernel(ABC):
    """Template: ``run`` times the analysis and converts exceptions into FAILED results."""

    name = "kernel"

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def _execute(self, model: AnalysisModel) -> KernelResult:
        """Run the analysis; implementations fill everything except ``runtime``."""

    def run(self, model: AnalysisModel) -> KernelResult:
        start = time.perf_counter()
        try:
            result = self._execute(model)
        except Exception as exc:  # noqa: BLE001 - reported back to the caller as a FAILED result
            result = KernelResult(self.name, KernelStatus.FAILED, message=f"{type(exc).__name__}: {exc}")
        result.runtime = time.perf_counter() - start
        return result


class ProcessRunner:
    """Thin wrapper over subprocess so external execution can be substituted in tests."""

    def run(self, args: Sequence[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
        return subprocess.run(
            list(args), cwd=str(cwd), capture_output=True, text=True, timeout=timeout, check=False
        )


def gauss2_heights(laminate: Laminate) -> np.ndarray:
    """Two Gauss heights per ply (mid +/- h_k / (2 sqrt 3)), bottom ply first."""
    z = []
    for z_b, z_t in laminate.ply_bounds():
        mid, half = 0.5 * (z_b + z_t), 0.5 * (z_t - z_b)
        z.extend([mid - half / math.sqrt(3.0), mid + half / math.sqrt(3.0)])
    return np.array(z)

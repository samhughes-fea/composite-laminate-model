"""Adapter over the in-house solver pipeline."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..laminate_benchmark_stress import ThicknessProfile
from ..laminate_solver_pipeline import StructuralGlobalSolver
from ..laminate_stress_recovery import ExactStressRecoverer, StressRecoverer
from .base import GLOBAL_COMPONENTS, MATERIAL_COMPONENTS, AnalysisModel, FEAKernel, KernelResult, KernelStatus, gauss2_heights


class NativePythonKernelAdapter(FEAKernel):
    """Adapter over ``StructuralGlobalSolver`` and a ``StressRecoverer``."""

    name = "native"

    def __init__(self, solver: Optional[StructuralGlobalSolver] = None, recoverer: Optional[StressRecoverer] = None):
        self.solver = solver or StructuralGlobalSolver()
        self.recoverer = recoverer or ExactStressRecoverer()

    def _execute(self, model: AnalysisModel) -> KernelResult:
        solution = self.solver.solve(model.mesh, model.boundary_conditions, model.load_case)
        profiles: Dict[int, Dict[str, ThicknessProfile]] = {}
        for eid in model.monitor_elements:
            element = model.mesh.element(eid)
            u_e = solution.element_displacements(eid)
            state = model.load_case.state_for(eid)
            z_g = gauss2_heights(element.laminate)
            points = self.recoverer.at_heights(element, u_e, z_g, state=state)
            glob = np.array([[*p.stress_global, *p.shear_global] for p in points])
            mids = self.recoverer.ply_midplane_states(element, u_e, state=state)
            mat = np.array([[*p.stress_material, *p.shear_material] for p in mids])
            mat_g = np.array([[*p.stress_material, *p.shear_material] for p in points])
            profiles[eid] = {
                "global": ThicknessProfile(eid, z_g, glob, GLOBAL_COMPONENTS, "global", self.name),
                "material": ThicknessProfile(eid, z_g, mat_g, MATERIAL_COMPONENTS, "material", self.name),
                "material_midply": ThicknessProfile(
                    eid, element.laminate.ply_midplanes(), mat, MATERIAL_COMPONENTS, "material", self.name
                ),
            }
        return KernelResult(
            self.name,
            KernelStatus.SUCCESS,
            displacements=solution.displacement_map(),
            stress_profiles=profiles,
            info={"solve_time": solution.solve_time, **solution.info},
        )

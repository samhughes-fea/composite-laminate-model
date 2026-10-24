"""In-plane (membrane) laminate analysis on a single free Quad4 element.

A symmetric laminate under uniform N = [Nx, Ny, Nxy] and a uniform temperature
change has the same strain everywhere, so one element held only against
rigid-body motion reproduces it exactly. Ply stresses for unit loads are found
once; any combined load is then a linear combination, which makes failure
envelopes and parameter sweeps cheap.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from laminate_modelling_package import (
    DOF,
    BoundaryConditions,
    ExactStressRecoverer,
    FEAMesh,
    FSDTModel,
    HygrothermalState,
    LoadCase,
    StructuralGlobalSolver,
)


@dataclass
class MembraneCell:
    """Ply stresses (material axes, Voigt 6) at 3 heights per ply for unit Nx, Ny, Nxy and a thermal state."""

    laminate: object
    ply_index: np.ndarray  # (m,)
    unit: np.ndarray  # (3, m, 6): stress per N/m of Nx, Ny, Nxy
    residual: np.ndarray  # (m, 6): stress from the hygrothermal state alone

    @classmethod
    def build(cls, laminate, state: Optional[HygrothermalState] = None) -> "MembraneCell":
        unit = []
        for load in np.eye(3):
            plies, stress = _solve(laminate, load, None)
            unit.append(stress)
        if state is None:
            residual = np.zeros_like(unit[0])
        else:
            _, residual = _solve(laminate, np.zeros(3), state)
        return cls(laminate, plies, np.array(unit), residual)

    def stresses(self, N, include_residual: bool = True) -> np.ndarray:
        s = np.tensordot(np.asarray(N, dtype=float), self.unit, axes=1)
        return s + self.residual if include_residual else s

    def max_index(self, criterion, N, include_residual: bool = True) -> Tuple[float, int]:
        stresses = self.stresses(N, include_residual)
        best, ply = -np.inf, -1
        for k, s in zip(self.ply_index, stresses):
            index = criterion.evaluate(s, self.laminate.plies[k].material.strength).index
            if index > best:
                best, ply = index, int(k)
        return best, ply

    def failure_load(self, criterion, direction, include_residual: bool = True, tol: float = 1e-4) -> Tuple[float, int]:
        """Scale factor on ``direction`` (N/m) at first-ply failure, and the failing ply."""
        direction = np.asarray(direction, dtype=float)
        if self.max_index(criterion, 0 * direction, include_residual)[0] >= 1.0:
            return 0.0, self.max_index(criterion, 0 * direction, include_residual)[1]
        lo, hi = 0.0, 1.0
        while self.max_index(criterion, hi * direction, include_residual)[0] < 1.0:
            lo, hi = hi, 2.0 * hi
        while hi - lo > tol * hi:
            mid = 0.5 * (lo + hi)
            if self.max_index(criterion, mid * direction, include_residual)[0] < 1.0:
                lo = mid
            else:
                hi = mid
        return hi, self.max_index(criterion, hi * direction, include_residual)[1]


def _solve(laminate, N, state):
    mesh = FEAMesh.rectangular_plate(1.0, 1.0, 1, 1, laminate, FSDTModel())
    n1, n2, n3, n4 = 1, 2, 4, 3  # corners: (0,0), (1,0), (1,1), (0,1)
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    bcs.fix([n1], [DOF.UX, DOF.UY]).fix([n2], [DOF.UY])
    nx, ny, nxy = N
    loads = LoadCase()
    edge_loads = {
        (n2, n3): (nx, nxy),  # x = 1, outward normal +x: traction (Nx, Nxy)
        (n4, n1): (-nx, -nxy),  # x = 0
        (n3, n4): (nxy, ny),  # y = 1, outward normal +y: traction (Nxy, Ny)
        (n1, n2): (-nxy, -ny),  # y = 0
    }
    for nodes, (fx, fy) in edge_loads.items():
        for node in nodes:
            if fx:
                loads.add_nodal_load(node, DOF.UX, fx / 2)
            if fy:
                loads.add_nodal_load(node, DOF.UY, fy / 2)
    if state is not None:
        loads.set_hygrothermal(state)
    result = StructuralGlobalSolver().solve(mesh, bcs, loads)
    element = mesh.elements[0]
    profile = ExactStressRecoverer().through_thickness_profile(
        element, result.element_displacements(element.element_id), state=state or HygrothermalState())
    return np.array([p.ply_index for p in profile]), np.array([p.material_stress_3d() for p in profile])

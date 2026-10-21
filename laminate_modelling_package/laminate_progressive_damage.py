"""Progressive damage: smeared-crack stiffness degradation and the incremental analysis loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .laminate_failure_criteria import FailureCriterion, FailureMode, FailureResult, HashinCriterion
from .laminate_materials_3d import ElasticProperties3D
from .laminate_mechanics import Laminate
from .laminate_solver_pipeline import BoundaryConditions, FEAMesh, LoadCase, SolutionResult, StructuralGlobalSolver
from .laminate_stress_recovery import ExactStressRecoverer


@dataclass(frozen=True)
class DamageState:
    """Monotonic damage-mode flags for one ply of one element."""

    fiber_tension: bool = False
    fiber_compression: bool = False
    matrix_tension: bool = False
    matrix_compression: bool = False
    shear: bool = False
    delamination: bool = False

    @property
    def fiber(self) -> bool:
        return self.fiber_tension or self.fiber_compression

    @property
    def matrix(self) -> bool:
        return self.matrix_tension or self.matrix_compression

    @property
    def any(self) -> bool:
        return self.fiber or self.matrix or self.shear or self.delamination

    def merge(self, other: "DamageState") -> "DamageState":
        return DamageState(*(a or b for a, b in zip(self._flags(), other._flags())))

    def _flags(self) -> Tuple[bool, ...]:
        return (self.fiber_tension, self.fiber_compression, self.matrix_tension,
                self.matrix_compression, self.shear, self.delamination)

    def labels(self) -> List[str]:
        names = ("fiber_tension", "fiber_compression", "matrix_tension", "matrix_compression", "shear", "delamination")
        return [n for n, f in zip(names, self._flags()) if f]

    @classmethod
    def from_failure(cls, result: FailureResult) -> "DamageState":
        """Map a failure result onto damage flags.

        Mode-separated criteria flag every mode whose own index reached 1. Interactive
        criteria (Tsai-Wu, Tsai-Hill) flag the dominant contribution among fibre /
        matrix / shear / interlaminar, using the stress sign stored by the caller.
        """
        if not result.failed:
            return cls()
        modes = result.active_modes() if result.mode is not FailureMode.INTERACTIVE else set()
        if result.mode is not FailureMode.INTERACTIVE and not modes:
            modes = {result.mode}
        return cls(
            fiber_tension=FailureMode.FIBER_TENSION in modes,
            fiber_compression=FailureMode.FIBER_COMPRESSION in modes,
            matrix_tension=FailureMode.MATRIX_TENSION in modes or FailureMode.TRANSVERSE_NORMAL in modes,
            matrix_compression=FailureMode.MATRIX_COMPRESSION in modes,
            shear=FailureMode.IN_PLANE_SHEAR in modes,
            delamination=FailureMode.DELAMINATION in modes,
        )

    @classmethod
    def from_interactive(cls, result: FailureResult, stress: np.ndarray) -> "DamageState":
        if not result.failed:
            return cls()
        dominant = max(result.mode_indices, key=lambda k: abs(result.mode_indices[k]))
        s1, s2 = float(stress[0]), float(stress[1])
        return cls(
            fiber_tension=dominant == "fiber" and s1 >= 0.0,
            fiber_compression=dominant == "fiber" and s1 < 0.0,
            matrix_tension=dominant == "matrix" and s2 >= 0.0,
            matrix_compression=dominant == "matrix" and s2 < 0.0,
            shear=dominant == "shear",
            delamination=dominant == "interlaminar",
        )


class DegradationStrategy(ABC):
    """Maps damage flags onto degraded elastic constants."""

    @abstractmethod
    def degrade(self, elastic: ElasticProperties3D, damage: DamageState) -> ElasticProperties3D:
        """Return the degraded elastic constants for the given damage state."""


class SmearedCrackDegradation(DegradationStrategy):
    """Direction-selective smeared-crack stiffness reduction.

    * Matrix cracking (tension or compression): E2, G12, G23 and the Poisson ratios
      v12, v23 are knocked down; E1 (fibre direction) is preserved.
    * Fibre failure: E1, G12, G13, v12, v13 are knocked down.
    * In-plane shear failure: G12 is knocked down.
    * Delamination: E3, G13, G23, v13, v23 are knocked down.

    A degraded modulus becomes ``max(residual_fraction * E_virgin, floor)`` where the
    floor is ``minimum_fraction`` of the virgin value, so a fraction of zero means
    "set to the residual floor" and keeps the stiffness matrix non-singular.
    Degraded Poisson ratios are multiplied by ``residual_fraction``.
    """

    def __init__(self, residual_fraction: float = 0.01, minimum_fraction: float = 1.0e-6, compressive_matrix_fraction: Optional[float] = None):
        if not 0.0 <= residual_fraction <= 1.0:
            raise ValueError("residual_fraction must lie in [0, 1]")
        if not 0.0 < minimum_fraction <= 1.0:
            raise ValueError("minimum_fraction must lie in (0, 1]")
        self.residual_fraction = float(residual_fraction)
        self.minimum_fraction = float(minimum_fraction)
        self.compressive_matrix_fraction = (
            self.residual_fraction if compressive_matrix_fraction is None else float(compressive_matrix_fraction)
        )

    def _knock(self, value: float, fraction: float) -> float:
        return max(fraction * value, self.minimum_fraction * value)

    def degrade(self, elastic: ElasticProperties3D, damage: DamageState) -> ElasticProperties3D:
        if not damage.any:
            return elastic
        values = elastic.as_dict()
        knocked: Dict[str, float] = {}

        def apply(names: Sequence[str], fraction: float) -> None:
            for name in names:
                knocked[name] = min(knocked.get(name, 1.0), fraction)

        r = self.residual_fraction
        if damage.matrix_tension:
            apply(("E2", "G12", "G23", "v12", "v23"), r)
        if damage.matrix_compression:
            apply(("E2", "G12", "G23", "v12", "v23"), self.compressive_matrix_fraction)
        if damage.fiber:
            apply(("E1", "G12", "G13", "v12", "v13"), r)
        if damage.shear:
            apply(("G12",), r)
        if damage.delamination:
            apply(("E3", "G13", "G23", "v13", "v23"), r)
        for name, fraction in knocked.items():
            if name.startswith("v"):
                values[name] = values[name] * fraction
            else:
                values[name] = self._knock(values[name], fraction)
        return ElasticProperties3D(**values)


# ---------------------------------------------------------------------------
# Incremental progressive damage analysis
# ---------------------------------------------------------------------------
@dataclass
class DamageStepRecord:
    load_factor: float
    iterations: int
    newly_failed: int
    total_damaged_plies: int
    max_failure_index: float
    max_displacement: float
    damage_events: List[Tuple[int, int, Tuple[str, ...]]] = field(default_factory=list)


@dataclass
class ProgressiveDamageResult:
    steps: List[DamageStepRecord]
    damage: Dict[Tuple[int, int], DamageState]
    final_solution: Optional[SolutionResult]
    mesh: FEAMesh
    first_ply_failure_factor: Optional[float]

    @property
    def damaged_plies(self) -> int:
        return sum(1 for d in self.damage.values() if d.any)


class ProgressiveDamageAnalysis:
    """Load-stepped analysis with equilibrium re-solves after each new damage event.

    At every load factor the model is solved, ply stresses are recovered at the
    element centre (bottom, middle and top of each ply), the failure criterion
    is evaluated, newly failed plies are degraded and the step is re-solved until
    no new damage appears or ``max_iterations`` is reached.
    """

    def __init__(
        self,
        criterion: Optional[FailureCriterion] = None,
        degradation: Optional[DegradationStrategy] = None,
        solver: Optional[StructuralGlobalSolver] = None,
        recoverer: Optional[ExactStressRecoverer] = None,
        max_iterations: int = 20,
        points_per_ply: int = 3,
    ):
        self.criterion = criterion or HashinCriterion("3d")
        self.degradation = degradation or SmearedCrackDegradation()
        self.solver = solver or StructuralGlobalSolver()
        self.recoverer = recoverer or ExactStressRecoverer()
        self.max_iterations = int(max_iterations)
        self.points_per_ply = int(points_per_ply)

    def _damage_from(self, result: FailureResult, stress: np.ndarray) -> DamageState:
        if result.mode is FailureMode.INTERACTIVE:
            return DamageState.from_interactive(result, stress)
        return DamageState.from_failure(result)

    def _evaluate(self, mesh: FEAMesh, solution: SolutionResult, load_case: LoadCase, virgin: Dict[int, Laminate]):
        """Return {(element, ply): DamageState} for failed plies and the max failure index."""
        found: Dict[Tuple[int, int], DamageState] = {}
        max_index = 0.0
        for element in mesh.elements:
            u_e = solution.element_displacements(element.element_id)
            state = load_case.state_for(element.element_id)
            profile = self.recoverer.through_thickness_profile(element, u_e, state=state, points_per_ply=self.points_per_ply)
            for point in profile:
                strength = virgin[element.element_id].plies[point.ply_index].material.strength
                stress = point.material_stress_3d()
                result = self.criterion.evaluate(stress, strength)
                max_index = max(max_index, result.index)
                damage = self._damage_from(result, stress)
                if damage.any:
                    key = (element.element_id, point.ply_index)
                    found[key] = found.get(key, DamageState()).merge(damage)
        return found, max_index

    def _apply(self, mesh: FEAMesh, virgin: Dict[int, Laminate], damage: Dict[Tuple[int, int], DamageState], keys) -> None:
        by_element: Dict[int, List[int]] = {}
        for e, k in keys:
            by_element.setdefault(e, []).append(k)
        for e in by_element:
            laminate = mesh.element(e).laminate
            for k in range(laminate.n_plies):
                state = damage.get((e, k))
                if state is None or not state.any:
                    continue
                virgin_ply = virgin[e].plies[k]
                elastic = self.degradation.degrade(virgin_ply.material.elastic, state)
                material = virgin_ply.material.with_elastic(elastic, suffix="+dmg")
                laminate = laminate.replace_ply(k, virgin_ply.with_material(material))
            mesh.set_element_laminate(e, laminate)

    def run(
        self,
        mesh: FEAMesh,
        bcs: BoundaryConditions,
        reference_load: LoadCase,
        load_factors: Sequence[float],
        scale_hygrothermal: bool = False,
    ) -> ProgressiveDamageResult:
        """Run the analysis on a copy of ``mesh``; ``reference_load`` is scaled by each factor."""
        mesh = mesh.copy()
        virgin = {e.element_id: e.laminate for e in mesh.elements}
        damage: Dict[Tuple[int, int], DamageState] = {}
        steps: List[DamageStepRecord] = []
        solution: Optional[SolutionResult] = None
        first_failure: Optional[float] = None
        for factor in load_factors:
            load_case = reference_load.scaled(factor, scale_hygrothermal)
            iterations = 0
            newly_total = 0
            events: List[Tuple[int, int, Tuple[str, ...]]] = []
            max_index = 0.0
            while True:
                solution = self.solver.solve(mesh, bcs, load_case)
                iterations += 1
                found, max_index = self._evaluate(mesh, solution, load_case, virgin)
                changed = []
                for key, state in found.items():
                    previous = damage.get(key, DamageState())
                    merged = previous.merge(state)
                    if merged != previous:
                        damage[key] = merged
                        changed.append(key)
                        events.append((key[0], key[1], tuple(merged.labels())))
                if changed and first_failure is None:
                    first_failure = float(factor)
                newly_total += len(changed)
                if not changed or iterations >= self.max_iterations:
                    break
                self._apply(mesh, virgin, damage, changed)
            steps.append(
                DamageStepRecord(
                    load_factor=float(factor),
                    iterations=iterations,
                    newly_failed=newly_total,
                    total_damaged_plies=sum(1 for d in damage.values() if d.any),
                    max_failure_index=float(max_index),
                    max_displacement=float(np.max(np.abs(solution.displacement_array()[:, :3]))),
                    damage_events=events,
                )
            )
        return ProgressiveDamageResult(steps, damage, solution, mesh, first_failure)

"""Mesh, boundary conditions, load cases and the native global solver.

Assembly uses a degree-of-freedom map to scatter element matrices into the global
stiffness. Every node has the six shell DOFs [u, v, w, theta_x, theta_y, theta_z];
nodes of enriched elements (full RZT, full layerwise) carry their extra in-plane
amplitudes after them. Single- and multi-point constraints are applied by row-column
elimination through a transformation

    U = T U_r + g,   K_r = T^T K T,   F_r = T^T (F - K g)

where prescribed DOFs drop out (their values sit in g) and each slave DOF of a
linear constraint u_s = sum c_i u_i + c0 is expressed through its masters.
Grounded springs are added to the diagonal of K before elimination.

One factorisation serves any number of load cases (``solve_many``), and linear
combinations of load cases are superposed exactly (``combine_solutions``).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .laminate_elements import DEFAULT_ELEMENTS, STANDARD_DOFS, ElementRegistry, ShellElement
from .laminate_hygrothermal import NULL_STATE, HygrothermalEngine, HygrothermalState
from .laminate_kinematics import KinematicModel
from .laminate_mechanics import Laminate

try:  # SciPy is optional: sparse assembly and factorisation when available.
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without SciPy
    _sp = None
    _spla = None
    HAVE_SCIPY = False


class DOF(IntEnum):
    UX = 0
    UY = 1
    UZ = 2
    RX = 3
    RY = 4
    RZ = 5


class EnrichedDOF(str, Enum):
    """Symbolic enrichment DOFs, expanded per node when constraints are applied.

    ``X`` / ``Y`` select the x / y components of every enrichment amplitude at the
    node (zig-zag or interface displacements along local x / y); ``ALL`` selects
    both. Nodes without enrichment simply have none, so these can be used on any mesh.
    On a simply supported edge, fix the component tangential to the edge.
    """

    X = "enriched_x"
    Y = "enriched_y"
    ALL = "enriched_all"


ALL_DOFS = tuple(DOF)
DofLike = Union[int, EnrichedDOF]
DofKey = Tuple[int, DofLike]


class SingularStiffnessError(RuntimeError):
    """Raised when the constrained stiffness matrix cannot be factorised."""


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
class FEAMesh:
    """Nodes and shell elements with a global DOF map.

    Elements are created by an ``ElementRegistry`` (Quad4, Tri3 and Quad8 by default,
    chosen by node count), so new element types plug in without touching the mesh.
    """

    def __init__(
        self,
        hygrothermal_engine: Optional[HygrothermalEngine] = None,
        element_registry: Optional[ElementRegistry] = None,
    ):
        self._nodes: Dict[int, np.ndarray] = {}
        self._elements: Dict[int, ShellElement] = {}
        self.hygrothermal_engine = hygrothermal_engine or HygrothermalEngine()
        self.element_registry = element_registry or DEFAULT_ELEMENTS
        self._layout: Optional[Tuple[Dict[int, int], Dict[int, int], int]] = None

    # ---------------------------------------------------------------- nodes
    def add_node(self, node_id: int, xyz: Sequence[float]) -> None:
        node_id = int(node_id)
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} already exists")
        xyz = np.asarray(xyz, dtype=float).ravel()
        if xyz.size == 2:
            xyz = np.append(xyz, 0.0)
        if xyz.size != 3:
            raise ValueError("Node coordinates need 2 or 3 components")
        self._nodes[node_id] = xyz
        self._layout = None

    def remove_nodes(self, node_ids: Sequence[int]) -> None:
        """Remove nodes that no element uses and renumber the DOF map."""
        drop = {int(n) for n in node_ids}
        used = {n for element in self._elements.values() for n in element.node_ids} & drop
        if used:
            raise ValueError(f"Nodes {sorted(used)} are used by elements and cannot be removed")
        self._nodes = {n: xyz for n, xyz in self._nodes.items() if n not in drop}
        self._layout = None

    @property
    def node_ids(self) -> List[int]:
        return list(self._nodes)

    def node_coordinates(self, node_id: int) -> np.ndarray:
        return self._nodes[node_id]

    def coordinates_array(self) -> np.ndarray:
        return np.array([self._nodes[n] for n in self._nodes])

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    # ----------------------------------------------------------- DOF layout
    def _dof_layout(self) -> Tuple[Dict[int, int], Dict[int, int], int]:
        if self._layout is None:
            count = {n: STANDARD_DOFS for n in self._nodes}
            signature: Dict[int, tuple] = {}
            for element in self._elements.values():
                d = element.dofs_per_node
                if d == STANDARD_DOFS:
                    continue
                sig = element.enrichment_signature
                for n in element.node_ids:
                    if n in signature and signature[n] != sig:
                        raise ValueError(
                            f"Node {n} is shared by elements with different enrichments ({signature[n]} and {sig}); "
                            "full layerwise elements must have the same number of layers across shared nodes"
                        )
                    signature[n] = sig
                    count[n] = d
            offsets, total = {}, 0
            for n in self._nodes:
                offsets[n] = total
                total += count[n]
            self._layout = (offsets, count, total)
        return self._layout

    @property
    def n_dof(self) -> int:
        return self._dof_layout()[2]

    def node_dof_count(self, node_id: int) -> int:
        return self._dof_layout()[1][int(node_id)]

    def dof_index(self, node_id: int, dof: int) -> int:
        offsets, count, _ = self._dof_layout()
        node_id, dof = int(node_id), int(dof)
        if node_id not in offsets:
            raise KeyError(f"Node {node_id} is not in the mesh")
        if not 0 <= dof < count[node_id]:
            raise KeyError(f"Node {node_id} has {count[node_id]} DOFs; DOF {dof} does not exist")
        return offsets[node_id] + dof

    def expand_dof(self, node_id: int, dof: DofLike) -> List[int]:
        """Global indices selected by a numeric or symbolic (``EnrichedDOF``) DOF at a node."""
        if isinstance(dof, EnrichedDOF):
            offsets, count, _ = self._dof_layout()
            base, n_extra = offsets[int(node_id)], count[int(node_id)] - STANDARD_DOFS
            start = {EnrichedDOF.X: 0, EnrichedDOF.Y: 1, EnrichedDOF.ALL: 0}[dof]
            step = 1 if dof is EnrichedDOF.ALL else 2
            return [base + STANDARD_DOFS + k for k in range(start, n_extra, step)]
        return [self.dof_index(node_id, dof)]

    def standard_dof_indices(self) -> np.ndarray:
        """(n_nodes, 6) global indices of the six shell DOFs of every node, in node order."""
        offsets, _, _ = self._dof_layout()
        return np.array([[offsets[n] + d for d in range(STANDARD_DOFS)] for n in self._nodes], dtype=int).reshape(-1, 6)

    def standard_array(self, vector: np.ndarray) -> np.ndarray:
        """(n_nodes, 6) view of a global vector restricted to the six shell DOFs."""
        return np.asarray(vector)[self.standard_dof_indices()]

    # --------------------------------------------------------- node queries
    def nodes_where(self, predicate: Callable[[np.ndarray], bool]) -> List[int]:
        return [n for n, xyz in self._nodes.items() if predicate(xyz)]

    def nodes_on_line(self, axis: int, value: float, tol: float = 1e-9) -> List[int]:
        """Nodes whose coordinate ``axis`` (0=x, 1=y, 2=z) equals ``value``."""
        return self.nodes_where(lambda p: abs(p[axis] - value) <= tol)

    def closest_node(self, point: Sequence[float]) -> int:
        point = np.asarray(point, dtype=float)
        if point.size == 2:
            point = np.append(point, 0.0)
        ids = self.node_ids
        distances = np.linalg.norm(self.coordinates_array() - point, axis=1)
        return ids[int(np.argmin(distances))]

    # ------------------------------------------------------------- elements
    def add_element(
        self,
        element_id: int,
        node_ids: Sequence[int],
        laminate: Laminate,
        kinematics: KinematicModel,
        element_type: Optional[str] = None,
        **element_options,
    ) -> ShellElement:
        """Create an element through the registry (type inferred from the node count)."""
        element_id = int(element_id)
        if element_id in self._elements:
            raise ValueError(f"Element {element_id} already exists")
        missing = [n for n in node_ids if n not in self._nodes]
        if missing:
            raise KeyError(f"Element {element_id} references unknown nodes {missing}")
        coords = np.array([self._nodes[n] for n in node_ids])
        element_options.setdefault("hygrothermal_engine", self.hygrothermal_engine)
        element = self.element_registry.create(element_id, node_ids, coords, laminate, kinematics, element_type, **element_options)
        self._elements[element_id] = element
        self._layout = None
        return element

    def insert_element(self, element: ShellElement) -> ShellElement:
        """Add an element built elsewhere (its nodes must already exist)."""
        if element.element_id in self._elements:
            raise ValueError(f"Element {element.element_id} already exists")
        missing = [n for n in element.node_ids if n not in self._nodes]
        if missing:
            raise KeyError(f"Element {element.element_id} references unknown nodes {missing}")
        self._elements[element.element_id] = element
        self._layout = None
        return element

    def replace_element(self, element: ShellElement) -> None:
        if element.element_id not in self._elements:
            raise KeyError(f"Element {element.element_id} is not in the mesh")
        self._elements[element.element_id] = element
        self._layout = None

    def set_element_laminate(self, element_id: int, laminate: Laminate) -> None:
        self.replace_element(self._elements[element_id].with_laminate(laminate))

    def set_kinematics(self, kinematics: KinematicModel) -> None:
        for element in list(self._elements.values()):
            self._elements[element.element_id] = element.with_kinematics(kinematics)
        self._layout = None

    def set_material_axis(self, material_axis, element_ids: Optional[Iterable[int]] = None) -> None:
        """Set the 0-degree reference direction (vector or orientation object) of elements."""
        ids = self.element_ids if element_ids is None else [int(e) for e in element_ids]
        for eid in ids:
            self._elements[eid] = self._elements[eid].with_material_axis(material_axis)

    def element(self, element_id: int) -> ShellElement:
        return self._elements[element_id]

    @property
    def elements(self) -> List[ShellElement]:
        return list(self._elements.values())

    @property
    def element_ids(self) -> List[int]:
        return list(self._elements)

    @property
    def n_elements(self) -> int:
        return len(self._elements)

    def element_dofs(self, element: Union[int, ShellElement]) -> np.ndarray:
        if not hasattr(element, "node_ids"):
            element = self._elements[int(element)]
        offsets, _, _ = self._dof_layout()
        d = element.dofs_per_node
        return np.array([offsets[n] + k for n in element.node_ids for k in range(d)], dtype=int)

    def copy(self) -> "FEAMesh":
        clone = FEAMesh(self.hygrothermal_engine, self.element_registry)
        clone._nodes = {k: v.copy() for k, v in self._nodes.items()}
        clone._elements = dict(self._elements)
        return clone

    # -------------------------------------------------------------- factory
    @classmethod
    def rectangular_plate(
        cls,
        length_x: float,
        length_y: float,
        nx: int,
        ny: int,
        laminate: Laminate,
        kinematics: KinematicModel,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        **element_options,
    ) -> "FEAMesh":
        """Structured nx-by-ny Quad4 mesh in the xy-plane (see ``RectangularPlateBuilder``)."""
        from .laminate_mesh_builders import RectangularPlateBuilder

        return RectangularPlateBuilder(length_x, length_y, nx, ny, origin=origin).build(laminate, kinematics, **element_options)


# ---------------------------------------------------------------------------
# Boundary conditions and loads
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinearConstraint:
    """Multi-point constraint u_slave = sum(coef_i * u_master_i) + constant."""

    slave: DofKey
    masters: Tuple[Tuple[DofKey, float], ...]
    constant: float = 0.0


def _dof_key(node: int, dof: DofLike) -> DofKey:
    return (int(node), dof if isinstance(dof, EnrichedDOF) else int(dof))


class BoundaryConditions:
    """Prescribed DOF values, linear multi-point constraints and grounded springs."""

    def __init__(self):
        self.prescribed: Dict[DofKey, float] = {}
        self.constraints: List[LinearConstraint] = []
        self.springs: Dict[Tuple[int, int], float] = {}

    def fix(self, node_ids: Iterable[int], dofs: Iterable[DofLike] = ALL_DOFS, value: float = 0.0) -> "BoundaryConditions":
        dofs = tuple(dofs)
        for n in node_ids:
            for d in dofs:
                self.prescribed[_dof_key(n, d)] = float(value)
        return self

    def clamp(self, node_ids: Iterable[int]) -> "BoundaryConditions":
        """Fix all six shell DOFs and every enrichment amplitude."""
        return self.fix(node_ids, ALL_DOFS + (EnrichedDOF.ALL,))

    def add_constraint(
        self, slave: DofKey, masters: Sequence[Tuple[DofKey, float]], constant: float = 0.0
    ) -> "BoundaryConditions":
        slave = (int(slave[0]), int(slave[1]))
        masters = tuple(((int(k[0]), int(k[1])), float(c)) for k, c in masters)
        self.constraints.append(LinearConstraint(slave, masters, float(constant)))
        return self

    def tie(self, slave: DofKey, master: DofKey) -> "BoundaryConditions":
        return self.add_constraint(slave, [(master, 1.0)])

    def fix_direction(
        self, node_ids: Iterable[int], direction: Sequence[float], rotation: bool = False, value: float = 0.0
    ) -> "BoundaryConditions":
        """Skew support: the displacement (or rotation) component along ``direction`` equals ``value``.

        Written as the constraint n . u = value with the component of largest |n| as slave,
        so supports on inclined or curved edges need no local coordinate systems.
        """
        n = np.asarray(direction, dtype=float)
        n = n / np.linalg.norm(n)
        base = 3 if rotation else 0
        s = int(np.argmax(np.abs(n)))
        for node in node_ids:
            masters = [((int(node), base + k), -n[k] / n[s]) for k in range(3) if k != s and n[k] != 0.0]
            self.add_constraint((int(node), base + s), masters, value / n[s])
        return self

    def add_spring(self, node_ids: Iterable[int], dofs: Iterable[int], stiffness: float) -> "BoundaryConditions":
        """Grounded spring of the given stiffness on each listed DOF (adds to existing springs)."""
        if stiffness < 0.0:
            raise ValueError("spring stiffness must be non-negative")
        dofs = tuple(int(d) for d in dofs)
        for node in node_ids:
            for d in dofs:
                key = (int(node), d)
                self.springs[key] = self.springs.get(key, 0.0) + float(stiffness)
        return self

    def constrained_keys(self) -> set:
        return set(self.prescribed) | {c.slave for c in self.constraints}

    def numeric_prescribed(self) -> Dict[Tuple[int, int], float]:
        """Prescribed values on the six shell DOFs only (what external solver decks can express)."""
        return {k: v for k, v in self.prescribed.items() if not isinstance(k[1], EnrichedDOF)}


PressureLike = Union[float, Callable[[np.ndarray], float]]


@dataclass(frozen=True)
class LineLoad:
    """Force (and optional moment) per unit length on every element edge whose nodes are all listed."""

    node_ids: frozenset
    force_per_length: Union[Tuple[float, ...], Callable[[np.ndarray], Sequence[float]]]

    def scaled(self, factor: float) -> "LineLoad":
        q = self.force_per_length
        if callable(q):
            return LineLoad(self.node_ids, lambda xyz, q=q, f=factor: f * np.asarray(q(xyz), dtype=float))
        return LineLoad(self.node_ids, tuple(factor * v for v in q))


@dataclass(frozen=True)
class BodyLoad:
    """Uniform acceleration (gravity, inertia relief) applied as rho_A * a."""

    acceleration: Tuple[float, float, float]
    element_ids: Optional[frozenset] = None

    def applies_to(self, element_id: int) -> bool:
        return self.element_ids is None or element_id in self.element_ids

    def scaled(self, factor: float) -> "BodyLoad":
        return BodyLoad(tuple(factor * a for a in self.acceleration), self.element_ids)


def _sum_pressures(parts: List[PressureLike]) -> PressureLike:
    numeric = sum(p for p in parts if not callable(p))
    fields_ = [p for p in parts if callable(p)]
    if not fields_:
        return float(numeric)
    return lambda xyz: numeric + sum(float(f(xyz)) for f in fields_)


def _scale_pressure(p: PressureLike, factor: float) -> PressureLike:
    if callable(p):
        return lambda xyz, p=p, f=factor: f * float(p(xyz))
    return p * factor


class LoadCase:
    """Nodal loads, pressures, line loads, body loads and hygrothermal states for one analysis."""

    def __init__(self, name: str = "LC1"):
        self.name = name
        self.nodal_loads: Dict[Tuple[int, int], float] = {}
        self.global_pressure: PressureLike = 0.0
        self.element_pressures: Dict[int, PressureLike] = {}
        self.line_loads: List[LineLoad] = []
        self.body_loads: List[BodyLoad] = []
        self.hygrothermal: HygrothermalState = NULL_STATE
        self.element_hygrothermal: Dict[int, HygrothermalState] = {}

    def add_nodal_load(self, node_id: int, dof: int, value: float) -> "LoadCase":
        key = (int(node_id), int(dof))
        self.nodal_loads[key] = self.nodal_loads.get(key, 0.0) + float(value)
        return self

    def add_pressure(self, pressure: PressureLike, element_ids: Optional[Iterable[int]] = None) -> "LoadCase":
        """Pressure along each element's +normal: a number, or a function of the global position.

        Applies to all elements when ``element_ids`` is omitted.
        """
        if element_ids is None:
            self.global_pressure = _sum_pressures([self.global_pressure, pressure])
        else:
            for e in element_ids:
                e = int(e)
                self.element_pressures[e] = _sum_pressures([self.element_pressures.get(e, 0.0), pressure])
        return self

    def add_line_load(
        self, node_ids: Iterable[int], force_per_length: Union[Sequence[float], Callable[[np.ndarray], Sequence[float]]]
    ) -> "LoadCase":
        """Line load on the element edges along ``node_ids`` (global [fx, fy, fz] or with [mx, my, mz])."""
        q = force_per_length if callable(force_per_length) else tuple(float(v) for v in force_per_length)
        if not callable(q) and len(q) not in (3, 6):
            raise ValueError("force_per_length needs 3 or 6 components")
        self.line_loads.append(LineLoad(frozenset(int(n) for n in node_ids), q))
        return self

    def add_body_load(self, acceleration: Sequence[float], element_ids: Optional[Iterable[int]] = None) -> "LoadCase":
        """Uniform acceleration, e.g. ``(0, 0, -9.81)`` for gravity in SI units."""
        a = tuple(float(v) for v in acceleration)
        if len(a) != 3:
            raise ValueError("acceleration needs 3 components")
        ids = None if element_ids is None else frozenset(int(e) for e in element_ids)
        self.body_loads.append(BodyLoad(a, ids))
        return self

    def set_hygrothermal(self, state: HygrothermalState, element_ids: Optional[Iterable[int]] = None) -> "LoadCase":
        if element_ids is None:
            self.hygrothermal = state
        else:
            for e in element_ids:
                self.element_hygrothermal[int(e)] = state
        return self

    def pressure_on(self, element_id: int) -> PressureLike:
        """Total pressure on an element: a float, or a function of position for pressure fields."""
        return _sum_pressures([self.global_pressure, self.element_pressures.get(element_id, 0.0)])

    def pressure_value(self, element: ShellElement) -> float:
        """Pressure on an element as one number (fields are evaluated at the centroid)."""
        p = self.pressure_on(element.element_id)
        return float(p(element.centroid)) if callable(p) else float(p)

    def has_pressure(self, element_id: int) -> bool:
        p = self.pressure_on(element_id)
        return callable(p) or p != 0.0

    def state_for(self, element_id: int) -> HygrothermalState:
        return self.element_hygrothermal.get(element_id, self.hygrothermal)

    def scaled(self, factor: float, scale_hygrothermal: bool = True) -> "LoadCase":
        new = LoadCase(f"{self.name}x{factor:g}")
        new.nodal_loads = {k: v * factor for k, v in self.nodal_loads.items()}
        new.global_pressure = _scale_pressure(self.global_pressure, factor)
        new.element_pressures = {k: _scale_pressure(v, factor) for k, v in self.element_pressures.items()}
        new.line_loads = [load.scaled(factor) for load in self.line_loads]
        new.body_loads = [load.scaled(factor) for load in self.body_loads]
        f = factor if scale_hygrothermal else 1.0
        new.hygrothermal = self.hygrothermal.scaled(f)
        new.element_hygrothermal = {k: v.scaled(f) for k, v in self.element_hygrothermal.items()}
        return new

    @classmethod
    def combine(cls, terms: Sequence[Tuple["LoadCase", float]], name: str = "combination") -> "LoadCase":
        """Linear combination sum(factor_i * case_i), hygrothermal states included."""
        out = cls(name)
        element_states: Dict[int, HygrothermalState] = {}
        ids = set()
        for case, _ in terms:
            ids |= set(case.element_hygrothermal)
        for case, factor in terms:
            s = case.scaled(factor)
            for k, v in s.nodal_loads.items():
                out.nodal_loads[k] = out.nodal_loads.get(k, 0.0) + v
            out.global_pressure = _sum_pressures([out.global_pressure, s.global_pressure])
            for e, p in s.element_pressures.items():
                out.element_pressures[e] = _sum_pressures([out.element_pressures.get(e, 0.0), p])
            out.line_loads += s.line_loads
            out.body_loads += s.body_loads
            out.hygrothermal = out.hygrothermal + s.hygrothermal
            for e in ids:
                element_states[e] = element_states.get(e, NULL_STATE) + s.state_for(e)
        for e in ids:
            out.element_hygrothermal[e] = element_states[e]
        return out


@dataclass(frozen=True)
class LoadCombination:
    """Named linear combination of load cases, e.g. 1.35 G + 1.5 Q."""

    name: str
    factors: Mapping[str, float]

    def terms(self, cases: Mapping[str, LoadCase]) -> List[Tuple[LoadCase, float]]:
        missing = [n for n in self.factors if n not in cases]
        if missing:
            raise KeyError(f"Combination '{self.name}' refers to unknown load cases {missing}")
        return [(cases[n], float(f)) for n, f in self.factors.items()]


# ---------------------------------------------------------------------------
# Linear algebra strategies
# ---------------------------------------------------------------------------
class LinearSystemSolver(ABC):
    """Strategy for solving the constrained symmetric positive definite system.

    ``pivot_tolerance`` is the smallest admissible ratio between the smallest and the
    largest factorisation pivot; a mechanism (unrestrained rigid-body mode) drives
    this ratio to round-off level (~1e-17) while well-posed shell models stay many
    orders of magnitude above it.
    """

    name = "linear-solver"

    def __init__(self, pivot_tolerance: float = 1e-14):
        self.pivot_tolerance = float(pivot_tolerance)

    @abstractmethod
    def factorize(self, K) -> Callable[[np.ndarray], np.ndarray]:
        """Factorise K once and return a function solving K x = F for one or more right-hand sides."""

    def solve(self, K, F: np.ndarray) -> np.ndarray:
        """Solve K x = F for a symmetric positive definite K (dense or sparse)."""
        return self.factorize(K)(F)

    def _check_pivots(self, pivots: np.ndarray) -> None:
        pivots = np.abs(np.asarray(pivots, dtype=float))
        if pivots.size and (not np.all(np.isfinite(pivots)) or pivots.min() <= self.pivot_tolerance * pivots.max()):
            raise SingularStiffnessError(
                "Stiffness matrix is singular (a rigid-body mode is unrestrained); check the constraints"
            )


class DenseLinearSolver(LinearSystemSolver):
    """Dense Cholesky factorisation (NumPy only)."""

    name = "numpy-dense"

    def factorize(self, K):
        K = K.toarray() if hasattr(K, "toarray") else np.asarray(K, dtype=float)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as exc:
            raise SingularStiffnessError("Stiffness matrix is not positive definite; check the constraints") from exc
        self._check_pivots(np.diag(L) ** 2)

        def solve(F):
            y = np.linalg.solve(L, np.asarray(F, dtype=float))
            return np.linalg.solve(L.T, y)

        return solve


class SparseLinearSolver(LinearSystemSolver):
    """Sparse LU factorisation (SuperLU via SciPy)."""

    name = "scipy-sparse"

    def __init__(self, pivot_tolerance: float = 1e-14):
        super().__init__(pivot_tolerance)
        if not HAVE_SCIPY:
            raise ImportError("SparseLinearSolver requires SciPy")

    def factorize(self, K):
        K = _sp.csc_matrix(K)
        try:
            lu = _spla.splu(K)
        except RuntimeError as exc:  # exactly singular
            raise SingularStiffnessError("Stiffness matrix is singular; check the constraints") from exc
        self._check_pivots(lu.U.diagonal())
        return lambda F: lu.solve(np.asarray(F, dtype=float))


def default_linear_solver() -> LinearSystemSolver:
    return SparseLinearSolver() if HAVE_SCIPY else DenseLinearSolver()


# ---------------------------------------------------------------------------
# Assembly and solution
# ---------------------------------------------------------------------------
class GlobalAssembler:
    """Scatter element matrices/vectors into global arrays via the DOF map."""

    def __init__(self, sparse: Optional[bool] = None):
        self.sparse = HAVE_SCIPY if sparse is None else bool(sparse)
        if self.sparse and not HAVE_SCIPY:
            raise ImportError("Sparse assembly requires SciPy")

    def assemble_stiffness(self, mesh: FEAMesh, bcs: Optional[BoundaryConditions] = None):
        n = mesh.n_dof
        rows, cols, vals = [], [], []
        for element in mesh.elements:
            dofs = mesh.element_dofs(element)
            K_e = element.stiffness_matrix()
            size = dofs.size
            rows.append(np.repeat(dofs, size))
            cols.append(np.tile(dofs, size))
            vals.append(K_e.ravel())
        if not rows:
            raise ValueError("Mesh has no elements")
        if bcs is not None and bcs.springs:
            idx = np.array([mesh.dof_index(nd, d) for nd, d in bcs.springs], dtype=int)
            rows.append(idx)
            cols.append(idx)
            vals.append(np.array(list(bcs.springs.values())))
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        if self.sparse:
            return _sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        K = np.zeros((n, n))
        np.add.at(K, (rows, cols), vals)
        return K

    def element_load_vector(self, element: ShellElement, load_case: LoadCase) -> np.ndarray:
        """Distributed loads on one element (pressure, body, hygrothermal) in global DOFs."""
        F = np.zeros(element.n_dof)
        if load_case.has_pressure(element.element_id):
            F += element.pressure_load_vector(load_case.pressure_on(element.element_id))
        for body in load_case.body_loads:
            if body.applies_to(element.element_id):
                F += element.body_load_vector(body.acceleration)
        state = load_case.state_for(element.element_id)
        if not state.is_null:
            F += element.hygrothermal_load_vector(state)
        return F

    def line_load_vector(self, mesh: FEAMesh, load_case: LoadCase) -> np.ndarray:
        """Consistent nodal loads of every line load; each mesh edge is loaded once."""
        F = np.zeros(mesh.n_dof)
        for load in load_case.line_loads:
            seen = set()
            for element in mesh.elements:
                for edge in element.edge_index(load.node_ids):
                    key = tuple(sorted(element.node_ids[i] for i in element.EDGES[edge]))
                    if key in seen:
                        continue
                    seen.add(key)
                    F[mesh.element_dofs(element)] += element.edge_load_vector(edge, load.force_per_length)
            if not seen:
                raise ValueError("A line load matches no element edge; list the nodes along the loaded edge")
        return F

    def assemble_loads(self, mesh: FEAMesh, load_case: LoadCase) -> np.ndarray:
        F = np.zeros(mesh.n_dof)
        for (node, dof), value in load_case.nodal_loads.items():
            F[mesh.dof_index(node, dof)] += value
        for element in mesh.elements:
            F[mesh.element_dofs(element)] += self.element_load_vector(element, load_case)
        if load_case.line_loads:
            F += self.line_load_vector(mesh, load_case)
        return F


class ConstraintEliminator:
    """Builds U = T U_r + g from prescribed values and linear constraints."""

    def build(self, mesh: FEAMesh, bcs: BoundaryConditions):
        n = mesh.n_dof
        prescribed: Dict[int, float] = {}
        for (node, dof), value in bcs.prescribed.items():
            for i in mesh.expand_dof(node, dof):
                prescribed[i] = value
        slaves = {}
        for c in bcs.constraints:
            s = mesh.dof_index(*c.slave)
            if s in prescribed:
                raise ValueError(f"DOF {c.slave} is both prescribed and a constraint slave")
            if s in slaves:
                raise ValueError(f"DOF {c.slave} is the slave of more than one constraint")
            slaves[s] = c
        free = [i for i in range(n) if i not in prescribed and i not in slaves]
        column = {dof: j for j, dof in enumerate(free)}
        g = np.zeros(n)
        for i, v in prescribed.items():
            g[i] = v
        rows, cols, vals = list(free), list(range(len(free))), [1.0] * len(free)
        for s, c in slaves.items():
            g[s] += c.constant
            for key, coef in c.masters:
                m = mesh.dof_index(*key)
                if m in slaves:
                    raise ValueError(f"Master DOF {key} is itself a constraint slave (chains are not supported)")
                if m in prescribed:
                    g[s] += coef * prescribed[m]
                else:
                    rows.append(s)
                    cols.append(column[m])
                    vals.append(coef)
        if HAVE_SCIPY:
            T = _sp.csr_matrix((vals, (rows, cols)), shape=(n, len(free)))
        else:
            T = np.zeros((n, len(free)))
            np.add.at(T, (np.array(rows), np.array(cols)), np.array(vals))
        return T, g


@dataclass
class SolutionResult:
    mesh: FEAMesh
    displacements: np.ndarray
    reactions: np.ndarray
    load_vector: np.ndarray
    solve_time: float
    load_case: Optional[LoadCase] = None
    info: Dict[str, object] = field(default_factory=dict)

    def node_displacement(self, node_id: int) -> np.ndarray:
        """[u, v, w, theta_x, theta_y, theta_z] of a node (global axes)."""
        base = self.mesh.dof_index(node_id, 0)
        return self.displacements[base : base + 6].copy()

    def node_enrichment(self, node_id: int) -> np.ndarray:
        """Enrichment amplitudes of a node, [a_1x, a_1y, a_2x, ...] (element local axes)."""
        base = self.mesh.dof_index(node_id, 0)
        return self.displacements[base + 6 : base + self.mesh.node_dof_count(node_id)].copy()

    def node_reaction(self, node_id: int) -> np.ndarray:
        base = self.mesh.dof_index(node_id, 0)
        return self.reactions[base : base + 6].copy()

    def displacement_array(self) -> np.ndarray:
        """(n_nodes, 6) displacements in mesh node order."""
        return self.mesh.standard_array(self.displacements).copy()

    def reaction_array(self) -> np.ndarray:
        return self.mesh.standard_array(self.reactions).copy()

    def load_array(self) -> np.ndarray:
        return self.mesh.standard_array(self.load_vector).copy()

    def displacement_map(self) -> Dict[int, np.ndarray]:
        return {n: self.node_displacement(n) for n in self.mesh.node_ids}

    def element_displacements(self, element_id: int) -> np.ndarray:
        return self.displacements[self.mesh.element_dofs(element_id)].copy()

    def max_abs(self, dof: int) -> float:
        return float(np.max(np.abs(self.displacement_array()[:, int(dof)])))

    @property
    def external_work(self) -> float:
        return 0.5 * float(self.displacements @ self.load_vector)


def combine_solutions(
    terms: Sequence[Tuple[SolutionResult, float]], load_case: Optional[LoadCase] = None, name: str = "combination"
) -> SolutionResult:
    """Exact linear superposition of solutions on the same mesh and constraints.

    Prescribed non-zero displacements are scaled with their solution, so a combination
    of cases that each prescribe a settlement scales the settlement too.
    """
    if not terms:
        raise ValueError("combine_solutions needs at least one term")
    mesh = terms[0][0].mesh
    U = sum(f * s.displacements for s, f in terms)
    R = sum(f * s.reactions for s, f in terms)
    F = sum(f * s.load_vector for s, f in terms)
    if load_case is None and all(s.load_case is not None for s, _ in terms):
        load_case = LoadCase.combine([(s.load_case, f) for s, f in terms], name)
    return SolutionResult(mesh, U, R, F, 0.0, load_case, {"combination": name, "terms": len(terms)})


class StructuralGlobalSolver:
    """Assemble, constrain and solve [K]{U} = {F} for a mesh."""

    def __init__(
        self,
        assembler: Optional[GlobalAssembler] = None,
        linear_solver: Optional[LinearSystemSolver] = None,
        eliminator: Optional[ConstraintEliminator] = None,
    ):
        self.assembler = assembler or GlobalAssembler()
        self.linear_solver = linear_solver or default_linear_solver()
        self.eliminator = eliminator or ConstraintEliminator()

    def solve(self, mesh: FEAMesh, bcs: BoundaryConditions, load_case: LoadCase) -> SolutionResult:
        return self.solve_many(mesh, bcs, [load_case])[0]

    def solve_many(self, mesh: FEAMesh, bcs: BoundaryConditions, load_cases: Sequence[LoadCase]) -> List[SolutionResult]:
        """Solve several load cases with one assembly and one factorisation."""
        start = time.perf_counter()
        K = self.assembler.assemble_stiffness(mesh, bcs)
        T, g = self.eliminator.build(mesh, bcs)
        K_r = T.T @ K @ T
        solve = self.linear_solver.factorize(K_r)
        Kg = np.asarray(K @ g).ravel()
        setup = time.perf_counter() - start
        results = []
        for load_case in load_cases:
            t0 = time.perf_counter()
            F = self.assembler.assemble_loads(mesh, load_case)
            F_r = np.asarray(T.T @ (F - Kg)).ravel()
            U = np.asarray(T @ solve(F_r)).ravel() + g
            R = np.asarray(K @ U).ravel() - F
            elapsed = time.perf_counter() - t0 + (setup if not results else 0.0)
            results.append(
                SolutionResult(
                    mesh=mesh,
                    displacements=U,
                    reactions=R,
                    load_vector=F,
                    solve_time=elapsed,
                    load_case=load_case,
                    info={"n_dof": mesh.n_dof, "n_free": int(T.shape[1]), "linear_solver": self.linear_solver.name},
                )
            )
        return results

    def solve_combinations(
        self,
        mesh: FEAMesh,
        bcs: BoundaryConditions,
        load_cases: Sequence[LoadCase],
        combinations: Sequence[LoadCombination],
    ) -> Dict[str, SolutionResult]:
        """Solve the basic cases once and superpose them into each combination.

        Returns every basic case (by name) and every combination (by name).
        """
        names = [c.name for c in load_cases]
        if len(set(names)) != len(names):
            raise ValueError("Load case names must be unique")
        basic = dict(zip(names, self.solve_many(mesh, bcs, load_cases)))
        cases = dict(zip(names, load_cases))
        out: Dict[str, SolutionResult] = dict(basic)
        for combo in combinations:
            terms = [(basic[case.name], factor) for case, factor in combo.terms(cases)]
            out[combo.name] = combine_solutions(terms, name=combo.name)
        return out

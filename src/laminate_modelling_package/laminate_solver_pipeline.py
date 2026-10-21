"""Mesh, boundary conditions, load cases and the native global solver.

Assembly uses a degree-of-freedom map (node order x 6 DOF) to scatter each 24x24
element matrix into the global stiffness topology. Single- and multi-point
constraints are applied by row-column elimination through a transformation

    U = T U_r + g,   K_r = T^T K T,   F_r = T^T (F - K g)

where prescribed DOFs drop out (their values sit in g) and each slave DOF of a
linear constraint u_s = sum c_i u_i + c0 is expressed through its masters.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .laminate_fea_engine import CompositePlateElement, KinematicModel
from .laminate_hygrothermal import NULL_STATE, HygrothermalEngine, HygrothermalState
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


ALL_DOFS = tuple(DOF)
DofKey = Tuple[int, int]


class SingularStiffnessError(RuntimeError):
    """Raised when the constrained stiffness matrix cannot be factorised."""


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
class FEAMesh:
    """Nodes and Quad4 composite elements with a global DOF map."""

    def __init__(self, hygrothermal_engine: Optional[HygrothermalEngine] = None):
        self._nodes: Dict[int, np.ndarray] = {}
        self._node_order: Dict[int, int] = {}
        self._elements: Dict[int, CompositePlateElement] = {}
        self.hygrothermal_engine = hygrothermal_engine or HygrothermalEngine()

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
        self._node_order[node_id] = len(self._nodes)
        self._nodes[node_id] = xyz

    def remove_nodes(self, node_ids: Sequence[int]) -> None:
        """Remove nodes that no element uses and renumber the DOF map."""
        drop = {int(n) for n in node_ids}
        used = {n for element in self._elements.values() for n in element.node_ids} & drop
        if used:
            raise ValueError(f"Nodes {sorted(used)} are used by elements and cannot be removed")
        self._nodes = {n: xyz for n, xyz in self._nodes.items() if n not in drop}
        self._node_order = {n: i for i, n in enumerate(self._nodes)}

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

    @property
    def n_dof(self) -> int:
        return 6 * len(self._nodes)

    def dof_index(self, node_id: int, dof: int) -> int:
        try:
            return 6 * self._node_order[int(node_id)] + int(dof)
        except KeyError as exc:
            raise KeyError(f"Node {node_id} is not in the mesh") from exc

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
        **element_options,
    ) -> CompositePlateElement:
        element_id = int(element_id)
        if element_id in self._elements:
            raise ValueError(f"Element {element_id} already exists")
        missing = [n for n in node_ids if n not in self._nodes]
        if missing:
            raise KeyError(f"Element {element_id} references unknown nodes {missing}")
        coords = np.array([self._nodes[n] for n in node_ids])
        element = CompositePlateElement(
            element_id, node_ids, coords, laminate, kinematics,
            hygrothermal_engine=self.hygrothermal_engine, **element_options,
        )
        self._elements[element_id] = element
        return element

    def replace_element(self, element: CompositePlateElement) -> None:
        if element.element_id not in self._elements:
            raise KeyError(f"Element {element.element_id} is not in the mesh")
        self._elements[element.element_id] = element

    def set_element_laminate(self, element_id: int, laminate: Laminate) -> None:
        self.replace_element(self._elements[element_id].with_laminate(laminate))

    def set_kinematics(self, kinematics: KinematicModel) -> None:
        for element in list(self._elements.values()):
            self._elements[element.element_id] = element.with_kinematics(kinematics)

    def element(self, element_id: int) -> CompositePlateElement:
        return self._elements[element_id]

    @property
    def elements(self) -> List[CompositePlateElement]:
        return list(self._elements.values())

    @property
    def element_ids(self) -> List[int]:
        return list(self._elements)

    @property
    def n_elements(self) -> int:
        return len(self._elements)

    def element_dofs(self, element: Union[int, CompositePlateElement]) -> np.ndarray:
        if not isinstance(element, CompositePlateElement):
            element = self._elements[int(element)]
        return np.array([self.dof_index(n, d) for n in element.node_ids for d in range(6)], dtype=int)

    def copy(self) -> "FEAMesh":
        clone = FEAMesh(self.hygrothermal_engine)
        clone._nodes = {k: v.copy() for k, v in self._nodes.items()}
        clone._node_order = dict(self._node_order)
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
        """Structured nx-by-ny Quad4 mesh in the xy-plane. Node ids start at 1, row by row."""
        if nx < 1 or ny < 1:
            raise ValueError("nx and ny must be >= 1")
        mesh = cls()
        x0, y0, z0 = (list(origin) + [0.0, 0.0, 0.0])[:3]
        xs = np.linspace(x0, x0 + length_x, nx + 1)
        ys = np.linspace(y0, y0 + length_y, ny + 1)
        node = 1
        for y in ys:
            for x in xs:
                mesh.add_node(node, (x, y, z0))
                node += 1
        element_id = 1
        for j in range(ny):
            for i in range(nx):
                n1 = j * (nx + 1) + i + 1
                mesh.add_element(
                    element_id, (n1, n1 + 1, n1 + nx + 2, n1 + nx + 1), laminate, kinematics, **element_options
                )
                element_id += 1
        return mesh


# ---------------------------------------------------------------------------
# Boundary conditions and loads
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LinearConstraint:
    """Multi-point constraint u_slave = sum(coef_i * u_master_i) + constant."""

    slave: DofKey
    masters: Tuple[Tuple[DofKey, float], ...]
    constant: float = 0.0


class BoundaryConditions:
    """Prescribed DOF values (single-point) and linear multi-point constraints."""

    def __init__(self):
        self.prescribed: Dict[DofKey, float] = {}
        self.constraints: List[LinearConstraint] = []

    def fix(self, node_ids: Iterable[int], dofs: Iterable[int] = ALL_DOFS, value: float = 0.0) -> "BoundaryConditions":
        dofs = tuple(int(d) for d in dofs)
        for n in node_ids:
            for d in dofs:
                self.prescribed[(int(n), d)] = float(value)
        return self

    def clamp(self, node_ids: Iterable[int]) -> "BoundaryConditions":
        return self.fix(node_ids, ALL_DOFS)

    def add_constraint(
        self, slave: DofKey, masters: Sequence[Tuple[DofKey, float]], constant: float = 0.0
    ) -> "BoundaryConditions":
        slave = (int(slave[0]), int(slave[1]))
        masters = tuple(((int(k[0]), int(k[1])), float(c)) for k, c in masters)
        self.constraints.append(LinearConstraint(slave, masters, float(constant)))
        return self

    def tie(self, slave: DofKey, master: DofKey) -> "BoundaryConditions":
        return self.add_constraint(slave, [(master, 1.0)])

    def constrained_keys(self) -> set:
        return set(self.prescribed) | {c.slave for c in self.constraints}


class LoadCase:
    """Nodal loads, element pressures and hygrothermal states for one analysis."""

    def __init__(self, name: str = "LC1"):
        self.name = name
        self.nodal_loads: Dict[DofKey, float] = {}
        self.element_pressures: Dict[int, float] = {}
        self.global_pressure: float = 0.0
        self.hygrothermal: HygrothermalState = NULL_STATE
        self.element_hygrothermal: Dict[int, HygrothermalState] = {}

    def add_nodal_load(self, node_id: int, dof: int, value: float) -> "LoadCase":
        key = (int(node_id), int(dof))
        self.nodal_loads[key] = self.nodal_loads.get(key, 0.0) + float(value)
        return self

    def add_pressure(self, pressure: float, element_ids: Optional[Iterable[int]] = None) -> "LoadCase":
        """Uniform pressure along each element's +normal (all elements when ids are omitted)."""
        if element_ids is None:
            self.global_pressure += float(pressure)
        else:
            for e in element_ids:
                self.element_pressures[int(e)] = self.element_pressures.get(int(e), 0.0) + float(pressure)
        return self

    def set_hygrothermal(self, state: HygrothermalState, element_ids: Optional[Iterable[int]] = None) -> "LoadCase":
        if element_ids is None:
            self.hygrothermal = state
        else:
            for e in element_ids:
                self.element_hygrothermal[int(e)] = state
        return self

    def pressure_on(self, element_id: int) -> float:
        return self.global_pressure + self.element_pressures.get(element_id, 0.0)

    def state_for(self, element_id: int) -> HygrothermalState:
        return self.element_hygrothermal.get(element_id, self.hygrothermal)

    def scaled(self, factor: float, scale_hygrothermal: bool = True) -> "LoadCase":
        new = LoadCase(f"{self.name}x{factor:g}")
        new.nodal_loads = {k: v * factor for k, v in self.nodal_loads.items()}
        new.element_pressures = {k: v * factor for k, v in self.element_pressures.items()}
        new.global_pressure = self.global_pressure * factor
        f = factor if scale_hygrothermal else 1.0
        new.hygrothermal = self.hygrothermal.scaled(f)
        new.element_hygrothermal = {k: v.scaled(f) for k, v in self.element_hygrothermal.items()}
        return new


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
    def solve(self, K, F: np.ndarray) -> np.ndarray:
        """Solve K x = F for a symmetric positive definite K (dense or sparse)."""

    def _check_pivots(self, pivots: np.ndarray) -> None:
        pivots = np.abs(np.asarray(pivots, dtype=float))
        if pivots.size and (not np.all(np.isfinite(pivots)) or pivots.min() <= self.pivot_tolerance * pivots.max()):
            raise SingularStiffnessError(
                "Stiffness matrix is singular (a rigid-body mode is unrestrained); check the constraints"
            )


class DenseLinearSolver(LinearSystemSolver):
    """Dense Cholesky factorisation (NumPy only)."""

    name = "numpy-dense"

    def solve(self, K, F):
        K = K.toarray() if hasattr(K, "toarray") else np.asarray(K, dtype=float)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as exc:
            raise SingularStiffnessError("Stiffness matrix is not positive definite; check the constraints") from exc
        self._check_pivots(np.diag(L) ** 2)
        y = np.linalg.solve(L, F)
        return np.linalg.solve(L.T, y)


class SparseLinearSolver(LinearSystemSolver):
    """Sparse LU factorisation (SuperLU via SciPy)."""

    name = "scipy-sparse"

    def __init__(self, pivot_tolerance: float = 1e-14):
        super().__init__(pivot_tolerance)
        if not HAVE_SCIPY:
            raise ImportError("SparseLinearSolver requires SciPy")

    def solve(self, K, F):
        K = _sp.csc_matrix(K)
        try:
            lu = _spla.splu(K)
        except RuntimeError as exc:  # exactly singular
            raise SingularStiffnessError("Stiffness matrix is singular; check the constraints") from exc
        self._check_pivots(lu.U.diagonal())
        return lu.solve(np.asarray(F, dtype=float))


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

    def assemble_stiffness(self, mesh: FEAMesh):
        n = mesh.n_dof
        rows, cols, vals = [], [], []
        for element in mesh.elements:
            dofs = mesh.element_dofs(element)
            K_e = element.stiffness_matrix()
            rows.append(np.repeat(dofs, 24))
            cols.append(np.tile(dofs, 24))
            vals.append(K_e.ravel())
        if not rows:
            raise ValueError("Mesh has no elements")
        rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        if self.sparse:
            return _sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
        K = np.zeros((n, n))
        np.add.at(K, (rows, cols), vals)
        return K

    def assemble_loads(self, mesh: FEAMesh, load_case: LoadCase) -> np.ndarray:
        F = np.zeros(mesh.n_dof)
        for (node, dof), value in load_case.nodal_loads.items():
            F[mesh.dof_index(node, dof)] += value
        for element in mesh.elements:
            dofs = mesh.element_dofs(element)
            pressure = load_case.pressure_on(element.element_id)
            if pressure != 0.0:
                F[dofs] += element.pressure_load_vector(pressure)
            state = load_case.state_for(element.element_id)
            if not state.is_null:
                F[dofs] += element.hygrothermal_load_vector(state)
        return F


class ConstraintEliminator:
    """Builds U = T U_r + g from prescribed values and linear constraints."""

    def build(self, mesh: FEAMesh, bcs: BoundaryConditions):
        n = mesh.n_dof
        prescribed = {mesh.dof_index(*k): v for k, v in bcs.prescribed.items()}
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
        base = self.mesh.dof_index(node_id, 0)
        return self.displacements[base : base + 6].copy()

    def node_reaction(self, node_id: int) -> np.ndarray:
        base = self.mesh.dof_index(node_id, 0)
        return self.reactions[base : base + 6].copy()

    def displacement_array(self) -> np.ndarray:
        """(n_nodes, 6) displacements in mesh node order."""
        return self.displacements.reshape(-1, 6).copy()

    def displacement_map(self) -> Dict[int, np.ndarray]:
        return {n: self.node_displacement(n) for n in self.mesh.node_ids}

    def element_displacements(self, element_id: int) -> np.ndarray:
        return self.displacements[self.mesh.element_dofs(element_id)].copy()

    def max_abs(self, dof: int) -> float:
        return float(np.max(np.abs(self.displacements[int(dof)::6])))

    @property
    def external_work(self) -> float:
        return 0.5 * float(self.displacements @ self.load_vector)


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
        start = time.perf_counter()
        K = self.assembler.assemble_stiffness(mesh)
        F = self.assembler.assemble_loads(mesh, load_case)
        T, g = self.eliminator.build(mesh, bcs)
        K_r = T.T @ K @ T
        F_r = T.T @ (F - K @ g)
        U_r = self.linear_solver.solve(K_r, np.asarray(F_r).ravel())
        U = np.asarray(T @ U_r).ravel() + g
        R = np.asarray(K @ U).ravel() - F
        elapsed = time.perf_counter() - start
        return SolutionResult(
            mesh=mesh,
            displacements=U,
            reactions=R,
            load_vector=F,
            solve_time=elapsed,
            load_case=load_case,
            info={"n_dof": mesh.n_dof, "n_free": int(T.shape[1]), "linear_solver": self.linear_solver.name},
        )

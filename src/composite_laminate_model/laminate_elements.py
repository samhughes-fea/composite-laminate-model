"""Flat composite shell elements behind one interface.

``ShellElement`` is the contract every consumer (mesh, assembler, stress recovery,
results, damage, solver decks) depends on. It owns everything that is common to the
element family - the local frame and material axis, the generalised section, the
B-operators of the enriched displacement field, stiffness, load vectors and strain
recovery - and leaves only the interpolation to the subclasses:

* ``Quad4ShellElement``  bilinear quad, MITC4 shear, optional incompatible membrane
                         modes (Wilson/Taylor QM6) for accurate in-plane bending.
* ``Tri3ShellElement``   linear triangle, constant-strain membrane, MITC3 shear.
* ``Quad8ShellElement``  serendipity quad (curved in-plane edges), 3x3 membrane and
                         bending, 2x2 transverse shear (selective reduced integration).

Every node carries [u, v, w, theta_x, theta_y, theta_z] plus, for enriched kinematic
theories (full RZT, full layerwise), two in-plane amplitudes per through-thickness
function. Sign convention: u = u0 + z theta_y, v = v0 - z theta_x.

Material axis
-------------
Ply angles are measured from the element's local x axis e1, which is the projection of
``material_axis`` onto the element plane. ``material_axis`` is a 3-vector or any object
with a ``direction_at(xyz)`` method (see ``laminate_orientation``), evaluated at the
element centroid; the default is global x. Enrichment amplitudes are stored in the
local axes of each element, which assumes the frame varies smoothly across the mesh.

All elements are flat: nodes are projected onto the element's mean plane, and the
out-of-plane distance of the nodes is reported as ``warping``.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .laminate_hygrothermal import HygrothermalEngine, HygrothermalState
from .laminate_kinematics import (
    GeneralizedSection,
    KinematicModel,
    SectionStiffness,
    generalized_hygrothermal,
)
from .laminate_mechanics import Laminate

STANDARD_DOFS = 6
WARPING_WARN_RATIO = 0.05

Pressure = Union[float, Callable[[np.ndarray], float]]


def _gauss_1d(order: int) -> Tuple[np.ndarray, np.ndarray]:
    pts, wts = np.polynomial.legendre.leggauss(order)
    return pts, wts


def _tensor_rule(order: int) -> List[Tuple[float, float, float]]:
    pts, wts = _gauss_1d(order)
    return [(float(x), float(y), float(wx * wy)) for x, wx in zip(pts, wts) for y, wy in zip(pts, wts)]


def _direction(axis, xyz: np.ndarray) -> np.ndarray:
    if axis is None:
        return np.array([1.0, 0.0, 0.0])
    if hasattr(axis, "direction_at"):
        return np.asarray(axis.direction_at(xyz), dtype=float)
    if callable(axis):
        return np.asarray(axis(xyz), dtype=float)
    return np.asarray(axis, dtype=float)


class ShellElement(ABC):
    """Base class of the flat composite shell elements."""

    TOPOLOGY: str = "shell"
    N_NODES: int = 0
    NATURAL_NODES: np.ndarray = np.zeros((0, 2))
    NATURAL_CENTROID: Tuple[float, float] = (0.0, 0.0)
    EDGES: Tuple[Tuple[int, ...], ...] = ()

    def __init__(
        self,
        element_id: int,
        node_ids: Sequence[int],
        coordinates: np.ndarray,
        laminate: Laminate,
        kinematics: KinematicModel,
        integration_order: Optional[int] = None,
        drilling_factor: float = 1.0e-3,
        hygrothermal_engine: Optional[HygrothermalEngine] = None,
        material_axis=None,
        warn_warping: bool = True,
    ):
        coordinates = np.asarray(coordinates, dtype=float)
        n = self.N_NODES
        if coordinates.shape == (n, 2):
            coordinates = np.hstack([coordinates, np.zeros((n, 1))])
        if coordinates.shape != (n, 3):
            raise ValueError(f"coordinates must have shape ({n}, 3) or ({n}, 2)")
        if len(node_ids) != n:
            raise ValueError(f"A {self.TOPOLOGY} element needs exactly {n} node ids")
        self.element_id = int(element_id)
        self.node_ids = tuple(int(v) for v in node_ids)
        self.coordinates = coordinates
        self.integration_order = self._check_order(integration_order)
        self.drilling_factor = float(drilling_factor)
        self.material_axis = material_axis
        self.warn_warping = bool(warn_warping)
        self._laminate = laminate
        self._kinematics = kinematics
        self._ht_engine = hygrothermal_engine or HygrothermalEngine()
        self._section: Optional[GeneralizedSection] = None
        self._K_local: Optional[np.ndarray] = None
        self._recovery_map: Optional[np.ndarray] = None
        self._build_local_frame()

    # ------------------------------------------------------- subclass hooks
    def _check_order(self, order: Optional[int]) -> Optional[int]:
        return order

    @abstractmethod
    def shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """N (n,) and natural derivatives dN (2, n) at (xi, eta)."""

    @abstractmethod
    def stiffness_rule(self) -> List[Tuple[float, float, float]]:
        """Points and weights for membrane, bending, enrichment and drilling terms."""

    def shear_rule(self) -> List[Tuple[float, float, float]]:
        """Points and weights for the transverse shear terms."""
        return self.stiffness_rule()

    def load_rule(self) -> List[Tuple[float, float, float]]:
        return self.stiffness_rule()

    def recovery_rule(self) -> List[Tuple[float, float, float]]:
        """Sampling points for results (the element's superconvergent points)."""
        return self.stiffness_rule()

    @abstractmethod
    def recovery_basis(self, xi: float, eta: float) -> np.ndarray:
        """Polynomial basis fitted to recovery-point values for extrapolation to nodes."""

    @abstractmethod
    def _normal_estimate(self) -> np.ndarray:
        """Unnormalised normal of the element mean plane (right-hand rule on the node order)."""

    @abstractmethod
    def _gamma0_operator(self, xi: float, eta: float) -> np.ndarray:
        """2 x (6n) mean transverse shear operator [gamma_yz; gamma_xz] on standard local DOFs."""

    def _incompatible_operator(self, xi: float, eta: float) -> Optional[np.ndarray]:
        """3 x n_int membrane strain operator of internal (condensed) modes, or None."""
        return None

    # ------------------------------------------------------------ geometry
    def _build_local_frame(self) -> None:
        X = self.coordinates
        normal = self._normal_estimate()
        norm = np.linalg.norm(normal)
        if norm <= 0.0:
            raise ValueError(f"Element {self.element_id} is degenerate (zero area)")
        e3 = normal / norm
        centre = X.mean(axis=0)
        reference = _direction(self.material_axis, centre)
        e1 = reference - np.dot(reference, e3) * e3
        if np.linalg.norm(e1) < 1e-8 * max(np.linalg.norm(reference), 1e-300):
            edge = X[1] - X[0]
            e1 = edge - np.dot(edge, e3) * e3
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(e3, e1)
        self.frame = np.vstack([e1, e2, e3])  # rows: local axes in global components
        local = (X - centre) @ self.frame.T
        self.local_xy = local[:, :2]
        self.warping = float(np.max(np.abs(local[:, 2])))
        for xi, eta, _ in self.stiffness_rule() + self.shear_rule():
            if self.jacobian(xi, eta)[1] <= 0.0:
                raise ValueError(f"Element {self.element_id} has a non-positive Jacobian; check node ordering")
        if self.warn_warping and self.warping_ratio > WARPING_WARN_RATIO:
            warnings.warn(
                f"Element {self.element_id} is warped: nodes sit {self.warping:.3g} off its mean plane "
                f"({100 * self.warping_ratio:.1f}% of its size). Refine the mesh on curved surfaces.",
                stacklevel=3,
            )

    def jacobian(self, xi: float, eta: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """Return (J, det J, dN/dx) where J = [[x,xi, y,xi], [x,eta, y,eta]]."""
        _, dN = self.shape_functions(xi, eta)
        J = dN @ self.local_xy
        detJ = float(np.linalg.det(J))
        dNdx = np.linalg.solve(J, dN)
        return J, detJ, dNdx

    @property
    def area(self) -> float:
        return float(sum(self.jacobian(xi, eta)[1] * w for xi, eta, w in self.stiffness_rule()))

    @property
    def characteristic_length(self) -> float:
        return float(np.sqrt(self.area))

    @property
    def warping_ratio(self) -> float:
        return self.warping / max(self.characteristic_length, 1e-300)

    def global_coordinates(self, xi: float, eta: float) -> np.ndarray:
        N, _ = self.shape_functions(xi, eta)
        return N @ self.coordinates

    @property
    def centroid(self) -> np.ndarray:
        return self.global_coordinates(*self.NATURAL_CENTROID)

    def quality(self) -> Dict[str, float]:
        """Aspect ratio, warping ratio and Jacobian ratio (min/max det J over the rules)."""
        dets = [self.jacobian(xi, eta)[1] for xi, eta, _ in self.stiffness_rule()]
        corners = self.local_xy[: self.N_CORNERS]
        edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        return {
            "aspect_ratio": float(edges.max() / edges.min()),
            "warping_ratio": self.warping_ratio,
            "jacobian_ratio": float(min(dets) / max(dets)),
        }

    N_CORNERS: int = 0

    # ------------------------------------------------------- section / DOFs
    @property
    def laminate(self) -> Laminate:
        return self._laminate

    @property
    def kinematics(self) -> KinematicModel:
        return self._kinematics

    @property
    def hygrothermal_engine(self) -> HygrothermalEngine:
        return self._ht_engine

    @property
    def section(self) -> GeneralizedSection:
        if self._section is None:
            self._section = self._kinematics.generalized_section(self._laminate)
        return self._section

    @property
    def condensed_section(self) -> SectionStiffness:
        return self.section.condensed

    @property
    def n_enrichment(self) -> int:
        return self.section.n_functions

    @property
    def dofs_per_node(self) -> int:
        return STANDARD_DOFS + 2 * self.n_enrichment

    @property
    def n_dof(self) -> int:
        return self.N_NODES * self.dofs_per_node

    @property
    def enrichment_signature(self) -> Optional[tuple]:
        basis = self.section.basis
        return None if basis is None else basis.signature

    def _options(self) -> dict:
        return {
            "integration_order": self.integration_order,
            "drilling_factor": self.drilling_factor,
            "hygrothermal_engine": self._ht_engine,
            "material_axis": self.material_axis,
            "warn_warping": False,
        }

    def _clone(self, **changes) -> "ShellElement":
        args = {"laminate": self._laminate, "kinematics": self._kinematics}
        options = self._options()
        for key, value in changes.items():
            (args if key in args else options)[key] = value
        return type(self)(self.element_id, self.node_ids, self.coordinates, args["laminate"], args["kinematics"], **options)

    def with_laminate(self, laminate: Laminate) -> "ShellElement":
        return self._clone(laminate=laminate)

    def with_kinematics(self, kinematics: KinematicModel) -> "ShellElement":
        return self._clone(kinematics=kinematics)

    def with_material_axis(self, material_axis) -> "ShellElement":
        return self._clone(material_axis=material_axis)

    @property
    def transformation(self) -> np.ndarray:
        """Global -> local DOF rotation (block diagonal; enrichment amplitudes are local)."""
        d = self.dofs_per_node
        T = np.eye(self.n_dof)
        for i in range(self.N_NODES):
            c = d * i
            T[c : c + 3, c : c + 3] = self.frame
            T[c + 3 : c + 6, c + 3 : c + 6] = self.frame
        return T

    # ----------------------------------------------------------- operators
    def _expand_standard(self, rows: np.ndarray) -> np.ndarray:
        """Embed an operator on the 6n standard DOFs into the full local DOF vector."""
        d = self.dofs_per_node
        if d == STANDARD_DOFS:
            return rows
        out = np.zeros((rows.shape[0], self.n_dof))
        for i in range(self.N_NODES):
            out[:, d * i : d * i + 6] = rows[:, 6 * i : 6 * i + 6]
        return out

    def membrane_B(self, dNdx: np.ndarray) -> np.ndarray:
        n = self.N_NODES
        B = np.zeros((3, 6 * n))
        for i in range(n):
            c = 6 * i
            B[0, c] = dNdx[0, i]
            B[1, c + 1] = dNdx[1, i]
            B[2, c] = dNdx[1, i]
            B[2, c + 1] = dNdx[0, i]
        return B

    def bending_B(self, dNdx: np.ndarray) -> np.ndarray:
        """Curvatures [kx, ky, kxy] with beta_x = theta_y, beta_y = -theta_x."""
        n = self.N_NODES
        B = np.zeros((3, 6 * n))
        for i in range(n):
            c = 6 * i
            B[0, c + 4] = dNdx[0, i]
            B[1, c + 3] = -dNdx[1, i]
            B[2, c + 4] = dNdx[1, i]
            B[2, c + 3] = -dNdx[0, i]
        return B

    def direct_shear_B(self, xi: float, eta: float) -> np.ndarray:
        """Displacement-interpolated [gamma_yz, gamma_xz] = [w,y - theta_x, w,x + theta_y]."""
        N, _ = self.shape_functions(xi, eta)
        _, _, dNdx = self.jacobian(xi, eta)
        n = self.N_NODES
        B = np.zeros((2, 6 * n))
        for i in range(n):
            c = 6 * i
            B[0, c + 2] = dNdx[1, i]
            B[0, c + 3] = -N[i]
            B[1, c + 2] = dNdx[0, i]
            B[1, c + 4] = N[i]
        return B

    def drilling_B(self, xi: float, eta: float) -> np.ndarray:
        """Row tying theta_z to the in-plane rotation (1/2)(v,x - u,y), on the full local DOF vector."""
        N, _ = self.shape_functions(xi, eta)
        _, _, dNdx = self.jacobian(xi, eta)
        b = np.zeros(6 * self.N_NODES)
        for i in range(self.N_NODES):
            c = 6 * i
            b[c + 5] = N[i]
            b[c] = 0.5 * dNdx[1, i]
            b[c + 1] = -0.5 * dNdx[0, i]
        return self._expand_standard(b[None, :])[0]

    def in_plane_operator(self, xi: float, eta: float) -> np.ndarray:
        """B_e: (6+4m) x n_dof map from local DOFs to e = [eps0, kappa, grad a_1, ...]."""
        _, _, dNdx = self.jacobian(xi, eta)
        m, d = self.n_enrichment, self.dofs_per_node
        B = np.zeros((6 + 4 * m, self.n_dof))
        B[:6] = self._expand_standard(np.vstack([self.membrane_B(dNdx), self.bending_B(dNdx)]))
        for j in range(m):
            for i in range(self.N_NODES):
                cx = d * i + 6 + 2 * j
                B[6 + 4 * j, cx] = dNdx[0, i]
                B[7 + 4 * j, cx] = dNdx[1, i]
                B[8 + 4 * j, cx + 1] = dNdx[0, i]
                B[9 + 4 * j, cx + 1] = dNdx[1, i]
        return B

    def shear_operator(self, xi: float, eta: float) -> np.ndarray:
        """B_s: (2+2m) x n_dof map from local DOFs to s = [gamma0, a_1, ...]."""
        m, d = self.n_enrichment, self.dofs_per_node
        B = np.zeros((2 + 2 * m, self.n_dof))
        B[:2] = self._expand_standard(self._gamma0_operator(xi, eta))
        if m:
            N, _ = self.shape_functions(xi, eta)
            for j in range(m):
                for i in range(self.N_NODES):
                    cx = d * i + 6 + 2 * j
                    B[2 + 2 * j, cx] = N[i]
                    B[3 + 2 * j, cx + 1] = N[i]
        return B

    def shear_B(self, xi: float, eta: float) -> np.ndarray:
        """Mean transverse shear operator [gamma_yz; gamma_xz] on the full local DOF vector."""
        return self._expand_standard(self._gamma0_operator(xi, eta))

    def strain_operators(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(B_m, B_b, B_s) acting on the *global* element displacement vector."""
        T = self.transformation
        B_e = self.in_plane_operator(xi, eta)
        return B_e[:3] @ T, B_e[3:6] @ T, self.shear_B(xi, eta) @ T

    # ------------------------------------------------------------ matrices
    @property
    def drilling_modulus(self) -> float:
        """Penalty modulus (force per length) tying theta_z to the in-plane rotation."""
        A = self.section.A
        return self.drilling_factor * max(A[2, 2], np.max(np.abs(A)) * 1e-6)

    def gauss_points(self) -> list:
        """[(xi, eta, weight * detJ)] at the recovery points (used for results)."""
        return [(xi, eta, w * self.jacobian(xi, eta)[1]) for xi, eta, w in self.recovery_rule()]

    def _internal_matrices(self):
        """(K_aa, K_au) of the incompatible modes, or (None, None)."""
        if self._incompatible_operator(*self.NATURAL_CENTROID) is None:
            return None, None
        C = self.section.C_ee
        K_aa = K_au = None
        for xi, eta, w in self.stiffness_rule():
            _, detJ, _ = self.jacobian(xi, eta)
            G = self._incompatible_operator(xi, eta)
            B_a = np.zeros((C.shape[0], G.shape[1]))
            B_a[:3] = G
            B_e = self.in_plane_operator(xi, eta)
            aa = B_a.T @ C @ B_a * (w * detJ)
            au = B_a.T @ C @ B_e * (w * detJ)
            K_aa = aa if K_aa is None else K_aa + aa
            K_au = au if K_au is None else K_au + au
        return K_aa, K_au

    def local_stiffness_matrix(self) -> np.ndarray:
        if self._K_local is not None:
            return self._K_local
        section = self.section
        K = np.zeros((self.n_dof, self.n_dof))
        drill = self.drilling_modulus
        for xi, eta, w in self.stiffness_rule():
            _, detJ, _ = self.jacobian(xi, eta)
            B_e = self.in_plane_operator(xi, eta)
            b_d = self.drilling_B(xi, eta)
            K += B_e.T @ section.C_ee @ B_e * (w * detJ)
            K += drill * np.outer(b_d, b_d) * (w * detJ)
        for xi, eta, w in self.shear_rule():
            _, detJ, _ = self.jacobian(xi, eta)
            B_s = self.shear_operator(xi, eta)
            K += B_s.T @ section.C_ss @ B_s * (w * detJ)
        K_aa, K_au = self._internal_matrices()
        if K_aa is not None:
            self._recovery_map = -np.linalg.solve(K_aa, K_au)  # alpha = R u_local
            K = K + K_au.T @ self._recovery_map
        self._K_local = 0.5 * (K + K.T)
        return self._K_local

    def stiffness_matrix(self) -> np.ndarray:
        """Element stiffness in global DOFs: K = T^T K_local T."""
        T = self.transformation
        return T.T @ self.local_stiffness_matrix() @ T

    # --------------------------------------------------------------- loads
    def hygrothermal_resultants(self, state: HygrothermalState) -> Tuple[np.ndarray, np.ndarray]:
        return self._ht_engine.resultants(self._laminate, state)

    def generalized_hygrothermal(self, state: HygrothermalState) -> np.ndarray:
        return generalized_hygrothermal(self._laminate, self.section.basis, state, self._ht_engine)

    def hygrothermal_load_vector(self, state: HygrothermalState) -> np.ndarray:
        """Environmental load F_ht = int B_e^T N_ht,gen dA (global DOFs)."""
        F = np.zeros(self.n_dof)
        if state.is_null:
            return F
        N_gen = self.generalized_hygrothermal(state)
        for xi, eta, w in self.stiffness_rule():
            _, detJ, _ = self.jacobian(xi, eta)
            F += self.in_plane_operator(xi, eta).T @ N_gen * (w * detJ)
        return self.transformation.T @ F

    def pressure_load_vector(self, pressure: Pressure) -> np.ndarray:
        """Consistent load of a pressure acting along the element normal (+e3).

        ``pressure`` is a number or a function of the global position.
        """
        F = np.zeros(self.n_dof)
        if not callable(pressure) and pressure == 0.0:
            return F
        d = self.dofs_per_node
        for xi, eta, w in self.load_rule():
            N, _ = self.shape_functions(xi, eta)
            _, detJ, _ = self.jacobian(xi, eta)
            p = float(pressure(self.global_coordinates(xi, eta))) if callable(pressure) else float(pressure)
            F[2::d] += N * p * detJ * w
        return self.transformation.T @ F

    def body_load_vector(self, acceleration: Sequence[float]) -> np.ndarray:
        """Consistent load of a uniform acceleration field (e.g. gravity, in length/time^2).

        The force per unit area is rho_A a, with rho_A the laminate areal density. When
        the laminate is offset from the reference surface the in-plane part also gives
        a distributed moment about the reference surface.
        """
        a_local = self.frame @ np.asarray(acceleration, dtype=float)
        f = self._laminate.areal_density() * a_local
        m = self._laminate.offset * np.array([-f[1], f[0], 0.0])  # r x f with r = offset e3
        F = np.zeros(self.n_dof)
        d = self.dofs_per_node
        for xi, eta, w in self.load_rule():
            N, _ = self.shape_functions(xi, eta)
            _, detJ, _ = self.jacobian(xi, eta)
            for c in range(3):
                F[c::d] += N * f[c] * detJ * w
                F[3 + c :: d] += N * m[c] * detJ * w
        return self.transformation.T @ F

    def edge_shape_functions(self, edge: int, s: float) -> np.ndarray:
        """1D shape functions on an edge (s in [-1, 1]) for the nodes listed in ``EDGES[edge]``."""
        if len(self.EDGES[edge]) == 2:
            return np.array([0.5 * (1.0 - s), 0.5 * (1.0 + s)])
        return np.array([0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)])

    def _edge_derivatives(self, edge: int, s: float) -> np.ndarray:
        if len(self.EDGES[edge]) == 2:
            return np.array([-0.5, 0.5])
        return np.array([s - 0.5, -2.0 * s, s + 0.5])

    def edge_length(self, edge: int) -> float:
        nodes = self.EDGES[edge]
        X = self.coordinates[list(nodes)]
        pts, wts = _gauss_1d(3)
        return float(sum(w * np.linalg.norm(self._edge_derivatives(edge, s) @ X) for s, w in zip(pts, wts)))

    def edge_load_vector(
        self, edge: int, force_per_length: Union[Sequence[float], Callable[[np.ndarray], Sequence[float]]]
    ) -> np.ndarray:
        """Consistent load of a line load on one edge, in global components.

        ``force_per_length`` is a 3-vector [fx, fy, fz] or a 6-vector that adds
        distributed moments [mx, my, mz], constant or a function of the global position.
        """
        nodes = list(self.EDGES[edge])
        X = self.coordinates[nodes]
        F = np.zeros(self.n_dof)
        d = self.dofs_per_node
        pts, wts = _gauss_1d(3)
        for s, w in zip(pts, wts):
            N = self.edge_shape_functions(edge, s)
            ds = np.linalg.norm(self._edge_derivatives(edge, s) @ X)
            xyz = N @ X
            q = np.asarray(force_per_length(xyz) if callable(force_per_length) else force_per_length, dtype=float)
            if q.size not in (3, 6):
                raise ValueError("force_per_length needs 3 (forces) or 6 (forces and moments) components")
            for a, local in enumerate(nodes):
                F[d * local : d * local + q.size] += N[a] * q * ds * w
        return F

    def edge_index(self, node_ids: Iterable[int]) -> List[int]:
        """Edges whose nodes all lie in ``node_ids``."""
        chosen = {int(n) for n in node_ids}
        return [k for k, edge in enumerate(self.EDGES) if all(self.node_ids[i] in chosen for i in edge)]

    # ------------------------------------------------------- post-processing
    def strain_state(self, u_elem: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generalised strains (e, s) at (xi, eta): e = [eps0, kappa, grad a...], s = [gamma0, a...]."""
        u_local = self.transformation @ np.asarray(u_elem, dtype=float)
        e = self.in_plane_operator(xi, eta) @ u_local
        G = self._incompatible_operator(xi, eta)
        if G is not None:
            self.local_stiffness_matrix()  # builds the recovery map
            e = e.copy()
            e[:3] += G @ (self._recovery_map @ u_local)
        s = self.shear_operator(xi, eta) @ u_local
        return e, s

    def generalized_strains(
        self, u_elem: np.ndarray, xi: float = 0.0, eta: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mid-plane strains eps0, curvatures kappa and mean shear gamma0 in local axes."""
        e, s = self.strain_state(u_elem, xi, eta)
        return e[:3], e[3:6], s[:2]

    def section_resultants(
        self, e: np.ndarray, s: np.ndarray, state: Optional[HygrothermalState] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mechanical N, M and Q from the generalised strains."""
        section = self.section
        NM = section.C_ee[:6] @ e
        if state is not None and not state.is_null:
            NM = NM - self.generalized_hygrothermal(state)[:6]
        return NM[:3], NM[3:6], section.C_ss[:2] @ s

    def strain_energy(self, u_elem: np.ndarray) -> float:
        u_elem = np.asarray(u_elem, dtype=float)
        return 0.5 * float(u_elem @ self.stiffness_matrix() @ u_elem)

    def energy_split(self, u_elem: np.ndarray) -> Dict[str, float]:
        """Strain energy split into membrane, bending, coupling, shear, enrichment and drilling."""
        C = self.section.C_ee
        out = dict.fromkeys(("membrane", "bending", "coupling", "enrichment", "shear", "drilling"), 0.0)
        u_local = self.transformation @ np.asarray(u_elem, dtype=float)
        for xi, eta, w in self.stiffness_rule():
            _, detJ, _ = self.jacobian(xi, eta)
            e, _ = self.strain_state(u_elem, xi, eta)
            dA = w * detJ
            eps, kap = e[:3], e[3:6]
            out["membrane"] += 0.5 * eps @ C[:3, :3] @ eps * dA
            out["bending"] += 0.5 * kap @ C[3:6, 3:6] @ kap * dA
            out["coupling"] += eps @ C[:3, 3:6] @ kap * dA
            if e.size > 6:
                total = 0.5 * e @ C @ e
                out["enrichment"] += (total - 0.5 * e[:6] @ C[:6, :6] @ e[:6]) * dA
            out["drilling"] += 0.5 * self.drilling_modulus * (self.drilling_B(xi, eta) @ u_local) ** 2 * dA
        for xi, eta, w in self.shear_rule():
            _, detJ, _ = self.jacobian(xi, eta)
            s = self.shear_operator(xi, eta) @ u_local
            out["shear"] += 0.5 * s @ self.section.C_ss @ s * w * detJ
        return {k: float(v) for k, v in out.items()}

    def extrapolation_matrix(self) -> np.ndarray:
        """(n_nodes, n_recovery) map from recovery-point values to nodal values."""
        pts = [(xi, eta) for xi, eta, _ in self.recovery_rule()]
        P = np.array([self.recovery_basis(xi, eta) for xi, eta in pts])
        nodes = np.array([self.recovery_basis(xi, eta) for xi, eta in self.NATURAL_NODES])
        if len(pts) < P.shape[1]:
            return np.full((self.N_NODES, len(pts)), 1.0 / len(pts))
        return nodes @ np.linalg.pinv(P)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.element_id}, nodes={self.node_ids}, model={self._kinematics.name})"


# ---------------------------------------------------------------------------
# Quad4
# ---------------------------------------------------------------------------
_Q4_XI = np.array([-1.0, 1.0, 1.0, -1.0])
_Q4_ETA = np.array([-1.0, -1.0, 1.0, 1.0])


def shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinear shape functions N (4,) and natural derivatives dN (2, 4) = [dN/dxi; dN/deta]."""
    N = 0.25 * (1.0 + _Q4_XI * xi) * (1.0 + _Q4_ETA * eta)
    dN = np.vstack([0.25 * _Q4_XI * (1.0 + _Q4_ETA * eta), 0.25 * _Q4_ETA * (1.0 + _Q4_XI * xi)])
    return N, dN


class Quad4ShellElement(ShellElement):
    """4-node shell: bilinear membrane and bending, MITC4 shear, drilling penalty.

    ``membrane="incompatible"`` (default) adds four internal Wilson modes with Taylor's
    correction (QM6), condensed out element by element. They remove the shear locking
    of the bilinear membrane under in-plane bending and still pass the patch test.
    ``membrane="standard"`` gives the plain bilinear membrane.
    """

    TOPOLOGY = "quad4"
    N_NODES = 4
    N_CORNERS = 4
    NATURAL_NODES = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    NATURAL_CENTROID = (0.0, 0.0)
    EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))

    def __init__(self, *args, membrane: str = "incompatible", **kwargs):
        if membrane not in ("incompatible", "standard"):
            raise ValueError("membrane must be 'incompatible' or 'standard'")
        self.membrane = membrane
        super().__init__(*args, **kwargs)

    def _options(self) -> dict:
        return {**super()._options(), "membrane": self.membrane}

    def _check_order(self, order):
        order = 2 if order is None else int(order)
        if order not in (1, 2, 3):
            raise ValueError("integration_order must be 1, 2 or 3")
        return order

    def shape_functions(self, xi, eta):
        return shape_functions(xi, eta)

    def stiffness_rule(self):
        return _tensor_rule(self.integration_order)

    def recovery_basis(self, xi, eta):
        return np.array([1.0, xi, eta, xi * eta])

    def _normal_estimate(self):
        X = self.coordinates
        return np.cross(X[2] - X[0], X[3] - X[1])

    def _covariant_shear_row(self, xi: float, eta: float, direction: int) -> np.ndarray:
        """Row of the covariant shear strain gamma_{xi z} (direction 0) or gamma_{eta z} (1)."""
        N, dN = shape_functions(xi, eta)
        J = dN @ self.local_xy
        x_d, y_d = J[direction, 0], J[direction, 1]
        row = np.zeros(24)
        for i in range(4):
            c = 6 * i
            row[c + 2] = dN[direction, i]
            row[c + 4] = N[i] * x_d
            row[c + 3] = -N[i] * y_d
        return row

    def _gamma0_operator(self, xi, eta):
        """MITC4 assumed covariant shear field tied at the edge midpoints."""
        g_xi = 0.5 * (1.0 - eta) * self._covariant_shear_row(0.0, -1.0, 0) + 0.5 * (1.0 + eta) * self._covariant_shear_row(0.0, 1.0, 0)
        g_eta = 0.5 * (1.0 - xi) * self._covariant_shear_row(-1.0, 0.0, 1) + 0.5 * (1.0 + xi) * self._covariant_shear_row(1.0, 0.0, 1)
        J, _, _ = self.jacobian(xi, eta)
        cartesian = np.linalg.solve(J, np.vstack([g_xi, g_eta]))  # [gamma_xz; gamma_yz]
        return cartesian[[1, 0], :]

    def _incompatible_operator(self, xi, eta):
        if self.membrane != "incompatible":
            return None
        _, _, dN0 = self.jacobian(0.0, 0.0)
        J0, det0, _ = self.jacobian(0.0, 0.0)
        _, detJ, _ = self.jacobian(xi, eta)
        # natural derivatives of P1 = 1 - xi^2 and P2 = 1 - eta^2, mapped with J0 (Taylor)
        dP = np.array([[-2.0 * xi, 0.0], [0.0, -2.0 * eta]])  # rows d/dxi, d/deta; columns P1, P2
        dPdx = np.linalg.solve(J0, dP) * (det0 / detJ)  # rows d/dx, d/dy
        G = np.zeros((3, 4))
        G[0, 0:2] = dPdx[0]  # u modes -> eps_x
        G[1, 2:4] = dPdx[1]  # v modes -> eps_y
        G[2, 0:2] = dPdx[1]
        G[2, 2:4] = dPdx[0]
        return G


# Backwards-compatible name used throughout the first releases.
CompositePlateElement = Quad4ShellElement


# ---------------------------------------------------------------------------
# Tri3
# ---------------------------------------------------------------------------
class Tri3ShellElement(ShellElement):
    """3-node shell: constant-strain membrane, linear bending, MITC3 shear (Lee and Bathe 2004).

    Natural coordinates are the area coordinates (r, s) of nodes 2 and 3.
    """

    TOPOLOGY = "tri3"
    N_NODES = 3
    N_CORNERS = 3
    NATURAL_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    NATURAL_CENTROID = (1.0 / 3.0, 1.0 / 3.0)
    EDGES = ((0, 1), (1, 2), (2, 0))
    _RULE = [(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)]

    def shape_functions(self, xi, eta):
        N = np.array([1.0 - xi - eta, xi, eta])
        dN = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        return N, dN

    def stiffness_rule(self):
        return list(self._RULE)

    def recovery_basis(self, xi, eta):
        return np.array([1.0, xi, eta])

    def _normal_estimate(self):
        X = self.coordinates
        return np.cross(X[1] - X[0], X[2] - X[0])

    def _covariant_row(self, r: float, s: float, direction) -> np.ndarray:
        """Covariant shear strain row along natural direction 0 (r), 1 (s) or 'q' (s - r)."""
        N, dN = self.shape_functions(r, s)
        J = dN @ self.local_xy
        if direction == "q":
            dn, t = dN[1] - dN[0], J[1] - J[0]
        else:
            dn, t = dN[direction], J[direction]
        row = np.zeros(18)
        for i in range(3):
            c = 6 * i
            row[c + 2] = dn[i]
            row[c + 4] = N[i] * t[0]
            row[c + 3] = -N[i] * t[1]
        return row

    def _gamma0_operator(self, xi, eta):
        e_rt1 = self._covariant_row(0.5, 0.0, 0)
        e_st2 = self._covariant_row(0.0, 0.5, 1)
        e_qt3 = self._covariant_row(0.5, 0.5, "q")  # e_st - e_rt on the hypotenuse
        # tie the tangential strain e_st - e_rt at the hypotenuse midpoint
        c = e_st2 - e_rt1 - e_qt3
        g_r = e_rt1 + c * eta
        g_s = e_st2 - c * xi
        J, _, _ = self.jacobian(xi, eta)
        cartesian = np.linalg.solve(J, np.vstack([g_r, g_s]))  # [gamma_xz; gamma_yz]
        return cartesian[[1, 0], :]


# ---------------------------------------------------------------------------
# Quad8
# ---------------------------------------------------------------------------
_Q8_NODES = np.array(
    [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
)


class Quad8ShellElement(ShellElement):
    """8-node serendipity shell (Gmsh / CalculiX S8R node order: corners, then mid-sides).

    Membrane, bending and enrichment use 3x3 Gauss points; transverse shear uses 2x2
    (selective reduced integration), which removes shear locking. Results are sampled
    at the 2x2 points, the superconvergent points of quadratic elements. The element is
    flat, so curved edges must lie in its plane (e.g. hole boundaries in a plate).
    """

    TOPOLOGY = "quad8"
    N_NODES = 8
    N_CORNERS = 4
    NATURAL_NODES = _Q8_NODES
    NATURAL_CENTROID = (0.0, 0.0)
    EDGES = ((0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0))

    def shape_functions(self, xi, eta):
        N = np.zeros(8)
        dN = np.zeros((2, 8))
        for i, (a, b) in enumerate(_Q8_NODES):
            if i < 4:
                N[i] = 0.25 * (1 + a * xi) * (1 + b * eta) * (a * xi + b * eta - 1)
                dN[0, i] = 0.25 * a * (1 + b * eta) * (2 * a * xi + b * eta)
                dN[1, i] = 0.25 * b * (1 + a * xi) * (a * xi + 2 * b * eta)
            elif a == 0.0:
                N[i] = 0.5 * (1 - xi * xi) * (1 + b * eta)
                dN[0, i] = -xi * (1 + b * eta)
                dN[1, i] = 0.5 * b * (1 - xi * xi)
            else:
                N[i] = 0.5 * (1 + a * xi) * (1 - eta * eta)
                dN[0, i] = 0.5 * a * (1 - eta * eta)
                dN[1, i] = -eta * (1 + a * xi)
        return N, dN

    def stiffness_rule(self):
        return _tensor_rule(3)

    def shear_rule(self):
        return _tensor_rule(2)

    def recovery_rule(self):
        return _tensor_rule(2)

    def recovery_basis(self, xi, eta):
        return np.array([1.0, xi, eta, xi * eta])

    def _normal_estimate(self):
        X = self.coordinates
        return np.cross(X[2] - X[0], X[3] - X[1])

    def _gamma0_operator(self, xi, eta):
        return self.direct_shear_B(xi, eta)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
class ElementRegistry:
    """Topology name -> element class, with lookup by node count (open for extension)."""

    def __init__(self):
        self._classes: Dict[str, Type[ShellElement]] = {}

    def register(self, element_class: Type[ShellElement], name: Optional[str] = None) -> None:
        self._classes[(name or element_class.TOPOLOGY).lower()] = element_class

    def names(self) -> List[str]:
        return list(self._classes)

    def resolve(self, n_nodes: int, element_type: Optional[str] = None) -> Type[ShellElement]:
        if element_type is not None:
            try:
                cls = self._classes[element_type.lower()]
            except KeyError as exc:
                raise KeyError(f"Unknown element type '{element_type}'. Registered: {', '.join(self._classes)}") from exc
            if cls.N_NODES != n_nodes:
                raise ValueError(f"Element type '{element_type}' needs {cls.N_NODES} nodes, got {n_nodes}")
            return cls
        matches = [c for c in self._classes.values() if c.N_NODES == n_nodes]
        if not matches:
            raise ValueError(f"No registered element has {n_nodes} nodes")
        return matches[0]

    def create(
        self,
        element_id: int,
        node_ids: Sequence[int],
        coordinates: np.ndarray,
        laminate: Laminate,
        kinematics: KinematicModel,
        element_type: Optional[str] = None,
        **options,
    ) -> ShellElement:
        cls = self.resolve(len(node_ids), element_type)
        return cls(element_id, node_ids, coordinates, laminate, kinematics, **options)


DEFAULT_ELEMENTS = ElementRegistry()
DEFAULT_ELEMENTS.register(Quad4ShellElement)
DEFAULT_ELEMENTS.register(Tri3ShellElement)
DEFAULT_ELEMENTS.register(Quad8ShellElement)

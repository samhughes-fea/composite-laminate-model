"""Layered solid elements and shell-to-solid coupling.

Shell theories condense the thickness into resultants. Where the full 3D stress
state is the answer (thick sections, free edges, ply drops, joints, the region
round a hole) the laminate can be modelled with layered solids instead, and joined
to the surrounding shells.

Elements
--------
* ``LayeredHex8Element``  8-node brick with Wilson-Taylor incompatible modes (no
                          shear locking in bending, passes the patch test).
* ``LayeredHex20Element`` 20-node serendipity brick (CalculiX C3D20 node order).

The plies of the element's laminate are stacked through the element along its
natural zeta direction (bottom face nodes 0-3, top face 4-7), each over a share of
the thickness equal to its share of the laminate thickness, and each integrated with
its own Gauss points. One element can hold one ply or a whole stack. Ply angles are
measured about the stack direction from the element material axis (``material_axis``
projected normal to the stack, global x by default).

Every node keeps the six DOFs of the mesh so solids and shells share one DOF map.
Solids have no rotational stiffness: the solver holds the rotations of nodes used
only by solids at zero, and nodes shared with shells take their rotations from them.

Coupling
--------
``couple_shell_to_solid`` ties solid nodes on an interface to shell nodes with the
shell's first-order kinematics, u_solid = u_shell + theta_shell x (x_solid - x_shell),
as linear multi-point constraints. Enrichment amplitudes of full RZT and layerwise
shells are not transferred: use condensed kinematics next to a solid region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np

from .laminate_hygrothermal import NULL_STATE, HygrothermalEngine, HygrothermalState
from .laminate_materials_3d import rotate_stiffness_3d, stress_rotation_3d
from .laminate_mechanics import Laminate

STANDARD_DOFS = 6


def _direction(axis, xyz):
    if axis is None:
        return np.array([1.0, 0.0, 0.0])
    if hasattr(axis, "direction_at"):
        return np.asarray(axis.direction_at(xyz), dtype=float)
    if callable(axis):
        return np.asarray(axis(xyz), dtype=float)
    return np.asarray(axis, dtype=float)


@dataclass(frozen=True)
class SolidPoint:
    """Stress state at one point of a layered solid (Voigt order [11, 22, 33, 23, 13, 12])."""

    element_id: int
    ply_index: int
    xi: float
    eta: float
    zeta: float
    xyz: np.ndarray
    z: float  # height in the element laminate (Laminate.z_interfaces); solid_block offsets match the full laminate
    strain_local: np.ndarray  # element axes (e1 material axis, e3 stack direction)
    stress_local: np.ndarray
    stress_material: np.ndarray  # ply axes [s1, s2, s3, t23, t13, t12]
    stress_global: np.ndarray  # global x, y, z axes [sx, sy, sz, tyz, txz, txy]


def _voigt_rotation(F: np.ndarray) -> np.ndarray:
    """Stress Voigt transformation for a frame change sigma' = F sigma F^T (rows of F = new axes)."""
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    T = np.zeros((6, 6))
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            if k == l:
                T[I, J] = F[i, k] * F[j, l]
            else:
                T[I, J] = F[i, k] * F[j, l] + F[i, l] * F[j, k]
    return T


class LayeredSolidElement:
    """Base class of the layered solids."""

    IS_SOLID = True
    TOPOLOGY = "solid"
    N_NODES = 0
    NATURAL_NODES = np.zeros((0, 3))
    EDGES: Tuple = ()  # no line loads on solids
    TOP_FACE: Tuple[int, ...] = ()
    BOTTOM_FACE: Tuple[int, ...] = ()
    IN_PLANE_ORDER = 2
    PLY_ORDER = 2

    def __init__(
        self,
        element_id: int,
        node_ids: Sequence[int],
        coordinates: np.ndarray,
        laminate: Laminate,
        kinematics=None,
        material_axis=None,
        hygrothermal_engine: Optional[HygrothermalEngine] = None,
        gradient_origin: Optional[float] = None,
        **_ignored,
    ):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != (self.N_NODES, 3):
            raise ValueError(f"coordinates must have shape ({self.N_NODES}, 3)")
        if len(node_ids) != self.N_NODES:
            raise ValueError(f"A {self.TOPOLOGY} element needs exactly {self.N_NODES} node ids")
        self.element_id = int(element_id)
        self.node_ids = tuple(int(n) for n in node_ids)
        self.coordinates = coordinates
        self.material_axis = material_axis
        self.gradient_origin = gradient_origin
        self._laminate = laminate
        self._kinematics = kinematics
        self._ht_engine = hygrothermal_engine or HygrothermalEngine()
        self._K: Optional[np.ndarray] = None
        self._recovery = None
        self._Kaa = None
        self.frame = self.point_frame(0.0, 0.0, 0.0)
        for xi, eta, zeta, _, _ in self.integration_points():
            if self.jacobian(xi, eta, zeta)[1] <= 0.0:
                raise ValueError(f"Solid element {self.element_id} has a non-positive Jacobian; check node ordering")

    # ----------------------------------------------------------- interface
    laminate = property(lambda self: self._laminate)
    kinematics = property(lambda self: self._kinematics)
    hygrothermal_engine = property(lambda self: self._ht_engine)
    n_enrichment = 0
    enrichment_signature = None
    dofs_per_node = STANDARD_DOFS

    @property
    def n_dof(self) -> int:
        return STANDARD_DOFS * self.N_NODES

    def _clone(self, **changes):
        args = dict(laminate=self._laminate, kinematics=self._kinematics, material_axis=self.material_axis,
                    hygrothermal_engine=self._ht_engine, gradient_origin=self.gradient_origin)
        args.update(changes)
        return type(self)(self.element_id, self.node_ids, self.coordinates, **args)

    def with_laminate(self, laminate: Laminate):
        return self._clone(laminate=laminate)

    def with_kinematics(self, kinematics):
        return self  # solids have no shell kinematics

    def with_material_axis(self, material_axis):
        return self._clone(material_axis=material_axis)

    def edge_index(self, node_ids) -> List[int]:
        return []

    # ------------------------------------------------------------ geometry
    def _shape(self, xi: float, eta: float, zeta: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def shape_functions(self, xi: float, eta: float, zeta: float) -> Tuple[np.ndarray, np.ndarray]:
        """(N, dN/dxi) at a natural point, cached per element class (they do not depend on geometry)."""
        cache = type(self).__dict__.get("_shape_cache")
        if cache is None:
            cache = {}
            setattr(type(self), "_shape_cache", cache)
        key = (float(xi), float(eta), float(zeta))
        value = cache.get(key)
        if value is None:
            if len(cache) > 4096:
                cache.clear()
            value = self._shape(*key)
            cache[key] = value
        return value

    def jacobian(self, xi: float, eta: float, zeta: float):
        """(J, det J, dN/dX) with J[r, c] = dX_c / dxi_r."""
        _, dN = self.shape_functions(xi, eta, zeta)
        J = dN @ self.coordinates
        return J, float(np.linalg.det(J)), np.linalg.solve(J, dN)

    def global_coordinates(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        return self.shape_functions(xi, eta, zeta)[0] @ self.coordinates

    @property
    def centroid(self) -> np.ndarray:
        return self.global_coordinates(0.0, 0.0, 0.0)

    @property
    def volume(self) -> float:
        return float(sum(self.jacobian(x, e, z)[1] * w for x, e, z, w, _ in self.integration_points()))

    @property
    def area(self) -> float:
        """Mid-surface area (volume over mean thickness)."""
        return self.volume / max(self.thickness, 1e-300)

    @property
    def thickness(self) -> float:
        J, _, _ = self.jacobian(0.0, 0.0, 0.0)
        return float(2.0 * np.linalg.norm(J[2]))

    def point_frame(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """Rows e1 (material axis), e2, e3 (stack direction) at a point."""
        J, _, _ = self.jacobian(xi, eta, zeta)
        return self._frame_from(J, xi, eta, zeta)

    def _frame_from(self, J: np.ndarray, xi: float, eta: float, zeta: float) -> np.ndarray:
        e3 = J[2] / np.linalg.norm(J[2])
        if self.material_axis is None or not (callable(self.material_axis) or hasattr(self.material_axis, "direction_at")):
            reference = _direction(self.material_axis, None)
        else:
            reference = _direction(self.material_axis, self.global_coordinates(xi, eta, zeta))
        e1 = reference - np.dot(reference, e3) * e3
        if np.linalg.norm(e1) < 1e-8 * max(np.linalg.norm(reference), 1e-300):
            e1 = J[0] - np.dot(J[0], e3) * e3
        e1 /= np.linalg.norm(e1)
        return np.vstack([e1, np.cross(e3, e1), e3])

    # ---------------------------------------------------------- integration
    def ply_zeta_bounds(self) -> List[Tuple[float, float]]:
        t = np.array([p.thickness for p in self._laminate.plies])
        edges = -1.0 + 2.0 * np.concatenate([[0.0], np.cumsum(t)]) / t.sum()
        return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]

    def integration_points(self) -> List[Tuple[float, float, float, float, int]]:
        """(xi, eta, zeta, weight, ply index) - in-plane Gauss rule times a Gauss rule per ply."""
        g, w = np.polynomial.legendre.leggauss(self.IN_PLANE_ORDER)
        gz, wz = np.polynomial.legendre.leggauss(self.PLY_ORDER)
        out = []
        for k, (a, b) in enumerate(self.ply_zeta_bounds()):
            half = 0.5 * (b - a)
            for zz, wzz in zip(gz, wz):
                zeta = 0.5 * (a + b) + half * zz
                for x, wx in zip(g, w):
                    for e, we in zip(g, w):
                        out.append((float(x), float(e), float(zeta), float(wx * we * wzz * half), k))
        return out

    def _z_in_laminate(self, zeta: float) -> float:
        z = self._laminate.z_interfaces
        return float(z[0] + 0.5 * (zeta + 1.0) * (z[-1] - z[0]))

    # ------------------------------------------------------------ operators
    def strain_B(self, xi: float, eta: float, zeta: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """(B (6 x n_dof) giving local-frame engineering strains, frame, det J)."""
        J, detJ, dNdX = self.jacobian(xi, eta, zeta)
        F = self._frame_from(J, xi, eta, zeta)
        return self._B_from_gradients(F, dNdX), F, detJ

    def _B_from_gradients(self, F: np.ndarray, dNdX: np.ndarray) -> np.ndarray:
        n = dNdX.shape[1]
        g = F @ dNdX  # derivatives along the local axes, (3, n)
        pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        B = np.zeros((6, n, STANDARD_DOFS))
        for row, (a, b) in enumerate(pairs):
            block = np.outer(g[b], F[a])
            if a != b:
                block = block + np.outer(g[a], F[b])
            B[row, :, :3] = block
        return B.reshape(6, STANDARD_DOFS * n)

    def _incompatible_B(self, xi, eta, zeta, F) -> Optional[np.ndarray]:
        return None

    def ply_stiffness(self, k: int) -> np.ndarray:
        """6x6 stiffness of ply k in the element axes (rotated by the ply angle)."""
        ply = self._laminate.plies[k]
        return rotate_stiffness_3d(ply.material.elastic.stiffness_matrix(), ply.angle)

    def stiffness_matrix(self) -> np.ndarray:
        if self._K is not None:
            return self._K
        n = self.n_dof
        K = np.zeros((n, n))
        Kaa = Kau = None
        C = [self.ply_stiffness(k) for k in range(self._laminate.n_plies)]
        for xi, eta, zeta, w, k in self.integration_points():
            B, F, detJ = self.strain_B(xi, eta, zeta)
            dV = w * detJ
            K += B.T @ C[k] @ B * dV
            G = self._incompatible_B(xi, eta, zeta, F)
            if G is not None:
                aa, au = G.T @ C[k] @ G * dV, G.T @ C[k] @ B * dV
                Kaa = aa if Kaa is None else Kaa + aa
                Kau = au if Kau is None else Kau + au
        if Kaa is not None:
            self._Kaa = Kaa
            self._recovery = -np.linalg.solve(Kaa, Kau)
            K = K + Kau.T @ self._recovery
        self._K = 0.5 * (K + K.T)
        return self._K

    def local_stiffness_matrix(self) -> np.ndarray:
        return self.stiffness_matrix()

    def _free_strain_local(self, k: int, zeta: float, state: HygrothermalState) -> np.ndarray:
        if state.is_null:
            return np.zeros(6)
        ply = self._laminate.plies[k]
        origin = self._laminate.mid_plane if self.gradient_origin is None else self.gradient_origin
        return self._ht_engine.global_free_strain_3d(ply, state, self._z_in_laminate(zeta), origin)

    def strain_at(self, u_elem: np.ndarray, xi: float, eta: float, zeta: float,
                  state: HygrothermalState = NULL_STATE) -> Tuple[np.ndarray, np.ndarray]:
        """(local-frame total strain, frame) at a point, including the incompatible modes.

        The incompatible-mode amplitudes are recovered from the nodal displacements and,
        for a hygrothermal ``state``, from its free strains.
        """
        u = np.asarray(u_elem, dtype=float)
        B, F, _ = self.strain_B(xi, eta, zeta)
        eps = B @ u
        G = self._incompatible_B(xi, eta, zeta, F)
        if G is not None:
            self.stiffness_matrix()
            alpha = self._recovery @ u
            if not state.is_null:
                alpha = alpha + np.linalg.solve(self._Kaa, self._incompatible_load(state))
            eps = eps + G @ alpha
        return eps, F

    def _incompatible_load(self, state: HygrothermalState) -> np.ndarray:
        f_a = 0.0
        for xi, eta, zeta, w, k in self.integration_points():
            B, Fr, detJ = self.strain_B(xi, eta, zeta)
            G = self._incompatible_B(xi, eta, zeta, Fr)
            f_a = f_a + G.T @ (self.ply_stiffness(k) @ self._free_strain_local(k, zeta, state)) * (w * detJ)
        return f_a

    # --------------------------------------------------------------- loads
    def hygrothermal_load_vector(self, state: HygrothermalState) -> np.ndarray:
        """F = int B^T C eps_free dV (with the incompatible modes condensed consistently)."""
        F = np.zeros(self.n_dof)
        if state.is_null:
            return F
        f_a = None
        for xi, eta, zeta, w, k in self.integration_points():
            B, Fr, detJ = self.strain_B(xi, eta, zeta)
            sig0 = self.ply_stiffness(k) @ self._free_strain_local(k, zeta, state) * (w * detJ)
            F += B.T @ sig0
            G = self._incompatible_B(xi, eta, zeta, Fr)
            if G is not None:
                f_a = G.T @ sig0 if f_a is None else f_a + G.T @ sig0
        if f_a is not None:
            self.stiffness_matrix()
            F += self._recovery.T @ f_a
        return F

    def _face_rule(self):
        g, w = np.polynomial.legendre.leggauss(self.IN_PLANE_ORDER + 1)
        return [(float(x), float(e), float(wx * we)) for x, wx in zip(g, w) for e, we in zip(g, w)]

    def pressure_load_vector(self, pressure) -> np.ndarray:
        """Pressure on the top face (zeta = +1), acting along its outward normal (+ stack direction)."""
        F = np.zeros(self.n_dof)
        if not callable(pressure) and pressure == 0.0:
            return F
        for xi, eta, w in self._face_rule():
            N, dN = self.shape_functions(xi, eta, 1.0)
            a = dN[:2] @ self.coordinates
            n = np.cross(a[0], a[1])  # area-weighted normal of the top face
            xyz = N @ self.coordinates
            p = float(pressure(xyz)) if callable(pressure) else float(pressure)
            for i in range(self.N_NODES):
                F[STANDARD_DOFS * i : STANDARD_DOFS * i + 3] += N[i] * p * n * w
        return F

    def body_load_vector(self, acceleration) -> np.ndarray:
        a = np.asarray(acceleration, dtype=float)
        F = np.zeros(self.n_dof)
        for xi, eta, zeta, w, k in self.integration_points():
            N, _ = self.shape_functions(xi, eta, zeta)
            rho = self._laminate.plies[k].material.density
            dV = w * self.jacobian(xi, eta, zeta)[1]
            for i in range(self.N_NODES):
                F[STANDARD_DOFS * i : STANDARD_DOFS * i + 3] += N[i] * rho * a * dV
        return F

    def edge_load_vector(self, edge, force_per_length):  # pragma: no cover - EDGES is empty
        raise ValueError("Line loads are not defined on solid elements")

    # ------------------------------------------------------------- results
    def stress_at(
        self, u_elem: np.ndarray, xi: float, eta: float, zeta: float, state: HygrothermalState = NULL_STATE
    ) -> SolidPoint:
        k = self.ply_at(zeta)
        eps, F = self.strain_at(u_elem, xi, eta, zeta, state)
        mech = eps - self._free_strain_local(k, zeta, state)
        sig = self.ply_stiffness(k) @ mech
        ply = self._laminate.plies[k]
        return SolidPoint(
            self.element_id, k, xi, eta, zeta, self.global_coordinates(xi, eta, zeta), self._z_in_laminate(zeta),
            eps, sig, stress_rotation_3d(ply.angle) @ sig, _voigt_rotation(F.T) @ sig,
        )

    def ply_at(self, zeta: float) -> int:
        bounds = self.ply_zeta_bounds()
        for k, (a, b) in enumerate(bounds):
            if zeta <= b + 1e-12:
                return k
        return len(bounds) - 1

    def ply_midplane_stresses(self, u_elem, xi: float = 0.0, eta: float = 0.0, state=NULL_STATE) -> List[SolidPoint]:
        return [self.stress_at(u_elem, xi, eta, 0.5 * (a + b), state) for a, b in self.ply_zeta_bounds()]

    def through_thickness(self, u_elem, xi: float = 0.0, eta: float = 0.0, points_per_ply: int = 5,
                          state=NULL_STATE) -> List[SolidPoint]:
        out = []
        for a, b in self.ply_zeta_bounds():
            for zeta in np.linspace(a + 1e-9, b - 1e-9, points_per_ply):
                out.append(self.stress_at(u_elem, xi, eta, float(zeta), state))
        return out

    def strain_energy(self, u_elem) -> float:
        u = np.asarray(u_elem, dtype=float)
        return 0.5 * float(u @ self.stiffness_matrix() @ u)

    def quality(self) -> Dict[str, float]:
        dets = [self.jacobian(x, e, z)[1] for x, e, z, _, _ in self.integration_points()]
        return {"aspect_ratio": 1.0, "warping_ratio": 0.0, "jacobian_ratio": float(min(dets) / max(dets))}

    @property
    def characteristic_length(self) -> float:
        return float(np.sqrt(max(self.area, 0.0)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.element_id}, plies={self._laminate.n_plies})"


# ---------------------------------------------------------------------------
_H8 = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


class LayeredHex8Element(LayeredSolidElement):
    """8-node layered brick with nine Wilson-Taylor incompatible modes."""

    TOPOLOGY = "hex8"
    N_NODES = 8
    NATURAL_NODES = _H8
    TOP_FACE = (4, 5, 6, 7)
    BOTTOM_FACE = (0, 1, 2, 3)
    IN_PLANE_ORDER = 2
    PLY_ORDER = 2

    def _shape(self, xi, eta, zeta):
        a, b, c = _H8[:, 0], _H8[:, 1], _H8[:, 2]
        N = 0.125 * (1 + a * xi) * (1 + b * eta) * (1 + c * zeta)
        dN = 0.125 * np.vstack([
            a * (1 + b * eta) * (1 + c * zeta),
            b * (1 + a * xi) * (1 + c * zeta),
            c * (1 + a * xi) * (1 + b * eta),
        ])
        return N, dN

    def _incompatible_B(self, xi, eta, zeta, F):
        J0, det0, _ = self.jacobian(0.0, 0.0, 0.0)
        _, detJ, _ = self.jacobian(xi, eta, zeta)
        dP = np.diag([-2.0 * xi, -2.0 * eta, -2.0 * zeta])  # d(1 - xi_r^2)/dxi_r
        dPdX = np.linalg.solve(J0, dP) * (det0 / detJ)  # (3 global derivatives, 3 modes)
        g = F @ dPdX  # local derivatives of each mode
        G = np.zeros((6, 9))
        for m in range(3):
            for comp in range(3):  # mode m of displacement component comp (global)
                col = 3 * m + comp
                e = np.zeros(3)
                e[comp] = 1.0
                Fe = F @ e
                G[0, col] = Fe[0] * g[0, m]
                G[1, col] = Fe[1] * g[1, m]
                G[2, col] = Fe[2] * g[2, m]
                G[3, col] = Fe[1] * g[2, m] + Fe[2] * g[1, m]
                G[4, col] = Fe[0] * g[2, m] + Fe[2] * g[0, m]
                G[5, col] = Fe[0] * g[1, m] + Fe[1] * g[0, m]
        return G


_H20_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
_H20 = np.vstack([_H8, [0.5 * (_H8[a] + _H8[b]) for a, b in _H20_EDGES]])
_H20_ZERO = [int(np.where(row == 0.0)[0][0]) for row in _H20[8:]]


class LayeredHex20Element(LayeredSolidElement):
    """20-node layered serendipity brick (corners, then edge mid-points in C3D20 / VTK order)."""

    TOPOLOGY = "hex20"
    N_NODES = 20
    NATURAL_NODES = _H20
    TOP_FACE = (4, 5, 6, 7, 12, 13, 14, 15)
    BOTTOM_FACE = (0, 1, 2, 3, 8, 9, 10, 11)
    IN_PLANE_ORDER = 3
    PLY_ORDER = 2

    def _shape(self, xi, eta, zeta):
        p = np.array([xi, eta, zeta])
        n = _H20
        f = 1.0 + n * p  # (20, 3)
        N = np.zeros(20)
        dN = np.zeros((3, 20))
        c = slice(0, 8)
        s = n[c] @ p - 2.0
        N[c] = 0.125 * f[c].prod(axis=1) * s
        for r in range(3):
            others = np.prod(np.delete(f[c], r, axis=1), axis=1)
            dN[r, c] = 0.125 * n[c, r] * others * (s + f[c, r])
        for i in range(8, 20):
            zero = _H20_ZERO[i - 8]
            g = 1.0 - p[zero] ** 2
            a, b = [q for q in range(3) if q != zero]
            N[i] = 0.25 * g * f[i, a] * f[i, b]
            dN[zero, i] = -0.5 * p[zero] * f[i, a] * f[i, b]
            dN[a, i] = 0.25 * g * n[i, a] * f[i, b]
            dN[b, i] = 0.25 * g * n[i, b] * f[i, a]
        return N, dN


class SolidRegistry:
    def __init__(self):
        self._classes: Dict[str, Type[LayeredSolidElement]] = {}

    def register(self, cls: Type[LayeredSolidElement]) -> None:
        self._classes[cls.TOPOLOGY] = cls

    def resolve(self, n_nodes: int, element_type: Optional[str] = None) -> Type[LayeredSolidElement]:
        if element_type is not None:
            return self._classes[element_type]
        for cls in self._classes.values():
            if cls.N_NODES == n_nodes:
                return cls
        raise ValueError(f"No layered solid element has {n_nodes} nodes")


DEFAULT_SOLIDS = SolidRegistry()
DEFAULT_SOLIDS.register(LayeredHex8Element)
DEFAULT_SOLIDS.register(LayeredHex20Element)


# ---------------------------------------------------------------------------
# Structured blocks
# ---------------------------------------------------------------------------
def solid_block(
    mesh,
    laminate: Laminate,
    length_x: float,
    length_y: float,
    nx: int,
    ny: int,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    element: str = "hex8",
    layers: str = "ply",
    elements_per_ply: int = 1,
    material_axis=None,
    first_node: Optional[int] = None,
    first_element: Optional[int] = None,
):
    """Mesh a flat laminate as layered solids and add them to ``mesh``.

    The block spans [x0, x0 + length_x] x [y0, y0 + length_y] in plan and occupies the
    heights z0 + ``laminate.z_interfaces``, so ``origin`` is the shell reference surface:
    a shell with the same laminate at z0 lines up with the block. ``layers="ply"`` uses
    ``elements_per_ply`` elements through each ply; ``layers="laminate"`` uses one
    element through the whole laminate. Returns a ``NamedMesh`` with node sets
    ``x0``, ``x1``, ``y0``, ``y1``, ``bottom``, ``top`` and ``all`` and element sets
    ``all`` and ``ply<k>`` (ply index k of the laminate).
    """
    from .laminate_mechanics import Ply
    from .laminate_mesh_builders import NamedMesh

    if element not in ("hex8", "hex20"):
        raise ValueError("element must be 'hex8' or 'hex20'")
    if layers not in ("ply", "laminate"):
        raise ValueError("layers must be 'ply' or 'laminate'")
    if nx < 1 or ny < 1 or elements_per_ply < 1:
        raise ValueError("element counts must be >= 1")
    x0, y0, z0 = (list(origin) + [0.0, 0.0, 0.0])[:3]
    if layers == "ply":
        stacks, owner = [], []
        for k, ply in enumerate(laminate.plies):
            for _ in range(elements_per_ply):
                stacks.append([Ply(ply.material, ply.thickness / elements_per_ply, ply.angle, ply.name)])
                owner.append(k)
        # offsets keep each element's ply heights equal to their heights in the full laminate
        bottom, placed = laminate.z_interfaces[0], []
        for plies in stacks:
            t = sum(p.thickness for p in plies)
            placed.append(Laminate(plies, laminate.name, offset=bottom + 0.5 * t))
            bottom += t
        stacks = placed
    else:
        stacks, owner = [laminate], [None]
    heights = [z0 + laminate.z_interfaces[0]]
    for stack in stacks:
        heights.append(heights[-1] + stack.thickness)
    quadratic = element == "hex20"
    step = 2 if quadratic else 1
    imax, jmax, kmax = step * nx, step * ny, step * len(stacks)

    def z_of(k):
        if not quadratic:
            return heights[k]
        return heights[k // 2] if k % 2 == 0 else 0.5 * (heights[k // 2] + heights[k // 2 + 1])

    node = (max(mesh.node_ids) + 1 if mesh.node_ids else 1) if first_node is None else int(first_node)
    eid = (max(mesh.element_ids) + 1 if mesh.element_ids else 1) if first_element is None else int(first_element)
    grid: Dict[Tuple[int, int, int], int] = {}
    for k in range(kmax + 1):
        for j in range(jmax + 1):
            for i in range(imax + 1):
                if quadratic and (i % 2) + (j % 2) + (k % 2) > 1:
                    continue
                mesh.add_node(node, (x0 + length_x * i / imax, y0 + length_y * j / jmax, z_of(k)))
                grid[(i, j, k)] = node
                node += 1
    element_sets: Dict[str, List[int]] = {"all": []}
    for k, stack in enumerate(stacks):
        for j in range(ny):
            for i in range(nx):
                a, b, c = step * i, step * j, step * k
                d = step
                corners = [(a, b, c), (a + d, b, c), (a + d, b + d, c), (a, b + d, c),
                           (a, b, c + d), (a + d, b, c + d), (a + d, b + d, c + d), (a, b + d, c + d)]
                conn = [grid[p] for p in corners]
                if quadratic:
                    for p, q in _H20_EDGES:
                        P, Q = corners[p], corners[q]
                        conn.append(grid[tuple((P[r] + Q[r]) // 2 for r in range(3))])
                mesh.add_solid(eid, conn, stack, element, material_axis, gradient_origin=laminate.mid_plane)
                element_sets["all"].append(eid)
                if owner[k] is not None:
                    element_sets.setdefault(f"ply{owner[k]}", []).append(eid)
                eid += 1
    sets = {
        "x0": [n for (i, j, k), n in grid.items() if i == 0],
        "x1": [n for (i, j, k), n in grid.items() if i == imax],
        "y0": [n for (i, j, k), n in grid.items() if j == 0],
        "y1": [n for (i, j, k), n in grid.items() if j == jmax],
        "bottom": [n for (i, j, k), n in grid.items() if k == 0],
        "top": [n for (i, j, k), n in grid.items() if k == kmax],
        "all": list(grid.values()),
    }
    return NamedMesh(mesh, sets, element_sets)


# ---------------------------------------------------------------------------
# Coupling
# ---------------------------------------------------------------------------
def couple_shell_to_solid(mesh, bcs, shell_nodes: Iterable[int], solid_nodes: Iterable[int],
                          max_distance: Optional[float] = None, require_all: bool = True) -> Dict[int, int]:
    """Tie each solid node to the nearest shell node with rigid shell kinematics.

    u_solid = u_shell + theta_shell x (x_solid - x_shell), one constraint per
    translation. Solid nodes that are also shell nodes are skipped (they already share
    DOFs). Returns {solid node: shell node}. ``max_distance`` guards against pairing a
    solid node with a far-away shell node.

    Every shell node on the interface needs a column of solid nodes, otherwise its
    rotations are free and the joint acts as a hinge there: use Quad4 shells with Hex8
    solids and Quad8 shells with Hex20 solids, with matching spacing along the
    interface. ``require_all`` (default) raises when a shell node gets no solid node.
    """
    shell_nodes = [int(n) for n in shell_nodes]
    if not shell_nodes:
        raise ValueError("No shell nodes to couple to")
    S = np.array([mesh.node_coordinates(n) for n in shell_nodes])
    pairs = {}
    for j in (int(n) for n in solid_nodes):
        if j in shell_nodes:
            continue
        xj = mesh.node_coordinates(j)
        d = np.linalg.norm(S - xj, axis=1)
        k = int(np.argmin(d))
        if max_distance is not None and d[k] > max_distance:
            raise ValueError(f"Solid node {j} is {d[k]:.3g} from the nearest shell node (limit {max_distance:.3g})")
        s = shell_nodes[k]
        r = xj - S[k]
        # u_j = u_s + theta x r
        bcs.add_constraint((j, 0), [((s, 0), 1.0), ((s, 4), r[2]), ((s, 5), -r[1])])
        bcs.add_constraint((j, 1), [((s, 1), 1.0), ((s, 5), r[0]), ((s, 3), -r[2])])
        bcs.add_constraint((j, 2), [((s, 2), 1.0), ((s, 3), r[1]), ((s, 4), -r[0])])
        pairs[j] = s
    if require_all:
        lonely = sorted(set(shell_nodes) - set(pairs.values()) - set(pairs))
        if lonely:
            raise ValueError(
                f"Shell nodes {lonely[:8]} have no solid node to couple to; match the meshes along the interface "
                "(Quad4 with Hex8, Quad8 with Hex20) or pass require_all=False"
            )
    return pairs


__all__ = [
    "SolidPoint", "LayeredSolidElement", "LayeredHex8Element", "LayeredHex20Element", "SolidRegistry",
    "DEFAULT_SOLIDS", "couple_shell_to_solid", "solid_block",
]

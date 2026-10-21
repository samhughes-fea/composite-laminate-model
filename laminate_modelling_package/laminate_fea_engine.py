"""Kinematic laminate models and the Quad4 composite shell element.

Kinematic models (Strategy Pattern)
-----------------------------------
Each ``KinematicModel`` turns a ``Laminate`` into the shell section stiffness used
by the element - the 6x6 ABD matrix and the 2x2 transverse shear matrix A_s - and
describes how transverse shear is distributed through the thickness for stress
recovery:

* ``CLPTModel``          Kirchhoff-Love equivalent single layer (shear-rigid, enforced by penalty).
* ``FSDTModel``          Mindlin-Reissner with shear correction k = 5/6.
* ``RefinedZigZagModel`` Refined zig-zag theory: piecewise-linear zig-zag functions whose
                         slopes come from the interface compliance jumps, giving continuous
                         interlaminar shear stress and an energy-consistent A_s.
* ``LayerwiseModel``     Linear Lagrange layerwise theory with every interface an independent
                         kinematic plane; the global thickness matrices are condensed to the
                         shell section by exact projection (membrane/bending) and static
                         condensation (transverse shear).

Element
-------
``CompositePlateElement`` is a flat 4-node shell with 6 DOF per node
[u, v, w, theta_x, theta_y, theta_z] (24x24 stiffness). Membrane and bending use
bilinear B-matrices with 2x2 Gauss quadrature, transverse shear uses the MITC4
assumed covariant strain field (locking-free from thick to thin plates), and the
drilling rotation is stabilised by a small penalty tying theta_z to the in-plane
rotation (1/2)(v,x - u,y), which preserves all six rigid-body modes.

Sign convention: u = u0 + z*theta_y, v = v0 - z*theta_x.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .laminate_hygrothermal import HygrothermalEngine, HygrothermalState
from .laminate_mechanics import Laminate


# ---------------------------------------------------------------------------
# Section stiffness container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionStiffness:
    """Shell section constitutive matrices produced by a kinematic model."""

    abd: np.ndarray
    shear: np.ndarray
    model_name: str

    @property
    def A(self) -> np.ndarray:
        return self.abd[:3, :3]

    @property
    def B(self) -> np.ndarray:
        return self.abd[:3, 3:]

    @property
    def D(self) -> np.ndarray:
        return self.abd[3:, 3:]


def integrate_abd(laminate: Laminate) -> np.ndarray:
    """Classical laminate ABD matrix: A = sum Qb h, B = sum Qb (z_t^2-z_b^2)/2, D = sum Qb (z_t^3-z_b^3)/3."""
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    for ply, (z_b, z_t) in zip(laminate.plies, laminate.ply_bounds()):
        A += ply.Q_bar * (z_t - z_b)
        B += ply.Q_bar * (z_t**2 - z_b**2) / 2.0
        D += ply.Q_bar * (z_t**3 - z_b**3) / 3.0
    return np.block([[A, B], [B, D]])


def integrate_shear(laminate: Laminate) -> np.ndarray:
    """Raw transverse shear stiffness sum Qb_s h (no correction)."""
    return sum(p.Q_bar_s * p.thickness for p in laminate.plies)


def harmonic_shear_stiffness(laminate: Laminate) -> np.ndarray:
    """Series (uniform-stress) shear modulus matrix G = (sum S_k h_k / h)^-1, S_k = Qb_s^-1."""
    h = laminate.thickness
    S_mean = sum(np.linalg.inv(p.Q_bar_s) * p.thickness for p in laminate.plies) / h
    return np.linalg.inv(S_mean)


# ---------------------------------------------------------------------------
# Kinematic model interface
# ---------------------------------------------------------------------------
class KinematicModel(ABC):
    """Strategy interface mapping a laminate to shell section behaviour."""

    name: str = "kinematic-model"

    @abstractmethod
    def abd_matrix(self, laminate: Laminate) -> np.ndarray:
        """6x6 membrane-bending stiffness relating [N; M] to [eps0; kappa]."""

    @abstractmethod
    def transverse_shear_matrix(self, laminate: Laminate) -> np.ndarray:
        """2x2 matrix A_s relating shear forces [Q_yz, Q_xz] to mean shear strains."""

    def section_stiffness(self, laminate: Laminate) -> SectionStiffness:
        return SectionStiffness(self.abd_matrix(laminate), self.transverse_shear_matrix(laminate), self.name)

    def shear_strain_operator(self, laminate: Laminate, z: float, side: str = "above") -> np.ndarray:
        """2x2 matrix M(z) such that the local shear strain is gamma(z) = M(z) gamma0."""
        return np.eye(2)

    def transverse_shear_stress(
        self, laminate: Laminate, z: float, gamma0: np.ndarray, side: str = "above"
    ) -> np.ndarray:
        """Global transverse shear stress [tau_yz, tau_xz] at height z."""
        ply = laminate.plies[laminate.ply_index_at(z, side)]
        return ply.Q_bar_s @ (self.shear_strain_operator(laminate, z, side) @ np.asarray(gamma0, float))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class CLPTModel(KinematicModel):
    """Classical lamination (Kirchhoff) theory.

    The element is a C0 Mindlin element, so the Kirchhoff constraint gamma = 0 is
    imposed with a penalty shear stiffness ``penalty * sum(Qb_s h)``. MITC4 shear
    interpolation keeps this free of locking. Transverse shear stresses are recovered
    from the shear force resultant with the parabolic (equilibrium) profile
    tau(z) = 3/(2h) (1 - 4 z^2/h^2) Q.
    """

    name = "CLPT"

    def __init__(self, penalty: float = 1.0e3):
        if penalty <= 0.0:
            raise ValueError("penalty must be positive")
        self.penalty = float(penalty)

    def abd_matrix(self, laminate: Laminate) -> np.ndarray:
        return integrate_abd(laminate)

    def transverse_shear_matrix(self, laminate: Laminate) -> np.ndarray:
        return self.penalty * integrate_shear(laminate)

    def transverse_shear_stress(self, laminate, z, gamma0, side="above"):
        h = laminate.thickness
        resultant = self.transverse_shear_matrix(laminate) @ np.asarray(gamma0, float)
        return 1.5 / h * (1.0 - 4.0 * z * z / (h * h)) * resultant


class FSDTModel(KinematicModel):
    """First-order shear deformation (Mindlin-Reissner) theory: A_s = k * sum(Qb_s h)."""

    name = "FSDT"

    def __init__(self, shear_correction: float = 5.0 / 6.0):
        if not 0.0 < shear_correction <= 1.0:
            raise ValueError("shear_correction must lie in (0, 1]")
        self.shear_correction = float(shear_correction)

    def abd_matrix(self, laminate: Laminate) -> np.ndarray:
        return integrate_abd(laminate)

    def transverse_shear_matrix(self, laminate: Laminate) -> np.ndarray:
        return self.shear_correction * integrate_shear(laminate)


class RefinedZigZagModel(KinematicModel):
    """Refined zig-zag theory (Tessler / Di Sciuva / Gherlone).

    In-plane displacement: u(z) = u0 + z theta + phi(z) psi, with phi piecewise linear.
    Requiring the transverse shear stress to be continuous across every interface gives
    the layer shear strains gamma_k = S_k G gamma0 with S_k = (Qb_s^k)^-1 and
    G = (sum S_k h_k / h)^-1. The zig-zag slope of ply k is beta_k = S_k G - I, so the
    slope discontinuity at the interface between ply k-1 and ply k is

        psi_k = beta_k - beta_{k-1} = (S_k - S_{k-1}) G = Delta S G.

    phi vanishes on both faces by construction. In the shell element the zig-zag
    amplitudes are slaved to the mean shear strain, which yields A_s = h G; the
    generalised zig-zag stiffnesses are exposed via ``zigzag_stiffness`` for users
    building higher-order elements.
    """

    name = "RZT"

    def abd_matrix(self, laminate: Laminate) -> np.ndarray:
        return integrate_abd(laminate)

    def effective_shear_modulus(self, laminate: Laminate) -> np.ndarray:
        return harmonic_shear_stiffness(laminate)

    def transverse_shear_matrix(self, laminate: Laminate) -> np.ndarray:
        return laminate.thickness * self.effective_shear_modulus(laminate)

    def compliance_jumps(self, laminate: Laminate) -> np.ndarray:
        """(N-1, 2, 2) array of Delta S = Q_curr^-1 - Q_prev^-1 at each internal interface."""
        S = [np.linalg.inv(p.Q_bar_s) for p in laminate.plies]
        return np.array([S[k] - S[k - 1] for k in range(1, len(S))]).reshape(-1, 2, 2)

    def zigzag_slopes(self, laminate: Laminate) -> np.ndarray:
        """(N, 2, 2) zig-zag slope matrices beta_k = S_k G - I."""
        G = self.effective_shear_modulus(laminate)
        return np.array([np.linalg.inv(p.Q_bar_s) @ G - np.eye(2) for p in laminate.plies])

    def slope_discontinuities(self, laminate: Laminate) -> np.ndarray:
        """(N-1, 2, 2) slope jumps psi = Delta S G at the internal interfaces."""
        G = self.effective_shear_modulus(laminate)
        return np.array([dS @ G for dS in self.compliance_jumps(laminate)]).reshape(-1, 2, 2)

    def zigzag_interface_values(self, laminate: Laminate) -> np.ndarray:
        """(N+1, 2, 2) values of phi at the interfaces (zero at both faces)."""
        betas = self.zigzag_slopes(laminate)
        values = [np.zeros((2, 2))]
        for beta, ply in zip(betas, laminate.plies):
            values.append(values[-1] + beta * ply.thickness)
        return np.array(values)

    def zigzag_function(self, laminate: Laminate, z: float, side: str = "above") -> np.ndarray:
        k = laminate.ply_index_at(z, side)
        phi_bottom = self.zigzag_interface_values(laminate)[k]
        return phi_bottom + self.zigzag_slopes(laminate)[k] * (z - laminate.z_interfaces[k])

    def shear_strain_operator(self, laminate, z, side="above"):
        k = laminate.ply_index_at(z, side)
        return np.eye(2) + self.zigzag_slopes(laminate)[k]

    def zigzag_stiffness(self, laminate: Laminate) -> dict:
        """Generalised RZT section stiffnesses for the in-plane zig-zag field.

        The zig-zag displacement is [u_x, u_y] = H(z) psi with psi = [psi_yz, psi_xz]
        and H = P phi(z), P swapping the [yz, xz] rows into [x, y]. With the amplitude
        gradient g = [psi_yz,x, psi_yz,y, psi_xz,x, psi_xz,y], the extra in-plane strain
        is Phi(z) g. Returned: 'B_phi' = int Qb Phi dz, 'D_phi' = int z Qb Phi dz,
        'D_phiphi' = int Phi^T Qb Phi dz and the effective shear modulus 'G'.
        """
        G = self.effective_shear_modulus(laminate)
        betas = self.zigzag_slopes(laminate)
        phis = self.zigzag_interface_values(laminate)
        swap = np.array([[0.0, 1.0], [1.0, 0.0]])
        B_phi = np.zeros((3, 4))
        D_phi = np.zeros((3, 4))
        D_pp = np.zeros((4, 4))
        gauss = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        weights = np.array([5.0, 8.0, 5.0]) / 9.0
        for k, (ply, (z_b, z_t)) in enumerate(zip(laminate.plies, laminate.ply_bounds())):
            half = 0.5 * (z_t - z_b)
            for g, w in zip(gauss, weights):
                z = 0.5 * (z_t + z_b) + half * g
                H = swap @ (phis[k] + betas[k] * (z - z_b))
                Phi = np.array(
                    [
                        [H[0, 0], 0.0, H[0, 1], 0.0],
                        [0.0, H[1, 0], 0.0, H[1, 1]],
                        [H[1, 0], H[0, 0], H[1, 1], H[0, 1]],
                    ]
                )
                B_phi += ply.Q_bar @ Phi * w * half
                D_phi += ply.Q_bar @ Phi * z * w * half
                D_pp += Phi.T @ ply.Q_bar @ Phi * w * half
        return {"B_phi": B_phi, "D_phi": D_phi, "D_phiphi": D_pp, "G": G}


@dataclass(frozen=True)
class LayerwiseThicknessMatrices:
    """Global thickness matrices of the linear layerwise theory.

    ``membrane``: (3(N+1), 3(N+1)) blocks A^{IJ} = sum_k int Qb_k Phi_I Phi_J dz.
    ``shear``:    (2(N+2), 2(N+2)) matrix for unknowns [U_0..U_N, grad w] (2 components each).
    ``projection``: (3(N+1), 6) map from ESL [eps0; kappa] to interface strains.
    """

    membrane: np.ndarray
    shear: np.ndarray
    projection: np.ndarray
    z_interfaces: np.ndarray


class LayerwiseModel(KinematicModel):
    """Linear layerwise (Reddy-type) theory with independent interface planes.

    Each interface I carries its own in-plane displacement U_I(x, y); within ply k,
    u(z) = U_{k-1} Phi_bottom(z) + U_k Phi_top(z). The global thickness matrices
    couple every interface; they are condensed onto the 6-DOF shell section as:

    * ABD = P^T A_LW P, where P imposes the planar (ESL) interface field
      eps_I = eps0 + z_I kappa (exact for linear-in-z strains);
    * A_s by static condensation of the interface sliding modes for a prescribed
      mean shear strain gamma0 = (U_N - U_0)/h + grad w.
    """

    name = "Layerwise"

    def thickness_matrices(self, laminate: Laminate) -> LayerwiseThicknessMatrices:
        n_if = laminate.n_plies + 1
        membrane = np.zeros((3 * n_if, 3 * n_if))
        shear = np.zeros((2 * (n_if + 1), 2 * (n_if + 1)))
        g = slice(2 * n_if, 2 * n_if + 2)  # grad w block
        for k, ply in enumerate(laminate.plies):
            h = ply.thickness
            i, j = k, k + 1
            mi, mj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            membrane[mi, mi] += ply.Q_bar * h / 3.0
            membrane[mj, mj] += ply.Q_bar * h / 3.0
            membrane[mi, mj] += ply.Q_bar * h / 6.0
            membrane[mj, mi] += ply.Q_bar * h / 6.0
            # gamma_k = (U_j - U_i)/h + grad w; energy density integrated over the ply.
            si, sj = slice(2 * i, 2 * i + 2), slice(2 * j, 2 * j + 2)
            Qs = ply.Q_bar_s
            shear[si, si] += Qs / h
            shear[sj, sj] += Qs / h
            shear[si, sj] -= Qs / h
            shear[sj, si] -= Qs / h
            shear[sj, g] += Qs
            shear[g, sj] += Qs
            shear[si, g] -= Qs
            shear[g, si] -= Qs
            shear[g, g] += Qs * h
        projection = np.zeros((3 * n_if, 6))
        for I, z in enumerate(laminate.z_interfaces):
            projection[3 * I : 3 * I + 3, :3] = np.eye(3)
            projection[3 * I : 3 * I + 3, 3:] = z * np.eye(3)
        return LayerwiseThicknessMatrices(membrane, shear, projection, laminate.z_interfaces.copy())

    def abd_matrix(self, laminate: Laminate) -> np.ndarray:
        tm = self.thickness_matrices(laminate)
        return tm.projection.T @ tm.membrane @ tm.projection

    def _condensation(self, laminate: Laminate) -> Tuple[np.ndarray, np.ndarray]:
        """Return (A_s, slope operators) from condensing the layerwise shear field.

        Rigid in-plane translation of all interfaces and the (grad w, rotation) gauge
        carry no shear energy; both are removed by fixing U_0 = 0 and grad w = 0, which
        leaves the interface displacements U_1..U_N with the constraint U_N / h = gamma0.
        """
        tm = self.thickness_matrices(laminate)
        n_if = laminate.n_plies + 1
        keep = np.arange(2, 2 * n_if)  # U_1..U_N
        K = tm.shear[np.ix_(keep, keep)]
        C = np.zeros((2, keep.size))
        C[:, -2:] = np.eye(2) / laminate.thickness
        K_inv_Ct = np.linalg.solve(K, C.T)
        A_s = np.linalg.inv(C @ K_inv_Ct)
        # Optimal interface displacements per unit mean shear: q = K^-1 C^T A_s gamma0.
        q_per_gamma = K_inv_Ct @ A_s  # (2N, 2)
        U = np.vstack([np.zeros((2, 2)), q_per_gamma]).reshape(n_if, 2, 2)
        slopes = np.array(
            [(U[k + 1] - U[k]) / ply.thickness for k, ply in enumerate(laminate.plies)]
        )
        return A_s, slopes

    def transverse_shear_matrix(self, laminate: Laminate) -> np.ndarray:
        A_s, _ = self._condensation(laminate)
        return 0.5 * (A_s + A_s.T)

    def shear_strain_operator(self, laminate, z, side="above"):
        _, slopes = self._condensation(laminate)
        return slopes[laminate.ply_index_at(z, side)]


# ---------------------------------------------------------------------------
# Quad4 shell element
# ---------------------------------------------------------------------------
_GAUSS_POINTS = {
    1: (np.array([0.0]), np.array([2.0])),
    2: (np.array([-1.0, 1.0]) / np.sqrt(3.0), np.array([1.0, 1.0])),
    3: (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5.0, 8.0, 5.0]) / 9.0),
}
_NODE_XI = np.array([-1.0, 1.0, 1.0, -1.0])
_NODE_ETA = np.array([-1.0, -1.0, 1.0, 1.0])


def shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Bilinear shape functions N (4,) and natural derivatives dN (2, 4) = [dN/dxi; dN/deta]."""
    N = 0.25 * (1.0 + _NODE_XI * xi) * (1.0 + _NODE_ETA * eta)
    dN = np.vstack(
        [0.25 * _NODE_XI * (1.0 + _NODE_ETA * eta), 0.25 * _NODE_ETA * (1.0 + _NODE_XI * xi)]
    )
    return N, dN


class CompositePlateElement:
    """Flat 4-node composite shell (membrane + Mindlin plate + drilling), 24 DOF."""

    N_NODES = 4
    DOF_PER_NODE = 6
    N_DOF = 24

    def __init__(
        self,
        element_id: int,
        node_ids: Sequence[int],
        coordinates: np.ndarray,
        laminate: Laminate,
        kinematics: KinematicModel,
        integration_order: int = 2,
        drilling_factor: float = 1.0e-3,
        hygrothermal_engine: Optional[HygrothermalEngine] = None,
    ):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape == (4, 2):
            coordinates = np.hstack([coordinates, np.zeros((4, 1))])
        if coordinates.shape != (4, 3):
            raise ValueError("coordinates must have shape (4, 3) or (4, 2)")
        if len(node_ids) != 4:
            raise ValueError("A Quad4 element needs exactly four node ids")
        if integration_order not in _GAUSS_POINTS:
            raise ValueError("integration_order must be 1, 2 or 3")
        self.element_id = int(element_id)
        self.node_ids = tuple(int(n) for n in node_ids)
        self.coordinates = coordinates
        self.integration_order = integration_order
        self.drilling_factor = float(drilling_factor)
        self._laminate = laminate
        self._kinematics = kinematics
        self._ht_engine = hygrothermal_engine or HygrothermalEngine()
        self._section: Optional[SectionStiffness] = None
        self._build_local_frame()

    # ---------------------------------------------------------------- geometry
    def _build_local_frame(self) -> None:
        X = self.coordinates
        normal = np.cross(X[2] - X[0], X[3] - X[1])
        norm = np.linalg.norm(normal)
        if norm <= 0.0:
            raise ValueError(f"Element {self.element_id} is degenerate (zero area)")
        e3 = normal / norm
        reference = np.array([1.0, 0.0, 0.0])
        e1 = reference - np.dot(reference, e3) * e3
        if np.linalg.norm(e1) < 1e-8:
            edge = X[1] - X[0]
            e1 = edge - np.dot(edge, e3) * e3
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(e3, e1)
        self.frame = np.vstack([e1, e2, e3])  # rows: local axes in global components
        centre = X.mean(axis=0)
        local = (X - centre) @ self.frame.T
        self.local_xy = local[:, :2]
        self.warping = float(np.max(np.abs(local[:, 2])))
        T = np.zeros((24, 24))
        for i in range(8):
            T[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = self.frame
        self.transformation = T
        for xi, eta in self._gauss_points():
            if self.jacobian(xi, eta)[1] <= 0.0:
                raise ValueError(
                    f"Element {self.element_id} has a non-positive Jacobian; check node ordering"
                )

    def _gauss_points(self, order: Optional[int] = None):
        pts, wts = _GAUSS_POINTS[order or self.integration_order]
        for i, xi in enumerate(pts):
            for j, eta in enumerate(pts):
                yield xi, eta

    def _gauss_points_weights(self, order: Optional[int] = None):
        pts, wts = _GAUSS_POINTS[order or self.integration_order]
        for i, xi in enumerate(pts):
            for j, eta in enumerate(pts):
                yield xi, eta, wts[i] * wts[j]

    def jacobian(self, xi: float, eta: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """Return (J, det J, dN/dx) where J = [[x,xi, y,xi], [x,eta, y,eta]]."""
        _, dN = shape_functions(xi, eta)
        J = dN @ self.local_xy
        detJ = float(np.linalg.det(J))
        dNdx = np.linalg.solve(J, dN)
        return J, detJ, dNdx

    @property
    def area(self) -> float:
        return float(sum(self.jacobian(xi, eta)[1] * w for xi, eta, w in self._gauss_points_weights(2)))

    # ------------------------------------------------------- section properties
    @property
    def laminate(self) -> Laminate:
        return self._laminate

    @property
    def kinematics(self) -> KinematicModel:
        return self._kinematics

    @property
    def section(self) -> SectionStiffness:
        if self._section is None:
            self._section = self._kinematics.section_stiffness(self._laminate)
        return self._section

    def with_laminate(self, laminate: Laminate) -> "CompositePlateElement":
        return CompositePlateElement(
            self.element_id, self.node_ids, self.coordinates, laminate, self._kinematics,
            self.integration_order, self.drilling_factor, self._ht_engine,
        )

    def with_kinematics(self, kinematics: KinematicModel) -> "CompositePlateElement":
        return CompositePlateElement(
            self.element_id, self.node_ids, self.coordinates, self._laminate, kinematics,
            self.integration_order, self.drilling_factor, self._ht_engine,
        )

    # ------------------------------------------------------------ B matrices
    @staticmethod
    def membrane_B(dNdx: np.ndarray) -> np.ndarray:
        B = np.zeros((3, 24))
        for i in range(4):
            c = 6 * i
            B[0, c] = dNdx[0, i]
            B[1, c + 1] = dNdx[1, i]
            B[2, c] = dNdx[1, i]
            B[2, c + 1] = dNdx[0, i]
        return B

    @staticmethod
    def bending_B(dNdx: np.ndarray) -> np.ndarray:
        """Curvatures [kx, ky, kxy] with beta_x = theta_y, beta_y = -theta_x."""
        B = np.zeros((3, 24))
        for i in range(4):
            c = 6 * i
            B[0, c + 4] = dNdx[0, i]
            B[1, c + 3] = -dNdx[1, i]
            B[2, c + 4] = dNdx[1, i]
            B[2, c + 3] = -dNdx[0, i]
        return B

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

    def shear_B(self, xi: float, eta: float) -> np.ndarray:
        """MITC4 transverse shear strain-displacement matrix, rows [gamma_yz, gamma_xz]."""
        g_xi = 0.5 * (1.0 - eta) * self._covariant_shear_row(0.0, -1.0, 0) + 0.5 * (1.0 + eta) * self._covariant_shear_row(0.0, 1.0, 0)
        g_eta = 0.5 * (1.0 - xi) * self._covariant_shear_row(-1.0, 0.0, 1) + 0.5 * (1.0 + xi) * self._covariant_shear_row(1.0, 0.0, 1)
        J, _, _ = self.jacobian(xi, eta)
        cartesian = np.linalg.solve(J, np.vstack([g_xi, g_eta]))  # [gamma_xz; gamma_yz]
        return cartesian[[1, 0], :]

    def drilling_B(self, xi: float, eta: float) -> np.ndarray:
        N, _ = shape_functions(xi, eta)
        _, _, dNdx = self.jacobian(xi, eta)
        b = np.zeros(24)
        for i in range(4):
            c = 6 * i
            b[c + 5] = N[i]
            b[c] = 0.5 * dNdx[1, i]
            b[c + 1] = -0.5 * dNdx[0, i]
        return b

    def strain_operators(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(B_m, B_b, B_s) acting on the *global* element displacement vector."""
        _, _, dNdx = self.jacobian(xi, eta)
        T = self.transformation
        return self.membrane_B(dNdx) @ T, self.bending_B(dNdx) @ T, self.shear_B(xi, eta) @ T

    # ------------------------------------------------------------ matrices
    @property
    def drilling_modulus(self) -> float:
        """Penalty modulus (force per length) tying theta_z to the in-plane rotation."""
        A = self.section.A
        return self.drilling_factor * max(A[2, 2], np.max(np.abs(A)) * 1e-6)

    def gauss_points(self) -> list:
        """[(xi, eta, weight * detJ)] for the element's integration rule."""
        return [(xi, eta, w * self.jacobian(xi, eta)[1]) for xi, eta, w in self._gauss_points_weights()]

    def global_coordinates(self, xi: float, eta: float) -> np.ndarray:
        """Global position of the natural point (xi, eta) (bilinear map of the corner nodes)."""
        N, _ = shape_functions(xi, eta)
        return N @ self.coordinates

    def local_stiffness_matrix(self) -> np.ndarray:
        section = self.section
        K = np.zeros((24, 24))
        drill_modulus = self.drilling_modulus
        for xi, eta, w in self._gauss_points_weights():
            _, detJ, dNdx = self.jacobian(xi, eta)
            B_mb = np.vstack([self.membrane_B(dNdx), self.bending_B(dNdx)])
            B_s = self.shear_B(xi, eta)
            b_d = self.drilling_B(xi, eta)
            dA = detJ * w
            K += B_mb.T @ section.abd @ B_mb * dA
            K += B_s.T @ section.shear @ B_s * dA
            K += drill_modulus * np.outer(b_d, b_d) * dA
        return 0.5 * (K + K.T)

    def stiffness_matrix(self) -> np.ndarray:
        """24x24 element stiffness in global DOFs: K = T^T K_local T."""
        T = self.transformation
        return T.T @ self.local_stiffness_matrix() @ T

    def hygrothermal_resultants(self, state: HygrothermalState) -> Tuple[np.ndarray, np.ndarray]:
        return self._ht_engine.resultants(self._laminate, state)

    def hygrothermal_load_vector(self, state: HygrothermalState) -> np.ndarray:
        """24-component environmental load F_ht = int (B_m^T N_ht + B_b^T M_ht) dA (global DOFs)."""
        F = np.zeros(24)
        if state.is_null:
            return F
        N_ht, M_ht = self.hygrothermal_resultants(state)
        for xi, eta, w in self._gauss_points_weights():
            _, detJ, dNdx = self.jacobian(xi, eta)
            F += (self.membrane_B(dNdx).T @ N_ht + self.bending_B(dNdx).T @ M_ht) * detJ * w
        return self.transformation.T @ F

    def pressure_load_vector(self, pressure: float) -> np.ndarray:
        """Consistent load of a uniform pressure acting along the element normal (+e3)."""
        F = np.zeros(24)
        if pressure == 0.0:
            return F
        for xi, eta, w in self._gauss_points_weights():
            N, _ = shape_functions(xi, eta)
            _, detJ, _ = self.jacobian(xi, eta)
            F[2::6] += N * pressure * detJ * w
        return self.transformation.T @ F

    # ------------------------------------------------------- post-processing
    def generalized_strains(
        self, u_elem: np.ndarray, xi: float = 0.0, eta: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mid-plane strains eps0, curvatures kappa and mean shear gamma0 in local axes."""
        u_elem = np.asarray(u_elem, dtype=float)
        B_m, B_b, B_s = self.strain_operators(xi, eta)
        return B_m @ u_elem, B_b @ u_elem, B_s @ u_elem

    def strain_energy(self, u_elem: np.ndarray) -> float:
        u_elem = np.asarray(u_elem, dtype=float)
        return 0.5 * float(u_elem @ self.stiffness_matrix() @ u_elem)

    def __repr__(self) -> str:
        return f"CompositePlateElement(id={self.element_id}, nodes={self.node_ids}, model={self._kinematics.name})"

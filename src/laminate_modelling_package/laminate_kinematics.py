"""Kinematic laminate theories (Strategy Pattern).

A kinematic theory plays two roles, kept as separate interfaces so that callers
depend only on what they use (interface segregation):

``SectionModel``
    Turns a ``Laminate`` into shell section stiffness. Every theory provides the
    condensed equivalent-single-layer (ESL) section - the 6x6 ABD matrix and the
    2x2 transverse shear matrix A_s - which any 6-DOF shell (including the CalculiX
    and Nastran decks) can use. Theories with extra through-thickness freedom also
    provide a ``ThroughThicknessBasis``: in-plane displacement functions H_j(z) whose
    amplitudes a_j(x, y) become extra nodal DOFs of the shell element.

``ThroughThicknessRecovery``
    Describes how transverse shear is distributed through the thickness when stresses
    are recovered from the ESL strains.

Theories
--------
* ``CLPTModel``          Kirchhoff-Love, shear-rigid (penalty).
* ``FSDTModel``          Mindlin-Reissner with shear correction k = 5/6.
* ``RefinedZigZagModel`` Refined zig-zag theory. By default a full RZT element: the two
                         zig-zag amplitudes are independent nodal DOFs (8 DOF per node).
                         ``condensed=True`` slaves them to the mean shear strain instead,
                         which keeps 6 DOF per node and changes only A_s.
* ``LayerwiseModel``     Linear layerwise theory. By default a full layerwise element: the
                         in-plane displacement of every internal interface is an independent
                         field (6 + 2 (n_layers - 1) DOF per node). ``sublayers`` splits each
                         ply into numerical layers for thick plies. ``condensed=True`` gives
                         the 6-DOF section from static condensation.

Generalised section
-------------------
With enrichment amplitudes a_j = [a_jx, a_jy] the displacement field is

    u(z) = u0 + z beta + sum_j H_j(z) a_j,      w(z) = w0,

so the in-plane strain is eps(z) = E(z) e with e = [eps0, kappa, grad a_1, ...] and the
transverse shear strain is gamma(z) = G(z) s with s = [gamma0, a_1, ...]. The element
integrates C_ee = int E^T Qbar E dz and C_ss = int G^T Qbar_s G dz (no shear correction:
the enrichment carries the warping that the correction factor otherwise approximates).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .laminate_hygrothermal import HygrothermalEngine, HygrothermalState
from .laminate_mechanics import Laminate

# [u_x, u_y] <-> shear order [gamma_yz, gamma_xz]
SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])
_GAUSS3 = (np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5.0, 8.0, 5.0]) / 9.0)


# ---------------------------------------------------------------------------
# Section containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionStiffness:
    """Condensed (6-DOF) shell section matrices produced by a kinematic model."""

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


@dataclass(frozen=True)
class ThroughThicknessBasis:
    """Continuous, piecewise-linear in-plane enrichment functions H_j(z).

    ``z_nodes`` (n,) are the heights where the functions are tabulated; they include
    every ply interface. ``values`` (m, n, 2, 2) hold H_j at those heights, mapping the
    amplitude [a_x, a_y] to the displacement [u_x, u_y]. ``signature`` identifies the
    meaning of the amplitudes: elements may share enriched nodal DOFs only when their
    signatures match.
    """

    z_nodes: np.ndarray
    values: np.ndarray
    signature: Tuple
    labels: Tuple[str, ...] = ()

    @property
    def n_functions(self) -> int:
        return int(self.values.shape[0])

    def segment(self, z: float, side: str = "above") -> int:
        tol = 1e-12 * max(abs(self.z_nodes[-1] - self.z_nodes[0]), 1.0)
        if side == "above":
            i = int(np.searchsorted(self.z_nodes, z + tol, side="right")) - 1
        else:
            i = int(np.searchsorted(self.z_nodes, z - tol, side="left")) - 1
        return int(np.clip(i, 0, len(self.z_nodes) - 2))

    def evaluate(self, z: float, side: str = "above") -> Tuple[np.ndarray, np.ndarray]:
        """(H, dH/dz) at z, each of shape (m, 2, 2)."""
        i = self.segment(z, side)
        z0, z1 = self.z_nodes[i], self.z_nodes[i + 1]
        slope = (self.values[:, i + 1] - self.values[:, i]) / (z1 - z0)
        return self.values[:, i] + slope * (z - z0), slope


def membrane_operator(H: np.ndarray) -> np.ndarray:
    """3x4 matrix Phi with eps_inplane = Phi [a_x,x, a_x,y, a_y,x, a_y,y] for u = H a."""
    return np.array(
        [
            [H[0, 0], 0.0, H[0, 1], 0.0],
            [0.0, H[1, 0], 0.0, H[1, 1]],
            [H[1, 0], H[0, 0], H[1, 1], H[0, 1]],
        ]
    )


def strain_operators_at(basis: Optional[ThroughThicknessBasis], z: float, side: str = "above") -> Tuple[np.ndarray, np.ndarray]:
    """E(z) (3 x (6+4m)) and G(z) (2 x (2+2m)) of the generalised section."""
    m = 0 if basis is None else basis.n_functions
    E = np.zeros((3, 6 + 4 * m))
    E[:, :3] = np.eye(3)
    E[:, 3:6] = z * np.eye(3)
    G = np.zeros((2, 2 + 2 * m))
    G[:, :2] = np.eye(2)
    if m:
        H, dH = basis.evaluate(z, side)
        for j in range(m):
            E[:, 6 + 4 * j : 10 + 4 * j] = membrane_operator(H[j])
            G[:, 2 + 2 * j : 4 + 2 * j] = SWAP @ dH[j]
    return E, G


def _segments(laminate: Laminate, basis: Optional[ThroughThicknessBasis]) -> List[Tuple[float, float, int]]:
    """(z_bottom, z_top, ply index) intervals on which every operator is linear in z."""
    z = laminate.z_interfaces if basis is None else basis.z_nodes
    out = []
    for z0, z1 in zip(z[:-1], z[1:]):
        out.append((float(z0), float(z1), laminate.ply_index_at(0.5 * (z0 + z1))))
    return out


@dataclass(frozen=True)
class GeneralizedSection:
    """Section matrices of the (possibly enriched) shell.

    ``C_ee`` relates in-plane resultants to e = [eps0, kappa, grad a_1, ...] and ``C_ss``
    relates shear resultants to s = [gamma0, a_1, ...]. Without enrichment C_ee = ABD
    and C_ss = A_s, so the classical theories are the m = 0 case.
    """

    C_ee: np.ndarray
    C_ss: np.ndarray
    basis: Optional[ThroughThicknessBasis]
    condensed: SectionStiffness

    @property
    def n_functions(self) -> int:
        return 0 if self.basis is None else self.basis.n_functions

    @property
    def n_extra_dofs(self) -> int:
        return 2 * self.n_functions

    # condensed views kept for callers that only need ABD / A_s
    abd = property(lambda self: self.condensed.abd)
    shear = property(lambda self: self.condensed.shear)
    A = property(lambda self: self.condensed.A)
    B = property(lambda self: self.condensed.B)
    D = property(lambda self: self.condensed.D)
    model_name = property(lambda self: self.condensed.model_name)


def generalized_section(model: "SectionModel", laminate: Laminate) -> GeneralizedSection:
    condensed = model.section_stiffness(laminate)
    basis = model.enrichment(laminate)
    if basis is None:
        return GeneralizedSection(condensed.abd, condensed.shear, None, condensed)
    m = basis.n_functions
    C_ee = np.zeros((6 + 4 * m, 6 + 4 * m))
    C_ss = np.zeros((2 + 2 * m, 2 + 2 * m))
    pts, wts = _GAUSS3
    for z0, z1, k in _segments(laminate, basis):
        ply = laminate.plies[k]
        half, mid = 0.5 * (z1 - z0), 0.5 * (z1 + z0)
        for g, w in zip(pts, wts):
            E, G = strain_operators_at(basis, mid + half * g)
            C_ee += E.T @ ply.Q_bar @ E * (w * half)
            C_ss += G.T @ ply.Q_bar_s @ G * (w * half)
    return GeneralizedSection(0.5 * (C_ee + C_ee.T), 0.5 * (C_ss + C_ss.T), basis, condensed)


def generalized_hygrothermal(
    laminate: Laminate, basis: Optional[ThroughThicknessBasis], state: HygrothermalState, engine: HygrothermalEngine
) -> np.ndarray:
    """Environmental resultants int E(z)^T Qbar eps_ht(z) dz, length 6 + 4m."""
    m = 0 if basis is None else basis.n_functions
    out = np.zeros(6 + 4 * m)
    if state.is_null:
        return out
    if basis is None:
        N, M = engine.resultants(laminate, state)
        out[:3], out[3:6] = N, M
        return out
    pts, wts = _GAUSS3
    for z0, z1, k in _segments(laminate, basis):
        ply = laminate.plies[k]
        half, mid = 0.5 * (z1 - z0), 0.5 * (z1 + z0)
        for g, w in zip(pts, wts):
            z = mid + half * g
            E, _ = strain_operators_at(basis, z)
            out += E.T @ ply.Q_bar @ engine.global_free_strain(ply, state, z, laminate.offset) * (w * half)
    return out


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
# Interfaces
# ---------------------------------------------------------------------------
class SectionModel(ABC):
    """Maps a laminate to shell section stiffness."""

    name: str = "section-model"

    @abstractmethod
    def abd_matrix(self, laminate: Laminate) -> np.ndarray:
        """6x6 membrane-bending stiffness relating [N; M] to [eps0; kappa]."""

    @abstractmethod
    def transverse_shear_matrix(self, laminate: Laminate) -> np.ndarray:
        """2x2 matrix A_s relating shear forces [Q_yz, Q_xz] to mean shear strains."""

    def section_stiffness(self, laminate: Laminate) -> SectionStiffness:
        return SectionStiffness(self.abd_matrix(laminate), self.transverse_shear_matrix(laminate), self.name)

    def enrichment(self, laminate: Laminate) -> Optional[ThroughThicknessBasis]:
        """Through-thickness enrichment carried as extra nodal DOFs (None for 6-DOF theories)."""
        return None

    def generalized_section(self, laminate: Laminate) -> GeneralizedSection:
        return generalized_section(self, laminate)


class ThroughThicknessRecovery(ABC):
    """Distribution of transverse shear through the thickness for stress recovery."""

    def shear_strain_operator(self, laminate: Laminate, z: float, side: str = "above") -> np.ndarray:
        """2x2 matrix M(z) such that the local shear strain is gamma(z) = M(z) gamma0."""
        return np.eye(2)

    def transverse_shear_stress(
        self, laminate: Laminate, z: float, gamma0: np.ndarray, side: str = "above"
    ) -> np.ndarray:
        """Global transverse shear stress [tau_yz, tau_xz] at height z from the mean shear strain."""
        ply = laminate.plies[laminate.ply_index_at(z, side)]
        return ply.Q_bar_s @ (self.shear_strain_operator(laminate, z, side) @ np.asarray(gamma0, float))


class KinematicModel(SectionModel, ThroughThicknessRecovery):
    """A complete kinematic theory: section stiffness plus shear recovery."""

    name: str = "kinematic-model"

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# Theories
# ---------------------------------------------------------------------------
class CLPTModel(KinematicModel):
    """Classical lamination (Kirchhoff) theory.

    The element is a C0 Mindlin element, so the Kirchhoff constraint gamma = 0 is
    imposed with a penalty shear stiffness ``penalty * sum(Qb_s h)``. Assumed-strain
    shear interpolation keeps this free of locking. Transverse shear stresses are
    recovered from the shear force resultant with the parabolic (equilibrium) profile
    tau(z) = 3/(2h) (1 - 4 zeta^2/h^2) Q, zeta measured from the mid-plane. This is a
    different contract from the strain-based recovery of the other theories: the
    profile comes from the resultant, not from Qbar_s gamma(z).
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
        zeta = z - laminate.offset
        resultant = self.transverse_shear_matrix(laminate) @ np.asarray(gamma0, float)
        return 1.5 / h * (1.0 - 4.0 * zeta * zeta / (h * h)) * resultant


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

    phi vanishes on both faces by construction.

    By default the zig-zag amplitudes are independent nodal DOFs (a full RZT element,
    8 DOF per node), with gamma(z) = gamma0 + beta_k psi. ``condensed=True`` slaves
    psi = gamma0 instead, which keeps a 6-DOF shell with A_s = h G.
    Laminates whose plies all share one shear stiffness have no zig-zag (beta = 0) and
    fall back to 6 DOF per node.
    """

    name = "RZT"

    def __init__(self, condensed: bool = False):
        self.condensed = bool(condensed)

    def __repr__(self) -> str:
        return f"RefinedZigZagModel(condensed={self.condensed})"

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

    def enrichment(self, laminate: Laminate) -> Optional[ThroughThicknessBasis]:
        if self.condensed:
            return None
        betas = self.zigzag_slopes(laminate)
        if np.max(np.abs(betas)) < 1e-10:
            return None  # homogeneous shear stiffness: no zig-zag
        # amplitude a = SWAP psi drives [u_x, u_y]; H = SWAP phi SWAP so that SWAP H' = beta SWAP
        phi = self.zigzag_interface_values(laminate)
        values = np.array([[SWAP @ p @ SWAP for p in phi]])
        return ThroughThicknessBasis(laminate.z_interfaces.copy(), values, ("RZT",), ("zigzag",))

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
        B_phi = np.zeros((3, 4))
        D_phi = np.zeros((3, 4))
        D_pp = np.zeros((4, 4))
        gauss, weights = _GAUSS3
        for k, (ply, (z_b, z_t)) in enumerate(zip(laminate.plies, laminate.ply_bounds())):
            half = 0.5 * (z_t - z_b)
            for g, w in zip(gauss, weights):
                z = 0.5 * (z_t + z_b) + half * g
                Phi = membrane_operator(SWAP @ (phis[k] + betas[k] * (z - z_b)))
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
    """Linear (Reddy-type) layerwise theory with independent interface planes.

    Each interface I carries its own in-plane displacement U_I(x, y); within a layer,
    u(z) = U_bottom Phi_bottom(z) + U_top Phi_top(z), and w is constant through the
    thickness. The element writes this as the ESL field u0 + z beta plus hat functions
    at the internal interfaces, so the full layerwise space is spanned with 6 standard
    DOFs and 2 (n_layers - 1) enrichment DOFs per node. ``sublayers`` splits each ply
    into that many numerical layers.

    ``condensed=True`` instead reduces the theory to the 6-DOF section:

    * ABD = P^T A_LW P, where P imposes the planar (ESL) interface field
      eps_I = eps0 + z_I kappa (exact for linear-in-z strains);
    * A_s by static condensation of the interface sliding modes for a prescribed
      mean shear strain gamma0 = (U_N - U_0)/h + grad w.
    """

    name = "Layerwise"

    def __init__(self, condensed: bool = False, sublayers: int = 1):
        if int(sublayers) < 1:
            raise ValueError("sublayers must be >= 1")
        self.condensed = bool(condensed)
        self.sublayers = int(sublayers)

    def __repr__(self) -> str:
        return f"LayerwiseModel(condensed={self.condensed}, sublayers={self.sublayers})"

    def layer_heights(self, laminate: Laminate) -> np.ndarray:
        """Numerical layer interfaces (every ply split into ``sublayers`` layers)."""
        z = [laminate.z_interfaces[0]]
        for z_b, z_t in laminate.ply_bounds():
            z.extend(np.linspace(z_b, z_t, self.sublayers + 1)[1:])
        return np.array(z, dtype=float)

    def enrichment(self, laminate: Laminate) -> Optional[ThroughThicknessBasis]:
        if self.condensed:
            return None
        z = self.layer_heights(laminate)
        n_internal = len(z) - 2
        if n_internal < 1:
            return None  # a single layer is already spanned by u0 + z beta
        values = np.zeros((n_internal, len(z), 2, 2))
        for j in range(n_internal):
            values[j, j + 1] = np.eye(2)
        labels = tuple(f"interface_{j + 1}" for j in range(n_internal))
        return ThroughThicknessBasis(z, values, ("Layerwise", len(z)), labels)

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

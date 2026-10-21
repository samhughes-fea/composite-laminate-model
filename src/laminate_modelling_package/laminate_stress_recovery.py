"""Exact through-thickness stress recovery for the composite Quad4 element.

For a sampling point (xi, eta, z) the recoverer evaluates

    eps_total(z) = B_m U + z B_b U                      (global B-matrices, local shell axes)
    eps_mech(z)  = eps_total(z) - eps_HT(z)             (remove free hygrothermal expansion)
    sigma_xy     = Qbar_k eps_mech(z)
    sigma_12     = T_k sigma_xy                         (back to fibre axes)

and the transverse shear stresses from B_s U through the element's kinematic model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .laminate_fea_engine import CompositePlateElement
from .laminate_hygrothermal import NULL_STATE, HygrothermalEngine, HygrothermalState


@dataclass(frozen=True)
class PointStressState:
    """Strain and stress state at one through-thickness sampling point."""

    element_id: int
    ply_index: int
    z: float
    xi: float
    eta: float
    strain_total: np.ndarray  # [ex, ey, gxy] local shell axes
    strain_hygrothermal: np.ndarray
    strain_mechanical: np.ndarray
    stress_global: np.ndarray  # [sx, sy, txy]
    stress_material: np.ndarray  # [s1, s2, t12]
    shear_global: np.ndarray  # [t_yz, t_xz]
    shear_material: np.ndarray  # [t23, t13]

    def material_stress_3d(self) -> np.ndarray:
        """Material-axis stress in Voigt order [s1, s2, s3, t23, t13, t12] (s3 = 0 in a shell)."""
        s1, s2, t12 = self.stress_material
        t23, t13 = self.shear_material
        return np.array([s1, s2, 0.0, t23, t13, t12])

    def global_stress_3d(self) -> np.ndarray:
        """Global stress in Voigt order [sx, sy, sz, t_yz, t_xz, t_xy]."""
        sx, sy, txy = self.stress_global
        tyz, txz = self.shear_global
        return np.array([sx, sy, 0.0, tyz, txz, txy])


class ExactStressRecoverer:
    """Recovers ply stresses from element displacement vectors."""

    def __init__(self, hygrothermal_engine: Optional[HygrothermalEngine] = None):
        self.hygrothermal_engine = hygrothermal_engine or HygrothermalEngine()

    def recover(
        self,
        element: CompositePlateElement,
        u_elem: np.ndarray,
        z: float,
        xi: float = 0.0,
        eta: float = 0.0,
        state: HygrothermalState = NULL_STATE,
        side: str = "above",
    ) -> PointStressState:
        laminate = element.laminate
        k = laminate.ply_index_at(z, side)
        ply = laminate.plies[k]
        eps0, kappa, gamma0 = element.generalized_strains(u_elem, xi, eta)
        eps_total = eps0 + z * kappa
        eps_ht = self.hygrothermal_engine.global_free_strain(ply, state, z)
        eps_mech = eps_total - eps_ht
        sigma = ply.Q_bar @ eps_mech
        tau = element.kinematics.transverse_shear_stress(laminate, z, gamma0, side)
        return PointStressState(
            element_id=element.element_id,
            ply_index=k,
            z=float(z),
            xi=float(xi),
            eta=float(eta),
            strain_total=eps_total,
            strain_hygrothermal=eps_ht,
            strain_mechanical=eps_mech,
            stress_global=sigma,
            stress_material=ply.to_material_stress(sigma),
            shear_global=tau,
            shear_material=ply.to_material_shear(tau),
        )

    def ply_sampling_points(self, element: CompositePlateElement, points_per_ply: int = 3) -> List[tuple]:
        """(z, side) pairs: bottom, interior and top of every ply (interfaces evaluated in both plies)."""
        if points_per_ply < 1:
            raise ValueError("points_per_ply must be >= 1")
        samples = []
        for z_b, z_t in element.laminate.ply_bounds():
            if points_per_ply == 1:
                samples.append((0.5 * (z_b + z_t), "above"))
                continue
            for i, z in enumerate(np.linspace(z_b, z_t, points_per_ply)):
                side = "above" if i < points_per_ply - 1 else "below"
                samples.append((float(z), side))
        return samples

    def through_thickness_profile(
        self,
        element: CompositePlateElement,
        u_elem: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
        state: HygrothermalState = NULL_STATE,
        points_per_ply: int = 3,
    ) -> List[PointStressState]:
        return [
            self.recover(element, u_elem, z, xi, eta, state, side)
            for z, side in self.ply_sampling_points(element, points_per_ply)
        ]

    def at_heights(
        self,
        element: CompositePlateElement,
        u_elem: np.ndarray,
        z_values: Sequence[float],
        xi: float = 0.0,
        eta: float = 0.0,
        state: HygrothermalState = NULL_STATE,
    ) -> List[PointStressState]:
        return [self.recover(element, u_elem, float(z), xi, eta, state) for z in z_values]

    def ply_midplane_states(
        self,
        element: CompositePlateElement,
        u_elem: np.ndarray,
        xi: float = 0.0,
        eta: float = 0.0,
        state: HygrothermalState = NULL_STATE,
    ) -> List[PointStressState]:
        return self.at_heights(element, u_elem, element.laminate.ply_midplanes(), xi, eta, state)

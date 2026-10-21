"""Hygrothermal (thermal + moisture) expansion properties and load resultants.

The free expansion strain of a ply in material axes is

    eps_ht = alpha * dT(z) + beta * dC(z)

where alpha / beta are the coefficients of thermal / moisture expansion and dT(z),
dC(z) are the temperature and moisture changes, allowed to vary linearly through
the laminate thickness. The engine rotates these strains into global axes and
integrates the laminate force / moment resultants {N_ht}, {M_ht} exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from .laminate_materials_3d import strain_rotation_3d

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .laminate_mechanics import Laminate, Ply


@dataclass(frozen=True)
class HygrothermalProperties:
    """Coefficients of thermal (alpha) and moisture (beta) expansion in material axes.

    ``alpha3`` / ``beta3`` default to the transverse values (transverse isotropy).
    ``reference_temperature`` is the stress-free temperature, recorded for decks
    written to external solvers; analyses here always work with temperature changes.
    """

    alpha1: float
    alpha2: float
    alpha3: Optional[float] = None
    beta1: float = 0.0
    beta2: float = 0.0
    beta3: Optional[float] = None
    reference_temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.alpha3 is None:
            object.__setattr__(self, "alpha3", float(self.alpha2))
        if self.beta3 is None:
            object.__setattr__(self, "beta3", float(self.beta2))
        for name in ("alpha1", "alpha2", "alpha3", "beta1", "beta2", "beta3"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    def alpha_vector(self) -> np.ndarray:
        """3D thermal expansion vector in Voigt order [11, 22, 33, 23, 13, 12]."""
        return np.array([self.alpha1, self.alpha2, self.alpha3, 0.0, 0.0, 0.0])

    def beta_vector(self) -> np.ndarray:
        """3D moisture expansion vector in Voigt order [11, 22, 33, 23, 13, 12]."""
        return np.array([self.beta1, self.beta2, self.beta3, 0.0, 0.0, 0.0])


ZERO_EXPANSION = HygrothermalProperties(alpha1=0.0, alpha2=0.0, alpha3=0.0)


@dataclass(frozen=True)
class HygrothermalState:
    """Environmental change relative to the stress-free state.

    The fields at height z (measured from the laminate mid-plane) are
    ``dT(z) = delta_temperature + temperature_gradient * z`` and
    ``dC(z) = delta_moisture + moisture_gradient * z``.
    """

    delta_temperature: float = 0.0
    delta_moisture: float = 0.0
    temperature_gradient: float = 0.0
    moisture_gradient: float = 0.0

    @property
    def is_null(self) -> bool:
        return (
            self.delta_temperature == 0.0
            and self.delta_moisture == 0.0
            and self.temperature_gradient == 0.0
            and self.moisture_gradient == 0.0
        )

    @property
    def is_uniform(self) -> bool:
        return self.temperature_gradient == 0.0 and self.moisture_gradient == 0.0

    def temperature_at(self, z: float) -> float:
        return self.delta_temperature + self.temperature_gradient * z

    def moisture_at(self, z: float) -> float:
        return self.delta_moisture + self.moisture_gradient * z

    def scaled(self, factor: float) -> "HygrothermalState":
        return HygrothermalState(
            self.delta_temperature * factor,
            self.delta_moisture * factor,
            self.temperature_gradient * factor,
            self.moisture_gradient * factor,
        )


NULL_STATE = HygrothermalState()

# Index map from 3D Voigt [11,22,33,23,13,12] to plane [11,22,12].
_PLANE_INDICES = (0, 1, 5)


class HygrothermalEngine:
    """Transforms ply expansion vectors into global strains and laminate resultants."""

    @staticmethod
    def properties_of(ply: "Ply") -> HygrothermalProperties:
        props = ply.material.hygrothermal
        return props if props is not None else ZERO_EXPANSION

    # ---------------------------------------------------------- free strains
    @staticmethod
    def material_free_strain(
        props: HygrothermalProperties, delta_temperature: float, delta_moisture: float
    ) -> np.ndarray:
        """3D free expansion strain in material axes (engineering shear, Voigt)."""
        return props.alpha_vector() * delta_temperature + props.beta_vector() * delta_moisture

    def global_free_strain_3d(
        self, ply: "Ply", state: HygrothermalState, z: float
    ) -> np.ndarray:
        """3D free strain at height z in global axes: eps_x = T_eps^-1 eps_1."""
        props = self.properties_of(ply)
        eps_material = self.material_free_strain(props, state.temperature_at(z), state.moisture_at(z))
        return np.linalg.solve(strain_rotation_3d(ply.angle), eps_material)

    def global_free_strain(self, ply: "Ply", state: HygrothermalState, z: float) -> np.ndarray:
        """In-plane free strain [eps_x, eps_y, gamma_xy] at height z in global axes."""
        return self.global_free_strain_3d(ply, state, z)[list(_PLANE_INDICES)]

    def global_expansion_coefficients(self, ply: "Ply") -> Tuple[np.ndarray, np.ndarray]:
        """Global in-plane (alpha_x, alpha_y, alpha_xy) and (beta_x, beta_y, beta_xy)."""
        props = self.properties_of(ply)
        T_inv = np.linalg.inv(strain_rotation_3d(ply.angle))
        alpha = (T_inv @ props.alpha_vector())[list(_PLANE_INDICES)]
        beta = (T_inv @ props.beta_vector())[list(_PLANE_INDICES)]
        return alpha, beta

    # ------------------------------------------------------------ resultants
    def resultants(
        self, laminate: "Laminate", state: HygrothermalState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact environmental force and moment resultants {N_ht}, {M_ht}.

        With eps_ht(z) = e0 + e1 z inside each ply (linear fields),
            N = sum_k Qbar_k (e0 h1 + e1 h2),  M = sum_k Qbar_k (e0 h2 + e1 h3)
        where h1 = z_t - z_b, h2 = (z_t^2 - z_b^2)/2, h3 = (z_t^3 - z_b^3)/3.
        """
        N = np.zeros(3)
        M = np.zeros(3)
        if state.is_null:
            return N, M
        for ply, (z_b, z_t) in zip(laminate.plies, laminate.ply_bounds()):
            alpha, beta = self.global_expansion_coefficients(ply)
            e0 = alpha * state.delta_temperature + beta * state.delta_moisture
            e1 = alpha * state.temperature_gradient + beta * state.moisture_gradient
            h1 = z_t - z_b
            h2 = 0.5 * (z_t**2 - z_b**2)
            h3 = (z_t**3 - z_b**3) / 3.0
            N += ply.Q_bar @ (e0 * h1 + e1 * h2)
            M += ply.Q_bar @ (e0 * h2 + e1 * h3)
        return N, M

    def resultant_vector(self, laminate: "Laminate", state: HygrothermalState) -> np.ndarray:
        """Stacked 6-vector [N_ht; M_ht]."""
        N, M = self.resultants(laminate, state)
        return np.concatenate([N, M])

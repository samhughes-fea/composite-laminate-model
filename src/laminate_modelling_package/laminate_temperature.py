"""Temperature-dependent lamina properties.

Composite properties fall with temperature, the matrix-dominated ones (E2, G12,
Yc, S12) fastest. A ``TemperatureDependentMaterial`` holds material cards measured
at several temperatures and interpolates every elastic, strength and expansion
constant linearly between them. The analysis stays linear: properties are taken at
the operating temperature of each element and held fixed for the solve.

    table = TemperatureDependentMaterial({20.0: rt_card, 120.0: hot_card})
    card = table.at_temperature(80.0)   # MaterialCard named 'IM7 [T=80]'
    hot_laminate = laminate_at_temperature(lam, {"IM7": table}, 80.0)
    apply_operating_temperature(mesh, load_case, {"IM7": table}, reference_temperature=20.0)

Temperatures are absolute operating temperatures in whatever unit the table uses;
the hygrothermal load still works with changes from the stress-free state.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Mapping, Optional

import numpy as np

from .laminate_hygrothermal import HygrothermalProperties
from .laminate_materials import MaterialCard
from .laminate_materials_3d import ElasticProperties3D, StrengthProperties3D
from .laminate_mechanics import Laminate, Ply


def _interpolate(cls, values, weights):
    data = {}
    for f in fields(cls):
        column = [getattr(v, f.name) for v in values]
        if f.name == "reference_temperature":
            data[f.name] = column[0]
        else:
            data[f.name] = float(np.dot(weights, column))
    return cls(**data)


class TemperatureDependentMaterial:
    """Material cards at several temperatures with linear interpolation between them.

    ``extrapolate=False`` (the default) raises outside the tabulated range instead of
    guessing; with True the end cards are held constant.
    """

    def __init__(self, cards: Mapping[float, MaterialCard], name: Optional[str] = None, extrapolate: bool = False):
        if len(cards) < 1:
            raise ValueError("Give at least one temperature and card")
        self.temperatures = np.array(sorted(float(t) for t in cards))
        self.cards = [cards[t] for t in sorted(cards)]
        self.name = name or self.cards[0].name
        self.extrapolate = bool(extrapolate)
        hyg = [c.hygrothermal is None for c in self.cards]
        if any(hyg) and not all(hyg):
            raise ValueError("Either every card or no card must have hygrothermal properties")

    @property
    def temperature_range(self):
        return float(self.temperatures[0]), float(self.temperatures[-1])

    def _weights(self, temperature: float) -> np.ndarray:
        T = float(temperature)
        lo, hi = self.temperature_range
        if not lo - 1e-12 <= T <= hi + 1e-12:
            if not self.extrapolate:
                raise ValueError(f"{self.name}: temperature {T:g} is outside the tabulated range [{lo:g}, {hi:g}]")
            T = min(max(T, lo), hi)
        w = np.zeros(len(self.temperatures))
        if len(w) == 1:
            w[0] = 1.0
            return w
        i = int(np.clip(np.searchsorted(self.temperatures, T) - 1, 0, len(w) - 2))
        t0, t1 = self.temperatures[i], self.temperatures[i + 1]
        s = (T - t0) / (t1 - t0)
        w[i], w[i + 1] = 1.0 - s, s
        return w

    def at_temperature(self, temperature: float) -> MaterialCard:
        w = self._weights(temperature)
        elastic = _interpolate(ElasticProperties3D, [c.elastic for c in self.cards], w)
        strength = _interpolate(StrengthProperties3D, [c.strength for c in self.cards], w)
        hyg = None
        if self.cards[0].hygrothermal is not None:
            hyg = _interpolate(HygrothermalProperties, [c.hygrothermal for c in self.cards], w)
        density = float(np.dot(w, [c.density for c in self.cards]))
        return replace(
            self.cards[0],
            name=f"{self.name} [T={float(temperature):g}]",
            elastic=elastic,
            strength=strength,
            hygrothermal=hyg,
            density=density,
        )

    def __repr__(self) -> str:
        lo, hi = self.temperature_range
        return f"TemperatureDependentMaterial({self.name!r}, {len(self.cards)} cards, {lo:g}..{hi:g})"


def _base_name(card: MaterialCard) -> str:
    return card.name.split(" [T=", 1)[0]


def laminate_at_temperature(
    laminate: Laminate, tables: Mapping[str, TemperatureDependentMaterial], temperature: float
) -> Laminate:
    """Replace each ply whose material name is in ``tables`` by the card at ``temperature``.

    Plies with other materials are kept. Names are matched without the `` [T=...]`` suffix,
    so a laminate already evaluated at one temperature can be re-evaluated.
    """
    plies = []
    for ply in laminate.plies:
        table = tables.get(_base_name(ply.material))
        if table is None:
            plies.append(ply)
        else:
            plies.append(Ply(table.at_temperature(temperature), ply.thickness, ply.angle))
    return laminate.with_plies(plies)


def apply_operating_temperature(
    mesh, load_case, tables: Mapping[str, TemperatureDependentMaterial], reference_temperature: float
) -> None:
    """Set each element's properties at its mid-plane operating temperature.

    The operating temperature of an element is ``reference_temperature`` plus the
    mean temperature change of its hygrothermal state in ``load_case``. Through-
    thickness gradients use the mean value (one property set per element).
    """
    for element in mesh.elements:
        T = reference_temperature + load_case.state_for(element.element_id).delta_temperature
        mesh.replace_element(element.with_laminate(laminate_at_temperature(element.laminate, tables, T)))

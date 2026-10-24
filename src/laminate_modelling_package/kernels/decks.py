"""Helpers shared by the external solver deck writers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..laminate_hygrothermal import HygrothermalState
from ..laminate_materials import MaterialCard
from ..laminate_mechanics import Laminate
from ..laminate_solver_pipeline import FEAMesh, GlobalAssembler, LoadCase
from .base import UnsupportedFeatureError


def equivalent_thermal_load(load_case: LoadCase, mesh: FEAMesh) -> Tuple[float, Optional[HygrothermalState]]:
    """Return (temperature load, state) for decks that only support uniform thermal loads.

    Moisture is folded into an equivalent expansion coefficient
    alpha_eq = alpha + beta dC / dT (or alpha_eq = beta dC with a unit temperature
    load when dT = 0), which reproduces the free strain exactly for uniform states.
    """
    states = {load_case.state_for(e) for e in mesh.element_ids}
    states.discard(HygrothermalState())
    if not states:
        return 0.0, None
    if len(states) > 1:
        raise UnsupportedFeatureError("This deck supports a single uniform hygrothermal state for all elements")
    state = states.pop()
    if not state.is_uniform:
        raise UnsupportedFeatureError("This deck does not support through-thickness temperature/moisture gradients")
    if state.delta_temperature != 0.0:
        return state.delta_temperature, state
    return 1.0, state


def element_temperature_field(load_case: LoadCase, mesh: FEAMesh) -> Optional[Dict[int, Tuple[float, float]]]:
    """Per-element (mean temperature change, gradient) for decks with element temperatures.

    Returns None without hygrothermal loads. Moisture cannot be represented element by
    element (it would need a different equivalent expansion coefficient per element),
    so moisture with varying states raises ``UnsupportedFeatureError``.
    """
    states = {e: load_case.state_for(e) for e in mesh.element_ids}
    if all(s.is_null for s in states.values()):
        return None
    if any(s.delta_moisture != 0.0 or s.moisture_gradient != 0.0 for s in states.values()):
        raise UnsupportedFeatureError("Element-wise temperature fields cannot carry moisture; use one uniform state")
    return {e: (s.delta_temperature, s.temperature_gradient) for e, s in states.items()}


def equivalent_alphas(card: MaterialCard, state: Optional[HygrothermalState]) -> Tuple[float, float, float]:
    props = card.hygrothermal
    if props is None or state is None:
        return 0.0, 0.0, 0.0
    alpha = np.array([props.alpha1, props.alpha2, props.alpha3])
    beta = np.array([props.beta1, props.beta2, props.beta3])
    if state.delta_temperature != 0.0:
        eq = alpha + beta * state.delta_moisture / state.delta_temperature
    else:
        eq = beta * state.delta_moisture
    return float(eq[0]), float(eq[1]), float(eq[2])


def material_key(card: MaterialCard) -> tuple:
    hyg = card.hygrothermal
    hyg_key = None if hyg is None else (hyg.alpha1, hyg.alpha2, hyg.alpha3, hyg.beta1, hyg.beta2, hyg.beta3)
    return (card.name, tuple(card.elastic.as_dict().values()), tuple(card.strength.as_dict().values()), hyg_key, card.density)


def laminate_key(laminate: Laminate) -> tuple:
    return (tuple((material_key(p.material), p.thickness, p.angle) for p in laminate.plies), laminate.offset)


def axis_key(element) -> tuple:
    """Rounded in-plane material axes (e1, e2) of an element."""
    return tuple(np.round(element.frame[:2].ravel(), 9) + 0.0)


class DeckCatalogue:
    """Deduplicates materials, orientations and laminates across a mesh.

    ``groups`` maps (laminate, material axes) to element ids, so elements with the same
    lay-up but different fibre directions get their own section and orientations.
    """

    def __init__(self, mesh: FEAMesh):
        self.materials: Dict[tuple, Tuple[str, MaterialCard]] = {}
        self.laminates: Dict[tuple, Tuple[str, Laminate, List[int]]] = {}
        self.groups: Dict[tuple, Tuple[str, Laminate, tuple, List[int]]] = {}
        self.orientations: Dict[Tuple[tuple, float], str] = {}
        for element in mesh.elements:
            key = laminate_key(element.laminate)
            if key not in self.laminates:
                self.laminates[key] = (f"LAM{len(self.laminates) + 1}", element.laminate, [])
            self.laminates[key][2].append(element.element_id)
            axes = axis_key(element)
            gkey = (key, axes)
            if gkey not in self.groups:
                self.groups[gkey] = (f"SEC{len(self.groups) + 1}", element.laminate, axes, [])
            self.groups[gkey][3].append(element.element_id)
            for ply in element.laminate.plies:
                mkey = material_key(ply.material)
                if mkey not in self.materials:
                    self.materials[mkey] = (f"M{len(self.materials) + 1}", ply.material)
                okey = (axes, ply.angle)
                if okey not in self.orientations:
                    self.orientations[okey] = f"OR{len(self.orientations) + 1}"

    def material_name(self, card: MaterialCard) -> str:
        return self.materials[material_key(card)][0]


def equivalent_nodal_forces(mesh: FEAMesh, load_case: LoadCase, body_loads: bool = True) -> Dict[int, np.ndarray]:
    """Nodal forces and moments from nodal, line and body loads (consistent element vectors).

    Pressures and hygrothermal states are written with the deck's own cards instead.
    ``body_loads=False`` leaves the body loads out, for decks that write them as
    accelerations (CalculiX ``GRAV``).
    """
    forces: Dict[int, np.ndarray] = {}
    for (node, dof), value in load_case.nodal_loads.items():
        forces.setdefault(node, np.zeros(6))[dof] += value
    vector = np.zeros(mesh.n_dof)
    assembler = GlobalAssembler(sparse=False)
    if load_case.line_loads:
        vector += assembler.line_load_vector(mesh, load_case)
    for body in load_case.body_loads if body_loads else ():
        for element in mesh.elements:
            if body.applies_to(element.element_id):
                vector[mesh.element_dofs(element)] += element.body_load_vector(body.acceleration)
    if np.any(vector):
        std = mesh.standard_array(vector)
        for node, row in zip(mesh.node_ids, std):
            if np.any(row):
                forces.setdefault(node, np.zeros(6))[:] += row
    return forces

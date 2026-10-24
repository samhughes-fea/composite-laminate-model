"""Ply-based layup definition: plies as zones on the mesh.

Real composite parts are specified ply by ply: each ply is a piece of material with
a fibre direction that covers part of the structure. Where plies stop, the laminate
gets thinner (ply drops, tapers), and where the fabric is draped over a curved tool
the fibre angle drifts from its nominal value. A ``Layup`` holds that ply book and
builds the laminate of every element from it:

    layup = Layup([
        PlyDefinition("skin_0", card, 0.25e-3, 0),
        PlyDefinition("pad_45", card, 0.25e-3, 45, elements="pad"),          # element set
        PlyDefinition("pad_m45", card, 0.25e-3, -45, region=lambda x: x[0] < 0.2),
        PlyDefinition("skin_90", card, 0.25e-3, 90, drape=ReferenceDirection((0, 1, 0))),
    ], reference="bottom")
    layup.apply(mesh, element_sets=named.element_sets)

Plies are stacked bottom to top in ``sequence`` order (definition order by default),
keeping only the plies that cover each element. ``reference`` fixes which surface
stays on the mesh when the thickness changes: ``"bottom"`` for a part laid up on a
tool under the mesh surface, ``"top"`` for the opposite, ``"mid"`` for a symmetric
taper.

Ply names are carried into each element's plies, so results can be collected per
named ply across the whole part (``Layup.ply_failure``).

Fibre angles
------------
A ply's ``angle`` is measured from the element material axis unless the ply has a
``drape``:

* ``ReferenceDirection(v)``: the ply's own 0-degree direction is ``v`` projected onto
  each element, independent of the element material axis (a ply coordinate system).
* ``AngleField(f)``: adds a deviation ``f(xyz)`` in degrees, for example the shear
  angle from a draping simulation.
* ``TabulatedDrape(points, deviations)``: deviations given at points (nearest point),
  for importing draping results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .laminate_materials import MaterialCard
from .laminate_mechanics import Laminate, Ply


# ---------------------------------------------------------------------------
# Drape fields
# ---------------------------------------------------------------------------
class ReferenceDirection:
    """The ply's 0-degree direction is ``direction`` projected onto the element plane."""

    def __init__(self, direction: Sequence[float]):
        d = np.asarray(direction, dtype=float)
        if d.shape != (3,) or np.linalg.norm(d) == 0.0:
            raise ValueError("direction must be a non-zero 3-vector")
        self.direction = d / np.linalg.norm(d)

    def angle_offset(self, element) -> float:
        e1, e2, e3 = element.frame
        d = self.direction - np.dot(self.direction, e3) * e3
        if np.linalg.norm(d) < 1e-8:
            raise ValueError(f"Reference direction is normal to element {element.element_id}")
        return float(np.degrees(np.arctan2(d @ e2, d @ e1)))


class AngleField:
    """Fibre angle deviation in degrees as a function of the global position."""

    def __init__(self, function: Callable[[np.ndarray], float]):
        self.function = function

    def angle_offset(self, element) -> float:
        return float(self.function(element.centroid))


class TabulatedDrape:
    """Fibre angle deviations given at points (nearest point to each element centroid)."""

    def __init__(self, points: Sequence[Sequence[float]], deviations: Sequence[float]):
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.deviations = np.asarray(deviations, dtype=float).ravel()
        if len(self.points) != len(self.deviations) or len(self.points) == 0:
            raise ValueError("points and deviations must be non-empty and the same length")

    def angle_offset(self, element) -> float:
        d = np.linalg.norm(self.points - element.centroid, axis=1)
        return float(self.deviations[int(np.argmin(d))])


# ---------------------------------------------------------------------------
# Plies and layup
# ---------------------------------------------------------------------------
ElementSelection = Union[None, str, Iterable[int]]


@dataclass
class PlyDefinition:
    """One ply of the ply book.

    ``elements`` (element ids or the name of an element set) and ``region`` (a
    predicate on the element centroid) restrict where the ply exists; with neither,
    it covers the whole mesh. With both, the ply covers their intersection.
    """

    name: str
    material: MaterialCard
    thickness: float
    angle: float = 0.0
    elements: ElementSelection = None
    region: Optional[Callable[[np.ndarray], bool]] = None
    sequence: Optional[float] = None
    drape: Optional[object] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Every ply needs a name")
        if self.thickness <= 0.0:
            raise ValueError(f"Ply '{self.name}' needs a positive thickness")
        if self.drape is not None and not hasattr(self.drape, "angle_offset"):
            raise TypeError("drape must provide angle_offset(element)")

    def _element_ids(self, element_sets: Mapping[str, Sequence[int]]) -> Optional[set]:
        if self.elements is None:
            return None
        if isinstance(self.elements, str):
            try:
                return {int(e) for e in element_sets[self.elements]}
            except KeyError as exc:
                raise KeyError(f"Ply '{self.name}' refers to unknown element set '{self.elements}'") from exc
        return {int(e) for e in self.elements}

    def covers(self, element, element_sets: Mapping[str, Sequence[int]] = None) -> bool:
        ids = self._element_ids(element_sets or {})
        if ids is not None and element.element_id not in ids:
            return False
        return self.region is None or bool(self.region(element.centroid))

    def angle_on(self, element) -> float:
        """Fibre angle measured from the element's material axis."""
        offset = 0.0 if self.drape is None else self.drape.angle_offset(element)
        angle = self.angle + offset
        return float((angle + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class PlyDrop:
    """A named ply that ends on the edge between two neighbouring elements."""

    ply: str
    inside: int  # element that has the ply
    outside: int  # neighbour without it
    nodes: Tuple[int, ...]  # the shared edge


class Layup:
    """A ply book that builds each element's laminate from the plies covering it."""

    REFERENCES = ("mid", "bottom", "top")

    def __init__(self, plies: Sequence[PlyDefinition], name: str = "layup", reference: str = "mid"):
        plies = list(plies)
        if not plies:
            raise ValueError("A layup needs at least one ply")
        names = [p.name for p in plies]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Ply names must be unique: {sorted(duplicates)}")
        if reference not in self.REFERENCES:
            raise ValueError(f"reference must be one of {self.REFERENCES}")
        order = sorted(range(len(plies)), key=lambda i: (plies[i].sequence if plies[i].sequence is not None else i, i))
        self.plies = [plies[i] for i in order]
        self.name = name
        self.reference = reference

    def __len__(self) -> int:
        return len(self.plies)

    def ply(self, name: str) -> PlyDefinition:
        for p in self.plies:
            if p.name == name:
                return p
        raise KeyError(f"No ply named '{name}'")

    # ---------------------------------------------------------- laminates
    def plies_on(self, element, element_sets: Mapping[str, Sequence[int]] = None) -> List[PlyDefinition]:
        return [p for p in self.plies if p.covers(element, element_sets)]

    def laminate_for(self, element, element_sets: Mapping[str, Sequence[int]] = None) -> Optional[Laminate]:
        """The element's laminate, or None when no ply covers it."""
        chosen = self.plies_on(element, element_sets)
        if not chosen:
            return None
        plies = [Ply(p.material, p.thickness, p.angle_on(element), p.name) for p in chosen]
        h = sum(p.thickness for p in chosen)
        offset = {"mid": 0.0, "bottom": 0.5 * h, "top": -0.5 * h}[self.reference]
        return Laminate(plies, f"{self.name}:" + "/".join(p.name for p in chosen), offset)

    def apply(
        self,
        mesh,
        element_sets: Mapping[str, Sequence[int]] = None,
        element_ids: Optional[Iterable[int]] = None,
        uncovered: str = "error",
    ) -> None:
        """Replace the laminate of each element (all shells by default) with the layup's.

        ``uncovered`` says what to do with an element no ply covers: ``"error"``
        (default) or ``"keep"`` its current laminate.
        """
        if uncovered not in ("error", "keep"):
            raise ValueError("uncovered must be 'error' or 'keep'")
        ids = [int(e) for e in element_ids] if element_ids is not None else [e.element_id for e in _shells(mesh)]
        missing = []
        for eid in ids:
            element = mesh.element(eid)
            laminate = self.laminate_for(element, element_sets)
            if laminate is None:
                missing.append(eid)
                continue
            mesh.replace_element(element.with_laminate(laminate))
        if missing and uncovered == "error":
            raise ValueError(f"No ply covers elements {missing[:10]}{'...' if len(missing) > 10 else ''}")

    # ------------------------------------------------------------ reports
    def coverage(self, mesh, element_sets: Mapping[str, Sequence[int]] = None) -> Dict[str, List[int]]:
        """Element ids covered by each ply."""
        return {p.name: [e.element_id for e in _shells(mesh) if p.covers(e, element_sets)] for p in self.plies}

    def ply_table(self, mesh, element_sets: Mapping[str, Sequence[int]] = None) -> List[dict]:
        """Area, mass and element count of each ply (a ply book summary)."""
        rows = []
        for p in self.plies:
            covered = [e for e in _shells(mesh) if p.covers(e, element_sets)]
            area = float(sum(e.area for e in covered))
            rows.append({
                "ply": p.name,
                "material": p.material.name,
                "angle": p.angle,
                "thickness": p.thickness,
                "elements": len(covered),
                "area": area,
                "mass": area * p.thickness * p.material.density,
            })
        return rows

    def thickness_map(self, mesh, element_sets: Mapping[str, Sequence[int]] = None) -> Dict[int, float]:
        return {e.element_id: sum(p.thickness for p in self.plies_on(e, element_sets)) for e in _shells(mesh)}

    def drops(self, mesh, element_sets: Mapping[str, Sequence[int]] = None) -> List[PlyDrop]:
        """Every place a ply ends between two elements that share an edge."""
        edges: Dict[frozenset, List[Tuple[int, Tuple[int, ...]]]] = {}
        for e in _shells(mesh):
            for edge in e.EDGES:
                nodes = tuple(e.node_ids[i] for i in edge)
                corners = frozenset((nodes[0], nodes[-1]))
                edges.setdefault(corners, []).append((e.element_id, nodes))
        cover = {p.name: set(ids) for p, ids in zip(self.plies, self.coverage(mesh, element_sets).values())}
        out = []
        for owners in edges.values():
            if len(owners) != 2:
                continue
            (a, nodes), (b, _) = owners
            for name, ids in cover.items():
                if (a in ids) != (b in ids):
                    inside, outside = (a, b) if a in ids else (b, a)
                    out.append(PlyDrop(name, inside, outside, nodes))
        return out

    def check_drops(self, mesh, max_per_step: int = 2, element_sets: Mapping[str, Sequence[int]] = None) -> List[str]:
        """Design-rule check: edges where more than ``max_per_step`` plies stop at once."""
        count: Dict[Tuple[int, int], List[str]] = {}
        for drop in self.drops(mesh, element_sets):
            key = (min(drop.inside, drop.outside), max(drop.inside, drop.outside))
            count.setdefault(key, []).append(drop.ply)
        return [
            f"Elements {a} and {b}: {len(names)} plies stop on the same edge ({', '.join(names)})"
            for (a, b), names in sorted(count.items())
            if len(names) > max_per_step
        ]

    def ply_failure(self, solution, criterion, load_case=None, recoverer=None) -> Dict[str, dict]:
        """Highest failure index of each named ply over the mesh (at the ply mid-planes).

        Returns {ply name: {"index", "element", "strength_ratio", "mode"}}. The
        hygrothermal states come from ``load_case`` (the solution's load case by default).
        """
        from .laminate_hygrothermal import NULL_STATE
        from .laminate_stress_recovery import ExactStressRecoverer

        recoverer = recoverer or ExactStressRecoverer(solution.mesh.hygrothermal_engine)
        load_case = load_case or solution.load_case
        worst: Dict[str, dict] = {}
        for element in _shells(solution.mesh):
            u = solution.element_displacements(element.element_id)
            state = load_case.state_for(element.element_id) if load_case is not None else NULL_STATE
            for point in recoverer.ply_midplane_states(element, u, state=state):
                ply = element.laminate.plies[point.ply_index]
                result = criterion.evaluate(point.material_stress_3d(), ply.material.strength)
                best = worst.get(ply.name)
                if best is None or result.index > best["index"]:
                    worst[ply.name] = {
                        "index": float(result.index),
                        "element": element.element_id,
                        "strength_ratio": float(result.strength_ratio),
                        "mode": result.mode.value,
                    }
        return worst


def _shells(mesh):
    return getattr(mesh, "shell_elements", mesh.elements)

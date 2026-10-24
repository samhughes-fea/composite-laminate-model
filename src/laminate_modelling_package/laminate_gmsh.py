"""Gmsh meshing: import Gmsh meshes into ``FEAMesh`` and build plate geometries.

Gmsh (https://gmsh.info) is an optional dependency, installed with
``pip install laminate_modelling_package[gmsh]``. It is imported only when one of
these classes is used.

Physical groups carry the model data:

* a physical **surface** names a laminate region. Its elements get the laminate
  mapped to that name (or the default laminate);
* a physical group of any dimension becomes a **node set** with the same name,
  so supports and loads can be applied by name (for example ``"left"``).

Supported surface elements are 3-node triangles (Tri3), 6-node triangles (Tri6,
curved), 4-node quads (Quad4) and 8-node serendipity quads (Quad8, ``Mesh.ElementOrder
= 2`` with ``Mesh.SecondOrderIncomplete = 1``), in any mix. 8-node quads become flat
Quad8 elements with ``curved=False``, or curved Quad8 shells (the default). Gmsh puts second-order
nodes on the CAD geometry, so the curved elements follow curved surfaces. The
builders take ``element="quad8_curved"`` (the default) ``| "quad4" | "tri3" | "quad8" |
"tri6" | "quad_dominant"`` and set the Gmsh options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .laminate_kinematics import KinematicModel
from .laminate_mechanics import Laminate
from .laminate_solver_pipeline import FEAMesh

GMSH_TRIANGLE, GMSH_QUAD4, GMSH_TRI6, GMSH_QUAD8 = 2, 3, 9, 16
_SUPPORTED = {GMSH_TRIANGLE: 3, GMSH_QUAD4: 4, GMSH_TRI6: 6, GMSH_QUAD8: 8}
_SURFACE_ELEMENT_TYPES = {2: "3-node triangle", 3: "4-node quadrangle", 9: "6-node triangle", 10: "9-node quadrangle",
                          16: "8-node quadrangle"}

# Gmsh options per element choice
MESH_OPTIONS = {
    "quad4": {"Mesh.RecombineAll": 1, "Mesh.Algorithm": 8, "Mesh.SubdivisionAlgorithm": 1, "Mesh.ElementOrder": 1},
    "quad_dominant": {"Mesh.RecombineAll": 1, "Mesh.Algorithm": 8, "Mesh.SubdivisionAlgorithm": 0, "Mesh.ElementOrder": 1},
    "tri3": {"Mesh.RecombineAll": 0, "Mesh.Algorithm": 6, "Mesh.ElementOrder": 1},
    "quad8": {"Mesh.RecombineAll": 1, "Mesh.Algorithm": 8, "Mesh.SubdivisionAlgorithm": 1, "Mesh.ElementOrder": 2,
              "Mesh.SecondOrderIncomplete": 1},
    "tri6": {"Mesh.RecombineAll": 0, "Mesh.Algorithm": 6, "Mesh.ElementOrder": 2},
}
MESH_OPTIONS["quad8_curved"] = dict(MESH_OPTIONS["quad8"])


def mesh_options(element: str) -> Dict[str, float]:
    try:
        return dict(MESH_OPTIONS[element])
    except KeyError as exc:
        raise ValueError(f"element must be one of {', '.join(MESH_OPTIONS)}") from exc


class GmshUnavailableError(ImportError):
    """Raised when the optional ``gmsh`` package is missing or cannot load."""


class UnsupportedMeshError(ValueError):
    """Raised when a Gmsh mesh contains elements the shell element cannot use."""


def _gmsh():
    try:
        import gmsh  # noqa: WPS433 - optional dependency, imported on use
    except (ImportError, OSError) as exc:  # OSError: the shared library failed to load
        raise GmshUnavailableError(
            "Gmsh meshing needs the gmsh package: pip install 'laminate_modelling_package[gmsh]'. "
            f"Import failed with: {exc}"
        ) from exc
    return gmsh


class GmshSession:
    """Context manager that initialises Gmsh once and finalises it on exit.

    If Gmsh is already initialised (for example by the caller's own script), the
    session reuses it and leaves it running.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._owner = False

    def __enter__(self):
        gmsh = _gmsh()
        if not gmsh.isInitialized():
            gmsh.initialize(readConfigFiles=False)
            self._owner = True
        gmsh.option.setNumber("General.Terminal", 1 if self.verbose else 0)
        return gmsh

    def __exit__(self, *exc) -> None:
        if self._owner:
            _gmsh().finalize()


@dataclass
class GmshMesh:
    """A mesh imported from Gmsh, with its named node and element sets."""

    mesh: FEAMesh
    node_sets: Dict[str, List[int]] = field(default_factory=dict)
    element_sets: Dict[str, List[int]] = field(default_factory=dict)

    def nodes(self, name: str) -> List[int]:
        try:
            return self.node_sets[name]
        except KeyError as exc:
            raise KeyError(f"No physical group named '{name}'; available: {sorted(self.node_sets)}") from exc

    def elements(self, name: str) -> List[int]:
        try:
            return self.element_sets[name]
        except KeyError as exc:
            raise KeyError(f"No physical surface named '{name}'; available: {sorted(self.element_sets)}") from exc


class GmshMeshImporter:
    """Builds an ``FEAMesh`` from the current Gmsh model, a ``.msh`` file or a geometry file.

    Parameters
    ----------
    laminates:
        One laminate for the whole mesh, or a mapping from physical-surface name to
        laminate. Elements outside every mapped surface use ``default_laminate``.
    kinematics:
        Kinematic model shared by every element.
    reference_normal:
        Elements whose normal points against this vector have their node order
        reversed, so pressure and the ply stack face a consistent side. Use
        ``None`` to keep Gmsh's orientation (for example on closed 3D surfaces).
    curved:
        Build 8-node quads as curved Quad8 shells (the default) instead of flat Quad8.
    """

    def __init__(
        self,
        laminates: Union[Laminate, Mapping[str, Laminate]],
        kinematics: KinematicModel,
        default_laminate: Optional[Laminate] = None,
        reference_normal: Optional[Sequence[float]] = (0.0, 0.0, 1.0),
        curved: bool = True,
        **element_options,
    ):
        self.curved = bool(curved)
        if isinstance(laminates, Laminate):
            self.laminates: Dict[str, Laminate] = {}
            self.default_laminate = laminates
        else:
            self.laminates = dict(laminates)
            self.default_laminate = default_laminate
        self.kinematics = kinematics
        self.reference_normal = None if reference_normal is None else np.asarray(reference_normal, dtype=float)
        self.element_options = element_options

    # ----------------------------------------------------------------- entry points
    def from_file(
        self, path: Union[str, Path], mesh_options: Optional[Mapping[str, float]] = None,
        element: Optional[str] = None,
    ) -> GmshMesh:
        """Read a ``.msh`` file, or open a geometry file (``.geo``, ``.step``, ...) and mesh it.

        ``element`` picks the element family for geometry files (curved Quad8 by
        default); ``mesh_options`` overrides or adds Gmsh options. A ``.msh`` file keeps
        its own elements, and its 8-node quads follow the importer's ``curved`` unless
        ``element`` says otherwise.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with GmshSession() as gmsh:
            gmsh.clear()
            gmsh.open(str(path))
            if path.suffix.lower() != ".msh":
                element = element or "quad8_curved"
                options = globals()["mesh_options"](element)
                options.update(mesh_options or {})
                for name, value in options.items():
                    gmsh.option.setNumber(name, value)
                gmsh.model.mesh.generate(2)
            return self.from_model(curved={"quad8_curved": True, "quad8": False}.get(element))

    def from_model(self, curved: Optional[bool] = None) -> GmshMesh:
        """Convert the mesh of the current Gmsh model (already generated) into an ``FEAMesh``.

        ``curved`` overrides the importer's choice of flat or curved Quad8.
        """
        curved = self.curved if curved is None else bool(curved)
        gmsh = _gmsh()
        if not gmsh.isInitialized():
            raise RuntimeError("Gmsh is not initialised; build or open a model first")
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise UnsupportedMeshError("The Gmsh model has no mesh nodes; call gmsh.model.mesh.generate(2) first")
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        mesh = FEAMesh()
        for tag, xyz in sorted(zip((int(t) for t in node_tags), coords), key=lambda item: item[0]):
            mesh.add_node(tag, xyz)

        surface_names = self._surface_physical_names(gmsh)
        element_sets: Dict[str, List[int]] = {}
        element_id = 1
        for _, surface in gmsh.model.getEntities(2):
            types, _, connectivity = gmsh.model.mesh.getElements(2, surface)
            for etype, conn in zip(types, connectivity):
                if etype not in _SUPPORTED:
                    raise UnsupportedMeshError(
                        f"Surface {surface} has {_SURFACE_ELEMENT_TYPES.get(etype, f'type-{etype}')} elements. "
                        "Supported are 3- and 6-node triangles, 4-node quads and 8-node quads; for quadratic quads set "
                        "Mesh.SecondOrderIncomplete = 1 with Mesh.RecombineAll = 1"
                    )
                names = surface_names.get(surface, [])
                laminate = self._laminate_for(surface, names)
                for cell in np.asarray(conn, dtype=int).reshape(-1, _SUPPORTED[etype]):
                    nodes = self._oriented(mesh, [int(n) for n in cell])
                    element_type = "quad8_curved" if curved and etype == GMSH_QUAD8 else None
                    mesh.add_element(element_id, nodes, laminate, self.kinematics, element_type=element_type,
                                     **self.element_options)
                    for name in names:
                        element_sets.setdefault(name, []).append(element_id)
                    element_id += 1
        if element_id == 1:
            raise UnsupportedMeshError("The Gmsh model has no surface elements")
        self._drop_orphan_nodes(mesh)
        node_sets = self._node_sets(gmsh, set(mesh.node_ids))
        return GmshMesh(mesh, node_sets, element_sets)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _surface_physical_names(gmsh) -> Dict[int, List[str]]:
        names: Dict[int, List[str]] = {}
        for dim, tag in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(dim, tag) or f"surface_{tag}"
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
                names.setdefault(int(entity), []).append(name)
        return names

    def _laminate_for(self, surface: int, names: Iterable[str]) -> Laminate:
        matched = [self.laminates[n] for n in names if n in self.laminates]
        if len({id(lam) for lam in matched}) > 1:
            raise ValueError(f"Surface {surface} belongs to several physical groups with different laminates: {list(names)}")
        if matched:
            return matched[0]
        if self.default_laminate is None:
            raise ValueError(
                f"Surface {surface} (physical groups {list(names) or 'none'}) has no laminate; "
                f"map one of {list(names)} or pass default_laminate"
            )
        return self.default_laminate

    def _oriented(self, mesh: FEAMesh, nodes: List[int]) -> List[int]:
        if self.reference_normal is None:
            return nodes
        x = np.array([mesh.node_coordinates(n) for n in nodes])
        if len(nodes) in (3, 6):
            normal = np.cross(x[1] - x[0], x[2] - x[0])
        else:
            normal = np.cross(x[2] - x[0], x[3] - x[1])
        if np.dot(normal, self.reference_normal) >= 0.0:
            return nodes
        if len(nodes) == 3:
            return [nodes[0], nodes[2], nodes[1]]
        if len(nodes) == 4:
            return [nodes[0], nodes[3], nodes[2], nodes[1]]
        if len(nodes) == 6:  # Tri6: mid-side nodes follow their edges
            return [nodes[0], nodes[2], nodes[1], nodes[5], nodes[4], nodes[3]]
        # Quad8: reverse the corners and carry the mid-side nodes with their edges
        return [nodes[0], nodes[3], nodes[2], nodes[1], nodes[7], nodes[6], nodes[5], nodes[4]]

    @staticmethod
    def _drop_orphan_nodes(mesh: FEAMesh) -> None:
        """Remove nodes no element uses (Gmsh keeps geometry points such as hole centres)."""
        used = {n for element in mesh.elements for n in element.node_ids}
        orphans = [n for n in mesh.node_ids if n not in used]
        if orphans:
            mesh.remove_nodes(orphans)

    @staticmethod
    def _node_sets(gmsh, valid: set) -> Dict[str, List[int]]:
        sets: Dict[str, List[int]] = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag) or f"group_{dim}_{tag}"
            tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
            nodes = sorted({int(t) for t in tags} & valid)
            sets[name] = sorted(set(sets.get(name, [])) | set(nodes))
        return sets


@dataclass(frozen=True)
class CircularHole:
    """A circular cut-out at (x, y) with radius ``radius``."""

    x: float
    y: float
    radius: float


class GmshPlateBuilder:
    """Meshes a flat rectangular plate with optional circular holes.

    Named node sets on the result: ``left`` (x = 0), ``right`` (x = length),
    ``bottom`` (y = 0), ``top`` (y = width), ``hole_1``, ``hole_2``, ... and
    ``plate`` (the whole surface, also the element set name). The element size
    shrinks to ``hole_size`` on the hole edges and grows to ``size`` over
    ``transition`` hole radii. ``element`` chooses curved Quad8 (default), Quad4, Tri3, flat Quad8, Tri6
    or a quad-dominant Quad4/Tri3 mix.
    """

    def __init__(
        self,
        length: float,
        width: float,
        holes: Sequence[CircularHole] = (),
        size: Optional[float] = None,
        hole_size: Optional[float] = None,
        transition: float = 3.0,
        element: str = "quad8_curved",
    ):
        self.element = element
        mesh_options(element)  # validates the choice
        if length <= 0 or width <= 0:
            raise ValueError("Plate length and width must be positive")
        self.length, self.width = float(length), float(width)
        self.holes = tuple(holes)
        for hole in self.holes:
            inside = hole.radius < hole.x < self.length - hole.radius and hole.radius < hole.y < self.width - hole.radius
            if hole.radius <= 0 or not inside:
                raise ValueError(f"{hole} must have a positive radius and lie inside the plate")
        self.size = float(size) if size else min(length, width) / 20.0
        self.hole_size = float(hole_size) if hole_size else self.size / 4.0
        self.transition = float(transition)

    def build(self, importer: GmshMeshImporter, save_msh: Optional[Union[str, Path]] = None) -> GmshMesh:
        with GmshSession() as gmsh:
            gmsh.clear()
            gmsh.model.add("plate")
            self._geometry(gmsh)
            self._sizing(gmsh)
            for name, value in mesh_options(self.element).items():
                gmsh.option.setNumber(name, value)
            gmsh.model.mesh.generate(2)
            if save_msh is not None:
                gmsh.write(str(save_msh))
            return importer.from_model(curved={"quad8_curved": True, "quad8": False}.get(self.element))

    def _geometry(self, gmsh) -> None:
        occ = gmsh.model.occ
        plate = occ.addRectangle(0.0, 0.0, 0.0, self.length, self.width)
        if self.holes:
            disks = [(2, occ.addDisk(h.x, h.y, 0.0, h.radius, h.radius)) for h in self.holes]
            occ.cut([(2, plate)], disks)
        occ.synchronize()
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, surfaces, name="plate")
        tol = 1e-6 * max(self.length, self.width)
        edges: Dict[str, List[int]] = {"left": [], "right": [], "bottom": [], "top": []}
        holes: Dict[int, List[int]] = {i: [] for i in range(len(self.holes))}
        for _, curve in gmsh.model.getEntities(1):
            # Classify each curve by a point in the middle of its parameter range.
            lo, hi = gmsh.model.getParametrizationBounds(1, curve)
            x, y, _ = gmsh.model.getValue(1, curve, [0.5 * (lo[0] + hi[0])])
            on_hole = [abs(np.hypot(x - h.x, y - h.y) - h.radius) < 1e-6 * h.radius for h in self.holes]
            if any(on_hole):
                holes[on_hole.index(True)].append(curve)
            elif abs(x) < tol:
                edges["left"].append(curve)
            elif abs(x - self.length) < tol:
                edges["right"].append(curve)
            elif abs(y) < tol:
                edges["bottom"].append(curve)
            elif abs(y - self.width) < tol:
                edges["top"].append(curve)
            else:
                raise RuntimeError(f"Could not classify curve {curve} at ({x}, {y})")
        for name, curves in edges.items():
            gmsh.model.addPhysicalGroup(1, curves, name=name)
        for i, curves in holes.items():
            gmsh.model.addPhysicalGroup(1, curves, name=f"hole_{i + 1}")
        self._hole_curves = [c for curves in holes.values() for c in curves]

    def _sizing(self, gmsh) -> None:
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        field = gmsh.model.mesh.field
        if not self.holes:
            constant = field.add("MathEval")
            field.setString(constant, "F", repr(self.size))
            field.setAsBackgroundMesh(constant)
            return
        distance = field.add("Distance")
        field.setNumbers(distance, "CurvesList", self._hole_curves)
        field.setNumber(distance, "Sampling", 200)
        radius = min(h.radius for h in self.holes)
        threshold = field.add("Threshold")
        field.setNumber(threshold, "InField", distance)
        field.setNumber(threshold, "SizeMin", self.hole_size)
        field.setNumber(threshold, "SizeMax", self.size)
        field.setNumber(threshold, "DistMin", 0.0)
        field.setNumber(threshold, "DistMax", self.transition * radius)
        field.setAsBackgroundMesh(threshold)

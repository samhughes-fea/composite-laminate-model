"""Gmsh meshing: import Gmsh quad meshes into ``FEAMesh`` and build common plate geometries.

Gmsh (https://gmsh.info) is an optional dependency, installed with
``pip install laminate_modelling_package[gmsh]``. It is imported only when one of
these classes is used.

Physical groups carry the model data:

* a physical **surface** names a laminate region. Its elements get the laminate
  mapped to that name (or the default laminate);
* a physical group of any dimension becomes a **node set** with the same name,
  so supports and loads can be applied by name (for example ``"left"``).

Only 4-node quadrilaterals are used by the shell element, so a mesh must be
recombined into quads (``Mesh.RecombineAll = 1``). ``GmshPlateBuilder`` does
this for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .laminate_fea_engine import KinematicModel
from .laminate_mechanics import Laminate
from .laminate_solver_pipeline import FEAMesh

GMSH_TRIANGLE, GMSH_QUAD4 = 2, 3
_SURFACE_ELEMENT_TYPES = {2: "3-node triangle", 3: "4-node quadrangle", 9: "6-node triangle", 10: "9-node quadrangle",
                          16: "8-node quadrangle"}


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
    """

    def __init__(
        self,
        laminates: Union[Laminate, Mapping[str, Laminate]],
        kinematics: KinematicModel,
        default_laminate: Optional[Laminate] = None,
        reference_normal: Optional[Sequence[float]] = (0.0, 0.0, 1.0),
        **element_options,
    ):
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
    def from_file(self, path: Union[str, Path], mesh_options: Optional[Mapping[str, float]] = None) -> GmshMesh:
        """Read a ``.msh`` file, or open a geometry file (``.geo``, ``.step``, ...) and mesh it in quads."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with GmshSession() as gmsh:
            gmsh.clear()
            gmsh.open(str(path))
            if path.suffix.lower() != ".msh":
                options = {"Mesh.RecombineAll": 1, "Mesh.Algorithm": 8, "Mesh.SubdivisionAlgorithm": 1}
                options.update(mesh_options or {})
                for name, value in options.items():
                    gmsh.option.setNumber(name, value)
                gmsh.model.mesh.generate(2)
            return self.from_model()

    def from_model(self) -> GmshMesh:
        """Convert the mesh of the current Gmsh model (already generated) into an ``FEAMesh``."""
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
                if etype != GMSH_QUAD4:
                    raise UnsupportedMeshError(
                        f"Surface {surface} has {_SURFACE_ELEMENT_TYPES.get(etype, f'type-{etype}')} elements. "
                        "The shell element needs 4-node quads: set Mesh.RecombineAll = 1 (and Mesh.ElementOrder = 1)"
                    )
                names = surface_names.get(surface, [])
                laminate = self._laminate_for(surface, names)
                for quad in np.asarray(conn, dtype=int).reshape(-1, 4):
                    nodes = self._oriented(mesh, [int(n) for n in quad])
                    mesh.add_element(element_id, nodes, laminate, self.kinematics, **self.element_options)
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
        normal = np.cross(x[2] - x[0], x[3] - x[1])
        if np.dot(normal, self.reference_normal) < 0.0:
            return [nodes[0], nodes[3], nodes[2], nodes[1]]
        return nodes

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
    """Meshes a flat rectangular plate with optional circular holes into Quad4 elements.

    Named node sets on the result: ``left`` (x = 0), ``right`` (x = length),
    ``bottom`` (y = 0), ``top`` (y = width), ``hole_1``, ``hole_2``, ... and
    ``plate`` (the whole surface, also the element set name). The element size
    shrinks to ``hole_size`` on the hole edges and grows to ``size`` over
    ``transition`` hole radii.
    """

    def __init__(
        self,
        length: float,
        width: float,
        holes: Sequence[CircularHole] = (),
        size: Optional[float] = None,
        hole_size: Optional[float] = None,
        transition: float = 3.0,
    ):
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
            for name, value in {"Mesh.RecombineAll": 1, "Mesh.Algorithm": 8, "Mesh.SubdivisionAlgorithm": 1,
                                "Mesh.ElementOrder": 1}.items():
                gmsh.option.setNumber(name, value)
            gmsh.model.mesh.generate(2)
            if save_msh is not None:
                gmsh.write(str(save_msh))
            return importer.from_model()

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

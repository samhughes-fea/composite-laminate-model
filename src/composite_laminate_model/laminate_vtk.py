"""VTK export for ParaView and other VTK viewers.

``write_vtu`` writes one solved (or unsolved) model as an XML unstructured grid
(``.vtu``). Every element keeps its own cell type and node order, so curved Quad8,
Tri6 and Hex20 cells render with their mid-side nodes:

=================  ===============  =========
element            VTK cell         type id
=================  ===============  =========
Tri3               VTK_TRIANGLE     5
Quad4              VTK_QUAD         9
Tri6               VTK_QUADRATIC_TRIANGLE  22
Quad8 (flat, curved)  VTK_QUADRATIC_QUAD  23
Hex8               VTK_HEXAHEDRON   12
Hex20              VTK_QUADRATIC_HEXAHEDRON  25
=================  ===============  =========

Point data: displacement, rotation, reaction force and moment, and the shell
resultants N, M, Q extrapolated to the nodes and averaged. Cell data: element id,
laminate index, thickness, number of plies, normal and material axis, element
resultants and strain energy (shells), mean stress (solids), and the worst failure
index with its ply when a failure criterion is given.

``layered=True`` also writes ``<name>_plies.vtu``: every ply of every element as its
own 3D cell (a wedge or hexahedron through the ply thickness, along the shell
normal, scaled by ``thickness_scale`` so thin laminates can be seen). Each ply cell
carries the ply index, angle, fibre direction, material-axis stresses at mid-ply and
the ply's failure index, so a single ply can be picked out with a threshold filter.

``write_pvd`` writes several load cases or combinations as one ``.pvd`` collection
(one ``.vtu`` per case), which ParaView opens as a time series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
from xml.sax.saxutils import quoteattr

import numpy as np

VTK_TYPES = {"tri3": 5, "quad4": 9, "tri6": 22, "quad8": 23, "quad8_curved": 23, "hex8": 12, "hex20": 25}
VTK_WEDGE, VTK_HEXAHEDRON = 13, 12


def _cell_type(element) -> int:
    try:
        return VTK_TYPES[element.TOPOLOGY]
    except KeyError as exc:
        raise ValueError(f"No VTK cell type for '{element.TOPOLOGY}' elements") from exc


def _is_solid(element) -> bool:
    return bool(getattr(element, "IS_SOLID", False))


def _normal(element, xi: float, eta: float) -> np.ndarray:
    if hasattr(element, "point_frame") and not _is_solid(element):
        return np.asarray(element.point_frame(xi, eta))[2]
    return np.asarray(element.frame)[2]


def _fmt(values: np.ndarray, integer: bool = False) -> str:
    values = np.asarray(values).ravel()
    if integer:
        return " ".join(str(int(v)) for v in values)
    values = np.where(np.isfinite(values), values, np.nan)
    return " ".join(f"{v:.9g}" for v in values)


class _Grid:
    """Minimal XML UnstructuredGrid writer (ASCII)."""

    def __init__(self, points: np.ndarray, connectivity: List[Sequence[int]], types: Sequence[int]):
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.connectivity = [list(map(int, c)) for c in connectivity]
        self.types = [int(t) for t in types]
        self.point_data: Dict[str, np.ndarray] = {}
        self.cell_data: Dict[str, np.ndarray] = {}

    def add_point(self, name: str, values) -> None:
        values = np.asarray(values)
        if values.shape[0] != len(self.points):
            raise ValueError(f"point field '{name}' has {values.shape[0]} rows for {len(self.points)} points")
        self.point_data[name] = values

    def add_cell(self, name: str, values) -> None:
        values = np.asarray(values)
        if values.shape[0] != len(self.types):
            raise ValueError(f"cell field '{name}' has {values.shape[0]} rows for {len(self.types)} cells")
        self.cell_data[name] = values

    @staticmethod
    def _array(name: str, values: np.ndarray) -> List[str]:
        values = np.asarray(values)
        integer = values.dtype.kind in "iub"
        vtype = "Int32" if integer else "Float64"
        ncomp = 1 if values.ndim == 1 else values.shape[1]
        return [f'        <DataArray type="{vtype}" Name={quoteattr(name)} NumberOfComponents="{ncomp}" format="ascii">',
                "          " + _fmt(values, integer), "        </DataArray>"]

    def write(self, path: Path) -> Path:
        offsets = np.cumsum([len(c) for c in self.connectivity])
        lines = ['<?xml version="1.0"?>',
                 '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">',
                 "  <UnstructuredGrid>",
                 f'    <Piece NumberOfPoints="{len(self.points)}" NumberOfCells="{len(self.types)}">']
        if self.point_data:
            lines.append("      <PointData>")
            for name, values in self.point_data.items():
                lines += self._array(name, values)
            lines.append("      </PointData>")
        if self.cell_data:
            lines.append("      <CellData>")
            for name, values in self.cell_data.items():
                lines += self._array(name, values)
            lines.append("      </CellData>")
        lines += ["      <Points>", '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">',
                  "          " + _fmt(self.points), "        </DataArray>", "      </Points>", "      <Cells>",
                  '        <DataArray type="Int32" Name="connectivity" format="ascii">',
                  "          " + _fmt(np.concatenate([np.asarray(c) for c in self.connectivity]) if self.connectivity
                                      else np.zeros(0), True),
                  "        </DataArray>",
                  '        <DataArray type="Int32" Name="offsets" format="ascii">',
                  "          " + _fmt(offsets, True), "        </DataArray>",
                  '        <DataArray type="UInt8" Name="types" format="ascii">',
                  "          " + _fmt(np.array(self.types), True), "        </DataArray>",
                  "      </Cells>", "    </Piece>", "  </UnstructuredGrid>", "</VTKFile>"]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path


class VTKExporter:
    """Writes a mesh, or a solution and its results, to VTK files.

    ``source`` is a ``SolutionResult`` or an ``FEAMesh`` (geometry and lay-up only).
    ``criterion`` adds failure indices; ``load_case`` and ``recoverer`` are passed to
    the ``ResultsExtractor`` that computes the fields.
    """

    def __init__(self, source, criterion=None, load_case=None, recoverer=None):
        from .laminate_results import ResultsExtractor

        if hasattr(source, "displacements"):
            self.solution = source
            self.mesh = source.mesh
            self.results = ResultsExtractor(source, load_case, recoverer, criterion)
        else:
            self.solution = None
            self.mesh = source
            self.results = None
        self.criterion = criterion
        laminates: Dict[int, int] = {}
        self._laminate_index = {}
        for element in self.mesh.elements:
            key = id(element.laminate)
            laminates.setdefault(key, len(laminates))
            self._laminate_index[element.element_id] = laminates[key]

    # -------------------------------------------------------------- model
    def model_grid(self) -> _Grid:
        mesh = self.mesh
        node_ids = mesh.node_ids
        index = {n: i for i, n in enumerate(node_ids)}
        elements = mesh.elements
        grid = _Grid(mesh.coordinates_array(), [[index[n] for n in e.node_ids] for e in elements],
                     [_cell_type(e) for e in elements])
        grid.add_point("node_id", np.array(node_ids, dtype=int))
        grid.add_cell("element_id", np.array([e.element_id for e in elements], dtype=int))
        grid.add_cell("solid", np.array([int(_is_solid(e)) for e in elements], dtype=int))
        grid.add_cell("laminate", np.array([self._laminate_index[e.element_id] for e in elements], dtype=int))
        grid.add_cell("thickness", np.array([e.laminate.thickness for e in elements]))
        grid.add_cell("n_plies", np.array([e.laminate.n_plies for e in elements], dtype=int))
        grid.add_cell("normal", np.array([np.asarray(e.frame)[2] for e in elements]))
        grid.add_cell("material_axis", np.array([np.asarray(e.frame)[0] for e in elements]))
        if self.solution is None:
            return grid
        sol = self.solution
        disp, reac = sol.displacement_array(), sol.reaction_array()
        grid.add_point("displacement", disp[:, :3])
        grid.add_point("rotation", disp[:, 3:])
        grid.add_point("reaction_force", reac[:, :3])
        grid.add_point("reaction_moment", reac[:, 3:])
        shells = mesh.shell_elements
        n = len(elements)
        row = {e.element_id: j for j, e in enumerate(elements)}
        if shells:
            nodal = self.results.nodes()
            grid.add_point("N", nodal.N)
            grid.add_point("M", nodal.M)
            grid.add_point("Q", nodal.Q)
            ef = self.results.elements()
            for name, values in (("element_N", ef.N), ("element_M", ef.M), ("element_Q", ef.Q)):
                full = np.full((n, values.shape[1]), np.nan)
                for eid, v in zip(ef.element_ids, values):
                    full[row[int(eid)]] = v
                grid.add_cell(name, full)
        energy = np.array([e.strain_energy(sol.element_displacements(e.element_id)) for e in elements])
        grid.add_cell("strain_energy", energy)
        if mesh.has_solids:
            sp = self.results.solid_points(positions=("mid",))
            mean = np.full((n, 6), np.nan)
            for eid in np.unique(sp.element_ids):
                mean[row[int(eid)]] = sp.stress_global[sp.element_ids == eid].mean(axis=0)
            grid.add_cell("solid_stress", mean)
        if self.criterion is not None:
            fi, ply = np.full(n, np.nan), np.full(n, -1, dtype=int)
            for field_ in (self.results.ply_points(), self.results.solid_points()):
                for eid in np.unique(field_.element_ids):
                    mask = field_.element_ids == eid
                    k = int(np.argmax(field_.failure_index[mask]))
                    fi[row[int(eid)]] = field_.failure_index[mask][k]
                    ply[row[int(eid)]] = field_.ply_index[mask][k]
            grid.add_cell("failure_index", fi)
            grid.add_cell("critical_ply", ply)
        return grid

    # -------------------------------------------------------------- plies
    def _ply_values(self):
        """{(element, ply): (5 or 6 material stresses at mid-ply, failure index)}."""
        if self.solution is None:
            return {}, {}
        stress, fi = {}, {}
        if self.mesh.shell_elements:
            pp = self.results.ply_points(positions=("mid",))
            planar = np.hstack([pp.stress_material, pp.shear_material])  # s11, s22, t12, t23, t13
            for key in set(zip(pp.element_ids.tolist(), pp.ply_index.tolist())):
                mask = (pp.element_ids == key[0]) & (pp.ply_index == key[1])
                s = planar[mask].mean(axis=0)
                stress[key] = np.array([s[0], s[1], 0.0, s[3], s[4], s[2]])
            if self.criterion is not None:
                full = self.results.ply_points()
                for key in set(zip(full.element_ids.tolist(), full.ply_index.tolist())):
                    mask = (full.element_ids == key[0]) & (full.ply_index == key[1])
                    fi[key] = float(full.failure_index[mask].max())
        if self.mesh.has_solids:
            sp = self.results.solid_points(positions=("mid",))
            for key in set(zip(sp.element_ids.tolist(), sp.ply_index.tolist())):
                mask = (sp.element_ids == key[0]) & (sp.ply_index == key[1])
                stress[key] = sp.stress_material[mask].mean(axis=0)
            if self.criterion is not None:
                full = self.results.solid_points()
                for key in set(zip(full.element_ids.tolist(), full.ply_index.tolist())):
                    mask = (full.element_ids == key[0]) & (full.ply_index == key[1])
                    fi[key] = float(full.failure_index[mask].max())
        return stress, fi

    def ply_grid(self, thickness_scale: float = 1.0, deformation_scale: float = 0.0) -> _Grid:
        """One 3D cell per ply of every element (see the module docstring).

        ``deformation_scale`` > 0 draws the plies on the deformed shape (translations
        times the scale), which helps when the viewer cannot warp the layered cells.
        """
        points: List[np.ndarray] = []
        conn: List[List[int]] = []
        types: List[int] = []
        rows = {k: [] for k in ("eid", "ply", "angle", "t", "fibre", "z")}
        disp = None
        if self.solution is not None and deformation_scale:
            disp = {n: self.solution.node_displacement(n)[:3] * deformation_scale for n in self.mesh.node_ids}
        for element in self.mesh.elements:
            lam = element.laminate
            if _is_solid(element):
                corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
                bounds = element.ply_zeta_bounds()
                for k, (a, b) in enumerate(bounds):
                    base = len(points)
                    for zeta in (a, b):
                        for xi, eta in corners:
                            x = element.global_coordinates(xi, eta, zeta)
                            if disp is not None:
                                N, _ = element.shape_functions(xi, eta, zeta)
                                x = x + sum(N[i] * disp[n] for i, n in enumerate(element.node_ids))
                            points.append(x)
                    conn.append(list(range(base, base + 8)))
                    types.append(VTK_HEXAHEDRON)
                    frame = element.point_frame(0.0, 0.0, 0.5 * (a + b))
                    self._ply_row(rows, element, k, frame, 0.5 * (a + b))
                continue
            corners = element.NATURAL_NODES[: element.N_CORNERS]
            cell = VTK_HEXAHEDRON if element.N_CORNERS == 4 else VTK_WEDGE
            mid = lam.mid_plane
            for k, (z_b, z_t) in enumerate(lam.ply_bounds()):
                base = len(points)
                for z in (z_b, z_t):  # bottom face first, counter-clockwise about the normal
                    zs = mid + thickness_scale * (z - mid)
                    for xi, eta in corners:
                        x = element.global_coordinates(xi, eta) + zs * _normal(element, xi, eta)
                        if disp is not None:
                            N, _ = element.shape_functions(xi, eta)
                            x = x + sum(N[i] * disp[n] for i, n in enumerate(element.node_ids))
                        points.append(x)
                n = element.N_CORNERS
                conn.append(list(range(base, base + 2 * n)))
                types.append(cell)
                self._ply_row(rows, element, k, np.asarray(element.frame), 0.5 * (z_b + z_t))
        grid = _Grid(np.array(points) if points else np.zeros((0, 3)), conn, types)
        grid.add_cell("element_id", np.array(rows["eid"], dtype=int))
        grid.add_cell("ply", np.array(rows["ply"], dtype=int))
        grid.add_cell("angle", np.array(rows["angle"]))
        grid.add_cell("ply_thickness", np.array(rows["t"]))
        grid.add_cell("fibre_direction", np.array(rows["fibre"]).reshape(-1, 3))
        grid.add_cell("z", np.array(rows["z"]))
        grid.add_cell("laminate", np.array([self._laminate_index[e] for e in rows["eid"]], dtype=int))
        if self.solution is not None:
            stress, fi = self._ply_values()
            keys = list(zip(rows["eid"], rows["ply"]))
            grid.add_cell("stress_material", np.array([stress.get(k, np.full(6, np.nan)) for k in keys]).reshape(-1, 6))
            if self.criterion is not None:
                grid.add_cell("failure_index", np.array([fi.get(k, np.nan) for k in keys]))
        return grid

    @staticmethod
    def _ply_row(rows, element, k, frame, z) -> None:
        ply = element.laminate.plies[k]
        t = np.radians(ply.angle)
        fibre = np.cos(t) * frame[0] + np.sin(t) * frame[1]
        rows["eid"].append(element.element_id)
        rows["ply"].append(k)
        rows["angle"].append(ply.angle)
        rows["t"].append(ply.thickness)
        rows["fibre"].append(fibre)
        rows["z"].append(z)

    # -------------------------------------------------------------- files
    def write(self, path, layered: bool = False, thickness_scale: float = 1.0,
              deformation_scale: float = 0.0) -> Dict[str, Path]:
        """Write ``path`` (``.vtu``) and, with ``layered``, ``<stem>_plies.vtu``; returns {kind: path}."""
        path = Path(path)
        if path.suffix != ".vtu":
            path = path.with_suffix(".vtu")
        out = {"model": self.model_grid().write(path)}
        if layered:
            out["plies"] = self.ply_grid(thickness_scale, deformation_scale).write(
                path.with_name(path.stem + "_plies.vtu"))
        return out


def write_vtu(path, source, criterion=None, layered: bool = False, thickness_scale: float = 1.0,
              deformation_scale: float = 0.0, load_case=None, recoverer=None) -> Dict[str, Path]:
    """Write a mesh or solution to ``path`` (see ``VTKExporter``); returns {kind: path}."""
    return VTKExporter(source, criterion, load_case, recoverer).write(path, layered, thickness_scale, deformation_scale)


def write_pvd(path, solutions: Union[Mapping[str, object], Sequence[object]], criterion=None, layered: bool = False,
              thickness_scale: float = 1.0) -> Path:
    """Write each solution to its own ``.vtu`` next to ``path`` and a ``.pvd`` collection of them.

    ``solutions`` is a mapping {name: SolutionResult} (e.g. from ``solve_combinations``)
    or a sequence; each case becomes one time step (0, 1, 2, ...) in ParaView, named in
    the file comments.
    """
    path = Path(path)
    if path.suffix != ".pvd":
        path = path.with_suffix(".pvd")
    items: List[Tuple[str, object]] = list(solutions.items()) if isinstance(solutions, Mapping) else \
        [(getattr(getattr(s, "load_case", None), "name", None) or f"case{i}", s) for i, s in enumerate(solutions)]
    lines = ['<?xml version="1.0"?>', '<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">',
             "  <Collection>"]
    for step, (name, solution) in enumerate(items):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))
        files = write_vtu(path.with_name(f"{path.stem}_{step:03d}_{safe}.vtu"), solution, criterion, layered,
                          thickness_scale)
        lines.append(f"    <!-- {step}: {name} -->")
        lines.append(f'    <DataSet timestep="{step}" group="" part="0" file={quoteattr(files["model"].name)}/>')
        if layered:
            lines.append(f'    <DataSet timestep="{step}" group="" part="1" file={quoteattr(files["plies"].name)}/>')
    lines += ["  </Collection>", "</VTKFile>"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


__all__ = ["VTK_TYPES", "VTKExporter", "write_vtu", "write_pvd"]

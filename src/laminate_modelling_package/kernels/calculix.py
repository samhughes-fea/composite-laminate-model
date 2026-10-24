"""CalculiX kernel: composite S8R/S6 shell decks, ``ccx`` execution and ``.dat`` parsing."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..laminate_benchmark_stress import ThicknessProfile
from ..laminate_solver_pipeline import BoundaryConditions
from .base import (
    MATERIAL_COMPONENTS,
    AnalysisModel,
    FEAKernel,
    KernelResult,
    KernelStatus,
    ProcessRunner,
    UnsupportedFeatureError,
    gauss2_heights,
)
from .decks import DeckCatalogue, equivalent_alphas, equivalent_nodal_forces, equivalent_thermal_load

# corner-pair edges that receive inserted mid-side nodes, per native topology
_MIDSIDE_EDGES = {"quad4": ((0, 1), (1, 2), (2, 3), (3, 0)), "tri3": ((0, 1), (1, 2), (2, 0))}
_CCX_TYPE = {"quad4": "S8R", "quad8": "S8R", "quad8_curved": "S8R", "tri3": "S6", "tri6": "S6"}


@dataclass
class CalculiXDeck:
    path: Path
    corner_nodes: List[int]
    midside_nodes: Dict[Tuple[int, int], int]
    layer_heights: Dict[int, np.ndarray]


class CalculiXDeckWriter:
    """Writes composite shell decks for CalculiX.

    CalculiX only accepts ``*SHELL SECTION, COMPOSITE`` on S8R/S6 elements, so every
    Quad4 is promoted to an S8R and every Tri3 to an S6 by inserting shared mid-side
    nodes; Quad8 (flat or curved) and Tri6 elements are written as S8R and S6 directly. Inserted mid-side nodes inherit a
    prescribed DOF when both corner nodes of the edge prescribe it (value = average).
    Each ply orientation is a rectangular system whose first two axes are the element's
    material axes (the projection of its ``material_axis``), rotated by the ply angle
    about the shell normal, so curved and rotated meshes keep the native fibre directions.
    Laminate offsets become ``OFFSET`` (fraction of the thickness, opposite sign).
    """

    def write(self, model: AnalysisModel, directory: Path, jobname: str) -> CalculiXDeck:
        mesh, bcs, loads = model.mesh, model.boundary_conditions, model.load_case
        catalogue = DeckCatalogue(mesh)
        temperature_load, state = equivalent_thermal_load(loads, mesh)
        next_id = max(mesh.node_ids) + 1
        midside: Dict[Tuple[int, int], int] = {}
        coords: Dict[int, np.ndarray] = {n: mesh.node_coordinates(n) for n in mesh.node_ids}
        connectivity: Dict[str, Dict[int, List[int]]] = {"S8R": {}, "S6": {}}
        for element in mesh.elements:
            n = element.node_ids
            topo = element.TOPOLOGY
            if topo not in _CCX_TYPE:
                raise UnsupportedFeatureError(f"CalculiX decks do not support '{topo}' elements")
            if topo not in _MIDSIDE_EDGES:  # already quadratic: written directly
                connectivity[_CCX_TYPE[topo]][element.element_id] = list(n)
                continue
            mids = []
            for a, b in _MIDSIDE_EDGES[topo]:
                key = (min(n[a], n[b]), max(n[a], n[b]))
                if key not in midside:
                    midside[key] = next_id
                    coords[next_id] = 0.5 * (coords[n[a]] + coords[n[b]])
                    next_id += 1
                mids.append(midside[key])
            connectivity[_CCX_TYPE[topo]][element.element_id] = list(n) + mids

        lines: List[str] = [f"** {model.title} - written by laminate_modelling_package", "*HEADING", model.title]
        lines.append("*NODE, NSET=NALL")
        lines += [f"{nid}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}" for nid, xyz in coords.items()]
        for ccx_type, conn in connectivity.items():
            if conn:
                lines.append(f"*ELEMENT, TYPE={ccx_type}, ELSET=EALL")
                lines += [f"{eid}, " + ", ".join(str(v) for v in c) for eid, c in conn.items()]
        for name, laminate, axes, eids in catalogue.groups.values():
            lines.append(f"*ELSET, ELSET={name}")
            lines += self._chunk(eids)
        if model.monitor_elements:
            lines.append("*ELSET, ELSET=EMON")
            lines += self._chunk(model.monitor_elements)
        lines += self._springs(bcs, max(mesh.element_ids) + 1)

        for name, card in catalogue.materials.values():
            e = card.elastic
            lines.append(f"** material {name}: {card.name}")
            lines.append(f"*MATERIAL, NAME={name}")
            lines.append("*ELASTIC, TYPE=ENGINEERING CONSTANTS")
            lines.append(f"{e.E1:.10g}, {e.E2:.10g}, {e.E3:.10g}, {e.v12:.10g}, {e.v13:.10g}, {e.v23:.10g}, {e.G12:.10g}, {e.G13:.10g}")
            lines.append(f"{e.G23:.10g}, 0.")
            a1, a2, a3 = equivalent_alphas(card, state)
            if state is not None:
                lines.append("*EXPANSION, TYPE=ORTHO, ZERO=0.")
                lines.append(f"{a1:.10g}, {a2:.10g}, {a3:.10g}, 0.")
            if card.density > 0.0:
                lines += ["*DENSITY", f"{card.density:.10g}"]
        for (axes, angle), name in catalogue.orientations.items():
            a = ", ".join(f"{v:.10g}" for v in axes)
            lines += [f"*ORIENTATION, NAME={name}, SYSTEM=RECTANGULAR", a, f"3, {angle:.10g}"]

        layer_heights: Dict[int, np.ndarray] = {}
        for name, laminate, axes, eids in catalogue.groups.values():
            offset = 0.0 - laminate.offset / laminate.thickness
            lines.append(f"*SHELL SECTION, ELSET={name}, COMPOSITE, OFFSET={offset:.10g}")
            for ply in laminate.plies:
                orientation = catalogue.orientations[(axes, ply.angle)]
                lines.append(f"{ply.thickness:.10g}, , {catalogue.material_name(ply.material)}, {orientation}")
            for eid in eids:
                layer_heights[eid] = gauss2_heights(laminate)

        prescribed = self._expand_prescribed(bcs, midside)
        if prescribed:
            lines.append("*BOUNDARY")
            for (node, dof), value in sorted(prescribed.items()):
                lines.append(f"{node}, {dof + 1}, {dof + 1}, {value:.12g}")
        for c in bcs.constraints:
            if c.constant != 0.0:
                raise UnsupportedFeatureError("CalculiX *EQUATION constraints must be homogeneous")
            lines += ["*EQUATION", str(1 + len(c.masters))]
            terms = [(c.slave, 1.0)] + [(k, -coef) for k, coef in c.masters]
            lines.append(", ".join(f"{k[0]}, {k[1] + 1}, {coef:.12g}" for k, coef in terms))

        if state is not None:
            lines += ["*INITIAL CONDITIONS, TYPE=TEMPERATURE", "NALL, 0."]
        lines += ["*STEP", "*STATIC"]
        forces = equivalent_nodal_forces(mesh, loads, body_loads=False)
        cloads = [(n, d, v) for n, vec in sorted(forces.items()) for d, v in enumerate(vec) if v != 0.0]
        if cloads:
            lines.append("*CLOAD")
            lines += [f"{n}, {d + 1}, {v:.12g}" for n, d, v in cloads]
        pressures = {e.element_id: loads.pressure_value(e) for e in mesh.elements if loads.has_pressure(e.element_id)}
        pressures = {e: p for e, p in pressures.items() if p != 0.0}
        if pressures:
            lines.append("*DLOAD")
            lines += [f"{e}, P, {p:.12g}" for e, p in pressures.items()]
        lines += self._gravity(mesh, loads)
        if state is not None:
            lines += ["*TEMPERATURE", f"NALL, {temperature_load:.12g}"]
        lines += ["*NODE PRINT, NSET=NALL", "U"]
        if model.monitor_elements:
            lines += ["*EL PRINT, ELSET=EMON", "S"]
        lines.append("*END STEP")

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{jobname}.inp"
        path.write_text("\n".join(lines) + "\n")
        return CalculiXDeck(path, mesh.node_ids, midside, layer_heights)

    @staticmethod
    def _chunk(ids: Sequence[int], per_line: int = 16) -> List[str]:
        ids = list(ids)
        return [", ".join(str(i) for i in ids[k : k + per_line]) for k in range(0, len(ids), per_line)]

    @staticmethod
    def _springs(bcs: BoundaryConditions, first_id: int) -> List[str]:
        lines = []
        for k, ((node, dof), stiffness) in enumerate(sorted(bcs.springs.items())):
            if dof > 2:
                raise UnsupportedFeatureError("CalculiX SPRING1 elements support translational DOFs only")
            name = f"SPR{k + 1}"
            lines += [f"*ELEMENT, TYPE=SPRING1, ELSET={name}", f"{first_id + k}, {node}",
                      f"*SPRING, ELSET={name}", str(dof + 1), f"{stiffness:.12g}"]
        return lines

    @staticmethod
    def _gravity(mesh, loads) -> List[str]:
        """Body loads as ``GRAV`` distributed loads.

        CalculiX integrates them over the expanded (3D) elements, so laminate offsets
        and ply densities are taken into account, and no nodal moments are written
        (moments on shell nodes need mean-rotation MPCs, which ccx cannot cascade at
        kinked edges such as a sharp trailing edge).
        """
        lines: List[str] = []
        for body in loads.body_loads:
            a = np.asarray(body.acceleration, dtype=float)
            g = float(np.linalg.norm(a))
            if g == 0.0:
                continue
            card = f"GRAV, {g:.12g}, {a[0] / g:.12g}, {a[1] / g:.12g}, {a[2] / g:.12g}"
            lines.append("*DLOAD")
            if body.element_ids is None:
                lines.append(f"EALL, {card}")
            else:
                lines += [f"{eid}, {card}" for eid in sorted(body.element_ids)]
        return lines

    @staticmethod
    def _expand_prescribed(bcs: BoundaryConditions, midside: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], float]:
        numeric = bcs.numeric_prescribed()
        prescribed = dict(numeric)
        for (a, b), m in midside.items():
            for dof in range(6):
                if (a, dof) in numeric and (b, dof) in numeric:
                    prescribed[(m, dof)] = 0.5 * (numeric[(a, dof)] + numeric[(b, dof)])
        return prescribed


class CalculiXResultReader:
    """Parses ``*NODE PRINT`` displacements and ``*EL PRINT`` stresses from a ``.dat`` file."""

    _FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

    def read(self, dat_path: Path) -> Tuple[Dict[int, np.ndarray], Dict[int, List[Tuple[int, np.ndarray]]]]:
        displacements: Dict[int, np.ndarray] = {}
        stresses: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        section = None
        number = re.compile(self._FLOAT)
        for raw in dat_path.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("displacements"):
                section, displacements = "u", {}
                continue
            if lower.startswith("stresses"):
                section, stresses = "s", {}
                continue
            if lower[0].isalpha():
                section = None
                continue
            tokens = line.split()
            if section == "u" and len(tokens) >= 4:
                displacements[int(tokens[0])] = np.array([float(t) for t in tokens[1:4]])
            elif section == "s" and len(tokens) >= 8 and all(number.fullmatch(t) for t in tokens[:8]):
                eid, ip = int(tokens[0]), int(tokens[1])
                stresses.setdefault(eid, []).append((ip, np.array([float(t) for t in tokens[2:8]])))
        return displacements, stresses

    @staticmethod
    def thickness_profile(element_id: int, rows: List[Tuple[int, np.ndarray]], heights: np.ndarray) -> ThicknessProfile:
        """Average the in-plane integration points of each layer level.

        CalculiX expands S8R into C3D20R (2x2x2 points per layer) and S6 into C3D15
        (wedge points per layer), numbering points layer by layer from the bottom, so
        point p (1-based) sits at level (p-1) // points_per_level. The result is in
        material axes (CalculiX reports layer stresses in each layer's orientation).
        """
        rows = sorted(rows, key=lambda r: r[0])
        n_levels = len(heights)
        per_level = max(1, len(rows) // max(n_levels, 1))
        levels = np.zeros((n_levels, 6))
        counts = np.zeros(n_levels)
        for ip, s in rows:
            level = (ip - 1) // per_level
            if level < n_levels:
                levels[level] += s
                counts[level] += 1
        if np.any(counts == 0):
            raise ValueError(f"Incomplete CalculiX integration point data for element {element_id}")
        levels /= counts[:, None]
        # For oriented composite layers ccx prints stresses in the layer (material) axes:
        # s11, s22, s33, s12, s13, s23 -> s11, s22, t12, t23, t13
        values = levels[:, [0, 1, 3, 5, 4]]
        return ThicknessProfile(element_id, heights, values, MATERIAL_COMPONENTS, "material", "calculix")


class CalculiXKernelAdapter(FEAKernel):
    name = "calculix"

    def __init__(
        self,
        executable: str = "ccx",
        work_dir: Optional[os.PathLike] = None,
        timeout: float = 600.0,
        writer: Optional[CalculiXDeckWriter] = None,
        reader: Optional[CalculiXResultReader] = None,
        runner: Optional[ProcessRunner] = None,
        threads: int = 1,
    ):
        self.executable = executable
        self.work_dir = Path(work_dir) if work_dir else None
        self.timeout = timeout
        self.writer = writer or CalculiXDeckWriter()
        self.reader = reader or CalculiXResultReader()
        self.runner = runner or ProcessRunner()
        self.threads = threads

    def executable_path(self) -> Optional[str]:
        return shutil.which(self.executable)

    def is_available(self) -> bool:
        return self.executable_path() is not None

    def _execute(self, model: AnalysisModel) -> KernelResult:
        directory = self.work_dir or Path(tempfile.mkdtemp(prefix="ccx_"))
        jobname = re.sub(r"[^A-Za-z0-9_]", "_", model.title) or "job"
        deck = self.writer.write(model, directory, jobname)
        files = {"deck": deck.path}
        if not self.is_available():
            return KernelResult(self.name, KernelStatus.SKIPPED, files=files,
                                message=f"'{self.executable}' not found on PATH; deck written to {deck.path}")
        os.environ.setdefault("OMP_NUM_THREADS", str(self.threads))
        start = time.perf_counter()
        completed = self.runner.run([self.executable_path(), jobname], directory, self.timeout)
        solve_time = time.perf_counter() - start
        dat = directory / f"{jobname}.dat"
        files.update({"dat": dat, "log": directory / f"{jobname}.sta"})
        if completed.returncode != 0 or not dat.exists() or "*ERROR" in (completed.stdout or ""):
            tail = "\n".join((completed.stdout or "").splitlines()[-15:])
            return KernelResult(self.name, KernelStatus.FAILED, files=files,
                                message=f"ccx exited with code {completed.returncode}:\n{tail}")
        displacements, stresses = self.reader.read(dat)
        nodal = {}
        for nid in deck.corner_nodes:
            u = displacements.get(nid)
            if u is None:
                raise RuntimeError(f"CalculiX output has no displacement for node {nid}")
            nodal[nid] = np.concatenate([u, np.full(3, np.nan)])  # rotations are not printed
        profiles = {
            eid: {"material": self.reader.thickness_profile(eid, stresses[eid], deck.layer_heights[eid])}
            for eid in model.monitor_elements
            if eid in stresses
        }
        return KernelResult(self.name, KernelStatus.SUCCESS, nodal, profiles, files=files,
                            info={"solve_time": solve_time, "n_nodes_s8r": len(deck.corner_nodes) + len(deck.midside_nodes)})

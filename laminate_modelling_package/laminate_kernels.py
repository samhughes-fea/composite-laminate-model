"""Pluggable FEA kernels (Adapter + Strategy Patterns).

All kernels consume the same ``AnalysisModel`` and return a ``KernelResult``:

* ``NativePythonKernelAdapter``  wraps the in-house NumPy/SciPy solver pipeline.
* ``CalculiXKernelAdapter``      writes an Abaqus-style ``.inp`` deck (composite S8R shells),
                                 runs ``ccx`` and parses the ``.dat`` tables.
* ``NastranKernelAdapter``       writes a fixed-field MSC/NX Nastran bulk deck (GRID, CQUAD4,
                                 PCOMP, MAT8, ...), runs ``nastran`` and parses the ``.f06``.

External kernels always write their deck. When the executable is not installed
they return status ``SKIPPED`` (with the deck path) instead of raising, so a
benchmark run completes on any machine.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .laminate_benchmark_stress import ThicknessProfile
from .laminate_hygrothermal import HygrothermalState
from .laminate_materials import MaterialCard
from .laminate_mechanics import Laminate
from .laminate_solver_pipeline import BoundaryConditions, FEAMesh, LoadCase, StructuralGlobalSolver
from .laminate_stress_recovery import ExactStressRecoverer

GLOBAL_COMPONENTS = ("sxx", "syy", "txy", "tyz", "txz")
MATERIAL_COMPONENTS = ("s11", "s22", "t12", "t23", "t13")


class KernelStatus(Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"


class UnsupportedFeatureError(ValueError):
    """The analysis model uses a feature the target solver deck cannot express."""


@dataclass
class AnalysisModel:
    """Solver-independent description of one linear static analysis."""

    mesh: FEAMesh
    boundary_conditions: BoundaryConditions
    load_case: LoadCase
    title: str = "laminate_model"
    monitor_elements: Tuple[int, ...] = ()


@dataclass
class KernelResult:
    kernel: str
    status: KernelStatus
    displacements: Dict[int, np.ndarray] = field(default_factory=dict)
    stress_profiles: Dict[int, Dict[str, ThicknessProfile]] = field(default_factory=dict)
    runtime: float = 0.0
    message: str = ""
    files: Dict[str, Path] = field(default_factory=dict)
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is KernelStatus.SUCCESS

    def max_abs_displacement(self, dof: int) -> float:
        values = [abs(v[dof]) for v in self.displacements.values() if np.isfinite(v[dof])]
        return float(max(values)) if values else float("nan")

    def profile(self, element_id: int, frame: str) -> Optional[ThicknessProfile]:
        return self.stress_profiles.get(element_id, {}).get(frame)


# ---------------------------------------------------------------------------
# Kernel interface
# ---------------------------------------------------------------------------
class FEAKernel(ABC):
    """Template: ``run`` times the analysis and converts exceptions into FAILED results."""

    name = "kernel"

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def _execute(self, model: AnalysisModel) -> KernelResult:
        """Run the analysis; implementations fill everything except ``runtime``."""

    def run(self, model: AnalysisModel) -> KernelResult:
        start = time.perf_counter()
        try:
            result = self._execute(model)
        except Exception as exc:  # noqa: BLE001 - reported back to the caller as a FAILED result
            result = KernelResult(self.name, KernelStatus.FAILED, message=f"{type(exc).__name__}: {exc}")
        result.runtime = time.perf_counter() - start
        return result


class ProcessRunner:
    """Thin wrapper over subprocess so external execution can be substituted in tests."""

    def run(self, args: Sequence[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
        return subprocess.run(
            list(args), cwd=str(cwd), capture_output=True, text=True, timeout=timeout, check=False
        )


def gauss2_heights(laminate: Laminate) -> np.ndarray:
    """Two Gauss heights per ply (mid +/- h_k / (2 sqrt 3)), bottom ply first."""
    z = []
    for z_b, z_t in laminate.ply_bounds():
        mid, half = 0.5 * (z_b + z_t), 0.5 * (z_t - z_b)
        z.extend([mid - half / math.sqrt(3.0), mid + half / math.sqrt(3.0)])
    return np.array(z)


# ---------------------------------------------------------------------------
# Native kernel
# ---------------------------------------------------------------------------
class NativePythonKernelAdapter(FEAKernel):
    """Adapter over ``StructuralGlobalSolver`` and ``ExactStressRecoverer``."""

    name = "native"

    def __init__(self, solver: Optional[StructuralGlobalSolver] = None, recoverer: Optional[ExactStressRecoverer] = None):
        self.solver = solver or StructuralGlobalSolver()
        self.recoverer = recoverer or ExactStressRecoverer()

    def _execute(self, model: AnalysisModel) -> KernelResult:
        solution = self.solver.solve(model.mesh, model.boundary_conditions, model.load_case)
        profiles: Dict[int, Dict[str, ThicknessProfile]] = {}
        for eid in model.monitor_elements:
            element = model.mesh.element(eid)
            u_e = solution.element_displacements(eid)
            state = model.load_case.state_for(eid)
            z_g = gauss2_heights(element.laminate)
            points = self.recoverer.at_heights(element, u_e, z_g, state=state)
            glob = np.array([[*p.stress_global, *p.shear_global] for p in points])
            mids = self.recoverer.ply_midplane_states(element, u_e, state=state)
            mat = np.array([[*p.stress_material, *p.shear_material] for p in mids])
            mat_g = np.array([[*p.stress_material, *p.shear_material] for p in points])
            profiles[eid] = {
                "global": ThicknessProfile(eid, z_g, glob, GLOBAL_COMPONENTS, "global", self.name),
                "material": ThicknessProfile(eid, z_g, mat_g, MATERIAL_COMPONENTS, "material", self.name),
                "material_midply": ThicknessProfile(
                    eid, element.laminate.ply_midplanes(), mat, MATERIAL_COMPONENTS, "material", self.name
                ),
            }
        return KernelResult(
            self.name,
            KernelStatus.SUCCESS,
            displacements=solution.displacement_map(),
            stress_profiles=profiles,
            info={"solve_time": solution.solve_time, **solution.info},
        )


# ---------------------------------------------------------------------------
# Shared helpers for external decks
# ---------------------------------------------------------------------------
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
        raise UnsupportedFeatureError("External decks support a single uniform hygrothermal state for all elements")
    state = states.pop()
    if not state.is_uniform:
        raise UnsupportedFeatureError("External decks do not support through-thickness temperature/moisture gradients")
    if state.delta_temperature != 0.0:
        return state.delta_temperature, state
    return 1.0, state


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


def _material_key(card: MaterialCard) -> tuple:
    hyg = card.hygrothermal
    hyg_key = None if hyg is None else (hyg.alpha1, hyg.alpha2, hyg.alpha3, hyg.beta1, hyg.beta2, hyg.beta3)
    return (card.name, tuple(card.elastic.as_dict().values()), tuple(card.strength.as_dict().values()), hyg_key, card.density)


def _laminate_key(laminate: Laminate) -> tuple:
    return tuple((_material_key(p.material), p.thickness, p.angle) for p in laminate.plies)


class _DeckCatalogue:
    """Deduplicates materials, orientations and laminates across a mesh."""

    def __init__(self, mesh: FEAMesh):
        self.materials: Dict[tuple, Tuple[str, MaterialCard]] = {}
        self.angles: Dict[float, str] = {}
        self.laminates: Dict[tuple, Tuple[str, Laminate, List[int]]] = {}
        for element in mesh.elements:
            key = _laminate_key(element.laminate)
            if key not in self.laminates:
                self.laminates[key] = (f"LAM{len(self.laminates) + 1}", element.laminate, [])
            self.laminates[key][2].append(element.element_id)
            for ply in element.laminate.plies:
                mkey = _material_key(ply.material)
                if mkey not in self.materials:
                    self.materials[mkey] = (f"M{len(self.materials) + 1}", ply.material)
                if ply.angle not in self.angles:
                    self.angles[ply.angle] = f"OR{len(self.angles) + 1}"

    def material_name(self, card: MaterialCard) -> str:
        return self.materials[_material_key(card)][0]


# ---------------------------------------------------------------------------
# CalculiX
# ---------------------------------------------------------------------------
@dataclass
class CalculiXDeck:
    path: Path
    corner_nodes: List[int]
    midside_nodes: Dict[Tuple[int, int], int]
    layer_heights: Dict[int, np.ndarray]


class CalculiXDeckWriter:
    """Writes composite shell decks for CalculiX.

    CalculiX only accepts ``*SHELL SECTION, COMPOSITE`` on S8R/S6 elements, so every
    Quad4 is promoted to an S8R by inserting shared mid-side nodes. Mid-side nodes
    inherit a prescribed DOF when both corner nodes of the edge prescribe it (value =
    average). Ply orientations are rectangular systems rotated about global z, which
    matches the native material axes for meshes lying in planes parallel to xy.
    """

    def write(self, model: AnalysisModel, directory: Path, jobname: str) -> CalculiXDeck:
        mesh, bcs, loads = model.mesh, model.boundary_conditions, model.load_case
        catalogue = _DeckCatalogue(mesh)
        temperature_load, state = equivalent_thermal_load(loads, mesh)
        next_id = max(mesh.node_ids) + 1
        midside: Dict[Tuple[int, int], int] = {}
        coords: Dict[int, np.ndarray] = {n: mesh.node_coordinates(n) for n in mesh.node_ids}
        connectivity: Dict[int, List[int]] = {}
        for element in mesh.elements:
            n = element.node_ids
            mids = []
            for a, b in ((n[0], n[1]), (n[1], n[2]), (n[2], n[3]), (n[3], n[0])):
                key = (min(a, b), max(a, b))
                if key not in midside:
                    midside[key] = next_id
                    coords[next_id] = 0.5 * (coords[a] + coords[b])
                    next_id += 1
                mids.append(midside[key])
            connectivity[element.element_id] = list(n) + mids

        lines: List[str] = [f"** {model.title} - written by laminate_modelling_package", "*HEADING", model.title]
        lines.append("*NODE, NSET=NALL")
        lines += [f"{nid}, {xyz[0]:.12g}, {xyz[1]:.12g}, {xyz[2]:.12g}" for nid, xyz in coords.items()]
        lines.append("*ELEMENT, TYPE=S8R, ELSET=EALL")
        lines += [f"{eid}, " + ", ".join(str(v) for v in conn) for eid, conn in connectivity.items()]
        for name, laminate, eids in catalogue.laminates.values():
            lines.append(f"*ELSET, ELSET={name}")
            lines += self._chunk(eids)
        if model.monitor_elements:
            lines.append("*ELSET, ELSET=EMON")
            lines += self._chunk(model.monitor_elements)

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
        for angle, name in catalogue.angles.items():
            lines += [f"*ORIENTATION, NAME={name}, SYSTEM=RECTANGULAR", "1., 0., 0., 0., 1., 0.", f"3, {angle:.10g}"]

        layer_heights: Dict[int, np.ndarray] = {}
        for name, laminate, eids in catalogue.laminates.values():
            lines.append(f"*SHELL SECTION, ELSET={name}, COMPOSITE, OFFSET=0.")
            for ply in laminate.plies:
                lines.append(f"{ply.thickness:.10g}, , {catalogue.material_name(ply.material)}, {catalogue.angles[ply.angle]}")
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
        if loads.nodal_loads:
            lines.append("*CLOAD")
            lines += [f"{n}, {d + 1}, {v:.12g}" for (n, d), v in sorted(loads.nodal_loads.items()) if v != 0.0]
        pressures = {e: loads.pressure_on(e) for e in mesh.element_ids if loads.pressure_on(e) != 0.0}
        if pressures:
            lines.append("*DLOAD")
            lines += [f"{e}, P, {p:.12g}" for e, p in pressures.items()]
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
    def _expand_prescribed(bcs: BoundaryConditions, midside: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], float]:
        prescribed = dict(bcs.prescribed)
        for (a, b), m in midside.items():
            for dof in range(6):
                if (a, dof) in bcs.prescribed and (b, dof) in bcs.prescribed:
                    prescribed[(m, dof)] = 0.5 * (bcs.prescribed[(a, dof)] + bcs.prescribed[(b, dof)])
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
        """Average the four in-plane integration points of each layer level (C3D20R 2x2x2 per layer).

        The result is in material axes (CalculiX reports layer stresses in the
        orientation assigned to each layer).

        CalculiX numbers the points xi fastest, then eta, then zeta, layer by layer,
        so point p (1-based) sits at level (p-1)//4 counted from the bottom.
        """
        rows = sorted(rows, key=lambda r: r[0])
        n_levels = len(heights)
        levels = np.zeros((n_levels, 6))
        counts = np.zeros(n_levels)
        for ip, s in rows:
            level = (ip - 1) // 4
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


# ---------------------------------------------------------------------------
# Nastran
# ---------------------------------------------------------------------------
class NastranFieldFormatter:
    """Small-field (8-character) Nastran bulk data formatting."""

    WIDTH = 8

    @classmethod
    def real(cls, value: float) -> str:
        """Most accurate 8-character representation (fixed point or 1.234+5 shorthand)."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Nastran fields must be finite")
        if value == 0.0:
            return "0."
        candidates = []
        for precision in range(0, 8):
            text = f"{value:.{precision}f}"
            if "." not in text:
                text += "."
            text = text.rstrip("0") if text.split(".")[1] else text
            if len(text) <= cls.WIDTH and float(text) != 0.0:
                candidates.append(text)
        for precision in range(0, 7):
            mantissa, exponent = f"{value:.{precision}e}".split("e")
            if "." not in mantissa:
                mantissa += "."
            exp = int(exponent)
            text = f"{mantissa}{'+' if exp >= 0 else '-'}{abs(exp)}"
            if len(text) <= cls.WIDTH:
                candidates.append(text)
        if not candidates:
            raise ValueError(f"Cannot represent {value} in an 8-character field")
        return min(candidates, key=lambda t: (abs(cls.parse_real(t) - value) / abs(value), len(t)))

    @staticmethod
    def parse_real(text: str) -> float:
        """Parse a Nastran real, including the exponent shorthand '1.5+11'."""
        text = text.strip()
        match = re.fullmatch(r"([-+]?\d*\.\d*)([-+]\d+)", text)
        return float(f"{match.group(1)}e{match.group(2)}") if match else float(text)

    @classmethod
    def field(cls, value) -> str:
        if value is None:
            text = ""
        elif isinstance(value, (bool, np.bool_)):
            raise TypeError("booleans are not valid Nastran fields")
        elif isinstance(value, (int, np.integer)):
            text = str(int(value))
        elif isinstance(value, (float, np.floating)):
            text = cls.real(value)
        else:
            text = str(value)
        if len(text) > cls.WIDTH:
            raise ValueError(f"Field '{text}' exceeds {cls.WIDTH} characters")
        return text.ljust(cls.WIDTH)

    @classmethod
    def card(cls, name: str, values: Sequence) -> List[str]:
        """Format a card over as many lines as needed (8 data fields per line)."""
        lines = []
        values = list(values)
        first = True
        while first or values:
            chunk, values = values[:8], values[8:]
            head = name if first else "+"
            line = cls.field(head) + "".join(cls.field(v) for v in chunk)
            lines.append(line.rstrip())
            first = False
        return lines


@dataclass
class NastranDeck:
    path: Path
    ply_heights: Dict[int, np.ndarray]


class NastranDeckWriter:
    """Writes SOL 101 bulk data: GRID, CQUAD4 (MCID 0), PCOMP, MAT8, SPC1/SPC, MPC, FORCE, MOMENT, PLOAD2, TEMPD."""

    SPC_ID, LOAD_ID, TEMP_ID, MPC_ID = 1, 2, 3, 4

    def __init__(self, formatter: type = NastranFieldFormatter):
        self.fmt = formatter

    def write(self, model: AnalysisModel, directory: Path, jobname: str) -> NastranDeck:
        mesh, bcs, loads = model.mesh, model.boundary_conditions, model.load_case
        catalogue = _DeckCatalogue(mesh)
        temperature_load, state = equivalent_thermal_load(loads, mesh)
        card = self.fmt.card
        has_loads = bool(loads.nodal_loads) or any(loads.pressure_on(e) != 0.0 for e in mesh.element_ids)

        lines = ["$ " + model.title + " - written by laminate_modelling_package", "SOL 101", "CEND",
                 f"TITLE = {model.title}", "ECHO = NONE", "SUBCASE 1", f"  SPC = {self.SPC_ID}"]
        if bcs.constraints:
            lines.append(f"  MPC = {self.MPC_ID}")
        if has_loads:
            lines.append(f"  LOAD = {self.LOAD_ID}")
        if state is not None:
            lines.append(f"  TEMPERATURE(LOAD) = {self.TEMP_ID}")
        lines += ["  DISPLACEMENT(PRINT) = ALL", "  STRESS(PRINT) = ALL", "BEGIN BULK",
                  "PARAM   AUTOSPC YES", "PARAM   POST    -1"]

        for nid in mesh.node_ids:
            x, y, z = mesh.node_coordinates(nid)
            lines += card("GRID", [nid, None, float(x), float(y), float(z)])
        laminate_pid = {}
        for pid, (name, laminate, eids) in enumerate(catalogue.laminates.values(), start=1):
            laminate_pid[name] = pid
            for eid in eids:
                n = mesh.element(eid).node_ids
                lines += card("CQUAD4", [eid, pid, *n, 0])

        ply_heights: Dict[int, np.ndarray] = {}
        for name, laminate, eids in catalogue.laminates.values():
            pid = laminate_pid[name]
            values = [pid, float(-0.5 * laminate.thickness), None, None, None, 0.0, None, None]
            for ply in laminate.plies:
                mid = int(catalogue.material_name(ply.material)[1:])
                values += [mid, float(ply.thickness), float(ply.angle), "YES"]
            lines += card("PCOMP", values)
            for eid in eids:
                ply_heights[eid] = laminate.ply_midplanes()

        for name, material in catalogue.materials.values():
            mid = int(name[1:])
            e, s = material.elastic, material.strength
            a1, a2, _ = equivalent_alphas(material, state)
            lines.append(f"$ MAT8 {mid}: {material.name}")
            lines += card("MAT8", [mid, e.E1, e.E2, e.v12, e.G12, e.G13, e.G23, float(material.density),
                                   float(a1), float(a2), 0.0, s.Xt, s.Xc, s.Yt, s.Yc, s.S12])

        lines += self._spc_cards(bcs)
        for c in bcs.constraints:
            if c.constant != 0.0:
                raise UnsupportedFeatureError("Nastran MPC equations must be homogeneous")
            terms = [(c.slave, 1.0)] + [(k, -coef) for k, coef in c.masters]
            fields_ = [[node, dof + 1, float(coef)] for (node, dof), coef in terms]
            blank = [None, None, None]
            values = [self.MPC_ID, *fields_[0], *(fields_[1] if len(fields_) > 1 else blank), None]
            for i in range(2, len(fields_), 2):
                second = fields_[i + 1] if i + 1 < len(fields_) else blank
                values += [None, *fields_[i], *second, None]
            lines += card("MPC", values)

        forces: Dict[int, np.ndarray] = {}
        for (node, dof), value in loads.nodal_loads.items():
            forces.setdefault(node, np.zeros(6))[dof] += value
        for node, vec in sorted(forces.items()):
            if np.any(vec[:3]):
                lines += card("FORCE", [self.LOAD_ID, node, 0, 1.0, *map(float, vec[:3])])
            if np.any(vec[3:]):
                lines += card("MOMENT", [self.LOAD_ID, node, 0, 1.0, *map(float, vec[3:])])
        for eid in mesh.element_ids:
            p = loads.pressure_on(eid)
            if p != 0.0:
                lines += card("PLOAD2", [self.LOAD_ID, float(p), eid])
        if state is not None:
            lines += card("TEMPD", [self.TEMP_ID, float(temperature_load)])
        lines.append("ENDDATA")

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{jobname}.bdf"
        path.write_text("\n".join(lines) + "\n")
        return NastranDeck(path, ply_heights)

    def _spc_cards(self, bcs: BoundaryConditions) -> List[str]:
        lines = []
        zero: Dict[str, List[int]] = {}
        by_node: Dict[int, List[int]] = {}
        for (node, dof), value in bcs.prescribed.items():
            if value == 0.0:
                by_node.setdefault(node, []).append(dof + 1)
            else:
                lines += self.fmt.card("SPC", [self.SPC_ID, node, dof + 1, float(value)])
        for node, comps in by_node.items():
            zero.setdefault("".join(str(c) for c in sorted(set(comps))), []).append(node)
        for comps, nodes in zero.items():
            lines += self.fmt.card("SPC1", [self.SPC_ID, int(comps), *sorted(nodes)])
        return lines


class NastranF06Reader:
    """Parses displacement vectors and layered composite stresses from an ``.f06`` file."""

    _FLOAT = re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$")

    def _is_float(self, token: str) -> bool:
        return bool(self._FLOAT.match(token))

    def read(self, f06_path: Path) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[int, np.ndarray]], List[str]]:
        text = Path(f06_path).read_text(errors="replace")
        displacements: Dict[int, np.ndarray] = {}
        ply_stresses: Dict[int, Dict[int, np.ndarray]] = {}
        fatal = [line.strip() for line in text.splitlines() if "FATAL" in line.upper()]
        section = None
        for raw in text.splitlines():
            squashed = raw.replace(" ", "").upper()
            if "DISPLACEMENTVECTOR" in squashed:
                section = "disp"
                continue
            if "STRESSESINLAYEREDCOMPOSITEELEMENTS" in squashed:
                section = "ply"
                continue
            if raw.startswith("1"):  # carriage control: new page
                section = None
                continue
            tokens = raw.split()
            if not tokens:
                continue
            if section == "disp" and len(tokens) == 8 and tokens[0].isdigit() and tokens[1] == "G":
                if all(self._is_float(t) for t in tokens[2:]):
                    displacements[int(tokens[0])] = np.array([float(t) for t in tokens[2:]])
            elif section == "ply" and len(tokens) >= 7 and tokens[0].isdigit() and tokens[1].isdigit():
                if all(self._is_float(t) for t in tokens[2:7]):
                    s1, s2, t12, t1z, t2z = (float(t) for t in tokens[2:7])
                    ply_stresses.setdefault(int(tokens[0]), {})[int(tokens[1])] = np.array([s1, s2, t12, t2z, t1z])
        return displacements, ply_stresses, fatal

    @staticmethod
    def thickness_profile(element_id: int, plies: Dict[int, np.ndarray], heights: np.ndarray) -> ThicknessProfile:
        order = sorted(plies)
        values = np.array([plies[k] for k in order])
        return ThicknessProfile(element_id, heights[: len(order)], values, MATERIAL_COMPONENTS, "material", "nastran")


class NastranKernelAdapter(FEAKernel):
    name = "nastran"

    def __init__(
        self,
        executable: str = "nastran",
        work_dir: Optional[os.PathLike] = None,
        timeout: float = 1800.0,
        writer: Optional[NastranDeckWriter] = None,
        reader: Optional[NastranF06Reader] = None,
        runner: Optional[ProcessRunner] = None,
        extra_arguments: Sequence[str] = ("batch=no", "scr=yes"),
    ):
        self.executable = executable
        self.work_dir = Path(work_dir) if work_dir else None
        self.timeout = timeout
        self.writer = writer or NastranDeckWriter()
        self.reader = reader or NastranF06Reader()
        self.runner = runner or ProcessRunner()
        self.extra_arguments = tuple(extra_arguments)

    def executable_path(self) -> Optional[str]:
        return shutil.which(self.executable)

    def is_available(self) -> bool:
        return self.executable_path() is not None

    def _execute(self, model: AnalysisModel) -> KernelResult:
        directory = self.work_dir or Path(tempfile.mkdtemp(prefix="nastran_"))
        jobname = re.sub(r"[^A-Za-z0-9_]", "_", model.title).lower() or "job"
        deck = self.writer.write(model, directory, jobname)
        files = {"deck": deck.path}
        if not self.is_available():
            return KernelResult(self.name, KernelStatus.SKIPPED, files=files,
                                message=f"'{self.executable}' not found on PATH; deck written to {deck.path}")
        start = time.perf_counter()
        completed = self.runner.run([self.executable_path(), deck.path.name, *self.extra_arguments], directory, self.timeout)
        solve_time = time.perf_counter() - start
        f06 = directory / f"{jobname}.f06"
        files["f06"] = f06
        if not f06.exists():
            return KernelResult(self.name, KernelStatus.FAILED, files=files,
                                message=f"nastran exited with code {completed.returncode} and produced no .f06")
        displacements, ply_stresses, fatal = self.reader.read(f06)
        if fatal:
            return KernelResult(self.name, KernelStatus.FAILED, files=files, message="; ".join(fatal[:5]))
        missing = [n for n in model.mesh.node_ids if n not in displacements]
        if missing:
            raise RuntimeError(f"Nastran output lacks displacements for {len(missing)} nodes")
        profiles = {
            eid: {"material_midply": self.reader.thickness_profile(eid, ply_stresses[eid], deck.ply_heights[eid])}
            for eid in model.monitor_elements
            if eid in ply_stresses
        }
        nodal = {n: displacements[n] for n in model.mesh.node_ids}
        return KernelResult(self.name, KernelStatus.SUCCESS, nodal, profiles, files=files, info={"solve_time": solve_time})


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class KernelRegistry:
    """Name -> factory registry so new kernels plug in without touching callers (OCP)."""

    def __init__(self):
        self._factories: Dict[str, Callable[..., FEAKernel]] = {}

    def register(self, name: str, factory: Callable[..., FEAKernel]) -> None:
        self._factories[name.lower()] = factory

    def create(self, name: str, **options) -> FEAKernel:
        try:
            return self._factories[name.lower()](**options)
        except KeyError as exc:
            raise KeyError(f"Unknown kernel '{name}'. Registered: {', '.join(self._factories)}") from exc

    def names(self) -> List[str]:
        return list(self._factories)


DEFAULT_KERNELS = KernelRegistry()
DEFAULT_KERNELS.register("native", NativePythonKernelAdapter)
DEFAULT_KERNELS.register("calculix", CalculiXKernelAdapter)
DEFAULT_KERNELS.register("nastran", NastranKernelAdapter)


def create_kernel(name: str, **options) -> FEAKernel:
    return DEFAULT_KERNELS.create(name, **options)

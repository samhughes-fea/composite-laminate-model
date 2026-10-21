"""Nastran kernel: SOL 101 bulk data decks, execution and ``.f06`` parsing."""

from __future__ import annotations

import math
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
from ..laminate_hygrothermal import HygrothermalState
from ..laminate_solver_pipeline import BoundaryConditions
from .base import (
    MATERIAL_COMPONENTS,
    AnalysisModel,
    FEAKernel,
    KernelResult,
    KernelStatus,
    ProcessRunner,
    UnsupportedFeatureError,
)
from .decks import (
    DeckCatalogue,
    element_temperature_field,
    equivalent_alphas,
    equivalent_nodal_forces,
    equivalent_thermal_load,
)


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


_CARD = {"quad4": "CQUAD4", "tri3": "CTRIA3", "quad8": "CQUAD8"}


class NastranDeckWriter:
    """Writes SOL 101 bulk data.

    Cards: GRID, CQUAD4 / CTRIA3 / CQUAD8, PCOMP, MAT8, CORD2R (material axes), SPC1/SPC,
    MPC, CELAS2 (springs), FORCE, MOMENT, PLOAD2, TEMPD and TEMPP1. Elements whose
    material axis is the projection of global x use MCID 0; others reference a CORD2R
    whose x axis is the element's material axis. Laminate offsets become PCOMP Z0.
    One uniform hygrothermal state is written as TEMPD (moisture folded into the
    expansion coefficients); element-wise temperatures and through-thickness gradients
    are written as TEMPP1.
    """

    SPC_ID, LOAD_ID, TEMP_ID, MPC_ID = 1, 2, 3, 4

    def __init__(self, formatter: type = NastranFieldFormatter):
        self.fmt = formatter

    @staticmethod
    def _uses_basic_x(element) -> bool:
        e3 = element.frame[2]
        x = np.array([1.0, 0.0, 0.0])
        proj = x - np.dot(x, e3) * e3
        if np.linalg.norm(proj) < 1e-8:
            return False
        return bool(np.allclose(proj / np.linalg.norm(proj), element.frame[0], atol=1e-9))

    def _thermal(self, model: AnalysisModel):
        """('uniform', dT, state) | ('field', {eid: (T, dT/dz)}, None) | (None, 0, None)."""
        try:
            temperature_load, state = equivalent_thermal_load(model.load_case, model.mesh)
            return ("uniform" if state is not None else None), temperature_load, state
        except UnsupportedFeatureError:
            return "field", element_temperature_field(model.load_case, model.mesh), None

    def write(self, model: AnalysisModel, directory: Path, jobname: str) -> NastranDeck:
        mesh, bcs, loads = model.mesh, model.boundary_conditions, model.load_case
        catalogue = DeckCatalogue(mesh)
        thermal, temperature_data, state = self._thermal(model)
        alpha_state = state if thermal == "uniform" else (HygrothermalState(1.0) if thermal == "field" else None)
        card = self.fmt.card
        forces = equivalent_nodal_forces(mesh, loads)
        has_pressure = any(loads.has_pressure(e) for e in mesh.element_ids)
        has_loads = bool(forces) or has_pressure

        lines = ["$ " + model.title + " - written by laminate_modelling_package", "SOL 101", "CEND",
                 f"TITLE = {model.title}", "ECHO = NONE", "SUBCASE 1", f"  SPC = {self.SPC_ID}"]
        if bcs.constraints:
            lines.append(f"  MPC = {self.MPC_ID}")
        if has_loads:
            lines.append(f"  LOAD = {self.LOAD_ID}")
        if thermal is not None:
            lines.append(f"  TEMPERATURE(LOAD) = {self.TEMP_ID}")
        lines += ["  DISPLACEMENT(PRINT) = ALL", "  STRESS(PRINT) = ALL", "BEGIN BULK",
                  "PARAM   AUTOSPC YES", "PARAM   POST    -1"]

        for nid in mesh.node_ids:
            x, y, z = mesh.node_coordinates(nid)
            lines += card("GRID", [nid, None, float(x), float(y), float(z)])

        cords: Dict[tuple, int] = {}
        mcid: Dict[int, int] = {}
        for element in mesh.elements:
            if self._uses_basic_x(element):
                mcid[element.element_id] = 0
                continue
            key = tuple(np.round(element.frame[0], 9) + 0.0)
            if key not in cords:
                cid = len(cords) + 1
                cords[key] = cid
                e1, e3 = element.frame[0], element.frame[2]
                lines += card("CORD2R", [cid, 0, 0.0, 0.0, 0.0, *map(float, e3), *map(float, e1)])
            mcid[element.element_id] = cords[key]

        group_pid = {}
        for pid, (name, laminate, axes, eids) in enumerate(catalogue.groups.values(), start=1):
            group_pid[name] = pid
            for eid in eids:
                element = mesh.element(eid)
                kind = _CARD.get(element.TOPOLOGY)
                if kind is None:
                    raise UnsupportedFeatureError(f"Nastran decks do not support '{element.TOPOLOGY}' elements")
                n = list(element.node_ids)
                if kind == "CQUAD8":
                    lines += card(kind, [eid, pid, *n, None, None, None, None, mcid[eid]])
                else:
                    lines += card(kind, [eid, pid, *n, mcid[eid]])

        ply_heights: Dict[int, np.ndarray] = {}
        for name, laminate, axes, eids in catalogue.groups.values():
            pid = group_pid[name]
            z0 = float(laminate.offset - 0.5 * laminate.thickness)
            values = [pid, z0, None, None, None, 0.0, None, None]
            for ply in laminate.plies:
                mid = int(catalogue.material_name(ply.material)[1:])
                values += [mid, float(ply.thickness), float(ply.angle), "YES"]
            lines += card("PCOMP", values)
            for eid in eids:
                ply_heights[eid] = laminate.ply_midplanes()

        for name, material in catalogue.materials.values():
            mid = int(name[1:])
            e, s = material.elastic, material.strength
            a1, a2, _ = equivalent_alphas(material, alpha_state)
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
        spring_id = max(mesh.element_ids) + 1
        for k, ((node, dof), stiffness) in enumerate(sorted(bcs.springs.items())):
            lines += card("CELAS2", [spring_id + k, float(stiffness), node, dof + 1])

        for node, vec in sorted(forces.items()):
            if np.any(vec[:3]):
                lines += card("FORCE", [self.LOAD_ID, node, 0, 1.0, *map(float, vec[:3])])
            if np.any(vec[3:]):
                lines += card("MOMENT", [self.LOAD_ID, node, 0, 1.0, *map(float, vec[3:])])
        for element in mesh.elements:
            if loads.has_pressure(element.element_id):
                p = loads.pressure_value(element)
                if p != 0.0:
                    lines += card("PLOAD2", [self.LOAD_ID, float(p), element.element_id])
        if thermal == "uniform":
            lines += card("TEMPD", [self.TEMP_ID, float(temperature_data)])
        elif thermal == "field":
            lines += card("TEMPD", [self.TEMP_ID, 0.0])
            for eid, (t_bar, t_prime) in temperature_data.items():
                if t_bar != 0.0 or t_prime != 0.0:
                    lines += card("TEMPP1", [self.TEMP_ID, eid, float(t_bar), float(t_prime)])
        lines.append("ENDDATA")

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{jobname}.bdf"
        path.write_text("\n".join(lines) + "\n")
        return NastranDeck(path, ply_heights)

    def _spc_cards(self, bcs: BoundaryConditions) -> List[str]:
        lines = []
        zero: Dict[str, List[int]] = {}
        by_node: Dict[int, List[int]] = {}
        for (node, dof), value in bcs.numeric_prescribed().items():
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

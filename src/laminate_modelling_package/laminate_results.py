"""Structured result extraction at every fidelity level.

``ResultsExtractor`` turns a ``SolutionResult`` into plain NumPy arrays at five levels:

1. Gauss point  - generalised strains and section resultants N, M, Q at each
                  integration point; ply stresses and failure indices at any z.
2. Element      - Gauss-averaged resultants, strain energy split
                  (membrane / bending / membrane-bending coupling / shear / drilling)
                  and the worst ply failure index.
3. Node         - displacements and reactions (exact nodal unknowns); resultants and
                  ply stresses extrapolated from the Gauss points with a bilinear fit
                  and averaged over the elements that share each node.
4. Ply / z      - any of the above evaluated at a chosen ply position.
5. Global       - total strain energy and its split, external work, applied and
                  reaction force/moment resultants, equilibrium residual, and peak
                  values with their locations.

Gauss points are the most accurate source for stresses (they are where the stiffness
was integrated); nodal stresses are smoothed for plotting. Section resultants are
mechanical: [N; M] = ABD [eps0; kappa] - [N_ht; M_ht] and Q = A_s gamma0.
All strains and resultants are in each element's local shell axes, which coincide
with global x, y for meshes in the xy-plane.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .laminate_failure_criteria import FailureCriterion
from .laminate_fea_engine import CompositePlateElement, shape_functions
from .laminate_hygrothermal import HygrothermalState
from .laminate_solver_pipeline import LoadCase, SolutionResult
from .laminate_stress_recovery import ExactStressRecoverer

PLY_POSITIONS = ("bottom", "mid", "top")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaussPointField:
    """Section-level results at every integration point (one row per point)."""

    element_ids: np.ndarray
    point_index: np.ndarray
    natural: np.ndarray  # (n, 2) xi, eta
    xyz: np.ndarray  # (n, 3)
    weight: np.ndarray  # quadrature weight x det J (area share)
    eps0: np.ndarray  # (n, 3) membrane strains
    kappa: np.ndarray  # (n, 3) curvatures
    gamma0: np.ndarray  # (n, 2) transverse shear strains [yz, xz]
    N: np.ndarray  # (n, 3) force resultants
    M: np.ndarray  # (n, 3) moment resultants
    Q: np.ndarray  # (n, 2) shear force resultants [Q_yz, Q_xz]


@dataclass(frozen=True)
class PlyPointField:
    """Ply stresses at Gauss points (one row per point and sampled height)."""

    element_ids: np.ndarray
    point_index: np.ndarray
    ply_index: np.ndarray
    z: np.ndarray
    xyz: np.ndarray
    stress_global: np.ndarray  # (m, 3) [sx, sy, txy]
    stress_material: np.ndarray  # (m, 3) [s11, s22, t12]
    shear_global: np.ndarray  # (m, 2) [t_yz, t_xz]
    shear_material: np.ndarray  # (m, 2) [t23, t13]
    failure_index: np.ndarray  # (m,) NaN when no criterion was given
    failure_mode: np.ndarray  # (m,) mode names ('' when no criterion)


@dataclass(frozen=True)
class ElementField:
    element_ids: np.ndarray
    centroid: np.ndarray
    area: np.ndarray
    N: np.ndarray
    M: np.ndarray
    Q: np.ndarray
    energy_membrane: np.ndarray
    energy_bending: np.ndarray
    energy_coupling: np.ndarray
    energy_shear: np.ndarray
    energy_drilling: np.ndarray
    energy_total: np.ndarray
    max_failure_index: np.ndarray  # NaN when no criterion was given
    critical_ply: np.ndarray  # -1 when no criterion was given


@dataclass(frozen=True)
class NodalField:
    node_ids: np.ndarray
    xyz: np.ndarray
    displacement: np.ndarray  # (n, 6)
    reaction: np.ndarray  # (n, 6)
    N: np.ndarray  # (n, 3) extrapolated and averaged
    M: np.ndarray
    Q: np.ndarray
    element_count: np.ndarray  # elements sharing the node


@dataclass(frozen=True)
class GlobalTotals:
    strain_energy: float
    energy_split: Dict[str, float]
    external_work: float
    applied_resultant: np.ndarray  # [Fx, Fy, Fz, Mx, My, Mz] about the origin
    reaction_resultant: np.ndarray
    equilibrium_residual: float  # |applied + reactions| / sum|nodal forces| (moments over model size)
    max_displacement: Dict[str, Tuple[float, int]]  # dof -> (signed value, node)
    max_failure: Optional[Tuple[float, int, int, str]]  # (index, element, ply, mode)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
def extrapolation_matrix(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    """(4, n_points) matrix mapping Gauss-point values to corner-node values.

    A bilinear field a + b xi + c eta + d xi eta is fitted to the point values in the
    least-squares sense and evaluated at the corners. For the 2x2 rule this is the
    classical exact extrapolation; for one point it returns the constant value.
    """
    P = np.array([[1.0, xi, eta, xi * eta] for xi, eta in points])
    corners = np.array([[1.0, x, e, x * e] for x, e in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    if len(points) < 4:
        return np.full((4, len(points)), 1.0 / len(points))
    return corners @ np.linalg.pinv(P)


class ResultsExtractor:
    """Builds result fields from a solved model. Fields are computed lazily and cached."""

    def __init__(
        self,
        solution: SolutionResult,
        load_case: Optional[LoadCase] = None,
        recoverer: Optional[ExactStressRecoverer] = None,
        criterion: Optional[FailureCriterion] = None,
    ):
        self.solution = solution
        self.mesh = solution.mesh
        self.load_case = load_case or solution.load_case or LoadCase()
        self.recoverer = recoverer or ExactStressRecoverer(self.mesh.hygrothermal_engine)
        self.criterion = criterion
        self._gauss: Optional[GaussPointField] = None
        self._elements: Optional[ElementField] = None
        self._plies: Optional[PlyPointField] = None

    # ------------------------------------------------------------- helpers
    def _state(self, element: CompositePlateElement) -> HygrothermalState:
        return self.load_case.state_for(element.element_id)

    def _section_resultants(self, element, eps0, kappa, gamma0):
        section = element.section
        N_ht, M_ht = element.hygrothermal_resultants(self._state(element))
        NM = section.abd @ np.concatenate([eps0, kappa]) - np.concatenate([N_ht, M_ht])
        return NM[:3], NM[3:], section.shear @ gamma0

    @staticmethod
    def ply_height(element: CompositePlateElement, ply: int, position: str) -> Tuple[float, str]:
        """z coordinate and interface side for a ply position: 'bottom', 'mid', 'top' or a float in [0, 1]."""
        z_b, z_t = element.laminate.ply_bounds()[ply]
        if position == "bottom":
            return z_b, "above"
        if position == "top":
            return z_t, "below"
        if position == "mid":
            return 0.5 * (z_b + z_t), "above"
        fraction = float(position)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("ply position fraction must lie in [0, 1]")
        return z_b + fraction * (z_t - z_b), ("below" if fraction == 1.0 else "above")

    # --------------------------------------------------------- Gauss level
    def gauss_points(self) -> GaussPointField:
        if self._gauss is not None:
            return self._gauss
        rows = {k: [] for k in ("eid", "idx", "nat", "xyz", "w", "e", "k", "g", "N", "M", "Q")}
        for element in self.mesh.elements:
            u = self.solution.element_displacements(element.element_id)
            for i, (xi, eta, w) in enumerate(element.gauss_points()):
                eps0, kappa, gamma0 = element.generalized_strains(u, xi, eta)
                N, M, Q = self._section_resultants(element, eps0, kappa, gamma0)
                for key, value in zip(rows, (element.element_id, i, (xi, eta), element.global_coordinates(xi, eta), w, eps0, kappa, gamma0, N, M, Q)):
                    rows[key].append(value)
        a = {k: np.array(v) for k, v in rows.items()}
        self._gauss = GaussPointField(a["eid"], a["idx"], a["nat"], a["xyz"], a["w"], a["e"], a["k"], a["g"], a["N"], a["M"], a["Q"])
        return self._gauss

    def ply_points(self, plies: Optional[Sequence[int]] = None, positions: Sequence = PLY_POSITIONS) -> PlyPointField:
        """Ply stresses (and failure indices if a criterion was given) at every Gauss point.

        The default call (all plies at bottom, mid and top) is cached.
        """
        default = plies is None and tuple(positions) == PLY_POSITIONS
        if default and self._plies is not None:
            return self._plies
        field_ = self._ply_points(plies, positions)
        if default:
            self._plies = field_
        return field_

    def _ply_points(self, plies, positions) -> PlyPointField:
        out = {k: [] for k in ("eid", "idx", "ply", "z", "xyz", "sg", "sm", "tg", "tm", "fi", "mode")}
        for element in self.mesh.elements:
            u = self.solution.element_displacements(element.element_id)
            state = self._state(element)
            ply_ids = range(element.laminate.n_plies) if plies is None else plies
            for i, (xi, eta, _) in enumerate(element.gauss_points()):
                xyz = element.global_coordinates(xi, eta)
                for k in ply_ids:
                    for position in positions:
                        z, side = self.ply_height(element, k, position)
                        p = self.recoverer.recover(element, u, z, xi, eta, state, side)
                        if self.criterion is not None:
                            res = self.criterion.evaluate(p.material_stress_3d(), element.laminate.plies[k].material.strength)
                            fi, mode = res.index, res.mode.value
                        else:
                            fi, mode = np.nan, ""
                        for key, value in zip(out, (element.element_id, i, k, z, xyz, p.stress_global, p.stress_material, p.shear_global, p.shear_material, fi, mode)):
                            out[key].append(value)
        a = {k: np.array(v) for k, v in out.items()}
        return PlyPointField(a["eid"], a["idx"], a["ply"], a["z"], a["xyz"], a["sg"], a["sm"], a["tg"], a["tm"], a["fi"], a["mode"])

    # ------------------------------------------------------- element level
    def elements(self) -> ElementField:
        if self._elements is not None:
            return self._elements
        gp = self.gauss_points()
        ids = self.mesh.element_ids
        n = len(ids)
        res = {k: np.zeros((n, d)) for k, d in (("N", 3), ("M", 3), ("Q", 2), ("c", 3))}
        energy = {k: np.zeros(n) for k in ("mem", "bend", "coup", "shear", "drill", "total")}
        area = np.zeros(n)
        fi = np.full(n, np.nan)
        crit = np.full(n, -1, dtype=int)
        for j, element in enumerate(self.mesh.elements):
            mask = gp.element_ids == element.element_id
            w = gp.weight[mask]
            area[j] = w.sum()
            res["c"][j] = w @ gp.xyz[mask] / area[j]
            res["N"][j] = w @ gp.N[mask] / area[j]
            res["M"][j] = w @ gp.M[mask] / area[j]
            res["Q"][j] = w @ gp.Q[mask] / area[j]
            s = element.section
            e0, k0, g0 = gp.eps0[mask], gp.kappa[mask], gp.gamma0[mask]
            energy["mem"][j] = 0.5 * np.sum(w * np.einsum("ni,ij,nj->n", e0, s.A, e0))
            energy["bend"][j] = 0.5 * np.sum(w * np.einsum("ni,ij,nj->n", k0, s.D, k0))
            energy["coup"][j] = np.sum(w * np.einsum("ni,ij,nj->n", e0, s.B, k0))
            energy["shear"][j] = 0.5 * np.sum(w * np.einsum("ni,ij,nj->n", g0, s.shear, g0))
            u = self.solution.element_displacements(element.element_id)
            u_local = element.transformation @ u
            drill = sum(wt * (element.drilling_B(xi, eta) @ u_local) ** 2 for xi, eta, wt in element.gauss_points())
            energy["drill"][j] = 0.5 * element.drilling_modulus * drill
            energy["total"][j] = element.strain_energy(u)
        if self.criterion is not None:
            pp = self.ply_points()
            for j, eid in enumerate(ids):
                mask = pp.element_ids == eid
                k = int(np.argmax(pp.failure_index[mask]))
                fi[j] = pp.failure_index[mask][k]
                crit[j] = pp.ply_index[mask][k]
        self._elements = ElementField(
            np.array(ids), res["c"], area, res["N"], res["M"], res["Q"],
            energy["mem"], energy["bend"], energy["coup"], energy["shear"], energy["drill"], energy["total"], fi, crit,
        )
        return self._elements

    # ---------------------------------------------------------- nodal level
    def _average_to_nodes(self, values_by_element: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """values_by_element[eid] has shape (n_gauss, k); returns node-averaged (n_nodes, k) and counts."""
        node_ids = self.mesh.node_ids
        index = {n: i for i, n in enumerate(node_ids)}
        width = next(iter(values_by_element.values())).shape[1]
        total = np.zeros((len(node_ids), width))
        count = np.zeros(len(node_ids))
        for element in self.mesh.elements:
            pts = [(xi, eta) for xi, eta, _ in element.gauss_points()]
            corner_values = extrapolation_matrix(pts) @ values_by_element[element.element_id]
            for local, nid in enumerate(element.node_ids):
                total[index[nid]] += corner_values[local]
                count[index[nid]] += 1
        return total / np.maximum(count, 1)[:, None], count

    def nodes(self) -> NodalField:
        gp = self.gauss_points()
        by_element = {
            eid: np.hstack([gp.N[gp.element_ids == eid], gp.M[gp.element_ids == eid], gp.Q[gp.element_ids == eid]])
            for eid in self.mesh.element_ids
        }
        averaged, count = self._average_to_nodes(by_element)
        sol = self.solution
        return NodalField(
            np.array(self.mesh.node_ids), self.mesh.coordinates_array(),
            sol.displacement_array(), sol.reactions.reshape(-1, 6).copy(),
            averaged[:, :3], averaged[:, 3:6], averaged[:, 6:8], count.astype(int),
        )

    def nodal_ply_stress(self, ply: int, position="mid", frame: str = "material") -> Tuple[np.ndarray, np.ndarray]:
        """Ply stresses extrapolated to the nodes and averaged: returns (node_ids, (n, 5) array).

        Columns are [s11, s22, t12, t23, t13] in the material frame or
        [sx, sy, txy, t_yz, t_xz] in the global frame.
        """
        if frame not in ("material", "global"):
            raise ValueError("frame must be 'material' or 'global'")
        pp = self.ply_points(plies=[ply], positions=[position])
        planar = pp.stress_material if frame == "material" else pp.stress_global
        shear = pp.shear_material if frame == "material" else pp.shear_global
        values = np.hstack([planar, shear])
        by_element = {eid: values[pp.element_ids == eid] for eid in self.mesh.element_ids}
        averaged, _ = self._average_to_nodes(by_element)
        return np.array(self.mesh.node_ids), averaged

    # --------------------------------------------------------- global level
    def _resultant(self, vector: np.ndarray) -> np.ndarray:
        xyz = self.mesh.coordinates_array()
        v = vector.reshape(-1, 6)
        force = v[:, :3].sum(axis=0)
        moment = (np.cross(xyz, v[:, :3]) + v[:, 3:]).sum(axis=0)
        return np.concatenate([force, moment])

    def totals(self) -> GlobalTotals:
        el = self.elements()
        split = {
            "membrane": float(el.energy_membrane.sum()),
            "bending": float(el.energy_bending.sum()),
            "coupling": float(el.energy_coupling.sum()),
            "shear": float(el.energy_shear.sum()),
            "drilling": float(el.energy_drilling.sum()),
        }
        applied = self._resultant(self.solution.load_vector)
        reaction = self._resultant(self.solution.reactions)
        forces = self.solution.load_vector.reshape(-1, 6)[:, :3]
        scale = max(np.abs(forces).sum(), np.linalg.norm(reaction[:3]), 1e-300)
        length = max(np.ptp(self.mesh.coordinates_array(), axis=0).max(), 1e-300)
        imbalance = applied + reaction
        residual = float(np.hypot(np.linalg.norm(imbalance[:3]), np.linalg.norm(imbalance[3:]) / length) / scale)
        disp = self.solution.displacement_array()
        node_ids = self.mesh.node_ids
        peaks = {}
        for d, name in enumerate(("UX", "UY", "UZ", "RX", "RY", "RZ")):
            i = int(np.argmax(np.abs(disp[:, d])))
            peaks[name] = (float(disp[i, d]), node_ids[i])
        max_failure = None
        if self.criterion is not None:
            pp = self.ply_points()
            i = int(np.argmax(pp.failure_index))
            max_failure = (float(pp.failure_index[i]), int(pp.element_ids[i]), int(pp.ply_index[i]), str(pp.failure_mode[i]))
        return GlobalTotals(
            strain_energy=float(el.energy_total.sum()),
            energy_split=split,
            external_work=self.solution.external_work,
            applied_resultant=applied,
            reaction_resultant=reaction,
            equilibrium_residual=residual,
            max_displacement=peaks,
            max_failure=max_failure,
        )

    # ---------------------------------------------------------------- export
    @staticmethod
    def _write_table(path: Path, columns: Dict[str, np.ndarray]) -> Path:
        names, data = [], []
        for name, values in columns.items():
            values = np.asarray(values)
            if values.ndim == 1:
                names.append(name)
                data.append(values)
            else:
                for c in range(values.shape[1]):
                    names.append(f"{name}_{c}")
                    data.append(values[:, c])
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(names)
            writer.writerows(zip(*data))
        return path

    def to_csv(self, directory, include_plies: bool = True) -> Dict[str, Path]:
        """Write every level to CSV files in ``directory``; returns {level: path}."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out = {}
        for level, obj in (("gauss_points", self.gauss_points()), ("elements", self.elements()), ("nodes", self.nodes())):
            out[level] = self._write_table(directory / f"{level}.csv", {f.name: getattr(obj, f.name) for f in fields(obj)})
        if include_plies:
            pp = self.ply_points()
            out["ply_points"] = self._write_table(directory / "ply_points.csv", {f.name: getattr(pp, f.name) for f in fields(pp)})
        t = self.totals()
        rows = [("strain_energy", t.strain_energy), ("external_work", t.external_work), ("equilibrium_residual", t.equilibrium_residual)]
        rows += [(f"energy_{k}", v) for k, v in t.energy_split.items()]
        rows += [(f"applied_{n}", v) for n, v in zip(("Fx", "Fy", "Fz", "Mx", "My", "Mz"), t.applied_resultant)]
        rows += [(f"reaction_{n}", v) for n, v in zip(("Fx", "Fy", "Fz", "Mx", "My", "Mz"), t.reaction_resultant)]
        rows += [(f"max_{k}", v[0]) for k, v in t.max_displacement.items()]
        path = directory / "totals.csv"
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerows([("quantity", "value"), *rows])
        out["totals"] = path
        return out

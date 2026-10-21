"""Transverse stresses from 3D equilibrium (post-processing).

Shell kinematics give accurate in-plane stresses but no transverse normal stress and,
for FSDT, a piecewise-constant transverse shear. Integrating the 3D equilibrium
equations through the thickness recovers both from the in-plane stress gradients:

    tau_xz(z) = - int_{z_b}^{z} (sigma_x,x + tau_xy,y) dz'
    tau_yz(z) = - int_{z_b}^{z} (tau_xy,x + sigma_y,y) dz'
    sigma_z(z) = sigma_z(z_b) - int_{z_b}^{z} (tau_xz,x + tau_yz,y) dz'

With sigma(z) = Qbar_k E(z) e the gradients follow from the derivatives of the
generalised strains e. A Quad4 element has no usable second derivatives of its own,
so the strains are first smoothed to the nodes (extrapolation from the recovery
points and averaging), differentiated with the element shape functions, smoothed
again and differentiated a second time. The profiles are exact polynomials in z
(quadratic shear, cubic normal stress per ply), continuous across every interface
for any kinematic theory.

The face conditions close the integrals: shear traction is zero on both faces and
sigma_z equals the applied pressure on the loaded face. Discretisation leaves a
small residual at the far face; it is removed by a linear correction through the
thickness so both faces satisfy their conditions exactly. The hygrothermal state is
assumed constant within each element (its strain gradients are ignored).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .laminate_elements import ShellElement
from .laminate_failure_criteria import DelaminationOnsetCriterion, FailureCriterion, FailureResult
from .laminate_hygrothermal import NULL_STATE, HygrothermalState
from .laminate_kinematics import strain_operators_at
from .laminate_solver_pipeline import LoadCase, SolutionResult
from .laminate_stress_recovery import ExactStressRecoverer, PointStressState, StressRecoverer


@dataclass(frozen=True)
class InterfaceCheck:
    """Delamination check at one ply interface (``interface`` k lies between plies k-1 and k)."""

    element_id: int
    interface: int
    z: float
    sigma_z: float
    tau_xz: float
    tau_yz: float
    result: FailureResult
    governing_ply: int


class EquilibriumStressRecoverer(StressRecoverer):
    """Adds sigma_z and equilibrium transverse shear to the constitutive in-plane stresses.

    ``pressure_face`` is the face the pressure acts on ("top" or "bottom"); the
    pressure is the load case's pressure at the point, positive along +normal.

    ``scale_to_shear_force`` keeps the equilibrium shape of tau(z) but scales each
    component so that its thickness integral equals the element's shear force
    Q = A_s gamma0. The element shear force is accurate right up to supports, where
    the recovered in-plane gradients are least reliable. It defaults to on for every
    theory except CLPT (whose shear force is a penalty artefact).
    """

    def __init__(
        self,
        solution: SolutionResult,
        load_case: Optional[LoadCase] = None,
        base: Optional[ExactStressRecoverer] = None,
        pressure_face: str = "top",
        scale_to_shear_force: Optional[bool] = None,
    ):
        if pressure_face not in ("top", "bottom"):
            raise ValueError("pressure_face must be 'top' or 'bottom'")
        self.scale_to_shear_force = scale_to_shear_force
        self.solution = solution
        self.mesh = solution.mesh
        self.load_case = load_case or solution.load_case or LoadCase()
        self.base = base or ExactStressRecoverer(self.mesh.hygrothermal_engine)
        self.pressure_face = pressure_face
        self._strain_nodes: Optional[Dict[int, np.ndarray]] = None
        self._gradient_nodes: Optional[Dict[int, np.ndarray]] = None

    # -------------------------------------------------------------- smoothing
    def _topology(self):
        """Elements around each node, the corner (vertex) nodes and the interior vertices."""
        if getattr(self, "_topo", None) is None:
            around: Dict[int, list] = {}
            edges: Dict[tuple, int] = {}
            vertices = set()
            for element in self.mesh.elements:
                for nid in element.node_ids:
                    around.setdefault(nid, []).append(element)
                vertices.update(element.node_ids[: element.N_CORNERS])
                for edge in element.EDGES:
                    key = tuple(sorted((element.node_ids[edge[0]], element.node_ids[edge[-1]])))
                    edges[key] = edges.get(key, 0) + 1
            boundary = {n for key, c in edges.items() if c == 1 for n in key}
            for element in self.mesh.elements:  # mid-side nodes of boundary edges
                for edge in element.EDGES:
                    key = tuple(sorted((element.node_ids[edge[0]], element.node_ids[edge[-1]])))
                    if edges[key] == 1:
                        boundary.update(element.node_ids[i] for i in edge)
            interior = {n for n in vertices if n not in boundary}
            self._topo = (around, interior)
        return self._topo

    def _smooth(self, values_at_points) -> Dict[int, np.ndarray]:
        """Superconvergent patch recovery (Zienkiewicz-Zhu) of a field sampled at the recovery points.

        Around every interior vertex a polynomial a + b s + c t + d s t (s, t in the
        tangent plane) is fitted to the recovery-point values of the surrounding
        elements and evaluated at the vertex. Boundary and mid-side nodes take the
        average of the neighbouring interior patches evaluated at their position, which
        keeps the recovered gradients accurate up to the edges.
        """
        around, interior = self._topology()
        samples = {}
        for element in self.mesh.elements:
            pts = np.array([element.global_coordinates(xi, eta) for xi, eta, _ in element.recovery_rule()])
            samples[element.element_id] = (pts, np.asarray(values_at_points(element), dtype=float))

        def fit(nid):
            elements = around[nid]
            frame = elements[0].frame
            origin = self.mesh.node_coordinates(nid)
            pts = np.vstack([samples[e.element_id][0] for e in elements])
            vals = np.vstack([samples[e.element_id][1] for e in elements])
            st = (pts - origin) @ frame[:2].T
            scale = max(np.abs(st).max(), 1e-300)
            st = st / scale
            P = np.column_stack([np.ones(len(st)), st[:, 0], st[:, 1], st[:, 0] * st[:, 1]])
            if len(st) < 6:
                P = P[:, :3] if len(st) >= 3 else P[:, :1]
            coef, *_ = np.linalg.lstsq(P, vals, rcond=None)
            return origin, frame, scale, coef

        def evaluate(patch, xyz):
            origin, frame, scale, coef = patch
            s, t = ((np.asarray(xyz) - origin) @ frame[:2].T) / scale
            row = np.array([1.0, s, t, s * t])[: coef.shape[0]]
            return row @ coef

        patches = {n: fit(n) for n in interior}
        out: Dict[int, np.ndarray] = {}
        for nid in self.mesh.node_ids:
            if nid not in around:
                continue
            xyz = self.mesh.node_coordinates(nid)
            if nid in patches:
                out[nid] = evaluate(patches[nid], xyz)
                continue
            neighbours = {m for e in around[nid] for m in e.node_ids if m in patches}
            if neighbours:
                out[nid] = np.mean([evaluate(patches[m], xyz) for m in sorted(neighbours)], axis=0)
            else:
                out[nid] = evaluate(fit(nid), xyz)
        return out

    def _nodal_strains(self) -> Dict[int, np.ndarray]:
        if self._strain_nodes is None:
            def strains(element):
                u = self.solution.element_displacements(element.element_id)
                return np.array([element.strain_state(u, xi, eta)[0] for xi, eta, _ in element.recovery_rule()])
            self._strain_nodes = self._smooth(strains)
        return self._strain_nodes

    def _element_field(self, element: ShellElement, nodal: Dict[int, np.ndarray]) -> np.ndarray:
        return np.array([nodal[n] for n in element.node_ids])

    def _first_derivatives(self, element: ShellElement, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
        values = self._element_field(element, self._nodal_strains())
        _, _, dNdx = element.jacobian(xi, eta)
        return dNdx[0] @ values, dNdx[1] @ values

    def _nodal_gradients(self) -> Dict[int, np.ndarray]:
        if self._gradient_nodes is None:
            def gradients(element):
                rows = []
                for xi, eta, _ in element.recovery_rule():
                    ex, ey = self._first_derivatives(element, xi, eta)
                    rows.append(np.concatenate([ex, ey]))
                return np.array(rows)
            self._gradient_nodes = self._smooth(gradients)
        return self._gradient_nodes

    def strain_derivatives(self, element: ShellElement, xi: float, eta: float):
        """(e_x, e_y, e_xx, e_xy, e_yy) of the smoothed generalised strain field (local axes)."""
        e_x, e_y = self._first_derivatives(element, xi, eta)
        grads = self._element_field(element, self._nodal_gradients())
        _, _, dNdx = element.jacobian(xi, eta)
        n = e_x.size
        d_x, d_y = grads[:, :n], grads[:, n:]
        e_xx = dNdx[0] @ d_x
        e_yy = dNdx[1] @ d_y
        e_xy = 0.5 * (dNdx[1] @ d_x + dNdx[0] @ d_y)
        return e_x, e_y, e_xx, e_xy, e_yy

    # ------------------------------------------------------------ integration
    def _segments(self, element: ShellElement):
        basis = element.section.basis
        lam = element.laminate
        z = lam.z_interfaces if basis is None else basis.z_nodes
        return [(float(z0), float(z1), lam.ply_index_at(0.5 * (z0 + z1))) for z0, z1 in zip(z[:-1], z[1:])]

    def _linear(self, element, k, z0, z1, rows_x, rows_y):
        """f(z) = rows_x(z) . a + rows_y(z) . b is linear on a segment: return (c0, c1) with f = c0 + c1 z."""
        Q = element.laminate.plies[k].Q_bar
        basis = element.section.basis

        def f(z):
            E, _ = strain_operators_at(basis, z, "above" if z < z1 else "below")
            QE = Q @ E
            return rows_x(QE), rows_y(QE)

        fa, fb = f(z0), f(z1)
        out = []
        for va, vb in zip(fa, fb):
            c1 = (vb - va) / (z1 - z0)
            out.append((va - c1 * z0, c1))
        return out

    def transverse_stresses(self, element: ShellElement, xi: float, eta: float, z: float) -> Tuple[float, float, float]:
        """(tau_yz, tau_xz, sigma_z) at height z, corrected to satisfy both face conditions."""
        e_x, e_y, e_xx, e_xy, e_yy = self.strain_derivatives(element, xi, eta)
        lam = element.laminate
        z_b, z_t = float(lam.z_interfaces[0]), float(lam.z_interfaces[-1])
        p = self.load_case.pressure_on(element.element_id)
        p = float(p(element.global_coordinates(xi, eta))) if callable(p) else float(p)
        s_bottom, s_top = (0.0, p) if self.pressure_face == "top" else (-p, 0.0)

        def integrate(z_end: float):
            tx = ty = 0.0  # tau_xz, tau_yz
            G = 0.0  # int g dz
            sz = s_bottom
            for z0, z1, k in self._segments(element):
                if z0 >= z_end:
                    break
                zz = min(z1, z_end)
                (ax, bx), (ay, by), (cg, dg) = self._linear(
                    element, k, z0, z1,
                    lambda QE: QE[0] @ e_x + QE[2] @ e_y,
                    lambda QE: QE[2] @ e_x + QE[1] @ e_y,
                ) + self._linear(element, k, z0, z1, lambda QE: QE[0] @ e_xx + 2.0 * QE[2] @ e_xy + QE[1] @ e_yy, lambda QE: 0.0)[:1]
                tx -= ax * (zz - z0) + bx * (zz**2 - z0**2) / 2.0
                ty -= ay * (zz - z0) + by * (zz**2 - z0**2) / 2.0
                sz += G * (zz - z0) + cg * (zz - z0) ** 2 / 2.0 + dg * ((zz**3 - z0**3) / 6.0 - z0**2 * (zz - z0) / 2.0)
                G += cg * (zz - z0) + dg * (zz**2 - z0**2) / 2.0
            return tx, ty, sz

        tx, ty, sz = integrate(z)
        tx_t, ty_t, sz_t = integrate(z_t)
        frac = (z - z_b) / (z_t - z_b)
        tx -= tx_t * frac
        ty -= ty_t * frac
        sz += (s_top - sz_t) * frac
        if self._scales(element):
            fx, fy = self._shear_scale(element, xi, eta, integrate, tx_t, ty_t, z_b, z_t)
            tx, ty = tx * fx, ty * fy
        return ty, tx, sz

    def _scales(self, element: ShellElement) -> bool:
        if self.scale_to_shear_force is not None:
            return bool(self.scale_to_shear_force)
        return element.kinematics.name != "CLPT"

    def _shear_scale(self, element, xi, eta, integrate, tx_t, ty_t, z_b, z_t):
        """Factors making int tau dz equal the element shear force [Q_xz, Q_yz]."""
        u = self.solution.element_displacements(element.element_id)
        e, s = element.strain_state(u, xi, eta)
        _, _, Q = element.section_resultants(e, s)
        pts, wts = np.polynomial.legendre.leggauss(4)
        ix = iy = 0.0
        for z0, z1, _ in self._segments(element):
            half, mid = 0.5 * (z1 - z0), 0.5 * (z1 + z0)
            for g, w in zip(pts, wts):
                zz = mid + half * g
                txz, tyz, _ = integrate(zz)
                f = (zz - z_b) / (z_t - z_b)
                ix += (txz - tx_t * f) * w * half
                iy += (tyz - ty_t * f) * w * half
        fx = Q[1] / ix if abs(ix) > 1e-12 * (abs(Q[1]) + 1e-300) and abs(ix) > 0.0 else 1.0
        fy = Q[0] / iy if abs(iy) > 1e-12 * (abs(Q[0]) + 1e-300) and abs(iy) > 0.0 else 1.0
        return fx, fy

    # ------------------------------------------------------------- interface
    def recover(
        self,
        element: ShellElement,
        u_elem: np.ndarray,
        z: float,
        xi: float = 0.0,
        eta: float = 0.0,
        state: HygrothermalState = NULL_STATE,
        side: str = "above",
    ) -> PointStressState:
        point = self.base.recover(element, u_elem, z, xi, eta, state, side)
        tyz, txz, sz = self.transverse_stresses(element, xi, eta, z)
        ply = element.laminate.plies[point.ply_index]
        tau = np.array([tyz, txz])
        return PointStressState(
            element_id=point.element_id,
            ply_index=point.ply_index,
            z=point.z,
            xi=point.xi,
            eta=point.eta,
            strain_total=point.strain_total,
            strain_hygrothermal=point.strain_hygrothermal,
            strain_mechanical=point.strain_mechanical,
            stress_global=point.stress_global,
            stress_material=point.stress_material,
            shear_global=tau,
            shear_material=ply.to_material_shear(tau),
            normal_stress=float(sz),
        )

    # ------------------------------------------------------------ delamination
    def interface_checks(
        self,
        element: ShellElement,
        xi: float = 0.0,
        eta: float = 0.0,
        criterion: Optional[FailureCriterion] = None,
    ) -> List[InterfaceCheck]:
        """Evaluate a delamination criterion at every internal ply interface.

        The interlaminar stresses are continuous, but their material-axis components
        and the strengths differ between the two plies, so the worse of the two plies
        governs each interface.
        """
        criterion = criterion or DelaminationOnsetCriterion()
        plies = element.laminate.plies
        out = []
        for k in range(1, len(plies)):
            z = float(element.laminate.z_interfaces[k])
            tyz, txz, sz = self.transverse_stresses(element, xi, eta, z)
            best = None
            for j in (k - 1, k):
                t23, t13 = plies[j].to_material_shear(np.array([tyz, txz]))
                stress = np.array([0.0, 0.0, sz, t23, t13, 0.0])
                result = criterion.evaluate(stress, plies[j].material.strength)
                if best is None or result.index > best[0].index:
                    best = (result, j)
            out.append(InterfaceCheck(element.element_id, k, z, float(sz), float(txz), float(tyz), best[0], best[1]))
        return out

    def critical_interface(self, criterion: Optional[FailureCriterion] = None, points=None) -> InterfaceCheck:
        """The highest delamination index over all elements (at the element recovery points)."""
        worst = None
        for element in self.solution.mesh.elements:
            sample = points if points is not None else [tuple(p[:2]) for p in element.gauss_points()]
            for xi, eta in sample:
                for check in self.interface_checks(element, xi, eta, criterion):
                    if worst is None or check.result.index > worst.result.index:
                        worst = check
        if worst is None:
            raise ValueError("The mesh has no ply interfaces (single-ply laminates cannot delaminate)")
        return worst

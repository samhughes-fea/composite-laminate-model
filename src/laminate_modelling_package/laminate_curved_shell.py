"""Curved quadratic shell elements: 8-node quad and 6-node triangle.

The flat elements project their nodes onto one mean plane, so a curved surface is
approximated by facets. These elements follow the curved geometry of their quadratic
interpolation instead. Every integration point has its own tangent frame (e1, e2,
e3) and the shell director d is interpolated from the surface normals at the nodes,
in the manner of degenerated-solid shells (Ahmad, Irons and Zienkiewicz) reduced to
stress resultants:

    membrane    eps_ab  = sym(e_a . u,b)
    bending     kap_ab  = sym(e_a . beta,b + d,a . u,b),   beta = sum N_i (theta_i x d_i)
    shear       gam_a3  = e_a . beta + d . u,a
    drilling    theta . e3 = (1/2)(e2 . u,x - e1 . u,y)   (penalty)

where derivatives are taken along the local tangent axes. The term d,a . u,b is the
curvature coupling: without it a rigid rotation of a curved element would bend it.
Rotations theta are global vectors, so no nodal frames are needed for the six
standard DOFs. Enrichment amplitudes of full RZT and layerwise models are stored in
a tangent frame at each node (the material axis projected normal to the nodal
director).

Integration is selective. The quad integrates membrane and transverse shear with
2 x 2 Gauss points, which removes membrane and shear locking in thin curved shells,
and bending, drilling and enrichment terms with 3 x 3 (``integration="selective"``,
the default; ``"full"`` uses 3 x 3 for the membrane too, ``"reduced"`` 2 x 2 for
everything, as S8R). The membrane part moved to 2 x 2 includes the membrane-bending
coupling of offset or unsymmetric laminates, so the element stiffness stays positive
semi-definite. A single selective quad keeps the serendipity quad's one
non-communicable membrane mode, as S8R does; a mesh of two or more has exactly six
rigid-body modes. The triangle uses
a 6-point rule for bending and 3 points for membrane and shear.
Results are sampled at the 2 x 2 points (quad) or the 3 points (triangle).

Nodal directors come from each element's own geometry. On a smooth surface meshed
with quadratic elements they agree closely between neighbours.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .laminate_elements import STANDARD_DOFS, ShellElement, _direction, _tensor_rule, _Q8_NODES

_TRI6_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
_TRI3_POINT = [(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)]


def _skew(v: np.ndarray) -> np.ndarray:
    """S(v) with S(v) w = v x w."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


class CurvedShellElement(ShellElement):
    """Base of the curved shells: point-wise frames and global-rotation kinematics."""

    CURVED = True

    def _check_order(self, order):
        return order

    # ---------------------------------------------------------------- geometry
    def _tangent_frame(self, xyz: np.ndarray, normal: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        e3 = normal / np.linalg.norm(normal)
        reference = _direction(self.material_axis, xyz)
        e1 = reference - np.dot(reference, e3) * e3
        if np.linalg.norm(e1) < 1e-8 * max(np.linalg.norm(reference), 1e-300):
            e1 = fallback - np.dot(fallback, e3) * e3
        e1 = e1 / np.linalg.norm(e1)
        return np.vstack([e1, np.cross(e3, e1), e3])

    def _surface(self, xi: float, eta: float):
        N, dN = self.shape_functions(xi, eta)
        a = dN @ self.coordinates  # covariant base vectors a_xi, a_eta
        return N, dN, a, np.cross(a[0], a[1])

    def _build_local_frame(self) -> None:
        self._points: Dict[Tuple[float, float], dict] = {}
        X = self.coordinates
        directors, frames = [], []
        for xi, eta in self.NATURAL_NODES:
            _, _, a, n = self._surface(xi, eta)
            if np.linalg.norm(n) <= 0.0:
                raise ValueError(f"Element {self.element_id} is degenerate at a node")
            frame = self._tangent_frame(self.global_coordinates(xi, eta), n, a[0])
            directors.append(frame[2])
            frames.append(frame[:2])
        self.directors = np.array(directors)
        self.nodal_frames = np.array(frames)  # (n, 2, 3): enrichment axes t1, t2 per node
        centre = self.global_coordinates(*self.NATURAL_CENTROID)
        _, _, a, n = self._surface(*self.NATURAL_CENTROID)
        self.frame = self._tangent_frame(centre, n, a[0])
        local = (X - centre) @ self.frame.T
        self.local_xy = local[:, :2]
        # deviation of the nodes from the centroid tangent plane, reported for information;
        # curved elements are meant to be curved, so no warping warning is raised.
        self.curvature_deviation = float(np.max(np.abs(local[:, 2])))
        self.warping = 0.0
        for xi, eta, _ in self.stiffness_rule() + self.shear_rule():
            if self._point(xi, eta)["detJ"] <= 0.0:
                raise ValueError(f"Element {self.element_id} has a non-positive Jacobian; check node ordering")

    def _point(self, xi: float, eta: float) -> dict:
        key = (round(float(xi), 13), round(float(eta), 13))
        cached = self._points.get(key)
        if cached is not None:
            return cached
        N, dN, a, n = self._surface(xi, eta)
        frame = self._tangent_frame(N @ self.coordinates, n, a[0])
        J = a @ frame[:2].T  # J[r, c] = a_r . e_c
        dNdx = np.linalg.solve(J, dN)
        d = N @ self.directors
        point = {
            "N": N,
            "dNdx": dNdx,
            "J": J,
            "detJ": float(np.linalg.det(J)),
            "frame": frame,
            "d": d,
            "dd": dNdx @ self.directors,  # d,x ; d,y
        }
        self._points[key] = point
        return point

    def jacobian(self, xi: float, eta: float):
        p = self._point(xi, eta)
        return p["J"], p["detJ"], p["dNdx"]

    def point_frame(self, xi: float, eta: float) -> np.ndarray:
        """Rows e1, e2, e3 of the tangent frame at (xi, eta)."""
        return self._point(xi, eta)["frame"]

    def _normal_estimate(self) -> np.ndarray:  # pragma: no cover - not used by curved elements
        return self._surface(*self.NATURAL_CENTROID)[3]

    @property
    def transformation(self) -> np.ndarray:
        """Identity: curved elements work directly in global DOFs."""
        return np.eye(self.n_dof)

    # --------------------------------------------------------------- operators
    def _standard_rows(self, xi: float, eta: float) -> np.ndarray:
        """9 x 6n rows on the standard DOFs: membrane (3), bending (3), shear [yz, xz] (2), drilling (1)."""
        p = self._point(xi, eta)
        N, dNdx, d, dd = p["N"], p["dNdx"], p["d"], p["dd"]
        e1, e2, e3 = p["frame"]
        n = self.N_NODES
        B = np.zeros((9, 6 * n))
        for i in range(n):
            u, t = slice(6 * i, 6 * i + 3), slice(6 * i + 3, 6 * i + 6)
            gx, gy = dNdx[0, i], dNdx[1, i]
            M = -_skew(self.directors[i])  # theta_i x d_i = M theta_i
            e1M, e2M = e1 @ M, e2 @ M
            B[0, u] = gx * e1
            B[1, u] = gy * e2
            B[2, u] = gy * e1 + gx * e2
            B[3, t] = gx * e1M
            B[3, u] = gx * dd[0]
            B[4, t] = gy * e2M
            B[4, u] = gy * dd[1]
            B[5, t] = gy * e1M + gx * e2M
            B[5, u] = gx * dd[1] + gy * dd[0]
            B[6, u] = gy * d
            B[6, t] = N[i] * e2M
            B[7, u] = gx * d
            B[7, t] = N[i] * e1M
            B[8, t] = N[i] * e3
            B[8, u] = -0.5 * (gx * e2 - gy * e1)
        return B

    def _enrichment_projection(self, xi: float, eta: float) -> np.ndarray:
        """(n, 2, 2) components of the nodal enrichment axes in the point frame: P[i, r, c] = e_r . t_c,i."""
        frame = self._point(xi, eta)["frame"]
        return np.einsum("rk,ick->irc", frame[:2], self.nodal_frames)

    def in_plane_operator(self, xi: float, eta: float) -> np.ndarray:
        m, d = self.n_enrichment, self.dofs_per_node
        B = np.zeros((6 + 4 * m, self.n_dof))
        B[:6] = self._expand_standard(self._standard_rows(xi, eta)[:6])
        if m:
            dNdx = self._point(xi, eta)["dNdx"]
            P = self._enrichment_projection(xi, eta)
            for j in range(m):
                for i in range(self.N_NODES):
                    cols = slice(d * i + 6 + 2 * j, d * i + 8 + 2 * j)
                    B[6 + 4 * j, cols] = dNdx[0, i] * P[i, 0]
                    B[7 + 4 * j, cols] = dNdx[1, i] * P[i, 0]
                    B[8 + 4 * j, cols] = dNdx[0, i] * P[i, 1]
                    B[9 + 4 * j, cols] = dNdx[1, i] * P[i, 1]
        return B

    def _gamma0_operator(self, xi: float, eta: float) -> np.ndarray:
        return self._standard_rows(xi, eta)[6:8]

    def shear_operator(self, xi: float, eta: float) -> np.ndarray:
        m, d = self.n_enrichment, self.dofs_per_node
        B = np.zeros((2 + 2 * m, self.n_dof))
        B[:2] = self._expand_standard(self._gamma0_operator(xi, eta))
        if m:
            N = self._point(xi, eta)["N"]
            P = self._enrichment_projection(xi, eta)
            for j in range(m):
                for i in range(self.N_NODES):
                    cols = slice(d * i + 6 + 2 * j, d * i + 8 + 2 * j)
                    B[2 + 2 * j, cols] = N[i] * P[i, 0]
                    B[3 + 2 * j, cols] = N[i] * P[i, 1]
        return B

    def drilling_B(self, xi: float, eta: float) -> np.ndarray:
        return self._expand_standard(self._standard_rows(xi, eta)[8:9])[0]

    def membrane_rule(self):
        """Points for the membrane-membrane stiffness (reduced to avoid membrane locking)."""
        return self.stiffness_rule()

    def local_stiffness_matrix(self) -> np.ndarray:
        if self._K_local is not None:
            return self._K_local
        K = super().local_stiffness_matrix().copy()
        full, reduced = self.stiffness_rule(), self.membrane_rule()
        if reduced != full:
            # The section energy splits into two non-negative parts,
            #   e^T C e = m^T A m + r^T (C_rr - C_rm A^-1 C_mr) r,   m = e_m + A^-1 C_mr r,
            # with e_m the membrane strains and r the rest (curvatures, enrichment). Only
            # the first part moves to the reduced rule. Swapping the bare membrane term
            # instead leaves the membrane-bending coupling of offset or unsymmetric
            # laminates on the full rule, and the element stiffness can turn indefinite.
            C = self.section.C_ee
            A = C[:3, :3]
            G = np.linalg.solve(A, C[:3, 3:])
            for rule, sign in ((full, -1.0), (reduced, 1.0)):
                for xi, eta, w in rule:
                    B_e = self.in_plane_operator(xi, eta)
                    B_m = B_e[:3] + G @ B_e[3:]
                    K += sign * (B_m.T @ A @ B_m) * (w * self._point(xi, eta)["detJ"])
        self._K_local = 0.5 * (K + K.T)
        return self._K_local

    # ------------------------------------------------------------------- loads
    def pressure_load_vector(self, pressure) -> np.ndarray:
        F = np.zeros(self.n_dof)
        if not callable(pressure) and pressure == 0.0:
            return F
        d = self.dofs_per_node
        for xi, eta, w in self.load_rule():
            p_ = self._point(xi, eta)
            xyz = p_["N"] @ self.coordinates
            p = float(pressure(xyz)) if callable(pressure) else float(pressure)
            f = p * p_["frame"][2] * p_["detJ"] * w
            for i in range(self.N_NODES):
                F[d * i : d * i + 3] += p_["N"][i] * f
        return F

    def body_load_vector(self, acceleration) -> np.ndarray:
        a = np.asarray(acceleration, dtype=float)
        f = self._laminate.areal_density() * a
        F = np.zeros(self.n_dof)
        d = self.dofs_per_node
        for xi, eta, w in self.load_rule():
            p_ = self._point(xi, eta)
            m = self._laminate.offset * np.cross(p_["frame"][2], f)
            dA = p_["detJ"] * w
            for i in range(self.N_NODES):
                F[d * i : d * i + 3] += p_["N"][i] * f * dA
                F[d * i + 3 : d * i + 6] += p_["N"][i] * m * dA
        return F

    def quality(self) -> Dict[str, float]:
        q = super().quality()
        q["curvature_ratio"] = self.curvature_deviation / max(self.characteristic_length, 1e-300)
        return q


class CurvedQuad8ShellElement(CurvedShellElement):
    """8-node curved serendipity shell (S8R node order: corners, then mid-sides)."""

    TOPOLOGY = "quad8_curved"
    N_NODES = 8
    N_CORNERS = 4
    NATURAL_NODES = _Q8_NODES
    NATURAL_CENTROID = (0.0, 0.0)
    EDGES = ((0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0))

    def __init__(self, *args, integration: str = "selective", **kwargs):
        if integration not in ("selective", "reduced", "full"):
            raise ValueError("integration must be 'selective', 'reduced' or 'full'")
        self.integration = integration
        super().__init__(*args, **kwargs)

    def _options(self) -> dict:
        return {**super()._options(), "integration": self.integration}

    def shape_functions(self, xi, eta):
        N = np.zeros(8)
        dN = np.zeros((2, 8))
        for i, (a, b) in enumerate(_Q8_NODES):
            if i < 4:
                N[i] = 0.25 * (1 + a * xi) * (1 + b * eta) * (a * xi + b * eta - 1)
                dN[0, i] = 0.25 * a * (1 + b * eta) * (2 * a * xi + b * eta)
                dN[1, i] = 0.25 * b * (1 + a * xi) * (a * xi + 2 * b * eta)
            elif a == 0.0:
                N[i] = 0.5 * (1 - xi * xi) * (1 + b * eta)
                dN[0, i] = -xi * (1 + b * eta)
                dN[1, i] = 0.5 * b * (1 - xi * xi)
            else:
                N[i] = 0.5 * (1 + a * xi) * (1 - eta * eta)
                dN[0, i] = 0.5 * a * (1 - eta * eta)
                dN[1, i] = -eta * (1 + a * xi)
        return N, dN

    def stiffness_rule(self):
        return _tensor_rule(2 if self.integration == "reduced" else 3)

    def membrane_rule(self):
        return _tensor_rule(3 if self.integration == "full" else 2)

    def shear_rule(self):
        return _tensor_rule(2)

    def load_rule(self):
        return _tensor_rule(3)

    def recovery_rule(self):
        return _tensor_rule(2)

    def recovery_basis(self, xi, eta):
        return np.array([1.0, xi, eta, xi * eta])


class Tri6ShellElement(CurvedShellElement):
    """6-node curved triangle (Gmsh / CalculiX S6 node order: corners, then mid-sides 12, 23, 31)."""

    TOPOLOGY = "tri6"
    N_NODES = 6
    N_CORNERS = 3
    NATURAL_NODES = _TRI6_NODES
    NATURAL_CENTROID = (1.0 / 3.0, 1.0 / 3.0)
    EDGES = ((0, 3, 1), (1, 4, 2), (2, 5, 0))

    def shape_functions(self, xi, eta):
        r, s = xi, eta
        t = 1.0 - r - s
        N = np.array([t * (2 * t - 1), r * (2 * r - 1), s * (2 * s - 1), 4 * r * t, 4 * r * s, 4 * s * t])
        dN = np.array(
            [
                [-(4 * t - 1), 4 * r - 1, 0.0, 4 * (t - r), 4 * s, -4 * s],
                [-(4 * t - 1), 0.0, 4 * s - 1, -4 * r, 4 * r, 4 * (t - s)],
            ]
        )
        return N, dN

    def stiffness_rule(self):
        return self.load_rule()

    def shear_rule(self):
        return list(_TRI3_POINT)

    def membrane_rule(self):
        return list(_TRI3_POINT)

    def recovery_rule(self):
        return list(_TRI3_POINT)

    def load_rule(self):
        # 6-point (degree 4) rule for consistent pressure and body loads
        a, b = 0.445948490915965, 0.091576213509771
        wa, wb = 0.223381589678011 / 2, 0.109951743655322 / 2
        return [(a, a, wa), (1 - 2 * a, a, wa), (a, 1 - 2 * a, wa), (b, b, wb), (1 - 2 * b, b, wb), (b, 1 - 2 * b, wb)]

    def recovery_basis(self, xi, eta):
        return np.array([1.0, xi, eta])

    # MITC6-type assumed transverse shear (rotated Raviart-Thomas space of degree 1)
    _G = 0.5 / np.sqrt(3.0)
    _EDGE_POINTS = (0.5 - _G, 0.5 + _G)

    @staticmethod
    def _rt1(r: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
        """Basis of the assumed covariant shear: e_r = phi_r . c, e_s = phi_s . c."""
        q = (s * r, s * s)
        phi_r = np.array([1.0, r, s, 0.0, 0.0, 0.0, q[0], q[1]])
        phi_s = np.array([0.0, 0.0, 0.0, 1.0, r, s, -r * r, -r * s])
        return phi_r, phi_s

    def _covariant_rows(self, r: float, s: float) -> np.ndarray:
        """Displacement-based covariant shear rows [e_r; e_s] (2 x 6n) at (r, s)."""
        rows = self._standard_rows(r, s)[6:8]  # [gamma_yz; gamma_xz] in the point frame
        J = self._point(r, s)["J"]
        return J @ rows[[1, 0]]

    def _mitc_coefficients(self) -> np.ndarray:
        cached = getattr(self, "_mitc_coeff", None)
        if cached is not None:
            return cached
        A, R = [], []
        for t in self._EDGE_POINTS:
            pr, ps = self._rt1(t, 0.0)  # edge 1-2: tangential e_r
            A.append(pr)
            R.append(self._covariant_rows(t, 0.0)[0])
            pr, ps = self._rt1(0.0, t)  # edge 3-1: tangential e_s
            A.append(ps)
            R.append(self._covariant_rows(0.0, t)[1])
            pr, ps = self._rt1(1.0 - t, t)  # edge 2-3: tangential e_s - e_r
            A.append(ps - pr)
            rows = self._covariant_rows(1.0 - t, t)
            R.append(rows[1] - rows[0])
        mean_r, mean_s = np.zeros(8), np.zeros(8)
        raw_r, raw_s = np.zeros(6 * self.N_NODES), np.zeros(6 * self.N_NODES)
        for r, s, w in self.load_rule():
            pr, ps = self._rt1(r, s)
            rows = self._covariant_rows(r, s)
            mean_r += pr * w
            mean_s += ps * w
            raw_r += rows[0] * w
            raw_s += rows[1] * w
        A += [mean_r, mean_s]
        R += [raw_r, raw_s]
        self._mitc_coeff = np.linalg.solve(np.array(A), np.array(R))
        return self._mitc_coeff

    def _gamma0_operator(self, xi, eta):
        """MITC6-type transverse shear: covariant strains tied to their tangential values
        at two points per edge and to their element means (a rotated Raviart-Thomas
        field of degree 1), which removes shear locking of the quadratic triangle."""
        c = self._mitc_coefficients()
        pr, ps = self._rt1(xi, eta)
        covariant = np.vstack([pr @ c, ps @ c])
        cartesian = np.linalg.solve(self._point(xi, eta)["J"], covariant)  # [gamma_xz; gamma_yz]
        return cartesian[[1, 0]]


__all__ = ["CurvedShellElement", "CurvedQuad8ShellElement", "Tri6ShellElement"]


def _register() -> None:
    from .laminate_elements import DEFAULT_ELEMENTS

    if "quad8_curved" not in DEFAULT_ELEMENTS.names():
        DEFAULT_ELEMENTS.register(CurvedQuad8ShellElement)
        DEFAULT_ELEMENTS.register(Tri6ShellElement)


_register()

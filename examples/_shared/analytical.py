"""Closed-form Navier solutions for simply supported cross-ply plates (Reddy, 2004)."""

import numpy as np


def navier_clpt_centre(D, a, b, q, terms=61):
    """Centre deflection under uniform pressure q, classical (Kirchhoff) plate theory."""
    w = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            al, be = m * np.pi / a, n * np.pi / b
            denom = D[0, 0] * al**4 + 2 * (D[0, 1] + 2 * D[2, 2]) * al**2 * be**2 + D[1, 1] * be**4
            w += 16 * q / (np.pi**2 * m * n) / denom * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
    return w


def navier_clpt_centre_moments(D, a, b, q, terms=199):
    """Centre bending moments [Mx, My] under uniform pressure q (classical plate theory)."""
    Mx = My = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            al, be = m * np.pi / a, n * np.pi / b
            W = 16 * q / (np.pi**2 * m * n) / (D[0, 0] * al**4 + 2 * (D[0, 1] + 2 * D[2, 2]) * al**2 * be**2 + D[1, 1] * be**4)
            s = np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
            Mx += (D[0, 0] * al**2 + D[0, 1] * be**2) * W * s
            My += (D[0, 1] * al**2 + D[1, 1] * be**2) * W * s
    return np.array([Mx, My])


def navier_fsdt_centre(D, As, a, b, q, terms=61):
    """Centre deflection under uniform pressure q, first-order shear deformation theory.

    ``As`` is the 2x2 transverse shear stiffness ordered [yz, xz].
    """
    A44, A55 = As[0, 0], As[1, 1]
    w = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            al, be = m * np.pi / a, n * np.pi / b
            K = np.array(
                [
                    [A55 * al**2 + A44 * be**2, A55 * al, A44 * be],
                    [A55 * al, D[0, 0] * al**2 + D[2, 2] * be**2 + A55, (D[0, 1] + D[2, 2]) * al * be],
                    [A44 * be, (D[0, 1] + D[2, 2]) * al * be, D[2, 2] * al**2 + D[1, 1] * be**2 + A44],
                ]
            )
            W = np.linalg.solve(K, [16 * q / (np.pi**2 * m * n), 0.0, 0.0])[0]
            w += W * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
    return w


def navier_clpt_moment_field(D, a, b, q, x, y, terms=99):
    """Bending moments [Mx, My] at points (x, y) under uniform pressure q (classical plate theory)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    Mx, My = np.zeros_like(x), np.zeros_like(x)
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            al, be = m * np.pi / a, n * np.pi / b
            W = 16 * q / (np.pi**2 * m * n) / (D[0, 0] * al**4 + 2 * (D[0, 1] + 2 * D[2, 2]) * al**2 * be**2 + D[1, 1] * be**4)
            s = np.sin(al * x) * np.sin(be * y)
            Mx += (D[0, 0] * al**2 + D[0, 1] * be**2) * W * s
            My += (D[0, 1] * al**2 + D[1, 1] * be**2) * W * s
    return np.stack([Mx, My], axis=-1)


def _rotated_stiffness_3d(laminate):
    from laminate_modelling_package.laminate_materials_3d import rotate_stiffness_3d

    return [rotate_stiffness_3d(p.material.elastic.stiffness_matrix(), p.angle) for p in laminate.plies]


def elasticity_3d_uniform(laminate, a, b, q, x, y, z, ply, terms=31, elements_per_ply=16):
    """3D elasticity solution of a simply supported cross-ply plate under uniform pressure q.

    The pressure is expanded in a double sine series, q = sum 16 q / (pi^2 m n) sin sin
    over odd m, n up to ``terms``. Each term is Pagano's problem (diaphragm supports,
    cross-ply plies), whose through-thickness amplitudes U(z), V(z), W(z) are solved here
    with quadratic finite elements in z (``elements_per_ply`` per ply). Unlike the
    transfer-matrix solution in ``PaganoPlate``, this stays well conditioned for the
    short wavelengths of the high terms. For m = n = 1 it reproduces ``PaganoPlate``
    to about 1e-5.

    Stresses are evaluated at the in-plane point (x, y) and heights ``z``, each taken in
    ply ``ply[i]`` (so interfaces can be sampled from either side). Returns a dict of
    arrays sigma_x, sigma_y, sigma_z, tau_xz, tau_yz, and the centre deflection
    w_centre. The pressure acts on the top face along +z, as in ``PaganoPlate``.
    """
    C = _rotated_stiffness_3d(laminate)
    z = np.asarray(z, dtype=float)
    ply = np.asarray(ply, dtype=int)
    # quadratic elements through each ply
    nodes, el_ply, conn = [laminate.ply_bounds()[0][0]], [], []
    for k, (zb, zt) in enumerate(laminate.ply_bounds()):
        edges = np.linspace(zb, zt, elements_per_ply + 1)
        for z0, z1 in zip(edges[:-1], edges[1:]):
            i0 = len(nodes) - 1
            nodes += [0.5 * (z0 + z1), z1]
            conn.append((i0, i0 + 1, i0 + 2))
            el_ply.append(k)
    nodes = np.array(nodes)
    n_dof = 3 * len(nodes)
    gp, gw = np.polynomial.legendre.leggauss(3)

    def parts(s, jac):
        """Strain amplitudes [e11, e22, e33, g23, g13, g12] = (B0 + p Bp + q Bq) X_e for
        u = U cos(px) sin(qy), v = V sin(px) cos(qy), w = W sin(px) sin(qy)."""
        Nv = np.array([0.5 * s * (s - 1), 1 - s * s, 0.5 * s * (s + 1)])
        dN = np.array([s - 0.5, -2 * s, s + 0.5]) / jac
        B0, Bp, Bq = np.zeros((6, 9)), np.zeros((6, 9)), np.zeros((6, 9))
        for i in range(3):
            u, v, w = 3 * i, 3 * i + 1, 3 * i + 2
            B0[2, w], B0[3, v], B0[4, u] = dN[i], dN[i], dN[i]
            Bp[0, u], Bp[4, w], Bp[5, v] = -Nv[i], Nv[i], Nv[i]
            Bq[1, v], Bq[3, w], Bq[5, u] = -Nv[i], Nv[i], Nv[i]
        return B0, Bp, Bq

    # K(p, q) = K00 + p (K0p) + q (K0q) + p^2 Kpp + pq Kpq + q^2 Kqq
    K = {key: np.zeros((n_dof, n_dof)) for key in ("00", "0p", "0q", "pp", "pq", "qq")}
    for c, k in zip(conn, el_ply):
        jac = 0.5 * (nodes[c[2]] - nodes[c[0]])
        dofs = np.array([3 * i + j for i in c for j in range(3)])
        idx = np.ix_(dofs, dofs)
        for s, wt in zip(gp, gw):
            B0, Bp, Bq = parts(s, jac)
            f = wt * jac
            K["00"][idx] += B0.T @ C[k] @ B0 * f
            K["0p"][idx] += (B0.T @ C[k] @ Bp + Bp.T @ C[k] @ B0) * f
            K["0q"][idx] += (B0.T @ C[k] @ Bq + Bq.T @ C[k] @ B0) * f
            K["pp"][idx] += Bp.T @ C[k] @ Bp * f
            K["pq"][idx] += (Bp.T @ C[k] @ Bq + Bq.T @ C[k] @ Bp) * f
            K["qq"][idx] += Bq.T @ C[k] @ Bq * f
    # stress operators at each sample point, in its own ply
    S0, Sp, Sq, gather = [], [], [], []
    for zi, ki in zip(z, ply):
        for c, k in zip(conn, el_ply):
            z0, z1 = nodes[c[0]], nodes[c[2]]
            if k == ki and z0 - 1e-12 <= zi <= z1 + 1e-12:
                B0, Bp, Bq = parts(2 * (zi - z0) / (z1 - z0) - 1, 0.5 * (z1 - z0))
                S0.append(C[k] @ B0)
                Sp.append(C[k] @ Bp)
                Sq.append(C[k] @ Bq)
                gather.append([3 * j + d for j in c for d in range(3)])
                break
        else:
            raise ValueError(f"z = {zi} is not in ply {ki}")
    S0, Sp, Sq, gather = np.array(S0), np.array(Sp), np.array(Sq), np.array(gather)
    out = {key: np.zeros(len(z)) for key in ("sigma_x", "sigma_y", "sigma_z", "tau_xz", "tau_yz")}
    w_centre = 0.0
    mid = int(np.argmin(np.abs(nodes - laminate.mid_plane)))
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            p, qq = m * np.pi / a, n * np.pi / b
            Kmn = K["00"] + p * K["0p"] + qq * K["0q"] + p * p * K["pp"] + p * qq * K["pq"] + qq * qq * K["qq"]
            F = np.zeros(n_dof)
            F[-1] = 16 * q / (np.pi**2 * m * n)
            X = np.linalg.solve(Kmn, F)
            w_centre += X[3 * mid + 2] * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)
            stress = np.einsum("pij,pj->pi", S0 + p * Sp + qq * Sq, X[gather])
            ss, sc, cs = np.sin(p * x) * np.sin(qq * y), np.sin(p * x) * np.cos(qq * y), np.cos(p * x) * np.sin(qq * y)
            out["sigma_x"] += stress[:, 0] * ss
            out["sigma_y"] += stress[:, 1] * ss
            out["sigma_z"] += stress[:, 2] * ss
            out["tau_yz"] += stress[:, 3] * sc
            out["tau_xz"] += stress[:, 4] * cs
    out["w_centre"] = w_centre
    return out

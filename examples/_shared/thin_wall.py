"""Thin-walled beam theory for closed multi-cell blade sections (a reference for the shell model).

Each skin and web segment is a laminate wall at its mid-thickness, carrying an axial
force per length Nx and a shear flow q, with the hoop stress free. From the wall
compliance a = A^-1:

    eps_x = a11 Nx + a16 q,   gamma = a16 Nx + a66 q.

The axial strain is eps_0 + kappa_x y - kappa_y x. The shear flow in each cell
follows from Bredt-Batho compatibility: the shear strain integrated round a cell is
twice its enclosed area times the twist rate. The section forces are then
N = sum Nx ds, Mx = sum Nx y ds, My = -sum Nx x ds and T = sum 2 A_cell q_cell.

This is classical thin-walled composite beam theory (free warping, no wall bending,
no transverse shear deformation). It is exact in the thin-wall limit for a long,
prismatic segment, which makes it the natural check on ``beam_section_properties``.
"""

import numpy as np

from composite_laminate_model import integrate_abd


def thin_wall_stiffness(airfoil, chord, pitch_axis, regions, webs, laminates, n=600):
    """4 x 4 section stiffness [N, Mx, My, Mz] about the pitch axis.

    ``regions`` and ``webs`` are the ``SkinRegion`` and ``ShearWeb`` lists given to
    ``BladeBuilder``; ``laminates`` maps region and web names to laminates, with
    ``_ss`` / ``_ps`` suffixes where the suction (upper) and pressure (lower) sides
    differ. Skins sit on the outer mould line and grow inward, as in ``BladeBuilder``;
    webs run between the mould lines. Ply angles turn about the outward normal, so
    on the skins, traversed anticlockwise, a16 changes sign.
    """
    cuts = [r.start for r in regions] + [regions[-1].end]
    xs = np.unique(np.concatenate([0.5 * (1 - np.cos(np.linspace(0, np.pi, n))), cuts]))
    bounds = [0.0] + sorted(w.position for w in webs) + [1.0]
    cells = list(zip(bounds[:-1], bounds[1:]))
    webs = sorted(webs, key=lambda w: w.position)

    def lam(xc, tag):
        name = next(r.name for r in regions if r.start <= xc <= r.end)
        return laminates.get(f"{name}_{tag}", laminates.get(name))

    def wall(laminate):
        a = np.linalg.inv(integrate_abd(laminate)[:3, :3])
        return a[0, 0], a[0, 2], a[2, 2]

    rows = []  # x, y, ds, a11, a16 along the traverse, a66, sign in each cell
    for side, tag in (("upper", "ss"), ("lower", "ps")):
        pts = np.column_stack([xs, airfoil.surface(side, xs)]) * chord
        for k in range(len(xs) - 1):
            p0, p1 = (pts[k + 1], pts[k]) if side == "upper" else (pts[k], pts[k + 1])  # anticlockwise
            xc = 0.5 * (xs[k] + xs[k + 1])
            laminate = lam(xc, tag)
            t = (p1 - p0) / np.linalg.norm(p1 - p0)
            mid = 0.5 * (p0 + p1) - np.array([t[1], -t[0]]) * 0.5 * laminate.thickness  # inward from the mould line
            a11, a16, a66 = wall(laminate)
            sign = [1.0 if c0 <= xc <= c1 else 0.0 for c0, c1 in cells]
            rows.append((mid[0] - pitch_axis * chord, mid[1], np.linalg.norm(p1 - p0), a11, -a16, a66, sign))
    for k, web in enumerate(webs):  # traversed upwards: anticlockwise for the cell ahead of the web
        a11, a16, a66 = wall(laminates[web.name])
        y0, y1 = (airfoil.surface(s, web.position) * chord for s in ("lower", "upper"))
        sign = [0.0] * len(cells)
        sign[k], sign[k + 1] = 1.0, -1.0
        ds = (y1 - y0) / 40
        for y in np.linspace(y0, y1, 41)[:-1]:
            rows.append(((web.position - pitch_axis) * chord, y + 0.5 * ds, ds, a11, a16, a66, sign))
    x, y, ds, a11, a16, a66 = (np.array([r[i] for r in rows]) for i in range(6))
    sign = np.array([r[6] for r in rows])
    area = np.zeros(len(cells))  # enclosed by the wall mid-lines
    for c, (c0, c1) in enumerate(cells):
        xx = np.linspace(c0, c1, 400)
        t_mean = np.array([0.5 * (lam(v, "ss").thickness + lam(v, "ps").thickness) for v in xx])
        h = (airfoil.surface("upper", xx) - airfoil.surface("lower", xx)) * chord - t_mean
        area[c] = np.sum(0.5 * (h[1:] + h[:-1]) * np.diff(xx)) * chord
    beta, a_shear = a16 / a11, a66 - a16**2 / a11
    M = (sign * (a_shear * ds)[:, None]).T @ sign
    S = np.zeros((4, 4))
    for col in range(4):
        e = np.eye(4)[col]
        eps = e[0] + e[1] * y - e[2] * x
        q_cell = np.linalg.solve(M, 2 * area * e[3] - sign.T @ (ds * beta * eps))
        q = sign @ q_cell
        Nx = (eps - a16 * q) / a11
        S[:, col] = [np.sum(Nx * ds), np.sum(Nx * y * ds), -np.sum(Nx * x * ds), np.sum(2 * area * q_cell)]
    return 0.5 * (S + S.T)

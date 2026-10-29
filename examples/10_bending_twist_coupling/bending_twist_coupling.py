"""Bending-twist coupling of an off-axis cantilever: the basis of aeroelastic tailoring.

A 300 x 50 mm cantilever strip of eight T300/5208 plies all at angle theta is
clamped at x = 0 and loaded by a transverse tip force. When the fibres are not
aligned with the span, D16 is non-zero, so bending the strip also twists it.
Sweeping theta shows how much twist a laminate designer can buy per unit of
bending, which is how wind-turbine and aircraft wings are tailored to unload
themselves.

Reference: the narrow-strip estimate from laminate theory, with d = D^-1,
    tip deflection w = d11 P L^3 / (3 b),  tip twist phi = d16 P L^2 / (4 b),
which assumes the strip is free to twist and bend anticlastically all the way
to the root. The clamp prevents that, so the FE answer is a little stiffer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from composite_laminate_model import (
    DOF,
    BoundaryConditions,
    FEAMesh,
    FSDTModel,
    Laminate,
    LoadCase,
    StructuralGlobalSolver,
    create_default_library,
    integrate_abd,
)

L, B, PLY_T, P = 0.3, 0.05, 0.125e-3, 0.1  # P = 0.1 N tip load (results scale linearly)
NX, NY = 30, 6
ANGLES = np.arange(0, 91, 7.5)


def cantilever(lam):
    mesh = FEAMesh.rectangular_plate(L, B, NX, NY, lam, FSDTModel())
    bcs = BoundaryConditions().fix(mesh.nodes_on_line(0, 0.0), [DOF.UX, DOF.UY, DOF.UZ, DOF.RX, DOF.RY, DOF.RZ])
    tip = sorted(mesh.nodes_on_line(0, L), key=lambda n: mesh.node_coordinates(n)[1])
    loads = LoadCase()
    for i, node in enumerate(tip):
        loads.add_nodal_load(node, DOF.UZ, P / NY * (0.5 if i in (0, NY) else 1.0))
    result = StructuralGlobalSolver().solve(mesh, bcs, loads)
    w = np.array([result.node_displacement(n)[2] for n in tip])
    return w.mean(), (w[-1] - w[0]) / B  # tip deflection, tip twist (rad)


def narrow_strip(lam):
    d = np.linalg.inv(integrate_abd(lam)[3:, 3:])
    return d[0, 0] * P * L**3 / (3 * B), d[0, 2] * P * L**2 / (4 * B)  # twist rate = kappa_xy / 2


def main():
    card = create_default_library().get("T300/5208")
    fe_w, fe_phi, cl_w, cl_phi = [], [], [], []
    print(f"{'theta':>6}{'w FE [mm]':>11}{'w strip':>10}{'twist FE [deg]':>16}{'twist strip':>13}")
    for theta in ANGLES:
        lam = Laminate.from_angles(card, [theta] * 8, PLY_T)
        w, phi = cantilever(lam)
        w0, phi0 = narrow_strip(lam)
        fe_w.append(w); fe_phi.append(phi); cl_w.append(w0); cl_phi.append(phi0)
        print(f"{theta:6.1f}{w * 1e3:11.3f}{w0 * 1e3:10.3f}{np.degrees(phi):16.3f}{np.degrees(phi0):13.3f}")
    fe_w, fe_phi, cl_w, cl_phi = map(np.array, (fe_w, fe_phi, cl_w, cl_phi))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    ax1.plot(ANGLES, fe_w * 1e3, "o", color=SERIES[0], label="FE tip deflection")
    ax1.plot(ANGLES, cl_w * 1e3, color=REFERENCE, lw=1.2, ls=(0, (4, 3)), label="narrow-strip estimate")
    ax1.set_xlabel("ply angle θ [deg]")
    ax1.set_ylabel(f"tip deflection per {P:g} N [mm]")
    ax1.set_title("Off-axis fibres soften bending")
    ax1.legend(loc="upper left")

    ax2.plot(ANGLES, np.degrees(fe_phi), "o", color=SERIES[1], label="FE tip twist")
    ax2.plot(ANGLES, np.degrees(cl_phi), color=REFERENCE, lw=1.2, ls=(0, (4, 3)), label="narrow-strip estimate")
    ax2.axhline(0, color=TEXT_SECONDARY, lw=0.8)
    best = int(np.argmax(np.abs(fe_phi)))
    ax2.set_title(f"Twist peaks at θ = {ANGLES[best]:.1f}°: {np.degrees(fe_phi[best]):.2f}° per {P:g} N")
    ax2.set_xlabel("ply angle θ [deg]")
    ax2.set_ylabel(f"tip twist per {P:g} N [deg]")
    ax2.legend(loc="lower right")
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(0, 91, 15))
    fig.text(0.0, -0.03, f"[θ]8 T300/5208, {PLY_T * 1e3:.3f} mm plies, {L * 1e3:.0f} x {B * 1e3:.0f} mm cantilever clamped at "
             f"x = 0, tip load spread over the free edge, {NX} x {NY} FSDT mesh. Twist = (w_tip,y=b - w_tip,y=0) / b.",
             fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "bending_twist_coupling.png")


if __name__ == "__main__":
    main()

"""Progressive damage of a [0/90]s strip in tension, from first-ply failure to fibre failure.

A 100 x 40 mm T300/5208 [0/90/90/0] strip is pulled along x by a line load
N_x on its right edge. ``ProgressiveDamageAnalysis`` steps the load up, checks
every ply with the 3D Hashin criterion, softens failed plies with
``SmearedCrackDegradation`` and re-solves each step until no new damage appears.

* The 90-degree plies crack in matrix tension first (first-ply failure). The
  laminate keeps carrying load through the 0-degree plies, but its axial
  stiffness drops, so the load-strain curve has a knee.
* Ultimate failure follows when the 0-degree plies reach the fibre tensile
  strength.

The reference is a classical-laminate-theory ply-discount analysis with the same
criterion and the same knockdown, worked on the laminate alone: find the load at
which the next ply fails, degrade it, repeat until a fibre fails. Each FE re-solve
is a linear static analysis with updated ply properties.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shared.plotting import REFERENCE, SERIES, TEXT_SECONDARY, plt, save
from composite_laminate_model import (
    DOF,
    DamageState,
    BoundaryConditions,
    FEAMesh,
    FSDTModel,
    HashinCriterion,
    Laminate,
    LoadCase,
    ProgressiveDamageAnalysis,
    SmearedCrackDegradation,
    create_default_library,
    integrate_abd,
)

L, B, PLY_T = 0.1, 0.04, 0.25e-3
REFERENCE_NX = 1.0e5  # N/m, scaled by the load factors


def tension_strip(laminate, n=8):
    mesh = FEAMesh.rectangular_plate(L, B, n, 2, laminate, FSDTModel())
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    bcs.fix(mesh.nodes_on_line(0, 0.0), [DOF.UX]).fix([mesh.closest_node((0.0, 0.0))], [DOF.UY])
    right = sorted(mesh.nodes_on_line(0, L), key=lambda nid: mesh.node_coordinates(nid)[1])
    loads = LoadCase()
    spacing = B / (len(right) - 1)
    for i, node in enumerate(right):
        loads.add_nodal_load(node, DOF.UX, REFERENCE_NX * spacing * (0.5 if i in (0, len(right) - 1) else 1.0))
    return mesh, bcs, loads, right


def clt_first_ply_failure(laminate):
    eps0 = np.linalg.solve(integrate_abd(laminate)[:3, :3], [1.0, 0.0, 0.0])
    ply = laminate.plies[1]
    return ply.material.strength.Yt / (ply.T @ ply.Q_bar @ eps0)[1]


def clt_ply_discount(laminate, criterion, degradation, n_max=20):
    """Ply-discount analysis on the laminate alone under N_x.

    Returns a list of (N_x at the event, axial compliance a11 after it, ply index,
    damage labels). Each step finds the lowest N_x at which a ply gains a
    new damage mode, degrades that ply and carries on until a fibre fails.
    """
    virgin, current = laminate, laminate
    damage = {k: DamageState() for k in range(laminate.n_plies)}
    events, n_prev = [], 0.0
    for _ in range(n_max):
        a = np.linalg.inv(integrate_abd(current)[:3, :3])
        eps0 = a @ [1.0, 0.0, 0.0]
        best = None
        for k, ply in enumerate(current.plies):
            s1, s2, t12 = ply.T @ ply.Q_bar @ eps0
            unit = np.array([s1, s2, 0.0, 0.0, 0.0, t12])
            strength = virgin.plies[k].material.strength
            ratio = criterion.evaluate(unit, strength).strength_ratio
            n_k = max(ratio, n_prev)
            new = damage[k].merge(DamageState.from_failure(criterion.evaluate(unit * n_k * (1 + 1e-9), strength)))
            if new != damage[k] and (best is None or n_k < best[0]):
                best = (n_k, k, new)
        if best is None:
            break
        n_prev, k, new = best
        before = damage[k]
        # the same knockdown hits every ply that fails at this load (the symmetric pair)
        for j, ply in enumerate(current.plies):
            if j == k or (ply.angle == current.plies[k].angle and damage[j] == before):
                damage[j] = new
                elastic = degradation.degrade(virgin.plies[j].material.elastic, new)
                current = current.replace_ply(j, virgin.plies[j].with_material(
                    virgin.plies[j].material.with_elastic(elastic, suffix="+dmg")))
        events.append((n_prev, np.linalg.inv(integrate_abd(current)[:3, :3])[0, 0], k, new.labels()))
        if new.fiber:
            break
    return events


def main():
    card = create_default_library().get("T300/5208")
    lam = Laminate.from_angles(card, [0, 90, 90, 0], PLY_T)
    mesh, bcs, loads, right = tension_strip(lam)
    n_fpf = clt_first_ply_failure(lam)
    factors = np.round(np.arange(0.25, 8.01, 0.25), 4)

    criterion, degradation = HashinCriterion("3d"), SmearedCrackDegradation(0.01)
    clt = clt_ply_discount(lam, criterion, degradation)
    analysis = ProgressiveDamageAnalysis(criterion=criterion, degradation=degradation)
    result = analysis.run(mesh, bcs, loads, load_factors=factors)

    nx = np.array([s.load_factor * REFERENCE_NX for s in result.steps]) / 1e3  # kN/m
    strain = np.array([s.max_displacement for s in result.steps]) / L * 100  # % (max |u| is the loaded edge)
    damaged = np.array([s.total_damaged_plies for s in result.steps])
    total_plies = mesh.n_elements * lam.n_plies
    fibre_step = next((s for s in result.steps if any("fiber" in m for _, _, modes in s.damage_events for m in modes)), None)
    print(f"CLT first-ply failure  N_x = {n_fpf / 1e3:.1f} kN/m")
    print(f"FE  first-ply failure  N_x = {result.first_ply_failure_factor * REFERENCE_NX / 1e3:.1f} kN/m")
    if fibre_step:
        print(f"FE  fibre failure      N_x = {fibre_step.load_factor * REFERENCE_NX / 1e3:.1f} kN/m")
    h = lam.thickness
    for n_ev, a11, k, labels in clt:
        print(f"CLT ply discount: N_x = {n_ev / 1e3:6.1f} kN/m  ply {k} ({lam.plies[k].angle:.0f} deg) {', '.join(labels)}; "
              f"axial modulus after {1 / (a11 * h) / 1e9:.1f} GPa")
    for s in result.steps:
        print(f"  N_x {s.load_factor * REFERENCE_NX / 1e3:7.1f} kN/m  max FI {s.max_failure_index:6.3f}  "
              f"damaged plies {s.total_damaged_plies:3d}  iterations {s.iterations}")

    keep = strain < 5 * strain[0] * factors[-1] / factors[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.3), gridspec_kw={"width_ratios": [1.3, 1.0]})
    ax1.plot(strain[keep], nx[keep], marker="o", ms=4, color=SERIES[0], label="FE, Hashin 3D + smeared crack")
    initial_slope = nx[0] / strain[0]
    ax1.plot([0, strain[keep].max()], [0, initial_slope * strain[keep].max()], ls=(0, (4, 3)), lw=1.0,
             color=REFERENCE, label="undamaged stiffness")
    # CLT ply-discount load path: at each event the strain jumps at constant load to
    # the degraded laminate's secant (the knockdown is instant), then follows it
    a_prev = np.linalg.inv(integrate_abd(lam)[:3, :3])[0, 0]
    clt_n, clt_e = [0.0], [0.0]
    for n_ev, a11, _, labels in clt:
        fibre = any("fiber" in m for m in labels)  # the laminate fails: no post-failure branch drawn
        clt_n += [n_ev] if fibre else [n_ev, n_ev]
        clt_e += [n_ev * a_prev] if fibre else [n_ev * a_prev, n_ev * a11]
        a_prev = a11
    fe_secant = nx[(damaged > 0) & keep][0] * 1e3 / (strain[(damaged > 0) & keep][0] / 100)
    print(f"Axial stiffness after matrix cracking: FE {fe_secant / lam.thickness / 1e9:.1f} GPa, "
          f"CLT {1 / (clt[0][1] * lam.thickness) / 1e9:.1f} GPa")
    ax1.plot(np.array(clt_e) * 100, np.array(clt_n) / 1e3, color=REFERENCE, lw=1.4, marker="s", ms=4, mfc="none",
             label="CLT ply discount")
    ax1.axhline(n_fpf / 1e3, color=TEXT_SECONDARY, lw=0.8, ls=(0, (1, 2)))
    ax1.text(0.0, n_fpf / 1e3, f" CLT first-ply failure {n_fpf / 1e3:.0f} kN/m", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    if fibre_step:
        ax1.axhline(fibre_step.load_factor * REFERENCE_NX / 1e3, color=TEXT_SECONDARY, lw=0.8, ls=(0, (1, 2)))
        ax1.text(0.0, fibre_step.load_factor * REFERENCE_NX / 1e3, " 0° plies reach fibre failure",
                 va="bottom", fontsize=8, color=TEXT_SECONDARY)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("average axial strain [%]")
    ax1.set_ylabel("N_x [kN/m]")
    cracked = (damaged > 0) & (damaged < total_plies) & keep
    if cracked.sum() >= 2:
        i, j = np.flatnonzero(cracked)[[0, -1]]
        drop = 100 * (1 - (nx[j] - nx[i]) / (strain[j] - strain[i]) / initial_slope)
        ax1.set_title(f"Matrix cracking cuts axial stiffness by {drop:.0f}%")
    else:
        ax1.set_title("Load path with progressive damage")
    ax1.legend(loc="lower right")

    ax2.step(nx, damaged / total_plies * 100, where="post", color=SERIES[1])
    ax2.set_xlabel("N_x [kN/m]")
    ax2.set_ylabel("damaged ply-elements [% of all]")
    ax2.set_ylim(0, 105)
    ax2.set_title("The 90° plies crack together, then the 0° plies")
    fig.text(0.0, -0.03, f"[0/90]s T300/5208, ply {PLY_T * 1e3:.2f} mm, {L * 1e3:.0f} x {B * 1e3:.0f} mm strip, "
             f"{mesh.n_elements} FSDT elements, residual stiffness 1% after failure.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "progressive_damage.png")


if __name__ == "__main__":
    main()

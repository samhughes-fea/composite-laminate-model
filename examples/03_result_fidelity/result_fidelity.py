"""Result fidelity levels: Gauss points, elements, nodes and global totals.

A thin (a/h = 100) simply supported [0/90/90/0] square plate under uniform pressure,
post-processed with ``ResultsExtractor`` at every level the package offers.

Figure 03a shows the same model three ways: nodal deflection, element-constant
bending moment Mx, and the nodal Mx that comes from extrapolating the Gauss-point
values to the corners and averaging across neighbouring elements.

Figure 03b measures each level against the Navier series solution at its own
sample locations as the mesh is refined. For this bilinear element the centroid
is the optimal (Barlow) sampling point for moments, converging at O(h^2), while
the 2x2 Gauss points and the averaged nodal values converge at O(h). It also splits the global strain energy
into membrane, bending, shear and drilling parts for a thin and a thick plate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from _shared.analytical import navier_clpt_moment_field
from _shared.plotting import REFERENCE, SEQUENTIAL, SERIES, TEXT_SECONDARY, plt, save, simply_supported
from composite_laminate_model import (
    CLPTModel,
    FEAMesh,
    FSDTModel,
    Laminate,
    LoadCase,
    ResultsExtractor,
    StructuralGlobalSolver,
    create_default_library,
    integrate_abd,
)

A, Q = 0.2, 1.0e3
STACK = [0, 90, 90, 0]


def solve(lam, n, kinematics):
    mesh = FEAMesh.rectangular_plate(A, A, n, n, lam, kinematics)
    result = StructuralGlobalSolver().solve(mesh, simply_supported(mesh, A, A), LoadCase().add_pressure(Q))
    return mesh, ResultsExtractor(result)


def relative_rms(values, reference):
    return float(np.sqrt(np.mean((values - reference) ** 2)) / np.max(np.abs(reference)))


def fields_figure(lam, mesh, ex):
    nodes, elements = ex.nodes(), ex.elements()
    x, y = nodes.xyz[:, 0] * 1e3, nodes.xyz[:, 1] * 1e3
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    ax = axes[0]
    cs = ax.tricontourf(x, y, nodes.displacement[:, 2] * 1e3, levels=12, cmap=SEQUENTIAL)
    fig.colorbar(cs, ax=ax, shrink=0.85, label="w [mm]")
    ax.set_title("Node: deflection w")

    m_all = np.concatenate([elements.M[:, 0], nodes.M[:, 0]])
    norm = Normalize(vmin=min(m_all.min(), 0.0), vmax=m_all.max())

    ax = axes[1]
    quads = [e.coordinates[:, :2] * 1e3 for e in mesh.elements]
    patches = PolyCollection(quads, array=elements.M[:, 0], cmap=SEQUENTIAL, norm=norm, edgecolors="#fcfcfb", linewidths=0.4)
    ax.add_collection(patches)
    fig.colorbar(patches, ax=ax, shrink=0.85, label="Mx [N·m/m]")
    ax.set_title("Element: Mx, one value each")

    ax = axes[2]
    cs = ax.tricontourf(x, y, nodes.M[:, 0], levels=np.linspace(norm.vmin, norm.vmax, 15), cmap=SEQUENTIAL, norm=norm)
    fig.colorbar(cs, ax=ax, shrink=0.85, label="Mx [N·m/m]", format="%.1f")
    ax.set_title("Node: Mx, averaged")

    for ax in axes:
        ax.set_xlim(0, A * 1e3)
        ax.set_ylim(0, A * 1e3)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    n = int(round(np.sqrt(len(mesh.elements))))
    fig.text(0.0, -0.02, f"[0/90/90/0] T300/5208, a = {A * 1e3:.0f} mm, a/h = 100, simply supported, q = {Q:g} Pa, "
             f"{n} x {n} mesh, CLPT.", fontsize=7.5, color=TEXT_SECONDARY)
    fig.tight_layout()
    save(fig, __file__, "result_fidelity_fields.png")


def convergence(lam, meshes):
    D = integrate_abd(lam)[3:, 3:]
    errors = {"Gauss point": [], "Element centroid": [], "Node (averaged)": []}
    for n in meshes:
        _, ex = solve(lam, n, CLPTModel())
        gp, el, nd = ex.gauss_points(), ex.elements(), ex.nodes()
        for label, field in (("Gauss point", gp), ("Element centroid", el), ("Node (averaged)", nd)):
            xyz = field.xyz if hasattr(field, "xyz") else field.centroid
            exact = navier_clpt_moment_field(D, A, A, Q, xyz[:, 0], xyz[:, 1])
            errors[label].append(relative_rms(field.M[:, :2], exact))
    return errors


def energy_split(card):
    cases = {}
    for ratio in (100, 5):
        h = A / ratio
        lam = Laminate.from_angles(card, STACK, h / len(STACK))
        _, ex = solve(lam, 16, FSDTModel())
        totals = ex.totals()
        cases[f"a/h = {ratio}"] = {k: v / totals.strain_energy for k, v in totals.energy_split.items()}
    return cases


def summary_figure(meshes, errors, split):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]})
    for colour, marker, (label, values) in zip(SERIES, ("o", "s", "^"), errors.items()):
        ax1.loglog(meshes, values, marker=marker, color=colour, label=label)
        ax1.annotate(f"{values[-1] * 100:.2f}%", (meshes[-1], values[-1]), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=8, color=TEXT_SECONDARY)
    h = np.array(meshes, dtype=float)
    ref = errors["Element centroid"][0] * (h / h[0]) ** -2
    ax1.loglog(h, ref, ls=(0, (4, 3)), lw=1.0, color=REFERENCE, label="slope 2 (O(h²))")
    ax1.set_xticks(meshes)
    ax1.set_xticklabels([str(m) for m in meshes])
    ax1.minorticks_off()
    ax1.set_xlabel("elements per side")
    ax1.set_ylabel("RMS error of [Mx, My] / peak Navier moment")
    ax1.set_title("Moments are most accurate at element centroids")
    ax1.legend(loc="lower left")

    parts = ["bending", "membrane", "coupling", "shear", "drilling"]
    labels = list(split)
    left = np.zeros(len(labels))
    for colour, part in zip(SERIES, parts):
        widths = np.array([split[c].get(part, 0.0) for c in labels])
        if np.all(widths < 1e-4):
            continue
        ax2.barh(labels, widths, left=left, color=colour, label=part, height=0.5, edgecolor="#fcfcfb", linewidth=2)
        for i, w in enumerate(widths):
            if w > 0.04:
                ax2.text(left[i] + w / 2, i, f"{w * 100:.0f}%", ha="center", va="center", fontsize=8.5, color="#ffffff" if part == "bending" else "#0b0b0b")
        left += widths
    ax2.set_axisbelow(True)
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax2.set_xlabel("share of total strain energy (FSDT, 16 x 16)")
    thick = split[labels[-1]].get("shear", 0.0)
    ax2.set_title(f"Totals: shear holds {thick * 100:.0f}% of the energy at {labels[-1]}")
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.tight_layout()
    save(fig, __file__, "result_fidelity_convergence.png")


def main():
    card = create_default_library().get("T300/5208")
    lam = Laminate.from_angles(card, STACK, (A / 100) / len(STACK))
    mesh, ex = solve(lam, 16, CLPTModel())
    fields_figure(lam, mesh, ex)

    meshes = [4, 8, 16, 32]
    errors = convergence(lam, meshes)
    split = energy_split(card)
    for label, values in errors.items():
        print(f"  {label:<18}" + "  ".join(f"{v * 100:6.2f}%" for v in values))
    for case, parts in split.items():
        print(f"  {case}: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in parts.items()))
    summary_figure(meshes, errors, split)


if __name__ == "__main__":
    main()

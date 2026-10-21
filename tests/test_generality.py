"""Ply-based layups, curved shells, layered solids with shell coupling, and VTK export."""

import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import navier_fsdt_centre, simply_supported
from laminate_modelling_package import (
    DOF,
    AngleField,
    BoundaryConditions,
    CalculiXDeckWriter,
    CalculiXKernelAdapter,
    NativePythonKernelAdapter,
    AnalysisModel,
    CylindricalPanelBuilder,
    ElasticProperties3D,
    FEAMesh,
    FSDTModel,
    HygrothermalState,
    Laminate,
    Layup,
    LoadCase,
    MaterialCard,
    ModelChecker,
    NastranDeckWriter,
    PaganoPlate,
    Ply,
    PlyDefinition,
    ProgressiveDamageAnalysis,
    RectangularPlateBuilder,
    ReferenceDirection,
    RefinedZigZagModel,
    ResultsExtractor,
    StrengthProperties3D,
    StructuralGlobalSolver,
    TabulatedDrape,
    UnsupportedFeatureError,
    couple_shell_to_solid,
    create_criterion,
    solid_block,
    write_pvd,
    write_vtu,
)


def zero_modes(K, tol=1e-9):
    s = 1.0 / np.sqrt(np.diag(K))
    e = np.linalg.eigvalsh(s[:, None] * K * s[None, :])
    return int(np.sum(np.abs(e) < tol * e.max()))


@pytest.fixture(scope="module")
def pagano_card():
    E2 = 1.0e9
    elastic = ElasticProperties3D(E1=25 * E2, E2=E2, E3=E2, v12=0.25, v13=0.25, v23=0.25,
                                  G12=0.5 * E2, G13=0.5 * E2, G23=0.2 * E2)
    return MaterialCard("pagano", elastic, StrengthProperties3D(1e9, 1e9, 5e7, 1e8, 6e7))


# ---------------------------------------------------------------------------
# Ply-based layup
# ---------------------------------------------------------------------------
@pytest.fixture()
def strip(cfrp):
    """10 x 2 Quad4 strip along x with an element set 'pad' on the first four columns."""
    named = RectangularPlateBuilder(0.1, 0.02, 10, 2).build_named(Laminate.from_angles(cfrp, [0], 1e-4), FSDTModel())
    pad = [e.element_id for e in named.mesh.elements if e.centroid[0] < 0.04]
    named.element_sets["pad"] = pad
    return named


def test_layup_builds_each_element_laminate_from_the_plies_covering_it(cfrp, strip):
    t = 0.125e-3
    layup = Layup([
        PlyDefinition("skin_0", cfrp, t, 0),
        PlyDefinition("pad_45", cfrp, t, 45, elements="pad"),
        PlyDefinition("pad_m45", cfrp, t, -45, region=lambda x: x[0] < 0.02),
        PlyDefinition("skin_90", cfrp, t, 90, sequence=10),
        PlyDefinition("core_0", cfrp, t, 0, sequence=2.5),
    ], reference="bottom")
    layup.apply(strip.mesh, strip.element_sets)
    root = strip.mesh.element(1)
    names = [p.name for p in root.laminate.plies]
    assert names == ["skin_0", "pad_45", "pad_m45", "core_0", "skin_90"]  # sequence order
    tip = [e for e in strip.mesh.elements if e.centroid[0] > 0.09][0]
    assert [p.name for p in tip.laminate.plies] == ["skin_0", "core_0", "skin_90"]
    # reference="bottom": every laminate starts at the mesh surface
    for e in strip.mesh.elements:
        assert e.laminate.z_interfaces[0] == pytest.approx(0.0, abs=1e-15)
    thickness = layup.thickness_map(strip.mesh, strip.element_sets)
    assert thickness[1] == pytest.approx(5 * t) and thickness[tip.element_id] == pytest.approx(3 * t)
    table = {row["ply"]: row for row in layup.ply_table(strip.mesh, strip.element_sets)}
    assert table["pad_45"]["area"] == pytest.approx(0.04 * 0.02)
    assert table["skin_0"]["mass"] == pytest.approx(0.1 * 0.02 * t * cfrp.density)
    # drops: pad_45 ends at x = 0.04 and pad_m45 at x = 0.02, two elements across each
    drops = layup.drops(strip.mesh, strip.element_sets)
    assert sorted({d.ply for d in drops}) == ["pad_45", "pad_m45"]
    assert len([d for d in drops if d.ply == "pad_45"]) == 2
    assert layup.check_drops(strip.mesh, max_per_step=2, element_sets=strip.element_sets) == []
    assert layup.check_drops(strip.mesh, max_per_step=0, element_sets=strip.element_sets)


def test_uniform_layup_matches_the_equivalent_laminate(cfrp, strip):
    t = 0.125e-3
    angles = [0, 45, -45, 90]
    layup = Layup([PlyDefinition(f"p{k}", cfrp, t, a) for k, a in enumerate(angles)])
    layup.apply(strip.mesh)
    lam = Laminate.from_angles(cfrp, angles, t)
    A_ref = FSDTModel().abd_matrix(lam)
    for e in strip.mesh.elements:
        assert np.allclose(FSDTModel().abd_matrix(e.laminate), A_ref)


def test_layup_errors_and_uncovered_elements(cfrp, strip):
    with pytest.raises(ValueError, match="unique"):
        Layup([PlyDefinition("a", cfrp, 1e-4), PlyDefinition("a", cfrp, 1e-4)])
    with pytest.raises(ValueError, match="positive thickness"):
        PlyDefinition("a", cfrp, 0.0)
    partial = Layup([PlyDefinition("pad", cfrp, 1e-4, elements="pad")])
    with pytest.raises(ValueError, match="No ply covers"):
        partial.apply(strip.mesh, strip.element_sets)
    with pytest.raises(KeyError, match="unknown element set"):
        partial.apply(strip.mesh, {})
    before = strip.mesh.element(10).laminate
    partial.apply(strip.mesh, strip.element_sets, uncovered="keep")
    assert strip.mesh.element(10).laminate is before


def test_drape_fields_change_the_fibre_angle(cfrp):
    named = RectangularPlateBuilder(0.1, 0.1, 2, 2).build_named(Laminate.from_angles(cfrp, [0], 1e-4), FSDTModel())
    mesh = named.mesh
    mesh.set_material_axis((1.0, 1.0, 0.0))  # element material axes at +45 degrees
    layup = Layup([
        PlyDefinition("global_x", cfrp, 1e-4, 0, drape=ReferenceDirection((1, 0, 0))),
        PlyDefinition("sheared", cfrp, 1e-4, 0, drape=AngleField(lambda x: 100.0 * x[0])),
        PlyDefinition("table", cfrp, 1e-4, 90, drape=TabulatedDrape([(0, 0, 0), (0.1, 0.1, 0)], [5.0, -5.0])),
    ])
    layup.apply(mesh)
    for e in mesh.elements:
        a = [p.angle for p in e.laminate.plies]
        assert a[0] == pytest.approx(-45.0)  # global x is -45 degrees from the element axis
        assert a[1] == pytest.approx(100.0 * e.centroid[0])
        assert a[2] == pytest.approx(95.0 if e.centroid[0] + e.centroid[1] < 0.1 else 85.0)


def test_ply_failure_reports_each_named_ply(cfrp, strip):
    t = 0.125e-3
    layup = Layup([PlyDefinition("bot", cfrp, t, 0), PlyDefinition("pad", cfrp, t, 90, elements="pad"),
                   PlyDefinition("top", cfrp, t, 0)])
    layup.apply(strip.mesh, strip.element_sets)
    bcs = BoundaryConditions().clamp(strip.nodes("left"))
    sol = StructuralGlobalSolver().solve(strip.mesh, bcs, LoadCase().add_line_load(strip.nodes("right"), (0, 0, 10.0)))
    worst = layup.ply_failure(sol, create_criterion("tsai_wu"))
    assert set(worst) == {"bot", "pad", "top"}
    assert worst["pad"]["element"] in strip.element_sets["pad"]
    assert all(v["index"] > 0 for v in worst.values())


# ---------------------------------------------------------------------------
# Curved shells
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element", ["quad8_curved", "tri6"])
def test_curved_elements_have_exactly_six_rigid_body_modes(cfrp, element):
    lam = Laminate.from_angles(cfrp, [0, 45, 90], 0.25e-3)
    for builder in (RectangularPlateBuilder(0.05, 0.03, 1, 1, element=element),
                    CylindricalPanelBuilder(0.1, 0.05, 1, 1, 0, 40, element=element)):
        e = builder.build(lam, FSDTModel()).elements[0]
        assert zero_modes(e.stiffness_matrix()) == 6


def scordelis_lo(element, n):
    E = 4.32e8
    card = MaterialCard("iso", ElasticProperties3D(E1=E, E2=E, E3=E, v12=0.0, v13=0.0, v23=0.0, G12=E / 2,
                                                   G13=E / 2, G23=E / 2), StrengthProperties3D(1e9, 1e9, 1e9, 1e9, 1e9),
                        density=360.0)
    lam = Laminate.from_angles(card, [0], 0.25)
    R, L = 25.0, 50.0
    named = CylindricalPanelBuilder(R, L, n, n, 50.0, 130.0, element=element).build_named(lam, FSDTModel())
    mesh = named.mesh
    bcs = BoundaryConditions().fix(named.nodes("x0") + named.nodes("x1"), [DOF.UY, DOF.UZ])
    bcs.fix([mesh.closest_node((L / 2, 0, R))], [DOF.UX])
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_body_load((0, 0, -1.0)))
    phi = np.radians(50)
    return sol.node_displacement(mesh.closest_node((L / 2, R * np.cos(phi), R * np.sin(phi))))[2] / -0.3024


def test_scordelis_lo_roof_with_curved_quad8_and_tri6():
    assert scordelis_lo("quad8_curved", 4) == pytest.approx(1.0, abs=0.015)
    assert scordelis_lo("quad8_curved", 8) == pytest.approx(1.0, abs=0.005)
    assert scordelis_lo("tri6", 16) == pytest.approx(1.0, abs=0.03)


def test_tri6_assumed_shear_does_not_lock_in_thin_plates(cfrp):
    a, h = 0.2, 0.2e-3
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], h / 4)
    kin = FSDTModel()
    exact = navier_fsdt_centre(kin.abd_matrix(lam)[3:, 3:], kin.transverse_shear_matrix(lam), a, a, 1e4)
    mesh = RectangularPlateBuilder(a, a, 8, 8, element="tri6").build(lam, kin)
    sol = StructuralGlobalSolver().solve(mesh, simply_supported(mesh, a, a), LoadCase().add_pressure(1e4))
    assert sol.node_displacement(mesh.closest_node((a / 2, a / 2)))[2] == pytest.approx(exact, rel=0.01)


def test_curved_elements_match_flat_quad8_with_zig_zag_kinematics(pagano_card):
    h = 0.01
    a = 4 * h
    lam = Laminate.from_angles(pagano_card, [0, 90, 0], h / 3)
    load = LoadCase().add_pressure(lambda x: np.sin(np.pi * x[0] / a) * np.sin(np.pi * x[1] / a))
    w = {}
    for element in ("quad8", "quad8_curved", "tri6"):
        m = RectangularPlateBuilder(a, a, 6, 6, element=element).build(lam, RefinedZigZagModel())
        s = StructuralGlobalSolver().solve(m, simply_supported(m, a, a), load)
        w[element] = s.node_displacement(m.closest_node((a / 2, a / 2)))[2]
    assert w["quad8_curved"] == pytest.approx(w["quad8"], rel=0.01)
    assert w["tri6"] == pytest.approx(w["quad8"], rel=0.02)
    assert w["quad8"] == pytest.approx(PaganoPlate(lam, a, a).centre_deflection(), rel=0.03)


@pytest.mark.parametrize("element", ["quad8_curved", "tri6"])
def test_curved_elements_are_written_to_both_decks(cfrp, element, tmp_path):
    lam = Laminate.from_angles(cfrp, [0, 90], 0.25e-3)
    named = CylindricalPanelBuilder(0.1, 0.1, 2, 2, 0, 60, element=element).build_named(lam, FSDTModel())
    model = AnalysisModel(named.mesh, BoundaryConditions().clamp(named.nodes("x0")), LoadCase().add_pressure(1e3))
    ccx = CalculiXDeckWriter().write(model, tmp_path, "job").path.read_text()
    bdf = NastranDeckWriter().write(model, tmp_path, "job").path.read_text()
    assert ("TYPE=S8R" if element == "quad8_curved" else "TYPE=S6") in ccx
    assert ("CQUAD8" if element == "quad8_curved" else "CTRIA6") in bdf
    assert not ccx.count("** midside")  # quadratic elements are written without inserted nodes


@pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX (ccx) is not installed")
def test_curved_quad8_agrees_with_calculix_on_a_clamped_panel(cfrp, tmp_path):
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.25e-3)
    named = CylindricalPanelBuilder(0.2, 0.2, 8, 8, 0, 60, element="quad8_curved").build_named(lam, FSDTModel())
    model = AnalysisModel(named.mesh, BoundaryConditions().clamp(named.nodes("x0") + named.nodes("x1")),
                          LoadCase().add_pressure(1e4))
    native = NativePythonKernelAdapter().run(model)
    ccx = CalculiXKernelAdapter(work_dir=tmp_path).run(model)
    assert ccx.ok, ccx.message
    assert native.max_abs_displacement(2) == pytest.approx(ccx.max_abs_displacement(2), rel=0.01)


# ---------------------------------------------------------------------------
# Layered solids
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("element", ["hex8", "hex20"])
def test_layered_solids_have_six_rigid_modes_and_pass_the_patch_test(cfrp, element):
    lam = Laminate.from_angles(cfrp, [0, 45, 90], 0.5e-3)
    mesh = FEAMesh()
    solid_block(mesh, lam, 0.01, 0.012, 1, 1, element=element, layers="laminate")
    e = mesh.elements[0]
    K = e.stiffness_matrix()
    idx = [i for i in range(K.shape[0]) if i % 6 < 3]
    assert zero_modes(K[np.ix_(idx, idx)], 1e-8) == 6
    # patch test on a distorted 2 x 2 x 2 block: linear displacements give exact uniform strain
    mesh = FEAMesh()
    # one fibre angle: with mixed angles a uniform strain is not in equilibrium across the interface
    named = solid_block(mesh, Laminate.from_angles(cfrp, [30, 30], 0.5e-3), 0.004, 0.004, 2, 2,
                        element=element, elements_per_ply=1)
    centre = [n for n in mesh.node_ids
              if np.allclose(mesh.node_coordinates(n)[:2], (0.002, 0.002))]
    for n in centre:  # move the interior column off the grid
        mesh._nodes[n] = mesh.node_coordinates(n) + np.array([3e-4, -2e-4, 0.0])
    for e in list(mesh.elements):
        mesh.replace_element(type(e)(e.element_id, e.node_ids, np.array([mesh.node_coordinates(n) for n in e.node_ids]),
                                     e.laminate, gradient_origin=e.gradient_origin))
    G = np.array([[1e-3, 2e-4, -1e-4], [3e-4, -5e-4, 2e-4], [1e-4, 4e-4, 7e-4]])
    boundary = set(named.nodes("x0") + named.nodes("x1") + named.nodes("y0") + named.nodes("y1")
                   + named.nodes("bottom") + named.nodes("top"))
    bcs = BoundaryConditions()
    for n in boundary:
        u = G @ mesh.node_coordinates(n)
        for d in range(3):
            bcs.fix([n], [d], float(u[d]))
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase())
    for n in set(mesh.node_ids) - boundary:
        assert np.allclose(sol.node_displacement(n)[:3], G @ mesh.node_coordinates(n), atol=1e-12)
    eps = 0.5 * (G + G.T)
    exact = np.array([eps[0, 0], eps[1, 1], eps[2, 2], 2 * eps[1, 2], 2 * eps[0, 2], 2 * eps[0, 1]])
    for e in mesh.solid_elements:
        u = sol.element_displacements(e.element_id)
        p = e.stress_at(u, 0.3, -0.4, 0.2)
        F = e.point_frame(0.3, -0.4, 0.2)
        local = F @ eps @ F.T
        assert np.allclose(e.strain_at(u, 0.3, -0.4, 0.2)[0],
                           [local[0, 0], local[1, 1], local[2, 2], 2 * local[1, 2], 2 * local[0, 2], 2 * local[0, 1]],
                           atol=1e-10)
        assert p.stress_global.shape == (6,)
    assert exact.shape == (6,)


@pytest.mark.parametrize("element, n, per_ply", [("hex8", 12, 4), ("hex20", 6, 1)])
def test_layered_solids_reproduce_the_pagano_solution(pagano_card, element, n, per_ply):
    h = 0.01
    a = 4 * h
    lam = Laminate.from_angles(pagano_card, [0, 90, 0], h / 3)
    mesh = FEAMesh()
    named = solid_block(mesh, lam, a, a, n, n, element=element, elements_per_ply=per_ply)
    bcs = BoundaryConditions()
    bcs.fix(named.nodes("x0") + named.nodes("x1"), (DOF.UY, DOF.UZ))
    bcs.fix(named.nodes("y0") + named.nodes("y1"), (DOF.UX, DOF.UZ))
    load = LoadCase().add_pressure(lambda x: np.sin(np.pi * x[0] / a) * np.sin(np.pi * x[1] / a))
    sol = StructuralGlobalSolver().solve(mesh, bcs, load)
    exact = PaganoPlate(lam, a, a)
    mid = [nd for nd in mesh.node_ids if np.allclose(mesh.node_coordinates(nd)[:2], (a / 2, a / 2))]
    w = np.mean([sol.node_displacement(nd)[2] for nd in mid])
    assert w == pytest.approx(exact.centre_deflection(), rel=0.015)
    # sigma_x on the top face at the centre and sigma_z at mid-thickness, straight from the solid
    prof = exact.profile(5)
    d = a / n / 2
    e = [el for el in mesh.solid_elements if np.allclose(el.centroid[:2], (a / 2 - d, a / 2 - d))]
    top = max(e, key=lambda el: el.centroid[2])
    p = top.stress_at(sol.element_displacements(top.element_id), 1.0, 1.0, 1.0)
    assert p.stress_global[0] == pytest.approx(prof.sigma_x[-1], rel=0.03)
    assert p.z == pytest.approx(lam.z_interfaces[-1])
    results = ResultsExtractor(sol, criterion=create_criterion("tsai_wu"))
    sp = results.solid_points()
    assert sp.stress_global.shape[1] == 6 and np.all(np.isfinite(sp.failure_index))
    totals = results.totals()
    assert totals.energy_split["solid"] == pytest.approx(sol.external_work, rel=1e-8)
    assert totals.max_failure is not None


def test_solid_thermal_expansion_is_stress_free(aluminium):
    lam = Laminate([Ply(aluminium, 1e-3), Ply(aluminium, 1e-3)])
    mesh = FEAMesh()
    named = solid_block(mesh, lam, 0.01, 0.01, 2, 2, element="hex8")
    corner = mesh.closest_node((0, 0, -1e-3))
    bcs = BoundaryConditions().fix([corner], (0, 1, 2))
    bcs.fix([mesh.closest_node((0.01, 0, -1e-3))], (1, 2)).fix([mesh.closest_node((0, 0.01, -1e-3))], (2,))
    alpha = aluminium.hygrothermal.alpha1
    uniform, graded = HygrothermalState(delta_temperature=50.0), HygrothermalState(50.0, temperature_gradient=2e4)
    sols = StructuralGlobalSolver().solve_many(mesh, bcs, [LoadCase().set_hygrothermal(s) for s in (uniform, graded)])
    for sol, state in zip(sols, (uniform, graded)):
        for e in mesh.solid_elements:
            for p in e.through_thickness(sol.element_displacements(e.element_id), state=state, points_per_ply=3):
                assert np.abs(p.stress_global).max() < 1e-9 * aluminium.elastic.E1 * alpha * 50.0
    far = mesh.closest_node((0.01, 0.01, 1e-3))
    assert sols[0].node_displacement(far)[:3] == pytest.approx(alpha * 50.0 * np.array([0.01, 0.01, 2e-3]), rel=1e-6)
    assert named.nodes("top")


@pytest.mark.parametrize("solid, shell", [("hex8", "quad4"), ("hex20", "quad8")])
def test_shell_to_solid_coupling_carries_bending(cfrp, solid, shell):
    lam = Laminate.from_angles(cfrp, [0, 90, 0, 90, 90, 0, 90, 0], 0.125e-3)
    L, b, Ls = 0.1, 0.02, 0.02
    nx = 25 if shell == "quad8" else 50
    ref = RectangularPlateBuilder(L, b, nx, 5, element=shell).build_named(lam, FSDTModel())
    s = StructuralGlobalSolver().solve(ref.mesh, BoundaryConditions().clamp(ref.nodes("left")),
                                       LoadCase().add_line_load(ref.nodes("right"), (0, 0, 50.0)))
    w_ref = np.mean([s.node_displacement(n)[2] for n in ref.nodes("right")])

    mesh = FEAMesh()
    block = solid_block(mesh, lam, Ls, b, 5, 5, element=solid)
    part = RectangularPlateBuilder(L - Ls, b, nx * 4 // 5, 5, origin=(Ls, 0, 0), element=shell).build_named(lam, FSDTModel())
    off = max(mesh.node_ids)
    for n in part.mesh.node_ids:
        mesh.add_node(n + off, part.mesh.node_coordinates(n))
    for e in part.mesh.elements:
        mesh.add_element(e.element_id + 1000, [n + off for n in e.node_ids], lam, FSDTModel())
    bcs = BoundaryConditions().fix(block.nodes("x0"), (0, 1, 2))
    pairs = couple_shell_to_solid(mesh, bcs, [n + off for n in part.nodes("left")], block.nodes("x1"))
    assert len(pairs) == len(block.nodes("x1"))
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_line_load([n + off for n in part.nodes("right")], (0, 0, 50.0)))
    w = np.mean([sol.node_displacement(n + off)[2] for n in part.nodes("right")])
    assert w == pytest.approx(w_ref, rel=0.01)
    # rotations of solid-only nodes are held at zero, shell rotations are free
    assert set(mesh.solid_only_nodes()) == set(block.nodes("all"))
    assert np.allclose([sol.node_displacement(n)[3:] for n in block.nodes("all")], 0.0)


def test_coupling_refuses_mismatched_interfaces_and_decks_refuse_solids(cfrp, tmp_path):
    lam = Laminate.from_angles(cfrp, [0, 90], 0.25e-3)
    mesh = FEAMesh()
    block = solid_block(mesh, lam, 0.01, 0.01, 2, 2, element="hex8")
    part = RectangularPlateBuilder(0.01, 0.01, 2, 2, origin=(0.01, 0, 0), element="quad8").build_named(lam, FSDTModel())
    off = max(mesh.node_ids)
    for n in part.mesh.node_ids:
        mesh.add_node(n + off, part.mesh.node_coordinates(n))
    for e in part.mesh.elements:
        mesh.add_element(e.element_id + 100, [n + off for n in e.node_ids], lam, FSDTModel())
    bcs = BoundaryConditions()
    with pytest.raises(ValueError, match="no solid node"):
        couple_shell_to_solid(mesh, bcs, [n + off for n in part.nodes("left")], block.nodes("x1"))
    with pytest.raises(ValueError, match="limit"):
        couple_shell_to_solid(mesh, BoundaryConditions(), [n + off for n in part.nodes("left")], block.nodes("x0"),
                              max_distance=1e-3)
    model = AnalysisModel(mesh, BoundaryConditions().fix(block.nodes("x0"), (0, 1, 2)), LoadCase())
    with pytest.raises(UnsupportedFeatureError, match="hex8"):
        CalculiXDeckWriter().write(model, tmp_path, "job")
    with pytest.raises(UnsupportedFeatureError, match="hex8"):
        NastranDeckWriter().write(model, tmp_path, "job")
    assert ModelChecker().check(mesh).issues is not None


def test_mesh_wide_pressure_loads_only_exposed_solid_faces(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90, 0], 0.25e-3)
    mesh = FEAMesh()
    solid_block(mesh, lam, 0.01, 0.01, 2, 2, element="hex20", elements_per_ply=2)
    assert len(mesh.covered_solids()) == 4 * 5
    from laminate_modelling_package import GlobalAssembler

    F = GlobalAssembler(sparse=False).assemble_loads(mesh, LoadCase().add_pressure(1000.0))
    assert mesh.standard_array(F)[:, 2].sum() == pytest.approx(1000.0 * 1e-4)


def test_progressive_damage_runs_on_solids(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90], 0.25e-3)
    mesh = FEAMesh()
    block = solid_block(mesh, lam, 0.02, 0.005, 4, 1, element="hex8")
    bcs = BoundaryConditions().fix(block.nodes("x0"), (0,)).fix([block.nodes("x0")[0]], (1, 2))
    bcs.fix(block.nodes("x0")[:2], (2,))
    load = LoadCase()
    for n in block.nodes("x1"):
        load.add_nodal_load(n, 0, 2.0e3 / len(block.nodes("x1")))
    result = ProgressiveDamageAnalysis(create_criterion("max_stress")).run(mesh, bcs, load, [0.2, 1.0])
    assert result.steps and result.first_ply_failure_factor is not None


# ---------------------------------------------------------------------------
# VTK export
# ---------------------------------------------------------------------------
def _read_vtu(path):
    piece = ET.parse(path).getroot().find("UnstructuredGrid/Piece")
    arrays = {a.get("Name"): a for a in piece.iter("DataArray") if a.get("Name")}
    types = [int(v) for v in arrays["types"].text.split()]
    return piece, arrays, types


@pytest.mark.parametrize("element, vtk_type, corners", [
    ("quad4", 9, 4), ("tri3", 5, 3), ("quad8", 23, 4), ("quad8_curved", 23, 4), ("tri6", 22, 3)])
def test_vtk_export_of_every_shell_type(cfrp, element, vtk_type, corners, tmp_path):
    lam = Laminate.from_angles(cfrp, [0, 45, -45, 90], 0.125e-3)
    named = CylindricalPanelBuilder(0.1, 0.2, 2, 3, 0, 60, element=element).build_named(lam, FSDTModel())
    sol = StructuralGlobalSolver().solve(named.mesh, BoundaryConditions().clamp(named.nodes("x0")),
                                         LoadCase().add_pressure(1e3))
    files = write_vtu(tmp_path / "panel", sol, create_criterion("tsai_wu"), layered=True, thickness_scale=5.0)
    piece, arrays, types = _read_vtu(files["model"])
    assert set(types) == {vtk_type} and int(piece.get("NumberOfCells")) == named.mesh.n_elements
    assert int(piece.get("NumberOfPoints")) == named.mesh.n_nodes
    disp = np.array(arrays["displacement"].text.split(), float).reshape(-1, 3)
    assert np.allclose(disp, sol.displacement_array()[:, :3], rtol=1e-8, atol=1e-15)
    for name in ("N", "M", "Q", "element_M", "strain_energy", "failure_index", "critical_ply", "normal"):
        assert name in arrays
    piece, arrays, types = _read_vtu(files["plies"])
    assert len(types) == 4 * named.mesh.n_elements
    assert set(types) == {12 if corners == 4 else 13}
    assert int(piece.get("NumberOfPoints")) == 2 * corners * len(types)
    fi = np.array(arrays["failure_index"].text.split(), float)
    assert np.all(np.isfinite(fi)) and fi.max() > 0
    assert sorted(set(np.array(arrays["ply"].text.split(), int))) == [0, 1, 2, 3]


def test_vtk_export_of_solids_mesh_only_and_collections(cfrp, tmp_path):
    lam = Laminate.from_angles(cfrp, [0, 90, 0], 0.25e-3)
    mesh = FEAMesh()
    block = solid_block(mesh, lam, 0.01, 0.01, 2, 2, element="hex20", elements_per_ply=2)
    bcs = BoundaryConditions().fix(block.nodes("x0"), (0, 1, 2))
    cases = [LoadCase("p").add_pressure(1e3), LoadCase("q").add_pressure(-2e3)]
    sols = StructuralGlobalSolver().solve_many(mesh, bcs, cases)
    files = write_vtu(tmp_path / "solid.vtu", sols[0], create_criterion("tsai_wu"), layered=True)
    _, arrays, types = _read_vtu(files["model"])
    assert set(types) == {25} and "solid_stress" in arrays
    _, arrays, types = _read_vtu(files["plies"])
    assert len(types) == mesh.n_elements and set(types) == {12}
    mesh_only = write_vtu(tmp_path / "mesh.vtu", mesh)
    _, arrays, _ = _read_vtu(mesh_only["model"])
    assert "displacement" not in arrays and "laminate" in arrays
    pvd = write_pvd(tmp_path / "cases", {c.name: s for c, s in zip(cases, sols)}, layered=True)
    sets = ET.parse(pvd).getroot().findall("Collection/DataSet")
    assert len(sets) == 4 and all((tmp_path / d.get("file")).exists() for d in sets)

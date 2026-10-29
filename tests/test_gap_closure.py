"""Element family, orientation, loads, 3D benchmark, delamination and model checks."""

import warnings

import numpy as np
import pytest

from conftest import navier_fsdt_centre, simply_supported
from composite_laminate_model import (
    DOF,
    BoundaryConditions,
    CalculiXDeckWriter,
    AnalysisModel,
    CylindricalDirection,
    CylindricalPanelBuilder,
    DelaminationOnsetCriterion,
    ElasticProperties3D,
    EnrichedDOF,
    EquilibriumStressRecoverer,
    FailureCriterion,
    FailureResult,
    FailureMode,
    FEAMesh,
    FixedDirection,
    FSDTModel,
    CLPTModel,
    HygrothermalState,
    Laminate,
    LayerwiseModel,
    LoadCase,
    LoadCombination,
    MaterialCard,
    ModelChecker,
    NastranDeckWriter,
    PaganoPlate,
    RectangularPlateBuilder,
    RefinedZigZagModel,
    StrengthProperties3D,
    StructuralGlobalSolver,
    TemperatureDependentMaterial,
    apply_operating_temperature,
    available_criteria,
    create_criterion,
    laminate_at_temperature,
    register_criterion,
)
from composite_laminate_model.laminate_mesh_builders import _structured

ELEMENTS = ("quad4", "tri3", "quad8")


@pytest.fixture(scope="module")
def pagano_card():
    E2 = 1.0e9
    elastic = ElasticProperties3D(E1=25 * E2, E2=E2, E3=E2, v12=0.25, v13=0.25, v23=0.25,
                                  G12=0.5 * E2, G13=0.5 * E2, G23=0.2 * E2)
    return MaterialCard("pagano", elastic, StrengthProperties3D(1e9, 1e9, 5e7, 1e8, 6e7, Zt=4e7, S13=6e7, S23=5e7))


def distorted(element, laminate, kinematics, n=3):
    """Unit square mesh with interior nodes moved (edges stay straight)."""

    def point(s, t):
        bump = 0.12 * np.sin(np.pi * s) * np.sin(np.pi * t)
        return np.array([s + bump * (1 - 2 * t), t + bump * (2 * s - 1), 0.0])

    return _structured(n, n, point, laminate, kinematics, element, ("left", "right", "bottom", "top"), {})


# ---------------------------------------------------------------- elements
@pytest.mark.parametrize("element", ELEMENTS)
def test_each_element_has_exactly_six_rigid_body_modes(cfrp, element):
    lam = Laminate.from_angles(cfrp, [0, 45, 90], 0.25e-3)
    mesh = RectangularPlateBuilder(0.05, 0.03, 1, 1, element=element).build(lam, FSDTModel())
    for e in mesh.elements:
        K = e.stiffness_matrix()
        scale = 1.0 / np.sqrt(np.diag(K))  # membrane, bending and drilling terms differ by many decades
        eig = np.linalg.eigvalsh(scale[:, None] * K * scale[None, :])
        assert np.sum(np.abs(eig) < 1e-9 * eig.max()) == 6


@pytest.mark.parametrize("element", ELEMENTS)
def test_membrane_patch_test_on_distorted_mesh(cfrp, element):
    lam = Laminate.from_angles(cfrp, [0, 30, -30, 90], 0.25e-3)
    named = distorted(element, lam, FSDTModel())
    mesh = named.mesh
    exx, eyy, gxy = 1e-3, -4e-4, 6e-4
    field = lambda x: (exx * x[0] + 0.5 * gxy * x[1], eyy * x[1] + 0.5 * gxy * x[0])  # noqa: E731
    boundary = set(named.nodes("left") + named.nodes("right") + named.nodes("bottom") + named.nodes("top"))
    bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY])
    for n in boundary:
        ux, uy = field(mesh.node_coordinates(n))
        bcs.fix([n], [DOF.UX], ux).fix([n], [DOF.UY], uy).fix([n], [DOF.RZ])
    solution = StructuralGlobalSolver().solve(mesh, bcs, LoadCase())
    for n in set(mesh.node_ids) - boundary:
        ux, uy = field(mesh.node_coordinates(n))
        u = solution.node_displacement(n)
        assert u[0] == pytest.approx(ux, abs=1e-12) and u[1] == pytest.approx(uy, abs=1e-12)
    for e in mesh.elements:
        eps = e.generalized_strains(solution.element_displacements(e.element_id))[0]
        np.testing.assert_allclose(eps, [exx, eyy, gxy], atol=1e-11)


def test_incompatible_quad4_and_quad8_bend_in_plane_like_a_beam(cfrp):
    """In-plane cantilever, one element through the depth: the standard Q4 locks, QM6 and Quad8 do not."""
    lam = Laminate.from_angles(cfrp, [0], 1.0e-3)
    L, H, P = 1.0, 0.1, 10.0  # 4 x 1 elements of aspect ratio 2.5
    E = cfrp.elastic.E1
    beam = P * L**3 / (3 * E * lam.thickness * H**3 / 12) + P * L / (5 / 6 * cfrp.elastic.G12 * lam.thickness * H)
    ratios = {}
    for label, element, opts in (("q4", "quad4", {"membrane": "standard"}), ("qm6", "quad4", {}), ("q8", "quad8", {})):
        named = RectangularPlateBuilder(L, H, 4, 1, element=element).build_named(lam, FSDTModel(), **opts)
        mesh = named.mesh
        bcs = BoundaryConditions().fix(mesh.node_ids, [DOF.UZ, DOF.RX, DOF.RY]).fix(named.nodes("left"), [DOF.UX, DOF.UY])
        loads = LoadCase().add_line_load(named.nodes("right"), (0.0, P / H, 0.0))
        sol = StructuralGlobalSolver().solve(mesh, bcs, loads)
        ratios[label] = np.mean([sol.node_displacement(n)[1] for n in named.nodes("right")]) / beam
    assert ratios["q4"] < 0.85
    assert ratios["qm6"] == pytest.approx(1.0, abs=0.04)
    assert ratios["q8"] == pytest.approx(1.0, abs=0.04)


@pytest.mark.parametrize("element, n, tol", [("quad8", 6, 2e-3), ("quad4", 12, 1.5e-2), ("tri3", 16, 6e-2)])
def test_elements_converge_to_navier_fsdt(cfrp, element, n, tol):
    a, q, h = 0.2, 1.0e4, 4e-3
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], h / 4)
    kin = FSDTModel()
    named = RectangularPlateBuilder(a, a, n, n, element=element).build_named(lam, kin)
    mesh = named.mesh
    sol = StructuralGlobalSolver().solve(mesh, simply_supported(mesh, a, a), LoadCase().add_pressure(q))
    w = sol.node_displacement(mesh.closest_node((a / 2, a / 2)))[2]
    exact = navier_fsdt_centre(kin.abd_matrix(lam)[3:, 3:], kin.transverse_shear_matrix(lam), a, a, q)
    assert w == pytest.approx(exact, rel=tol)


# ------------------------------------------------------------- orientation
def test_rotated_plate_with_material_axis_gives_the_same_answer(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 45, -45, 90], 0.25e-3)
    a, theta = 0.2, np.radians(30.0)
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    results = []
    for rotate in (False, True):
        mesh = RectangularPlateBuilder(a, a, 6, 6, element="quad4").build(lam, FSDTModel())  # rebuilt below by node count
        bcs = simply_supported(mesh, a, a, hard=False)
        centre = mesh.closest_node((a / 2, a / 2))
        if rotate:
            rotated = FEAMesh()
            for n in mesh.node_ids:
                rotated.add_node(n, R @ mesh.node_coordinates(n))
            for e in mesh.elements:
                rotated.add_element(e.element_id, e.node_ids, lam, FSDTModel(), material_axis=FixedDirection(R[:, 0]))
            mesh = rotated
        sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_pressure(1e3))
        results.append(sol.node_displacement(centre)[2])
    assert results[1] == pytest.approx(results[0], rel=1e-9)


def test_pressurised_barrel_with_hoop_plies_carries_pr_over_t(cfrp):
    R, L, p = 0.3, 0.6, 2.0e5
    lam = Laminate.from_angles(cfrp, [90, 0, 0, 90], 0.25e-3)
    named = CylindricalPanelBuilder(R, L, 6, 24, 0.0, 360.0).build_named(
        lam, FSDTModel(), material_axis=CylindricalDirection(reference="axial"))
    mesh = named.mesh
    # free ends: only rigid-body restraint, so the axial stress is zero and the hoop resultant is pR
    bcs = BoundaryConditions()
    for n in named.nodes("x0"):
        bcs.fix([n], [DOF.UX])
    top = mesh.closest_node((0.0, 0.0, R))
    bottom = mesh.closest_node((0.0, 0.0, -R))
    side = mesh.closest_node((0.0, R, 0.0))
    bcs.fix([top, bottom], [DOF.UY]).fix([side], [DOF.UZ])
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_pressure(p))
    mid = [e for e in mesh.elements if abs(e.centroid[0] - L / 2) < L / 6]
    for e in mid:
        eps = e.generalized_strains(sol.element_displacements(e.element_id))[0]
        N = FSDTModel().abd_matrix(lam)[:3, :3] @ eps
        assert N[1] == pytest.approx(p * R, rel=0.02)  # hoop (element y follows the hoop direction)
        assert abs(N[0]) < 0.02 * p * R
    radial = np.mean([np.hypot(*sol.node_displacement(n)[1:3]) for n in mesh.node_ids])
    assert radial > 0.0


# ------------------------------------------------------------------- offset
def test_offset_shifts_the_abd_matrix_by_the_parallel_axis_rule(cfrp):
    base = Laminate.from_angles(cfrp, [0, 45, 90], 0.25e-3)
    h = base.thickness
    for offset in (h / 2, -0.3 * h):
        lam = base.with_offset(offset)
        d = lam.mid_plane
        A, B, D = (FSDTModel().abd_matrix(base)[i:i + 3, j:j + 3] for i, j in ((0, 0), (0, 3), (3, 3)))
        Ao, Bo, Do = (FSDTModel().abd_matrix(lam)[i:i + 3, j:j + 3] for i, j in ((0, 0), (0, 3), (3, 3)))
        np.testing.assert_allclose(Ao, A, rtol=1e-12)
        np.testing.assert_allclose(Bo, B + d * A, atol=1e-9 * np.abs(A).max() * h)
        np.testing.assert_allclose(Do, D + 2 * d * B + d * d * A, atol=1e-9 * np.abs(A).max() * h * h)


# -------------------------------------------------------------------- loads
@pytest.fixture
def plate(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.5e-3)
    named = RectangularPlateBuilder(0.2, 0.1, 8, 4).build_named(lam, FSDTModel())
    return named, lam


def total_reaction(sol, mesh):
    return sum(sol.node_reaction(n)[:3] for n in mesh.node_ids)


def test_line_body_and_pressure_field_loads_are_in_equilibrium(plate):
    named, lam = plate
    mesh = named.mesh
    bcs = BoundaryConditions().clamp(named.nodes("left"))
    loads = LoadCase().add_line_load(named.nodes("right"), (0.0, 0.0, 50.0))
    loads.add_body_load((0.0, 0.0, -9.81))
    loads.add_pressure(lambda x: 1.0e3 * x[0] / 0.2)  # linear ramp
    sol = StructuralGlobalSolver().solve(mesh, bcs, loads)
    rho_a = lam.areal_density()
    expected = 50.0 * 0.1 - 9.81 * rho_a * 0.02 + 1.0e3 * 0.5 * 0.02
    assert -total_reaction(sol, mesh)[2] == pytest.approx(expected, rel=1e-9)


def test_springs_and_skew_supports(plate):
    named, _ = plate
    mesh = named.mesh
    k, F = 2.0e6, 300.0
    bcs = BoundaryConditions().clamp(named.nodes("left"))
    tip = mesh.closest_node((0.2, 0.05))
    stiff = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_nodal_load(tip, DOF.UZ, F))
    flex = F / stiff.node_displacement(tip)[2]
    bcs.add_spring([tip], [DOF.UZ], k)
    sprung = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_nodal_load(tip, DOF.UZ, F))
    assert sprung.node_displacement(tip)[2] == pytest.approx(F / (flex + k), rel=1e-6)
    # skew support: displacement along (1, 1, 0)/sqrt 2 held at zero on the right edge
    skew = BoundaryConditions().clamp(named.nodes("left")).fix_direction(named.nodes("right"), (1.0, 1.0, 0.0))
    loads = LoadCase().add_line_load(named.nodes("right"), (1.0e3, 0.0, 0.0))
    sol = StructuralGlobalSolver().solve(mesh, skew, loads)
    for n in named.nodes("right"):
        u = sol.node_displacement(n)
        assert abs(u[0] + u[1]) < 1e-9 * abs(u[0]) + 1e-18


def test_solve_many_and_combinations_match_direct_solves(plate):
    named, _ = plate
    mesh = named.mesh
    bcs = BoundaryConditions().clamp(named.nodes("left"))
    dead = LoadCase("dead").add_body_load((0.0, 0.0, -9.81))
    wind = LoadCase("wind").add_pressure(800.0)
    tip = LoadCase("tip").add_line_load(named.nodes("right"), (0.0, 0.0, -20.0))
    solver = StructuralGlobalSolver()
    many = solver.solve_many(mesh, bcs, [dead, wind, tip])
    for case, result in zip((dead, wind, tip), many):
        single = solver.solve(mesh, bcs, case).displacements
        np.testing.assert_allclose(result.displacements, single, rtol=0.0, atol=1e-10 * np.abs(single).max())
    combo = LoadCombination("ULS", {"dead": 1.35, "wind": 1.5, "tip": 1.0})
    out = solver.solve_combinations(mesh, bcs, [dead, wind, tip], [combo])
    direct = LoadCase("uls").add_body_load((0.0, 0.0, -1.35 * 9.81)).add_pressure(1.5 * 800.0)
    direct.add_line_load(named.nodes("right"), (0.0, 0.0, -20.0))
    expected = solver.solve(mesh, bcs, direct).displacements
    np.testing.assert_allclose(out["ULS"].displacements, expected, rtol=0.0, atol=1e-9 * np.abs(expected).max())


# -------------------------------------------------------- Pagano benchmark
def pagano_fe(card, kinematics, S, n=12, element="quad4"):
    h = 0.01
    a = S * h
    lam = Laminate.from_angles(card, [0, 90, 0], h / 3)
    mesh = RectangularPlateBuilder(a, a, n, n, element=element).build(lam, kinematics)
    sol = StructuralGlobalSolver().solve(
        mesh, simply_supported(mesh, a, a), LoadCase().add_pressure(lambda x: np.sin(np.pi * x[0] / a) * np.sin(np.pi * x[1] / a))
    )
    return lam, a, mesh, sol


def test_pagano_reference_values(pagano_card):
    h = 0.01
    lam = Laminate.from_angles(pagano_card, [0, 90, 0], h / 3)
    ref = PaganoPlate(lam, 4 * h, 4 * h).nondimensional()
    assert ref["w"] == pytest.approx(2.0059, rel=2e-3)
    assert ref["sigma_x_top"] == pytest.approx(0.8008, rel=2e-3)
    assert ref["tau_xz_mid"] == pytest.approx(0.2559, rel=2e-3)
    thin = PaganoPlate(lam, 100 * h, 100 * h).nondimensional()
    assert thin["w"] == pytest.approx(0.4347, rel=2e-3)


@pytest.mark.parametrize("kinematics, tol", [(RefinedZigZagModel(), 0.035), (LayerwiseModel(sublayers=2), 0.03)])
def test_zigzag_and_layerwise_match_pagano_on_a_thick_plate(pagano_card, kinematics, tol):
    lam, a, mesh, sol = pagano_fe(pagano_card, kinematics, S=4)
    exact = PaganoPlate(lam, a, a).centre_deflection()
    w = sol.node_displacement(mesh.closest_node((a / 2, a / 2)))[2]
    assert w == pytest.approx(exact, rel=tol)
    fsdt = pagano_fe(pagano_card, FSDTModel(), S=4)[3]
    w_fsdt = fsdt.node_displacement(mesh.closest_node((a / 2, a / 2)))[2]
    assert abs(w - exact) < 0.5 * abs(w_fsdt - exact)


def test_equilibrium_recovery_gives_pagano_sigma_z_and_shear(pagano_card):
    lam, a, mesh, sol = pagano_fe(pagano_card, RefinedZigZagModel(), S=10, n=12)
    ref = PaganoPlate(lam, a, a)
    loads = LoadCase().add_pressure(lambda x: np.sin(np.pi * x[0] / a) * np.sin(np.pi * x[1] / a))
    rec = EquilibriumStressRecoverer(sol, loads)
    h = lam.thickness
    # tau_xz on the mid-plane next to the x = 0 support, sigma_z at the centre
    edge = min(mesh.elements, key=lambda e: np.hypot(e.centroid[0], e.centroid[1] - a / 2))
    centre = min(mesh.elements, key=lambda e: np.hypot(e.centroid[0] - a / 2, e.centroid[1] - a / 2))
    x, y = edge.centroid[:2]
    exact_tau = ref.state_at(lam.mid_plane)[0][3] * np.cos(np.pi * x / a) * np.sin(np.pi * y / a)
    assert rec.transverse_stresses(edge, 0.0, 0.0, lam.mid_plane)[1] == pytest.approx(exact_tau, rel=0.04)
    xc, yc = centre.centroid[:2]
    shape = np.sin(np.pi * xc / a) * np.sin(np.pi * yc / a)
    for z in (lam.z_interfaces[1], lam.mid_plane, lam.z_interfaces[2]):
        exact_sz = ref.state_at(float(z))[0][5] * shape
        assert rec.transverse_stresses(centre, 0.0, 0.0, float(z))[2] == pytest.approx(exact_sz, rel=0.03, abs=0.01)
    top = rec.transverse_stresses(centre, 0.0, 0.0, float(lam.z_interfaces[-1]))
    assert top[2] == pytest.approx(shape, rel=1e-9) and abs(top[0]) < 1e-12 and abs(top[1]) < 1e-12
    assert h > 0


# ------------------------------------------------------------ delamination
def test_delamination_criterion_ignores_compression_and_scales_quadratically(pagano_card):
    st = pagano_card.strength
    crit = DelaminationOnsetCriterion()
    r = crit.evaluate([0, 0, 0.5 * st.Zt, 0, 0.5 * st.S13, 0], st)
    assert r.index == pytest.approx(0.5) and r.mode is FailureMode.DELAMINATION
    assert r.strength_ratio == pytest.approx(np.sqrt(2.0))
    assert crit.evaluate([0, 0, -st.Zt, 0, 0, 0], st).index == 0.0
    assert DelaminationOnsetCriterion(1.0).evaluate([0, 0, -st.Zc, 0, 0, 0], st).index == pytest.approx(1.0)
    assert r.active_modes() == set()


def test_criterion_registry(pagano_card):
    class Never(FailureCriterion):
        name = "never"

        def evaluate(self, stress, strength):
            return FailureResult(self.name, 0.0, float("inf"), FailureMode.NONE)

    assert "delamination" in available_criteria()
    register_criterion("never_fails", Never)
    assert isinstance(create_criterion("never_fails"), Never)
    with pytest.raises(KeyError, match="already registered"):
        register_criterion("never_fails", Never)
    register_criterion("never_fails", Never, overwrite=True)
    register_criterion("broken", lambda: object(), overwrite=True)
    with pytest.raises(TypeError):
        create_criterion("broken")


def test_interface_checks_find_the_critical_interface(pagano_card):
    lam, a, mesh, sol = pagano_fe(pagano_card, RefinedZigZagModel(), S=10, n=8)
    loads = LoadCase().add_pressure(lambda x: 1e6 * np.sin(np.pi * x[0] / a) * np.sin(np.pi * x[1] / a))
    sol = StructuralGlobalSolver().solve(mesh, simply_supported(mesh, a, a), loads)
    rec = EquilibriumStressRecoverer(sol, loads)
    checks = rec.interface_checks(mesh.elements[0])
    assert [c.interface for c in checks] == [1, 2]
    worst = rec.critical_interface()
    assert worst.result.index > 0.0
    # interlaminar shear peaks at the supports, so the critical point is in an edge element
    e = mesh.element(worst.element_id)
    assert min(e.centroid[0], a - e.centroid[0], e.centroid[1], a - e.centroid[1]) < a / 8


# ------------------------------------------------------- temperature tables
def test_temperature_dependent_material_interpolates(cfrp):
    hot_elastic = ElasticProperties3D(**{**cfrp.elastic.as_dict(), "E2": 0.6 * cfrp.elastic.E2, "G12": 0.5 * cfrp.elastic.G12})
    hot = MaterialCard(cfrp.name, hot_elastic, cfrp.strength, cfrp.hygrothermal, cfrp.density)
    table = TemperatureDependentMaterial({20.0: cfrp, 120.0: hot})
    mid = table.at_temperature(70.0)
    assert mid.elastic.E2 == pytest.approx(0.8 * cfrp.elastic.E2)
    assert mid.elastic.E1 == pytest.approx(cfrp.elastic.E1)
    assert mid.name == f"{cfrp.name} [T=70]"
    with pytest.raises(ValueError, match="outside"):
        table.at_temperature(150.0)
    assert TemperatureDependentMaterial({20.0: cfrp, 120.0: hot}, extrapolate=True).at_temperature(150.0).elastic.E2 == \
        pytest.approx(hot.elastic.E2)
    lam = Laminate.from_angles(cfrp, [0, 90], 0.25e-3)
    hot_lam = laminate_at_temperature(laminate_at_temperature(lam, {cfrp.name: table}, 70.0), {cfrp.name: table}, 120.0)
    assert all(p.material.elastic.E2 == pytest.approx(hot.elastic.E2) for p in hot_lam.plies)
    mesh = RectangularPlateBuilder(0.1, 0.1, 2, 2).build(lam, FSDTModel())
    loads = LoadCase().set_hygrothermal(HygrothermalState(50.0), element_ids=[1, 2])
    apply_operating_temperature(mesh, loads, {cfrp.name: table}, reference_temperature=20.0)
    assert mesh.element(1).laminate.plies[0].material.elastic.E2 == pytest.approx(0.8 * cfrp.elastic.E2)
    assert mesh.element(4).laminate.plies[0].material.elastic.E2 == pytest.approx(cfrp.elastic.E2)


# ------------------------------------------------------------- model checks
def test_model_checker_flags_common_mistakes(cfrp):
    lam = Laminate.from_angles(cfrp, [0, 90, 90, 0], 0.25e-3)
    good = RectangularPlateBuilder(0.2, 0.2, 4, 4).build(lam, FSDTModel())
    bcs = BoundaryConditions().fix(good.node_ids, [DOF.UZ]).fix([1], [DOF.UX, DOF.UY])
    assert ModelChecker().check(good, bcs).ok
    assert ModelChecker().check(good, bcs).issues == []
    # millimetre geometry with Pa moduli, long thin elements, CLPT on a thick plate
    mm = RectangularPlateBuilder(200.0, 20.0, 2, 4).build(Laminate.from_angles(cfrp, [0] * 8, 2.0), CLPTModel())
    report = ModelChecker().check(mm, BoundaryConditions().fix([1], [DOF.UZ]))
    assert {"units", "aspect_ratio", "clpt_thick_plate", "unsupported_rigid_body"} <= report.codes()
    assert not report.ok and "error" in str(report)


# ------------------------------------------------------------------- decks
@pytest.mark.parametrize("element, ccx_type, nastran_card", [("tri3", "S6", "CTRIA3"), ("quad8", "S8R", "CQUAD8")])
def test_decks_write_tri3_and_quad8(cfrp, tmp_path, element, ccx_type, nastran_card):
    lam = Laminate.from_angles(cfrp, [0, 90], 0.5e-3)
    mesh = RectangularPlateBuilder(0.2, 0.2, 3, 3, element=element).build(lam, FSDTModel())
    model = AnalysisModel(mesh, simply_supported(mesh, 0.2, 0.2, hard=False), LoadCase().add_pressure(1e3), "elements")
    ccx = CalculiXDeckWriter().write(model, tmp_path, "ccx").path.read_text()
    assert f"TYPE={ccx_type}" in ccx
    nas = NastranDeckWriter().write(model, tmp_path, "nas").path.read_text().splitlines()
    assert sum(line.startswith(nastran_card) for line in nas) == mesh.n_elements


def test_decks_carry_orientation_offset_springs_and_temperature_fields(cfrp, tmp_path):
    lam = Laminate.from_angles(cfrp, [0, 90], 0.5e-3).with_offset(0.5e-3)
    named = RectangularPlateBuilder(0.2, 0.2, 2, 2).build_named(lam, FSDTModel())
    mesh = named.mesh
    mesh.set_material_axis(FixedDirection((0.0, 1.0, 0.0)))
    bcs = simply_supported(mesh, 0.2, 0.2, hard=False).add_spring([mesh.closest_node((0.1, 0.1))], [DOF.UZ], 1e5)
    loads = LoadCase().add_pressure(1e3)
    for eid, dT in zip(mesh.element_ids, (10.0, 20.0, 30.0, 40.0)):
        loads.set_hygrothermal(HygrothermalState(dT, temperature_gradient=1e3), element_ids=[eid])
    model = AnalysisModel(mesh, bcs, loads, "features")
    nas = NastranDeckWriter().write(model, tmp_path, "nas").path.read_text()
    assert "CORD2R" in nas and "CELAS2" in nas and "TEMPP1" in nas
    ccx = CalculiXDeckWriter()
    with pytest.raises(Exception):
        ccx.write(model, tmp_path, "ccx")  # through-thickness gradients are not supported in CalculiX
    loads2 = LoadCase().add_pressure(1e3)
    text = ccx.write(AnalysisModel(mesh, bcs, loads2, "features2"), tmp_path, "ccx").path.read_text()
    assert "OFFSET=-0.5" in text and "SPRING1" in text
    assert "0, 1, 0, -1, 0, 0" in text

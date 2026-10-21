"""Blade builder, connections (RBE2 / RBE3) and beam section properties."""

import numpy as np
import pytest

from laminate_modelling_package import (
    DOF,
    Airfoil,
    BladeBuilder,
    BoundaryConditions,
    CalculiXDeckWriter,
    AnalysisModel,
    CylindricalPanelBuilder,
    FSDTModel,
    Laminate,
    LoadCase,
    Ply,
    ShearWeb,
    SkinRegion,
    Station,
    StructuralGlobalSolver,
    add_reference_node,
    beam_section_properties,
    distributing_link,
    rigid_link,
)

REGIONS = [SkinRegion("le_panel", 0.0, 0.15, 3), SkinRegion("spar_cap", 0.15, 0.45, 4),
           SkinRegion("te_panel", 0.45, 1.0, 5)]
WEBS = [ShearWeb("web_fwd", 0.15, 3), ShearWeb("web_aft", 0.45, 3)]


def laminates(cfrp):
    skin = Laminate.from_angles(cfrp, [45, -45, 0, -45, 45], 0.25e-3)
    return {
        "le_panel": skin, "te_panel": skin,
        "spar_cap": Laminate.from_angles(cfrp, [45, -45] + [0] * 8 + [-45, 45], 0.25e-3),
        "web_fwd": Laminate.from_angles(cfrp, [45, -45, -45, 45], 0.25e-3),
        "web_aft": Laminate.from_angles(cfrp, [45, -45, -45, 45], 0.25e-3),
    }


def test_airfoils():
    naca = Airfoil.naca4("0018")
    assert naca.max_thickness == pytest.approx(0.18, rel=2e-3)
    assert naca.surface("upper", 0.3) == pytest.approx(-naca.surface("lower", 0.3))
    cambered = Airfoil.naca4("4412")
    assert cambered.surface("upper", 0.4) + cambered.surface("lower", 0.4) == pytest.approx(0.08, abs=2e-3)
    circle = Airfoil.ellipse(1.0)
    assert circle.thickness(0.5) == pytest.approx(1.0, rel=1e-6)
    # Selig order or reversed gives the same surfaces
    pts = np.array([[1, 0], [0.5, 0.1], [0, 0], [0.5, -0.05], [1, 0]], float)
    assert Airfoil(pts).surface("upper", 0.5) == pytest.approx(Airfoil(pts[::-1]).surface("upper", 0.5))
    with pytest.raises(ValueError):
        Airfoil.naca4("18")


@pytest.mark.parametrize("element", ["quad4", "quad8", "quad8_curved"])
def test_blade_builder_regions_webs_and_normals(cfrp, element):
    af = Airfoil.naca4("0021")
    stations = [Station(0.0, 1.0, af, twist=0.0, pitch_axis=0.3), Station(3.0, 0.7, Airfoil.naca4("0015"), twist=-6.0,
                                                                           pitch_axis=0.3)]
    blade = BladeBuilder(stations, REGIONS, WEBS, n_span=6, element=element).build(laminates(cfrp), FSDTModel())
    mesh = blade.mesh
    sets = blade.element_sets
    for name in ("le_panel_ss", "le_panel_ps", "spar_cap_ss", "spar_cap_ps", "te_panel", "web_fwd", "web_aft",
                 "skin", "webs"):
        assert sets[name], name
    assert len(sets["spar_cap"]) == 2 * 4 * 6 and len(sets["web_fwd"]) == 3 * 6
    assert sorted(sets["all"]) == sorted(sets["skin"] + sets["webs"])
    assert {e.TOPOLOGY for e in mesh.elements} == {element}
    # skin normals point out of the section, web normals towards the trailing edge
    for eid in sets["skin"]:
        e = mesh.element(eid)
        assert e.frame[2][:2] @ (np.asarray(e.centroid)[:2] - _section_centre(blade, e.centroid[2])) > 0
    for eid in sets["web_aft"]:
        e = mesh.element(eid)
        a, b, w, chord, twist, *_ = BladeBuilder(stations, REGIONS, WEBS)._station_at(e.centroid[2])
        chord_dir = np.array([np.cos(np.radians(twist)), -np.sin(np.radians(twist))])
        assert e.frame[2][:2] @ chord_dir > 0.9
    # outer mould line reference: the laminate grows inward
    cap = mesh.element(sets["spar_cap_ss"][0])
    assert cap.laminate.z_interfaces[-1] == pytest.approx(0.0, abs=1e-15)
    assert abs(cap.frame[0][2]) > 0.99  # 0-degree plies run along the span
    # webs share nodes with the skins
    joint = set(blade.nodes("web_fwd_ss"))
    skin_nodes = {n for eid in sets["spar_cap_ss"] for n in mesh.element(eid).node_ids}
    assert joint <= skin_nodes
    assert len(blade.nodes("root")) == len(blade.nodes("tip"))
    # tip of the twisted, tapered blade: chord 0.7 rotated by -6 degrees
    le = [n for n in blade.nodes("le") if n in blade.nodes("tip")][0]
    x = mesh.node_coordinates(le)
    assert x[0] == pytest.approx(-0.3 * 0.7 * np.cos(np.radians(6)), abs=1e-4)
    assert x[1] == pytest.approx(-0.3 * 0.7 * np.sin(np.radians(6)), abs=1e-4)


def _section_centre(blade, z):
    pts = np.array([blade.mesh.node_coordinates(n)[:2] for n in blade.node_sets["all"]
                    if abs(blade.mesh.node_coordinates(n)[2] - z) < 0.3])
    return pts.mean(axis=0)


def test_blunt_trailing_edges_get_a_web_and_errors_are_clear(cfrp):
    pts = Airfoil.naca4("0018", closed_te=False)
    blade = BladeBuilder([Station(0, 1, pts), Station(1, 1, pts)], REGIONS, n_span=2).build(
        laminates(cfrp) | {"te_web": laminates(cfrp)["web_fwd"]}, FSDTModel())
    assert blade.element_sets["te_web"]
    with pytest.raises(ValueError, match="region boundary"):
        BladeBuilder([Station(0, 1, pts), Station(1, 1, pts)], REGIONS, [ShearWeb("w", 0.3)])
    with pytest.raises(ValueError, match="gap"):
        BladeBuilder([Station(0, 1, pts), Station(1, 1, pts)], [SkinRegion("a", 0, 0.4), SkinRegion("b", 0.5, 1)])
    with pytest.raises(KeyError, match="No laminate"):
        BladeBuilder([Station(0, 1, pts), Station(1, 1, pts)], REGIONS, n_span=1).build({}, FSDTModel())


@pytest.mark.parametrize("element", ["quad4", "quad8_curved"])
def test_beam_properties_of_a_thin_tube(aluminium, element):
    t, R = 2e-3, 0.25
    lam = Laminate([Ply(aluminium, t)])
    circle = Airfoil.ellipse(1.0, 161)
    blade = BladeBuilder([Station(0, 2 * R, circle, pitch_axis=0.5), Station(6, 2 * R, circle, pitch_axis=0.5)],
                         [SkinRegion("a", 0, 0.5, 8), SkinRegion("b", 0.5, 1.0, 8)], n_span=24, element=element,
                         skin_reference="mid").build(lam, FSDTModel())
    p = beam_section_properties(blade)
    E = aluminium.elastic.E1
    G = E / (2 * (1 + aluminium.elastic.v12))
    tol = 0.012 if element == "quad4" else 1e-3
    assert p.EA == pytest.approx(E * 2 * np.pi * R * t, rel=tol)
    assert p.EI_x == pytest.approx(E * np.pi * R**3 * t, rel=tol)
    assert p.EI_y == pytest.approx(E * np.pi * R**3 * t, rel=tol)
    assert p.GJ == pytest.approx(G * 2 * np.pi * R**3 * t, rel=tol)
    assert np.allclose(p.shear_centre, 0.0, atol=1e-6) and np.allclose(p.tension_centre, 0.0, atol=1e-6)
    assert p.mass_per_length == pytest.approx(aluminium.density * 2 * np.pi * R * t, rel=tol)
    assert np.allclose(p.bend_twist_coupling, 0.0, atol=1e-6)


def test_beam_properties_show_bend_twist_coupling_and_offset_centres(cfrp):
    af = Airfoil.naca4("0024")
    lams = laminates(cfrp)
    stations = [Station(0, 1.0, af, pitch_axis=0.25), Station(6, 1.0, af, pitch_axis=0.25)]
    blade = BladeBuilder(stations, REGIONS, WEBS, n_span=24).build(lams, FSDTModel())
    plain = beam_section_properties(blade)
    assert max(np.abs(plain.bend_twist_coupling)) < 5e-3
    # box spar between the webs: the shear centre lies aft of the quarter chord, inside the box
    assert 0.0 < plain.shear_centre[0] < 0.2 and abs(plain.shear_centre[1]) < 1e-3
    assert 0.0 < plain.tension_centre[0] < 0.2
    # off-axis cap fibres: opposite angles on the two sides couple flap bending and twist,
    # equal angles (mirrored by the opposite normals) couple extension and twist
    off = lambda a: Laminate.from_angles(cfrp, [a] * 12, 0.25e-3)  # noqa: E731
    bend_twist = beam_section_properties(BladeBuilder(stations, REGIONS, WEBS, n_span=24).build(
        dict(lams, spar_cap_ss=off(20), spar_cap_ps=off(-20)), FSDTModel()))
    ext_twist = beam_section_properties(BladeBuilder(stations, REGIONS, WEBS, n_span=24).build(
        dict(lams, spar_cap=off(20)), FSDTModel()))
    S = bend_twist.stiffness
    assert abs(bend_twist.bend_twist_coupling[0]) > 0.2
    assert abs(S[0, 3]) / np.sqrt(S[0, 0] * S[3, 3]) < 0.05
    S = ext_twist.stiffness
    assert abs(S[0, 3]) / np.sqrt(S[0, 0] * S[3, 3]) > 0.1 and abs(ext_twist.bend_twist_coupling[0]) < 0.05


def test_rigid_and_distributing_links(aluminium):
    lam = Laminate([Ply(aluminium, 2e-3)])
    R, L = 0.25, 3.0
    named = CylindricalPanelBuilder(R, L, 12, 16, 0, 360, element="quad8_curved").build_named(lam, FSDTModel())
    mesh = named.mesh
    bcs = BoundaryConditions().clamp(named.nodes("x0"))
    ref = add_reference_node(mesh, (L, 0.0, 0.0))
    distributing_link(mesh, bcs, ref, named.nodes("x1"))
    E = aluminium.elastic.E1
    sol = StructuralGlobalSolver().solve(mesh, bcs, LoadCase().add_nodal_load(ref, DOF.UZ, 100.0))
    # Euler-Bernoulli plus a small Timoshenko shear term for a thin tube
    I, A = np.pi * R**3 * 2e-3, 2 * np.pi * R * 2e-3
    G = E / (2 * (1 + aluminium.elastic.v12))
    w = 100.0 * L**3 / (3 * E * I) + 100.0 * L / (0.5 * G * A)
    assert sol.node_displacement(ref)[2] == pytest.approx(w, rel=0.02)
    # RBE2: a rigid end keeps the section plane and round
    bcs2 = BoundaryConditions().clamp(named.nodes("x0"))
    ref2 = add_reference_node(mesh, (L, 0.0, 0.0))
    rigid_link(mesh, bcs2, ref2, named.nodes("x1"))
    bcs2.fix([ref], range(6))  # the unused RBE3 node from above
    sol2 = StructuralGlobalSolver().solve(mesh, bcs2, LoadCase().add_nodal_load(ref2, DOF.RX, 50.0))
    twist = sol2.node_displacement(ref2)[3]
    assert twist == pytest.approx(50.0 * L / (G * 2 * np.pi * R**3 * 2e-3), rel=0.01)
    for n in named.nodes("x1"):
        x = mesh.node_coordinates(n) - mesh.node_coordinates(ref2)
        assert np.allclose(sol2.node_displacement(n)[:3], np.cross([twist, 0, 0], x), atol=1e-12)
    with pytest.raises(ValueError, match="collinear"):
        distributing_link(mesh, BoundaryConditions(), ref, named.nodes("x1")[:1] * 3)


def test_links_are_written_to_decks(aluminium, tmp_path):
    lam = Laminate([Ply(aluminium, 2e-3)])
    named = CylindricalPanelBuilder(0.25, 1.0, 4, 8, 0, 360).build_named(lam, FSDTModel())
    bcs = BoundaryConditions().clamp(named.nodes("x0"))
    ref = add_reference_node(named.mesh, (1.0, 0.0, 0.0))
    distributing_link(named.mesh, bcs, ref, named.nodes("x1"))
    deck = CalculiXDeckWriter().write(AnalysisModel(named.mesh, bcs, LoadCase().add_nodal_load(ref, 2, 1.0)),
                                      tmp_path, "job").path.read_text()
    assert deck.count("*EQUATION") == 6

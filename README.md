# composite-laminate-fea

[![tests](https://github.com/samhughes-fea/composite-laminate-fea/actions/workflows/tests.yml/badge.svg)](https://github.com/samhughes-fea/composite-laminate-fea/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/samhughes-fea/composite-laminate-fea/blob/main/LICENSE)

`laminate_modelling_package` (distributed as `composite-laminate-fea`) is a Python suite for the linear static analysis of
composite blade structures (wind and tidal turbine blades, helicopter blades and
aircraft wings: sandwich and monolithic skins, spar caps and shear webs), built on
general laminate mechanics from thin plates to thick laminates:

- a blade section builder: airfoils, chordwise skin regions, spar caps and shear webs,
  twist, taper and pitch axis, and beam section properties extracted from the shell
  model: the Timoshenko 6 × 6 stiffness with all its couplings (and the classical
  4 × 4 inside it), the Timoshenko-Vlasov 7 × 7 with the warping DOF, and the shear,
  tension and mass centres
- torsion beyond Saint-Venant: thin-walled section properties (multi-cell shear flows,
  warping function, shear centre, EΓ, and the 6 × 6 and 7 × 7 stiffness from thin-wall
  theory), restrained-warping beams with the bimoment
  (Vlasov, Benscoter), and Generalised Beam Theory with the section's distortional modes
  and ribs
- ply books: laminates defined ply by ply over element sets, with ply drops and
  per-ply results
- 3D orthotropic materials, plies and laminates, with temperature-dependent properties
- four kinematic theories: CLPT, FSDT, refined zig-zag (RZT) and layerwise, the
  last two with their own zig-zag / interface DOFs
- Tri3, Quad4 and Quad8 composite shell elements, and curved Quad8 / Tri6 shells,
  behind one `ShellElement` interface
- layered Hex8 / Hex20 solids for 3D detail (ply drops, bonded joints), tied into a
  shell model
- rigid (RBE2) and distributing (RBE3) links for load introduction and measurement
- material orientation on curved shells, laminate offsets, springs and skew supports
- pressure fields, line loads, body loads and hygrothermal (thermal and moisture) loads;
  many load cases with one factorisation, and load combinations
- through-thickness stress recovery, including σz and interlaminar shear from 3D equilibrium
- failure criteria (with a delamination onset check), smeared-crack progressive damage
- Pagano's exact 3D solution as a built-in benchmark
- structured and Gmsh meshing, and pre-solve model checks
- pluggable solver kernels: native NumPy/SciPy, CalculiX and MSC/NX Nastran
- ParaView (VTK) export of the model and of every ply

```
.
├── pyproject.toml                          package metadata (name composite-laminate-fea, version from __init__)
├── LICENSE                                 MIT
├── CHANGELOG.md
├── RELEASING.md                            how to cut a release (PyPI publishing is manual and off by default)
├── src/
│   └── laminate_modelling_package/
│       ├── __init__.py                     public API and __version__
│       ├── run_solver_benchmarks.py        benchmark runner (laminate-benchmarks command)
│       ├── laminate_materials_3d.py        ElasticProperties3D, StrengthProperties3D, 3D Bond rotations
│       ├── laminate_materials.py           MaterialCard, MaterialRepository / MaterialLibrary
│       ├── laminate_hygrothermal.py        HygrothermalProperties/State, HygrothermalEngine (N_ht, M_ht)
│       ├── laminate_mechanics.py           Ply (Q, Q_s, Q_bar, Q_bar_s), Laminate, Reuter transforms
│       ├── laminate_temperature.py         TemperatureDependentMaterial
│       ├── laminate_kinematics.py          CLPT, FSDT, RZT, layerwise; generalised section matrices
│       ├── laminate_elements.py            ShellElement, Tri3 / Quad4 / Quad8 shells, ElementRegistry
│       ├── laminate_curved_shell.py        curved Quad8 / Tri6 shells (normals from the surface)
│       ├── laminate_solid.py               layered Hex8 / Hex20 solids, solid_block, shell-solid coupling
│       ├── laminate_blade.py               Airfoil, Station, BladeBuilder, beam_section_properties
│       ├── laminate_thin_wall.py           ThinWalledSection: Saint-Venant shear flows, GJ, warping function, EΓ
│       ├── laminate_torsion.py             RestrainedTorsionBeam: Benscoter / Vlasov torsion, bimoment
│       ├── laminate_gbt.py                 GeneralizedBeam, SectionModes: GBT warping and distortion, ribs
│       ├── laminate_connectors.py          reference nodes, rigid_link (RBE2), distributing_link (RBE3)
│       ├── laminate_layup.py               PlyDefinition, Layup: ply books, ply drops, per-ply results
│       ├── laminate_orientation.py         FixedDirection, CylindricalDirection, FieldDirection
│       ├── laminate_fea_engine.py          compatibility imports for the two modules above
│       ├── laminate_stress_recovery.py     StressRecoverer, ExactStressRecoverer
│       ├── laminate_equilibrium.py         EquilibriumStressRecoverer (σz, interlaminar shear, delamination)
│       ├── laminate_elasticity_3d.py       PaganoPlate: exact 3D cross-ply solution
│       ├── laminate_failure_criteria.py    Max stress, Tsai-Hill, Tsai-Wu, Hashin, delamination; registry
│       ├── laminate_progressive_damage.py  SmearedCrackDegradation, ProgressiveDamageAnalysis
│       ├── laminate_solver_pipeline.py     FEAMesh, BoundaryConditions, LoadCase, LoadCombination, solver
│       ├── laminate_mesh_builders.py       RectangularPlateBuilder, CylindricalPanelBuilder
│       ├── laminate_gmsh.py                GmshMeshImporter, GmshPlateBuilder (optional Gmsh meshing)
│       ├── laminate_model_check.py         ModelChecker: quality, slenderness, units, supports
│       ├── kernels/                        FEAKernel + Native / CalculiX / Nastran adapters and deck writers
│       ├── laminate_kernels.py             compatibility imports for kernels/
│       ├── laminate_benchmark_stress.py    StressDistributionVerifier, ThicknessProfile
│       ├── laminate_results.py             ResultsExtractor: Gauss point, ply, element, node and global results
│       └── laminate_vtk.py                 write_vtu / write_pvd: ParaView files, one cell per ply
├── examples/                               twenty-eight worked examples with figures (see examples/README.md)
└── tests/                                  pytest suite
```

## Install

The package is not on PyPI yet. Install it from a clone:

```bash
pip install -e .             # NumPy only (dense solver)
pip install -e ".[sparse]"   # + SciPy sparse solver
pip install -e ".[gmsh]"     # + Gmsh meshing
pip install -e ".[examples]" # + SciPy, matplotlib and Gmsh for the examples
pip install -e ".[dev]"      # everything above, plus pytest, build and twine
pytest                       # run the test suite
```

The import name is `laminate_modelling_package`; the distribution name is
`composite-laminate-fea`.

Python 3.9 or newer. SciPy is optional: without it the solver falls back to dense Cholesky.

## Quick start

```python
from laminate_modelling_package import (
    DOF, BoundaryConditions, ExactStressRecoverer, FEAMesh, FSDTModel,
    HashinCriterion, HygrothermalState, Laminate, LoadCase,
    StructuralGlobalSolver, create_default_library,
)

card = create_default_library().get("T300/5208")
laminate = Laminate.from_angles(card, [0, 45, -45, 90], ply_thickness=0.25e-3)
mesh = FEAMesh.rectangular_plate(0.2, 0.2, 10, 10, laminate, FSDTModel())

bcs = BoundaryConditions()
edges = {n for axis in (0, 1) for v in (0.0, 0.2) for n in mesh.nodes_on_line(axis, v)}
bcs.fix(edges, [DOF.UZ])
bcs.fix([mesh.closest_node((0, 0))], [DOF.UX, DOF.UY])
bcs.fix([mesh.closest_node((0.2, 0))], [DOF.UY])

loads = LoadCase().add_pressure(1e3).set_hygrothermal(HygrothermalState(delta_temperature=-50))
result = StructuralGlobalSolver().solve(mesh, bcs, loads)

element = mesh.element(56)
profile = ExactStressRecoverer().through_thickness_profile(
    element, result.element_displacements(56), state=loads.state_for(56))
for p in profile:
    print(p.ply_index, p.z, p.stress_material, HashinCriterion("3d").evaluate(
        p.material_stress_3d(), laminate.plies[p.ply_index].material.strength).index)
```

## Blade structures

`BladeBuilder` meshes a blade or wing segment from stations along the span. Each
station gives a chord, an airfoil, a twist and a pitch axis; skin regions are
chordwise bands of the surface and webs sit at chord fractions. Laminates are
given by region name, with `_ss` / `_ps` suffixes where the suction and pressure
sides differ:

```python
from laminate_modelling_package import (
    Airfoil, BladeBuilder, FSDTModel, ShearWeb, SkinRegion, Station, beam_section_properties)

stations = [Station(0.0, 1.2, Airfoil.naca4("0024"), pitch_axis=0.35),
            Station(10.0, 0.6, Airfoil.naca4("0018"), twist=-6.0, pitch_axis=0.35)]
regions = [SkinRegion("le_panel", 0.0, 0.2, 4), SkinRegion("spar_cap", 0.2, 0.5, 6),
           SkinRegion("te_panel", 0.5, 1.0, 8)]
webs = [ShearWeb("web_fwd", 0.2, 4), ShearWeb("web_aft", 0.5, 4)]
blade = BladeBuilder(stations, regions, webs, n_span=40).build(
    {"le_panel": sandwich, "te_panel": sandwich, "spar_cap": cap, "web_fwd": web, "web_aft": web}, FSDTModel())

bcs = BoundaryConditions().clamp(blade.nodes("root"))
props = beam_section_properties(blade)   # of a prismatic segment
props.stiffness_6x6, props.stiffness, props.shear_centre, props.mass_per_length, props.bend_twist_coupling
props = beam_section_properties(blade, warping=True, end_ribs=0.2)   # adds stiffness_7x7, EGamma
```

**Section stiffness: 4 × 4, 6 × 6 and 7 × 7.** The matrices nest, and each adds a
deformation the one before leaves out:

| Matrix | Strains | Forces | Adds |
|---|---|---|---|
| `stiffness` (4 × 4, classical) | ε, κx, κy, κz | N, Mx, My, Mz | extension, bending, Saint-Venant torsion and their couplings |
| `stiffness_6x6` (Timoshenko) | γx, γy, ε, κx, κy, κz | Fx, Fy, N, Mx, My, Mz | transverse shear and its couplings with twist (the shear centre) and bending |
| `stiffness_7x7` (Timoshenko-Vlasov) | ... and κz′ | ... and the bimoment B | restrained warping: EΓ, with Benscoter's warping shear stiffness S alongside |

The 6 × 6 is in the BeamDyn / windIO order, about the reference point, with
γx = u_x′ − θy and γy = u_y′ + θx. The 4 × 4 is the 6 × 6 with the shear forces free,
so it is what an Euler-Bernoulli code wants. `transform_beam_stiffness(K, offset,
angle)` moves a 6 × 6 to another point or turns its axes. `ThinWalledSection` gives
the same matrices from thin-wall theory (`timoshenko_stiffness`,
`vlasov_stiffness`), from no shell model at all. The two share no code, and
[example 29](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/29_timoshenko_vlasov_stiffness/) compares them with each other
and with the published IEA 15 MW 6 × 6.

- **Geometry.** z is the span, x the chord (leading to trailing edge) and y the
  thickness, with the pitch axis at x = 0. `Airfoil.naca4(code)`, `Airfoil.ellipse(t)`
  or `Airfoil(points)` for any coordinates (Selig order), splined so nodes are spaced
  evenly along the arc. Blunt trailing edges get a closing web.
- **Laminates.** Skin normals point outwards and skins grow inwards from the outer
  mould line (`skin_reference="outer"`), then are meshed on their mid-surface (see
  Mesh surface); the 0° direction follows the span.
  Different angles on the two caps give bend-twist or extension-twist coupling.
- **Sets.** Element sets per region and side (`spar_cap_ss`, `te_panel`, `skin`,
  `webs`, ...) and node sets (`root`, `tip`, `le`, `te`, each web) for loads,
  supports and results.
- **Elements.** `element="quad8_curved"` (default), `"quad8"` or `"quad4"`. On the
  default mid-surface mesh all three keep the torsional stiffness of thick-walled
  sections; with skins left on the mould line, the flat elements lose it (see
  Limitations).
- **Mesh surface.** `mesh_surface="mid"` (default) meshes the skins on the
  laminates' mid-surface: skin nodes move inward from the mould line by the local
  offset, smoothed along the skin across region edges and ply drops
  (`mid_surface_smoothing`, two of the thickest skins by default), the webs end on
  the moved skins, and the offsets go (small residuals stay in the blends, so the
  plies keep their place). After a `Layup`, call `blade = blade.to_mid_surface()`:
  it moves the blade again from the mould line with the ply-by-ply thicknesses.
  `mesh_surface="outer"` keeps the nodes on the mould line and offsets the
  laminates, for when the true outer geometry matters. Example 26 compares the two.

`add_reference_node`, `rigid_link` (RBE2) and `distributing_link` (RBE3) bring
section loads from an aeroelastic code into the model, and measure section motion.
See examples [20](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/20_blade_beam_properties/),
[21](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/21_spar_cap_ply_drops/) and [23](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/23_blade_segment/).

## Torsion: Saint-Venant, restrained warping and distortion

`beam_section_properties` gives the Saint-Venant GJ of a segment. Three further
tools follow the torsional response beyond it, each adding one physical
contribution:

```python
from laminate_modelling_package import GeneralizedBeam, RestrainedTorsionBeam, ThinWalledSection

section = ThinWalledSection.from_blade_mesh(blade, z=5.0)       # or from_segments(...)
section.GJ, section.shear_flow, section.omega, section.shear_centre, section.EGamma

beam = RestrainedTorsionBeam.from_section(section, length=10.0)    # Benscoter; vlasov=True for Vlasov
r = beam.cantilever_tip_torque(1e3)      # twist, warping amplitude, bimoment, Saint-Venant / warping torque
r = beam.solve(distributed_torque=50.0, supports={0.0: (True, True)})

gbt = GeneralizedBeam(section, 10.0, warping="full", distortional_modes=8,
                      z_nodes=GeneralizedBeam.graded_nodes(10.0, 40, stations=[5.0]))
gbt.clamp(0.0).diaphragm(5.0).add_wall_load(10.0, gbt.walls_named("te_panel"), (0.0, -2e3))
sol = gbt.solve()
sol.mean_twist(z), sol.bimoment(z), sol.distortion(z, 0), sol.wall_resultants(z)
```

- **Saint-Venant** (`ThinWalledSection`). The section is a network of laminate walls.
  The warping function is the one that minimises the membrane shear energy, which
  gives the Bredt-Batho shear flows in any number of cells and the sectorial
  coordinate on open branches in one solve. GJ adds each wall's plate twisting. The
  shear centre is the pole that makes ω orthogonal to bending, and EΓ = ∫ Et ω² ds
  plus the walls' own bending.
- **Restrained warping** (`RestrainedTorsionBeam`). Benscoter's theory gives the
  warping its own amplitude φ(z), with shear stiffness S = Ip − J; Vlasov's ties it to
  the twist rate (S → ∞). Solved in closed form for cantilevers and by finite elements
  for any supports. Outputs: bimoment B = −EΓ φ', warping stress N_zz = B Et ω / EΓ,
  and the split of the torque into Saint-Venant and warping parts.
- **Distortion** (`GeneralizedBeam`, Generalised Beam Theory). Every section point
  gets an axial warping displacement and an in-plane displacement and rotation. The
  walls are Kirchhoff plate strips with their full stiffness, couplings included. The
  in-plane fields split into the three rigid modes and the distortional modes, which
  are free of work with the Saint-Venant shear flows and ordered from the softest.
  Choosing the modes gives each level: `warping="classical", distortional_modes=0` is
  Benscoter's beam (with Timoshenko bending), `warping="full"` adds shear lag,
  `distortional_modes=n` adds distortion, and `extensible=True` lets the walls stretch
  round the contour. Clamps, ribs (`diaphragm`), warping restraints, Saint-Venant
  torques, point forces and wall loads.

Examples [27](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/27_box_spar_torsion/) (box spar) and
[28](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/28_blade_torsion_hierarchy/) (three-cell blade section) check each layer
against closed forms and the shell model.

## Ply books, solids and ParaView

```python
from laminate_modelling_package import Layup, PlyDefinition, solid_block, couple_shell_to_solid, write_vtu

plies = [PlyDefinition("cap_12", glass, 1.0e-3, 0.0, elements="spar_cap", region=lambda x: x[2] < 4.0), ...]
layup = Layup(plies, reference="top")
layup.apply(mesh, blade.element_sets)
layup.check_drops(mesh, max_per_step=2, element_sets=blade.element_sets)
worst = layup.ply_failure(solution, TsaiWuCriterion())     # per named ply

block = solid_block(mesh, laminate, 0.01, 0.01, 14, 2, element="hex20")   # layered solids
couple_shell_to_solid(mesh, bcs, shell_edge_nodes, block.nodes("x0"))

write_vtu("out/blade", solution, TsaiWuCriterion(), layered=True)   # model and one cell per ply
```

- **Layups.** Plies are listed from the inside out; each covers an element set,
  optionally cut by a region function. Every element gets the stack of plies over
  it, `drops` lists the edges where plies stop and `ply_table` gives each ply's
  area and mass.
- **Solids.** `LayeredHex8Element` (with incompatible modes) and
  `LayeredHex20Element` take a laminate through their thickness and give the full
  3D stress state. `merge_coincident_nodes` joins solid blocks and
  `couple_shell_to_solid` ties shell edges to solid faces. See
  [example 22](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/22_ply_drop_solid_detail/).
- **ParaView.** `write_vtu` writes the deformed model with displacements, stress
  resultants and failure indices, plus a second file with one cell per ply,
  offset through the thickness; `write_pvd` collects several load cases.

## Results at every level

`ResultsExtractor` turns a solution into arrays at each level of fidelity:

```python
from laminate_modelling_package import ResultsExtractor, TsaiWuCriterion

ex = ResultsExtractor(result, load_case=loads, criterion=TsaiWuCriterion())
ex.gauss_points()   # strains and N, M, Q at the 2x2 Gauss points
ex.ply_points()     # ply stresses (global and fibre axes) and failure indices
ex.elements()       # element N, M, Q, strain-energy split, worst ply
ex.nodes()          # displacements, reactions, extrapolated and averaged N, M, Q
ex.totals()         # strain energy, external work, reaction sums, equilibrium residual
ex.to_csv("results/")
```

For the bilinear element, moments and stresses are most accurate at element
centroids (O(h²) convergence); nodal values are smoothed for plotting. See
[example 03](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/03_result_fidelity/).

The constitutive shell stresses have no σz, and FSDT's interlaminar shear jumps
between plies. `EquilibriumStressRecoverer` integrates the 3D equilibrium
equations through the thickness for any theory, and checks every ply interface
for delamination onset:

```python
from laminate_modelling_package import EquilibriumStressRecoverer

rec = EquilibriumStressRecoverer(result, loads)
tau_yz, tau_xz, sigma_z = rec.transverse_stresses(element, 0.0, 0.0, z)
worst = rec.critical_interface()     # DelaminationOnsetCriterion by default
```

See [example 02](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/02_through_thickness_stresses/) and
[example 17](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/17_pagano_benchmark/), which checks it against Pagano's
3D solution.

## Elements, orientation and loads

```python
from laminate_modelling_package import BladeBuilder, LoadCombination, RectangularPlateBuilder

mesh = RectangularPlateBuilder(0.4, 0.2, 20, 10, element="quad8").build(laminate, RefinedZigZagModel())
blade = BladeBuilder(stations, regions, webs, material_axis=(0.0, 0.0, 1.0))   # 0° along the span (default)

bcs.add_spring([node], [DOF.UZ], 5e5).fix_direction(edge_nodes, (1.0, 1.0, 0.0))
cases = [LoadCase("dead").add_body_load((0, 0, -9.81)),
         LoadCase("wind").add_pressure(lambda x: 800.0 * x[2]),
         LoadCase("tip").add_line_load(tip_nodes, (0.0, 0.0, -50.0))]
results = StructuralGlobalSolver().solve_combinations(
    mesh, bcs, cases, [LoadCombination("ULS", {"dead": 1.35, "wind": 1.5, "tip": 1.0})])
```

- **Elements.** The curved `"quad8_curved"` (default) and `"tri6"`, and the flat
  `"quad4"`, `"tri3"` and `"quad8"`, everywhere a mesh is built (`FEAMesh.rectangular_plate`
  stays a Quad4 helper); `FEAMesh.add_element` picks the type from the node count, and new
  element classes plug in through `ElementRegistry`.
- **Orientation.** `material_axis` (a vector, `CylindricalDirection`, or
  `FieldDirection(f)`) sets each element's 0° direction; by default it is the
  global x axis projected onto the element.
- **Offsets.** `Laminate(..., offset=d)` or `laminate.with_offset(d)` puts the
  reference surface at distance d below the mid-plane (d = h/2: reference on the
  bottom face).
- **Checks.** `ModelChecker().check(mesh, bcs)` reports distorted or warped
  elements, CLPT on thick plates, likely unit mix-ups and missing supports.

See [example 18](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/18_element_comparison/) and
[example 19](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/19_load_combinations/).

## Meshing with Gmsh

`FEAMesh.rectangular_plate` covers structured rectangles. For anything else,
mesh the geometry in [Gmsh](https://gmsh.info) and import it. Physical surfaces
choose the laminate of each region, and every physical group becomes a named
node set for supports and loads:

```python
from laminate_modelling_package import CircularHole, GmshMeshImporter, GmshPlateBuilder

importer = GmshMeshImporter({"skin": skin, "doubler": doubler}, FSDTModel())
gm = importer.from_file("panel.geo")        # or .step / .brep (meshed in quads), or a .msh
bcs.fix(gm.nodes("clamped_edge"), [DOF.UX, DOF.UY, DOF.UZ])

# A plate with holes, meshed in graded quads straight from Python:
gm = GmshPlateBuilder(0.3, 0.15, [CircularHole(0.15, 0.075, 0.005)], size=0.012, hole_size=0.0003).build(
    GmshMeshImporter(laminate, FSDTModel()))
```

Tri3, Quad4 and Quad8 cells (Gmsh types 2, 3 and 16) are imported, in any mix.
`GmshPlateBuilder(..., element="quad4")` and `from_file(..., element=...)` set the
Gmsh options (`"quad8_curved"` by default, `"tri3"`, `"quad4"`, `"quad8"`, `"tri6"` or
`"quad_dominant"`); 8-node quads read from a `.msh` file become curved Quad8 unless
the importer is given `curved=False`; for Quad8 in
your own models use `Mesh.ElementOrder = 2` with `Mesh.SecondOrderIncomplete = 1`. Gmsh is
optional: `pip install -e ".[gmsh]"`; headless Linux machines also need
`libglu1-mesa`. See [example 08](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/08_open_hole_gmsh/).

## Examples

[`examples/`](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/) holds twenty-nine worked examples, each in its own folder
with a script, its figures and a README that walks through the model. Examples
01–19 cover the laminate mechanics underneath: the kinematic theories,
through-thickness stresses, result fidelity, hygrothermal loads, failure criteria
and envelopes, progressive damage, Gmsh meshing, sandwich panels, bending-twist
coupling, design sweeps, solver scaling, a CalculiX cross-check, the Pagano 3D
benchmark, the element types and load combinations. Examples 20–23 are blade
structures: beam properties and bend-twist tailoring, spar cap ply drops, the 3D
stresses at a ply drop, and a sandwich blade segment checked against a beam. Examples
24 and 25 validate the blade tools on the IEA 15 MW reference blade: section
properties against PreComp and the published data, and a 20 m segment against
CalculiX. Example 26 compares skins meshed on the mould line with offset laminates
and on their mid-surface. Examples 27 and 28 take torsion beyond Saint-Venant on a
box spar and a three-cell blade section: restrained warping and the bimoment, then
GBT distortion and ribs, each checked against closed forms and the shell model. Run
them all with `python examples/run_all.py`.

## Benchmark runner

```bash
laminate-benchmarks --n 10 --work-dir benchmark_runs   # installed with pip install -e .
```

This builds a simply supported 200 mm square plate with the asymmetric stack `[0/45/-45/90/0/90]` in T300/5208, loaded by 1 kPa pressure and a 50 K cool-down. It then:

1. runs the native, CalculiX and Nastran kernels in turn, timed with `perf_counter`
2. prints a summary of the maximum absolute displacements and runtimes
3. compares the four native kinematic models
4. prints failure indices at the centre element
5. reports error statistics (MAE, max error, RMS) for displacements and ply-stress profiles from each external solver against the native engine

If `ccx` or `nastran` is not on the `PATH`, that kernel still writes its input deck and is reported as `skipped`, so the run always completes.

## Conventions

| Item | Convention |
|---|---|
| Units | Any consistent set. The default material library is SI (Pa, m, 1/K). |
| Material axes | 1 = fibre, 2 = in-plane transverse, 3 = thickness. 3D Voigt order is [11, 22, 33, 23, 13, 12]. |
| Ply angle | Degrees, counter-clockwise about the element normal from the element's material axis (by default global X projected onto the element plane; set with `material_axis`). |
| Stacking | Plies are listed bottom to top. z is measured from the reference surface, which is the mid-plane unless the laminate has an `offset`. |
| DOFs | Six per node: [u, v, w, θx, θy, θz], with u = u0 + z·θy and v = v0 − z·θx. Full RZT adds 2 zig-zag amplitudes per node and layerwise 2 per internal interface; fix them with `EnrichedDOF.X / Y / ALL`. |
| Shear vectors | [γyz, γxz] in global axes and [γ23, γ13] in material axes, so Q_s = diag(G23, G13). |
| Pressure | Positive pressure acts along the element normal (counter-clockwise node order). |

## Theory notes

- **Q̄ transform.** `Q̄ = T⁻¹ Q R T R⁻¹`, where R is the Reuter matrix. The tests check it against the Tsai–Pagano invariant form at every angle.
- **CLPT.** The element is C0, so the Kirchhoff constraint is imposed with a penalty shear stiffness (1000 × Σ Q̄_s h). MITC4 shear keeps this free of locking. Transverse shear stresses are recovered with the parabolic equilibrium profile.
- **FSDT.** A_s = k Σ Q̄_s h with k = 5/6.
- **Refined zig-zag (RZT).** u(z) = u0 + z·β + φ(z)·ψ, with the zig-zag function φ built from the ply shear moduli (Tessler, Di Sciuva and Gherlone). The amplitudes ψ are extra nodal DOFs, so the element carries the zig-zag through the thickness and its gradients. `RefinedZigZagModel(condensed=True)` ties ψ to the mean shear instead (6 DOFs per node, shear stiffness A_s = hG).
- **Layerwise.** Each internal interface carries its own in-plane displacement (hat functions in z), optionally with sub-layers inside each ply. `LayerwiseModel(condensed=True)` condenses them out into a 6-DOF section.
- **Generalised sections.** Every theory is written as ε(z) = E(z)·e and γ(z) = G(z)·s, with section matrices ∫EᵀQ̄E dz and ∫GᵀQ̄sG dz, so all theories share one element formulation.
- **Elements.** Flat facets with a drilling penalty tying θz to ½(v,x − u,y). Quad4: 2×2 Gauss, MITC4 shear and incompatible (QM6) membrane modes, which removes in-plane shear locking. Tri3: constant-strain membrane with MITC3 shear (stiffer; use where quads cannot go). Quad8: serendipity, 3×3 Gauss with 2×2 selective-reduced shear. Each element has exactly six rigid-body modes and passes the membrane patch test on distorted meshes. Curved Quad8 (the default): degenerated-shell kinematics with a tangent frame at every integration point; membrane (with its coupling to bending, so offset and unsymmetric laminates keep a positive semi-definite stiffness) and transverse shear on 2×2 points, bending and drilling on 3×3. A single element keeps one non-communicable membrane mode, as S8R does; two or more have exactly six rigid-body modes.
- **Equilibrium recovery.** In-plane strains are smoothed to the nodes by superconvergent patch recovery, differentiated twice, and the 3D equilibrium equations are integrated analytically through each ply. Face conditions are enforced exactly, and the shear profile is scaled to the element shear force.
- **Constraints.** Fixed DOFs, skew supports and multi-point constraints u_s = Σ c_i u_i + c0 are eliminated with U = T U_r + g. A free rigid-body mode raises `SingularStiffnessError`, detected from the factorisation pivots.

## External kernels

**CalculiX** (tested here against ccx 2.21):

- CalculiX only allows composite shell sections on S8R/S6 elements, so each Quad4 is promoted to S8R and each Tri3 to S6 by adding shared mid-side nodes; Quad8 maps to S8R directly.
- Each element group gets `*ORIENTATION` cards from its material axes and ply angles, `OFFSET` from the laminate offset, and springs become SPRING1 elements.
- A mid-side node takes a fixed DOF only when both corner nodes of its edge fix it.
- CalculiX reports stresses in each layer's own axes, so its profiles are compared in the material frame.
- Moisture loads are converted to an equivalent expansion coefficient α + β·ΔC/ΔT. This is exact for uniform states.

**Nastran:**

- Writes a fixed-field SOL 101 deck with GRID, CQUAD4 / CTRIA3 / CQUAD8, PCOMP (Z0 from the offset), MAT8, CORD2R material axes where needed, SPC1/SPC, MPC, CELAS2 springs, FORCE/MOMENT, PLOAD2 and TEMPD / TEMPP1 (per-element temperatures and gradients) cards.
- Runs `nastran <job>.bdf batch=no scr=yes` and reads the displacement and layered-composite stress tables from the `.f06`.
- The parser is tested against MSC-format excerpts, but the adapter has not been run against a licensed Nastran install yet.

CalculiX supports one uniform hygrothermal state; Nastran also takes per-element temperatures and gradients, but not per-element moisture. Anything a deck cannot express raises `UnsupportedFeatureError`. The decks describe the laminate, not the native kinematic model: each external solver uses its own shell theory, and constraints on RZT or layerwise DOFs are left out.

## Verification

Every example checks itself against a reference: closed-form or series
solutions, exact 3D elasticity, classical laminate and thin-walled beam theory,
CalculiX, PreComp and the published IEA 15 MW data (see the
[examples index](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/)). The test suite includes:

- Navier series solutions for simply supported cross-ply plates (CLPT and FSDT): the native solver is within 1% at a/h = 100 and a/h = 10, and each element type converges to them.
- Pagano's exact 3D solution: RZT and layerwise centre deflections within 2–3% at a/h = 4 (FSDT is 11% stiff), and equilibrium-recovered interlaminar shear and σz within 3–4%.
- Patch tests on distorted meshes, rigid-body modes, an in-plane cantilever against beam theory, a pressurised cylinder against pR/t, and a rotated plate giving the same answer as an unrotated one.
- Free thermal warping of unsymmetric laminates, which matches ABD⁻¹{N_ht; M_ht} exactly, with self-equilibrated residual stresses.
- Classical-laminate-theory ply stresses for laminates in tension.
- First-ply failure of a [0/90]s laminate in tension at the CLT-predicted load.
- Layered Hex8 and Hex20 solids against Pagano within about 1–2% (deflection and stresses), shell–solid coupling within 0.1% of an all-shell model, and curved Quad8 within 0.2% of CalculiX S8R on a curved panel.
- Beam section properties of a thin circular tube against closed-form EA, EI and GJ (exact with curved Quad8, about 1% with Quad4), and RBE2/RBE3 links against hand calculations.
- Timoshenko and Vlasov section stiffness (`tests/test_beam_stiffness.py`): thin-wall theory gives the thin tube's shear factor of 1/2 and the channel's textbook shear centre; on a box with no Poisson coupling the shell's 6 × 6 matches thin-wall theory term by term within 0.5%, and its fitted EΓ and warping shear stiffness within 6% and 15%.
- A direct comparison with CalculiX, which runs only when `ccx` is installed. On the benchmark plate, displacements agree within about 2% and in-plane ply stresses within about 3%.
- Thin-walled section properties against textbook closed forms: an isotropic box
  (Bredt GJ, Benscoter warping function, EΓ, Ip), an I-beam (open GJ and Γ) and a channel
  (shear centre and Γ), all exact. Benscoter and Vlasov finite elements against their
  closed forms, and GBT against shell models of clamped boxes under a tip torque and a
  web force couple, within 2% (examples 27 and 28 give the full comparison).
- The IEA 15 MW reference blade ([example 24](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/24_iea15mw_section_properties/)): from its published layup, the tension centre agrees with PreComp on the same sections within 0.1% of the chord at six stations, and EA, EI and mass are 0–2% below PreComp because the mid-surface mesh takes each wall's mid-line length where PreComp takes the mould line (on the mould line they agree within 0.3%). At the root tube the shell model matches the published EA, GJ and mass within 0.1%. The shear centre follows the published reference data rather than PreComp, and GJ lies between the two, close to the published data outboard, where PreComp's thin-wall torsion falls to half of it. A 20 m segment of the same blade ([example 25](https://github.com/samhughes-fea/composite-laminate-fea/tree/main/examples/25_blade_calculix_crosscheck/)) agrees with CalculiX S8R within 0.2% in deflection, 1.2% in twist and 6% in spar-cap fibre stress.

## Limitations

- The analysis is linear and small-deflection.
- Progressive damage is ply-by-ply stiffness knockdown (`SmearedCrackDegradation`), by design. There is no crack-band regularisation by fracture energy, cohesive zone or continuum damage mechanics, so post-failure softening is not mesh-objective. The knockdown is meant for first-ply-to-last-ply load paths and stiffness loss, not for crack growth.
- Tri3, Quad4 and Quad8 are flat facets (warped elements raise a warning); use `quad8_curved` or `tri6` where curvature matters. Tri6 still shows some membrane locking on strongly curved shells.
- Flat facets (Quad4, flat Quad8) with laminates offset from the reference surface under-predict the torsional stiffness of thick-walled closed sections: GJ of a tube is about 7% low at t/R = 0.02, 22% low at 0.04 and 35% low at 0.06 with the skin on its outer surface, and refining the mesh does not help. Bending and axial stiffness are not affected. Curved Quad8 is within 2–3%. `BladeBuilder` therefore meshes skins on their mid-surface by default, which brings every element to within 0.4% (examples 24 and 26); this applies to `mesh_surface="outer"` and to offset laminates on other meshes (Gmsh, plates).
- Mid-surface blade meshes are inset from the outer mould line by half the skin thickness, so surface pressures act on a slightly smaller perimeter, and the wall area is the mid-wall length × thickness. Thin-wall section tools such as PreComp take the mould-line length × thickness instead, so EA and mass from a mid-surface model are typically 1–2% below PreComp on thick blade sections (example 24).
- Gmsh cells other than Tri3, Tri6, Quad4 and Quad8 (for example 9-node quads) are rejected with a clear error.
- The shells carry no through-thickness normal strain, so σ3 comes from `EquilibriumStressRecoverer` (post-processing) or from layered solids.
- `beam_section_properties` needs a prismatic segment long enough for the end restraint of warping to die out before the middle fifth, where it measures: about 8 chords for uncoupled layups and 12 for bend-twist-coupled caps (at 6 chords a ±20° cap layup reads EI flap 9% and the coupling 23% high). A tapered blade is handled one station at a time.
- With no ribs (`end_ribs=0`, the default), the GJ from `beam_section_properties` includes the cross-section distortion that the clamped root and rigid tip drive over the segment, a few percent either way (examples 28 and 29); `end_ribs=0.2` holds the end fifths rigid in their plane and gives the Saint-Venant GJ (use it with Quad4 or curved Quad8: flat Quad8 on a faceted curved section reads stiff with ribs).
- The Timoshenko shear stiffness depends on how the section rotation is defined, as shear warps the section. `beam_section_properties` uses the stiffness-weighted rotation (the one that does work with the bending moment). With Poisson's ratio it reads a few percent stiffer than thin-wall theory (the Poisson strain of the bending stress varies along a beam in shear, as in Cowper's shear coefficient): +7% on the glass root tube of the IEA 15 MW blade, where the published 6 × 6 sides with thin-wall theory.
- `ThinWalledSection`, `RestrainedTorsionBeam` and `GeneralizedBeam` describe prismatic beams with straight walls between section points; the closed-form section properties (and the thin-wall 6 × 6 and 7 × 7) use the uncoupled wall stiffnesses, with each wall stretching and shearing without curving (GBT keeps all couplings). Off-axis walls' couplings (bend-twist, extension-twist, shear-bending) come only from the shell model (example 29).
- Equilibrium recovery assumes the hygrothermal state is constant within each element, and is least accurate right at supports and free edges, where it falls back on the element shear force.
- Tri3 is markedly stiffer than Quad4 for the same mesh; use it to fill regions that cannot take quads.

## License

MIT; see [LICENSE](https://github.com/samhughes-fea/composite-laminate-fea/blob/main/LICENSE).

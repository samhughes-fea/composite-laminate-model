# composite-laminate-fea

`laminate_modelling_package` is a Python suite for composite laminate mechanics, from thin plates to thick laminates:

- 3D orthotropic materials, plies and laminates, with temperature-dependent properties
- four kinematic theories: CLPT, FSDT, refined zig-zag (RZT) and layerwise, the
  last two with their own zig-zag / interface DOFs
- Tri3, Quad4 and Quad8 composite shell elements behind one `ShellElement` interface
- material orientation on curved shells, laminate offsets, springs and skew supports
- pressure fields, line loads, body loads and hygrothermal (thermal and moisture) loads;
  many load cases with one factorisation, and load combinations
- through-thickness stress recovery, including σz and interlaminar shear from 3D equilibrium
- failure criteria (with a delamination onset check), smeared-crack progressive damage
- Pagano's exact 3D solution as a built-in benchmark
- structured and Gmsh meshing, and pre-solve model checks
- pluggable solver kernels: native NumPy/SciPy, CalculiX and MSC/NX Nastran

```
.
├── setup.py
├── src/
│   ├── run_solver_benchmarks.py            benchmark runner (laminate-benchmarks command)
│   └── laminate_modelling_package/
│       ├── __init__.py                     public API
│       ├── laminate_materials_3d.py        ElasticProperties3D, StrengthProperties3D, 3D Bond rotations
│       ├── laminate_materials.py           MaterialCard, MaterialRepository / MaterialLibrary
│       ├── laminate_hygrothermal.py        HygrothermalProperties/State, HygrothermalEngine (N_ht, M_ht)
│       ├── laminate_mechanics.py           Ply (Q, Q_s, Q_bar, Q_bar_s), Laminate, Reuter transforms
│       ├── laminate_temperature.py         TemperatureDependentMaterial
│       ├── laminate_kinematics.py          CLPT, FSDT, RZT, layerwise; generalised section matrices
│       ├── laminate_elements.py            ShellElement, Tri3 / Quad4 / Quad8 shells, ElementRegistry
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
│       └── laminate_results.py             ResultsExtractor: Gauss point, ply, element, node and global results
├── examples/                               twenty worked examples with figures (see examples/README.md)
└── tests/                                  pytest suite
```

## Install

```bash
pip install -e .            # NumPy only (dense solver)
pip install -e ".[dev]"     # + SciPy sparse solver and pytest
pip install -e ".[gmsh]"     # + Gmsh meshing
pip install -e ".[examples]" # + matplotlib and Gmsh for the examples
pytest                      # run the test suite
```

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
[example 03](examples/03_result_fidelity/).

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

See [example 02](examples/02_through_thickness_stresses/) and
[example 17](examples/17_pagano_benchmark/), which checks it against Pagano's
3D solution.

## Elements, orientation and loads

```python
from laminate_modelling_package import (
    CylindricalDirection, CylindricalPanelBuilder, LoadCombination, RectangularPlateBuilder)

mesh = RectangularPlateBuilder(0.4, 0.2, 20, 10, element="quad8").build(laminate, RefinedZigZagModel())
barrel = CylindricalPanelBuilder(0.15, 0.6, 8, 32, 0.0, 360.0).build_named(
    laminate, FSDTModel(), material_axis=CylindricalDirection(reference="axial"))

bcs.add_spring([node], [DOF.UZ], 5e5).fix_direction(edge_nodes, (1.0, 1.0, 0.0))
cases = [LoadCase("dead").add_body_load((0, 0, -9.81)),
         LoadCase("wind").add_pressure(lambda x: 800.0 * x[2]),
         LoadCase("tip").add_line_load(tip_nodes, (0.0, 0.0, -50.0))]
results = StructuralGlobalSolver().solve_combinations(
    mesh, bcs, cases, [LoadCombination("ULS", {"dead": 1.35, "wind": 1.5, "tip": 1.0})])
```

- **Elements.** `"tri3"`, `"quad4"` (default) and `"quad8"` everywhere a mesh is
  built; `FEAMesh.add_element` picks the type from the node count, and new
  element classes plug in through `ElementRegistry`.
- **Orientation.** `material_axis` (a vector, `CylindricalDirection`, or
  `FieldDirection(f)`) sets each element's 0° direction; by default it is the
  global x axis projected onto the element.
- **Offsets.** `Laminate(..., offset=d)` or `laminate.with_offset(d)` puts the
  reference surface at distance d below the mid-plane (d = h/2: reference on the
  bottom face).
- **Checks.** `ModelChecker().check(mesh, bcs)` reports distorted or warped
  elements, CLPT on thick plates, likely unit mix-ups and missing supports.

See [example 18](examples/18_pressure_vessel_winding/),
[example 19](examples/19_element_comparison/) and
[example 20](examples/20_load_combinations/).

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
`GmshPlateBuilder(..., element="quad8")` and `from_file(..., element=...)` set the
Gmsh options (`"tri3"`, `"quad4"`, `"quad8"` or `"quad_dominant"`); for Quad8 in
your own models use `Mesh.ElementOrder = 2` with `Mesh.SecondOrderIncomplete = 1`. Gmsh is
optional: `pip install -e ".[gmsh]"`; headless Linux machines also need
`libglu1-mesa`. See [example 08](examples/08_open_hole_gmsh/).

## Examples

[`examples/`](examples/) holds twenty worked examples, each in its own folder
with a script, its figures and a README that walks through the model. They
cover the kinematic theories, through-thickness stresses, result fidelity,
hygrothermal loads, failure criteria and envelopes, progressive damage, Gmsh
meshing, sandwich panels, bending-twist coupling, design sweeps, solver scaling,
a CalculiX cross-check, the Pagano 3D benchmark, a wound pressure vessel, the
three element types and load combinations. Run them all with `python examples/run_all.py`.

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
- **Elements.** Flat facets with a drilling penalty tying θz to ½(v,x − u,y). Quad4: 2×2 Gauss, MITC4 shear and incompatible (QM6) membrane modes, which removes in-plane shear locking. Tri3: constant-strain membrane with MITC3 shear (stiffer; use where quads cannot go). Quad8: serendipity, 3×3 Gauss with 2×2 selective-reduced shear. Each element has exactly six rigid-body modes and passes the membrane patch test on distorted meshes.
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

The test suite includes:

- Navier series solutions for simply supported cross-ply plates (CLPT and FSDT): the native solver is within 1% at a/h = 100 and a/h = 10, and each element type converges to them.
- Pagano's exact 3D solution: RZT and layerwise centre deflections within 2–3% at a/h = 4 (FSDT is 11% stiff), and equilibrium-recovered interlaminar shear and σz within 3–4%.
- Patch tests on distorted meshes, rigid-body modes, an in-plane cantilever against beam theory, a pressurised cylinder against pR/t, and a rotated plate giving the same answer as an unrotated one.
- Free thermal warping of unsymmetric laminates, which matches ABD⁻¹{N_ht; M_ht} exactly, with self-equilibrated residual stresses.
- Classical-laminate-theory ply stresses for laminates in tension.
- First-ply failure of a [0/90]s laminate in tension at the CLT-predicted load.
- A direct comparison with CalculiX, which runs only when `ccx` is installed. On the benchmark plate, displacements agree within about 2% and in-plane ply stresses within about 3%.

## Limitations

- The analysis is linear and small-deflection.
- Elements are flat facets, so curved surfaces are approximated by facets. Elements warped by more than 5% of their size raise a warning.
- Gmsh cells other than Tri3, Quad4 and Quad8 (for example 6-node triangles or 9-node quads) are rejected with a clear error.
- The shell itself carries no through-thickness normal strain, so σ3 comes only from `EquilibriumStressRecoverer` (post-processing), and progressive damage uses the constitutive ply stresses.
- Equilibrium recovery assumes the hygrothermal state is constant within each element, and is least accurate right at supports and free edges, where it falls back on the element shear force.
- Tri3 is markedly stiffer than Quad4 for the same mesh; use it to fill regions that cannot take quads.

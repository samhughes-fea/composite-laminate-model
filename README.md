# composite-laminate-fea

`laminate_modelling_package` is a Python suite for composite laminate mechanics, from thin plates to thick laminates:

- 3D orthotropic materials, plies and laminates
- four kinematic theories: CLPT, FSDT, refined zig-zag and layerwise
- a 24-DOF Quad4 composite shell element
- hygrothermal (thermal and moisture) loads
- exact through-thickness stress recovery
- failure criteria and smeared-crack progressive damage
- pluggable solver kernels: native NumPy/SciPy, CalculiX and MSC/NX Nastran

```
.
├── setup.py
├── run_solver_benchmarks.py
├── laminate_modelling_package/
│   ├── __init__.py                     public API
│   ├── laminate_materials_3d.py        ElasticProperties3D, StrengthProperties3D, 3D Bond rotations
│   ├── laminate_materials.py           MaterialCard, MaterialRepository / MaterialLibrary
│   ├── laminate_hygrothermal.py        HygrothermalProperties/State, HygrothermalEngine (N_ht, M_ht)
│   ├── laminate_mechanics.py           Ply (Q, Q_s, Q_bar, Q_bar_s), Laminate, Reuter transforms
│   ├── laminate_fea_engine.py          KinematicModel strategies + CompositePlateElement (Quad4)
│   ├── laminate_stress_recovery.py     ExactStressRecoverer
│   ├── laminate_failure_criteria.py    Max stress, Tsai-Hill, Tsai-Wu (Hoffman F12), Hashin 2D/3D
│   ├── laminate_progressive_damage.py  SmearedCrackDegradation, ProgressiveDamageAnalysis
│   ├── laminate_solver_pipeline.py     FEAMesh, BoundaryConditions, LoadCase, StructuralGlobalSolver
│   ├── laminate_kernels.py             FEAKernel + Native / CalculiX / Nastran adapters
│   ├── laminate_benchmark_stress.py    StressDistributionVerifier, ThicknessProfile
│   └── laminate_results.py             ResultsExtractor: Gauss point, ply, element, node and global results
├── examples/                           seven worked examples with figures (see examples/README.md)
└── tests/                              pytest suite (161 tests)
```

## Install

```bash
pip install -e .            # NumPy only (dense solver)
pip install -e ".[dev]"     # + SciPy sparse solver and pytest
pip install -e ".[examples]" # + matplotlib for the examples
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

## Examples

[`examples/`](examples/) holds seven worked examples, each in its own folder
with a script, its figures and a README that walks through the model: kinematic
models against thickness, through-thickness stresses, result fidelity levels,
thermal warping, failure envelopes, progressive damage and a CalculiX
cross-check. Run them all with `python examples/run_all.py`.

## Benchmark runner

```bash
python run_solver_benchmarks.py --n 10 --work-dir benchmark_runs
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
| Ply angle | Degrees, measured counter-clockwise from the element x axis (global X projected onto the element plane). |
| Stacking | Plies are listed bottom (z = -h/2) to top. z is measured from the mid-plane. |
| DOFs | Six per node: [u, v, w, θx, θy, θz], with u = u0 + z·θy and v = v0 − z·θx. |
| Shear vectors | [γyz, γxz] in global axes and [γ23, γ13] in material axes, so Q_s = diag(G23, G13). |
| Pressure | Positive pressure acts along the element normal (counter-clockwise node order). |

## Theory notes

- **Q̄ transform.** `Q̄ = T⁻¹ Q R T R⁻¹`, where R is the Reuter matrix. The tests check it against the Tsai–Pagano invariant form at every angle.
- **CLPT.** The element is C0, so the Kirchhoff constraint is imposed with a penalty shear stiffness (1000 × Σ Q̄_s h). MITC4 shear keeps this free of locking. Transverse shear stresses are recovered with the parabolic equilibrium profile.
- **FSDT.** A_s = k Σ Q̄_s h with k = 5/6.
- **Refined zig-zag.** Each layer's shear strain is γ_k = S_k G γ0, where S_k is the inverse of the layer's Q̄_s and G = (Σ S_k h_k / h)⁻¹. The zig-zag slope jump at each interface is ψ = (S_k − S_{k−1}) G = ΔS·G. This makes interlaminar shear stress continuous. In the 6-DOF element the zig-zag amplitudes are tied to the mean shear, which gives A_s = hG. `zigzag_stiffness()` returns the generalised RZT matrices for anyone building a 7-DOF element.
- **Layerwise.** Each interface carries its own in-plane displacement field, and the global thickness matrices are assembled from these. Projecting onto a planar field reproduces the CLPT ABD exactly. Statically condensing the interface sliding modes gives A_s (it matches RZT, which the tests confirm).
- **Element.** Flat Quad4 with 2×2 Gauss quadrature, bilinear membrane and bending B-matrices, MITC4 assumed transverse shear, and a drilling penalty tying θz to ½(v,x − u,y). It has exactly six rigid-body modes and passes both the membrane and bending patch tests.
- **Constraints.** Fixed DOFs and multi-point constraints u_s = Σ c_i u_i + c0 are eliminated with U = T U_r + g. A free rigid-body mode raises `SingularStiffnessError`, detected from the factorisation pivots.

## External kernels

**CalculiX** (tested here against ccx 2.21):

- CalculiX only allows composite shell sections on S8R/S6 elements, so each Quad4 is promoted to S8R by adding shared mid-side nodes.
- A mid-side node takes a fixed DOF only when both corner nodes of its edge fix it.
- CalculiX reports stresses in each layer's own axes, so its profiles are compared in the material frame.
- Moisture loads are converted to an equivalent expansion coefficient α + β·ΔC/ΔT. This is exact for uniform states.

**Nastran:**

- Writes a fixed-field SOL 101 deck with GRID, CQUAD4 (MCID 0), PCOMP, MAT8, SPC1/SPC, MPC, FORCE/MOMENT, PLOAD2 and TEMPD cards.
- Runs `nastran <job>.bdf batch=no scr=yes` and reads the displacement and layered-composite stress tables from the `.f06`.
- The parser is tested against MSC-format excerpts, but the adapter has not been run against a licensed Nastran install yet.

Both external decks support only a single uniform hygrothermal state. Through-thickness gradients and per-element states raise `UnsupportedFeatureError`.

## Verification

The test suite includes:

- Navier series solutions for simply supported cross-ply plates (CLPT and FSDT): the native solver is within 1% at a/h = 100 and a/h = 10.
- Free thermal warping of unsymmetric laminates, which matches ABD⁻¹{N_ht; M_ht} exactly, with self-equilibrated residual stresses.
- Classical-laminate-theory ply stresses for laminates in tension.
- First-ply failure of a [0/90]s laminate in tension at the CLT-predicted load.
- A direct comparison with CalculiX, which runs only when `ccx` is installed. On the benchmark plate, displacements agree within about 2% and in-plane ply stresses within about 3%.

## Limitations

- The analysis is linear and small-deflection.
- Elements are flat facets. Ply orientations in the CalculiX deck assume shells lying in planes parallel to xy.
- Through-thickness normal stress σ3 is zero in the shell. The 3D failure criteria still accept full 3D stress vectors.

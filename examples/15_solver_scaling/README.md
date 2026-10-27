# 15 · Solver scaling

How long does a model take, and where does the time go? This example times the
same plate on meshes from 4 × 4 to 48 × 48 elements with each solver path.

![Runtime against degrees of freedom, and the native time split](figures/solver_scaling.png)

## The model

| Item | Value |
|---|---|
| Plate | 200 mm square [0/45/-45/90]s T300/5208, simply supported, 1 kPa pressure, FSDT |
| Meshes | 4², 8², 12², 16², 24², 32², 40², 48² elements (150 to 14,406 DOF) |
| Solvers | native sparse (SciPy SuperLU), native dense (NumPy Cholesky, up to 24²), CalculiX |

```bash
python examples/15_solver_scaling/solver_scaling.py
```

## Walkthrough

The linear solver is a strategy passed to `StructuralGlobalSolver`, so changing
the back end is one argument:

```python
StructuralGlobalSolver(linear_solver=SparseLinearSolver()).solve(mesh, bcs, loads)
StructuralGlobalSolver(linear_solver=DenseLinearSolver()).solve(mesh, bcs, loads)
CalculiXKernelAdapter(work_dir=...).run(AnalysisModel(mesh, bcs, loads, "plate"))
```

The assembly time is measured separately with
`GlobalAssembler().assemble_stiffness(mesh)`.

## Reading the figure

Wall times from the machine that built the figure (single runs, so expect some
noise):

| Mesh | DOF | Assembly [s] | Native sparse [s] | Native dense [s] | CalculiX [s] |
|---|---|---|---|---|---|
| 8² | 486 | 0.16 | 0.19 | 0.18 | 0.12 |
| 24² | 3,750 | 1.39 | 1.41 | 2.98 | 1.06 |
| 48² | 14,406 | 4.87 | 7.57 | – | 4.67 |

The native times are the whole solve (assembly, constraints, factorisation and
back-substitution), each on a freshly built mesh so no cached element data is
reused; the assembly column times the stiffness assembly alone.

- **Assembly is most of the native time.** Up to about 4,000 DOFs the sparse
  solve adds almost nothing to it; at 14,000 DOFs assembly is still two thirds of
  the total. The element stiffness loop runs in Python, so vectorising it, or
  caching identical element matrices on a regular mesh, is the obvious speed-up
  if large models become routine.
- **The sparse native solver scales almost linearly** with the number of DOFs
  over this range. The dense solver grows much faster and needs memory that
  grows with the square of the DOF count, so it is only for small models or
  machines without SciPy.
- **CalculiX is about as fast as the native solver here**, and faster on the
  largest mesh, even though its time includes writing the deck, starting `ccx`,
  parsing the text output, and an S8R mesh with about three times as many nodes.
  Its element loop is compiled Fortran.

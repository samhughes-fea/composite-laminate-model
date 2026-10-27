# 06 · Progressive damage of a cross-ply strip in tension

A [0/90]s laminate in tension does not fail all at once. The 90° plies crack
first, the laminate softens slightly, and it carries on until the 0° fibres
break. `ProgressiveDamageAnalysis` follows this sequence with a series of
linear static solves.

![Load against strain with first-ply and fibre failure marked, and the damaged ply count](figures/progressive_damage.png)

## The model

| Item | Value |
|---|---|
| Laminate | [0/90/90/0] T300/5208, 0.25 mm plies (h = 1 mm) |
| Specimen | 100 × 40 mm strip, 8 × 2 Quad4, FSDT |
| Supports | left edge held in x, one corner held in y, out-of-plane DOFs held |
| Load | line load N_x on the right edge, stepped from 25 to 800 kN/m |
| Criterion | `HashinCriterion("3d")` |
| Degradation | `SmearedCrackDegradation(0.01)`: failed moduli drop to 1% |

```bash
python examples/06_progressive_damage/progressive_damage.py
```

## Walkthrough

```python
analysis = ProgressiveDamageAnalysis(criterion=HashinCriterion("3d"),
                                     degradation=SmearedCrackDegradation(0.01))
result = analysis.run(mesh, bcs, loads, load_factors=np.arange(0.25, 8.01, 0.25))
```

At each load factor the analysis:

1. solves the current (possibly damaged) model with a linear static solve;
2. recovers ply stresses at the bottom, middle and top of every ply at each
   element centre and evaluates the criterion;
3. degrades the newly failed plies according to their failure mode (matrix
   cracking reduces E2, G12, G23 and ν12; fibre failure reduces E1);
4. re-solves the same load step until no new damage appears.

`result.steps` records the load factor, iterations, damaged ply count, maximum
failure index and maximum displacement of every step. `result.first_ply_failure_factor`
gives the first-ply failure load.

## The reference: ply discount on the laminate

The same criterion and knockdown can be worked by hand on the laminate, with no
mesh: under N_x alone every element of the strip has the same strains, so
classical laminate theory is exact. `clt_ply_discount` in the script finds the
lowest N_x at which a ply gains a new failure mode, knocks that ply (and its
symmetric partner) down with the same `SmearedCrackDegradation`, rebuilds A and
repeats until a fibre fails. This is the textbook ply-discount method.

## Reading the figure

| Event | CLT ply discount | FE (load steps of 25 kN/m) |
|---|---|---|
| First-ply failure: matrix tension in the 90° plies | 374.3 kN/m | 375 kN/m, the first step above |
| Axial modulus after the 90° plies crack | 96.0 → 90.9 GPa | 90.9 GPa |
| Fibre tension in the 0° plies (laminate failure) | 750.4 kN/m | 775 kN/m, the first step above |

- **Left:** the FE load-strain points lie on the CLT ply-discount line. Up to
  first-ply failure both follow the undamaged stiffness. At 375 kN/m the 90°
  plies crack, the knockdown is instant, so the strain jumps at constant load
  to the cracked laminate's line, 5% softer, which is the small share of the
  axial load the 90° plies carried. The 0° plies then take the load until their
  fibres fail at 750 kN/m.
- **Right:** all 32 ply-elements in the 90° plies (50% of the total) crack in
  the same step because the stress is uniform. The 0° plies follow when they
  reach fibre failure.

Both methods agree to the load step because the stress field is uniform. Where
it is not (holes, ply drops, joints), the FE analysis tracks damage element by
element, which the laminate-level method cannot.

The knockdown is instant and the post-failure branch is not drawn beyond fibre
failure. This ply-by-ply stiffness knockdown is the package's damage model by
design: it gives first-ply to last-ply load paths and stiffness loss, not crack
growth, so the softening after failure depends on the mesh.

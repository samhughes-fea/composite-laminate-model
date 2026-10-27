# 02 · Through-thickness stresses in a thick laminate

In-plane stresses come straight from the strains, but transverse shear stresses
depend on what the kinematic theory assumes through the thickness. This example
puts all four theories side by side in a thick [0/90/0] plate and checks them
against the exact 3D elasticity solution, with CalculiX overlaid when it is installed.

![σx and τxz through the thickness for each model](figures/through_thickness_stresses.png)

## The model

| Item | Value |
|---|---|
| Laminate | [0/90/0] T300/5208, three equal plies |
| Plate | 100 mm square, a/h = 5 (h = 20 mm) |
| Supports | simply supported on all edges |
| Load | uniform pressure q = 100 kPa |
| Mesh | 16 × 16 Quad4 |
| Sampling | σx at the element next to the plate centre; τxz at the element next to the x = 0 edge, where shear peaks; 9 heights per ply |
| Reference | exact 3D elasticity: Pagano's solution for each term of the double sine series of the uniform pressure (odd m, n ≤ 45), evaluated at the same points |

```bash
python examples/02_through_thickness_stresses/through_thickness_stresses.py
```

## Walkthrough

1. **Solve each model** as in example 01.
2. **Recover stresses at any height** with `ExactStressRecoverer`. At a ply
   interface the stress has two values, so `side` picks the ply below or above:

   ```python
   rec = ExactStressRecoverer()
   u_e = result.element_displacements(element.element_id)
   point = rec.recover(element, u_e, z, side="above")
   sigma_x = point.stress_global[0]   # [σx, σy, τxy]
   tau_xz = point.shear_global[1]     # [τyz, τxz]
   ```

3. **Exact reference.** A uniform pressure is the sum of doubly sinusoidal loads
   16q/(π²mn) sin(mπx/a) sin(nπy/a) over odd m and n, and each of those has
   Pagano's exact 3D solution. `_shared/analytical.py` sums them
   (`elasticity_3d_uniform`), solving each term through the thickness with fine
   quadratic elements in z, which stays well conditioned for the short-wavelength
   terms where the transfer matrix of `PaganoPlate` does not. For m = n = 1 it
   matches `PaganoPlate` to 1e-5, and the sum has converged to 0.1% by 31 terms.

4. **CalculiX overlay.** `CalculiXKernelAdapter` writes an S8R composite shell
   deck, runs `ccx` and reads the layer stresses back. CalculiX reports them in
   each ply's material axes, so the script rotates them to global axes ply by
   ply before plotting.

## Reading the figure

| Model | σx, top face [MPa] | vs 3D | τxz, mid-plane [MPa] | vs 3D |
|---|---|---|---|---|
| Exact 3D | 1.980 | | 0.2748 | |
| CLPT | 1.843 | −6.9% | 0.3561 (parabola) | +29.6% |
| FSDT | 1.654 | −16.4% | 0.1644 (constitutive) | −40.2% |
| RZT and layerwise | 1.902 | −3.9% | 0.2766 (constitutive) | +0.7% |
| FSDT + equilibrium | | | 0.3144 | +14.4% |
| RZT + equilibrium | | | 0.2852 | +3.8% |

The script prints this table.

- **σx (left)** jumps at the interfaces, because the 90° ply is much softer
  along x. Through each 0° ply the exact stress is not quite linear: at a/h = 5
  the plate is thick enough for 3D effects. RZT and layerwise (identical for this
  symmetric stack) follow the exact curve and are 4% low at the faces. FSDT is
  16% low: with one rotation for the whole stack it shifts bending moment from x
  to y at this thickness. CLPT happens to land closer (−7%), but only because its
  missing shear flexibility and its linear profile err in opposite directions.
  CalculiX's S8R shell (circles) sits with FSDT.
- **τxz (right)** is where the theories differ most. The thin lines are the shear
  stress straight from each shell theory's constitutive law:
  - **FSDT** has a constant shear strain, so its stress is piecewise constant
    and jumps at each interface (0.32 MPa in the 0° plies, 0.16 MPa in the
    90° ply). The exact interlaminar shear is continuous.
  - **RZT and layerwise** also jump: each ply's shear modulus multiplies a
    different zig-zag slope. Their constitutive shear is right in the middle ply
    (+0.7%), but a constant 0.19 MPa in the outer plies, where the exact stress
    runs from zero at the face to 0.27 MPa at the interface.
  - **CLPT** recovers τxz from equilibrium with one parabola through the whole
    thickness, ignoring the stiffness change between plies, and is 30% high at
    the mid-plane.
  - **CalculiX** uses the FSDT-type S8R shell, so its points follow the FSDT
    pattern of a ply-wise jump.
- **The thick dashed lines** come from `EquilibriumStressRecoverer`. It
  integrates the 3D equilibrium equations through the thickness using the
  in-plane stress gradients of the solved shell, so the profile is continuous,
  zero on both faces and parabolic within each ply. It also returns σz. From RZT
  it is within 4% of the exact curve everywhere; from FSDT, 14% high in the
  middle ply, because FSDT's in-plane stresses are less accurate. Example 17 makes
  the same comparison for a sinusoidal load, away from the supports.

```python
rec = EquilibriumStressRecoverer(result, load_case)
tau_yz, tau_xz, sigma_z = rec.transverse_stresses(element, 0.0, 0.0, z)
```

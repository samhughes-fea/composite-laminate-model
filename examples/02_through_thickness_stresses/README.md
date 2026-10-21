# 02 · Through-thickness stresses in a thick laminate

In-plane stresses come straight from the strains, but transverse shear stresses
depend on what the kinematic theory assumes through the thickness. This example
puts all four theories side by side in a thick [0/90/0] plate and overlays
CalculiX when it is installed.

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

3. **CalculiX overlay.** `CalculiXKernelAdapter` writes an S8R composite shell
   deck, runs `ccx` and reads the layer stresses back. CalculiX reports them in
   each ply's material axes, so the script rotates them to global axes ply by
   ply before plotting.

## Reading the figure

- **σx (left)** is linear within each ply and jumps at the interfaces, because
  the 90° ply is much softer along x. FSDT, RZT and layerwise agree with each
  other and with CalculiX. CLPT, which ignores shear flexibility, overshoots the
  outer-ply peak by about 11% at this thickness.
- **τxz (right)** is where the theories differ:
  - **FSDT** has a constant shear strain, so its stress is piecewise constant
    and jumps at each interface (about 0.32 MPa in the 0° plies, 0.16 MPa in the
    90° ply). Interlaminar shear stress should be continuous, so this is a known
    FSDT weakness.
  - **RZT and layerwise** give one continuous value through the thickness
    (about 0.23 MPa). In this 6-DOF shell both reduce to the same constant
    interlaminar shear stress.
  - **CLPT** recovers τxz from equilibrium with a parabolic profile, which is
    continuous, zero at the faces and peaks at mid-thickness.
  - **CalculiX** uses the FSDT-type S8R shell, so its points follow the FSDT
    pattern of a ply-wise jump.

None of the shell theories here gives the exact 3D profile, which is
continuous and parabolic within each ply. Recovering that from the 3D
equilibrium equations is the top item on the roadmap.

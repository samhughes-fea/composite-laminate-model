# 18 · Every shell element on the open-hole plate

The package has five shell elements behind one `ShellElement` interface: the flat
Tri3, Quad4 and Quad8, and the curved Tri6 and Quad8 (the default). This example
meshes the quasi-isotropic open-hole plate of example 08 with each of them at four
mesh sizes and compares the stress concentration at the hole edge with
Lekhnitskii's solution.

![The three meshes near the hole, and Kt error against degrees of freedom](figures/element_comparison.png)

## The model

| Item | Value |
|---|---|
| Laminate | [0/45/−45/90]s T300/5208, 0.125 mm plies |
| Plate | 300 × 150 mm, central hole d = 10 mm |
| Load | uniform Nx = 100 kN/m as a line load on the right edge |
| Supports | left edge held in x, one node held in y, out-of-plane DOFs fixed |
| Meshes | Gmsh, graded from 0.2–1.6 mm at the hole to 10–24 mm far away |
| Elements | `Tri3ShellElement`, `Quad4ShellElement`, `Quad8ShellElement`, and the curved `Tri6` and `Quad8` shells |
| Reference | Lekhnitskii Kt for an infinite orthotropic plate × finite-width correction = 3.01 |

```bash
python examples/18_element_comparison/element_comparison.py
```

## Walkthrough

1. **Choose the element when meshing.** `GmshPlateBuilder(..., element=...)`
   sets the Gmsh options for `"tri3"`, `"quad4"`, `"quad8"`, `"tri6"`, `"quad8_curved"`
   (the default) or `"quad_dominant"`,
   and the importer creates the matching element for each Gmsh cell:

   ```python
   builder = GmshPlateBuilder(L, W, [CircularHole(L / 2, W / 2, R)], size=size, hole_size=hole_size,
                              element="quad8")
   gm = builder.build(GmshMeshImporter(lam, FSDTModel()))
   ```

   Outside Gmsh, `RectangularPlateBuilder` and `CylindricalPanelBuilder` take the
   same `element` argument, and `FEAMesh.add_element` picks the element type from
   the number of nodes (`ElementRegistry`).

2. **Load with a line load.** `add_line_load` integrates the load along each
   element edge with that element's own shape functions, which matters for
   Quad8: its consistent edge loads are 1/6, 2/3, 1/6, not equal shares.

3. **Read Kt** from the averaged nodal Nx at the hole edge on the transverse
   centre line.

## Reading the figure

| Element | DOFs | Kt | Error | | DOFs | Kt | Error |
|---|---|---|---|---|---|---|---|
| Tri3 | 2,634 | 2.628 | −12.8% | | 9,204 | 2.878 | −4.5% |
| Quad4 | 8,652 | 2.866 | −4.9% | | 16,704 | 3.020 | +0.2% |
| Quad8 | 11,640 | 2.933 | −2.7% | | 25,356 | 3.017 | +0.1% |
| Tri6 | 9,954 | 3.003 | −0.4% | | 35,388 | 3.013 | −0.0% |
| Curved Quad8 | 11,640 | 2.932 | −2.7% | | 25,356 | 3.019 | +0.2% |

The reference is Kt = 3.014 (Lekhnitskii × finite-width correction). The script
prints all four meshes of each element.

- **Tri3** has a constant-strain membrane, so the steep stress gradient at the
  hole needs many more elements. Even at its finest mesh it is 4.5% low.
- **Quad4** uses incompatible modes for the membrane, so it captures the
  gradient much better per element. It reaches the reference by about 17,000
  DOFs.
- **Quad8 and curved Quad8** are quadratic, so they are the most accurate on a
  given mesh. On a flat plate the two give the same answer (the curved element
  differs only where the surface curves), and per DOF they are close to Quad4
  here, because they carry mid-side nodes on the same mesh.
- **Tri6** is the most accurate per DOF on this problem: within 0.4% at 10,000
  DOFs. Gmsh's triangles grade more smoothly into the hole than its recombined
  quads, and the quadratic triangle has no distortion penalty. On strongly
  curved shells it still shows some membrane locking, so it is best for flat or
  gently curved regions that quads cannot mesh well.
- Below about 0.5% the differences are within the accuracy of the reference
  itself: the finite-width correction is an approximation, which is why Quad4
  moves slightly away from it on the finest mesh.

The default, curved Quad8, is the safe choice for blade skins (curved, offset,
thick-walled sections, examples 24 and 26). On flat panels like this one, Quad4
is the cheapest way to an accurate stress concentration, and Tri3 is only for
filling regions that cannot be meshed with quads (`element="quad_dominant"`).

## Things to try

- Swap the laminate for `[0]8`, which has a much higher Kt (about 6.8 by Lekhnitskii) and a
  steeper gradient.
- Run with `element="quad_dominant"` to see the mixed mesh Gmsh produces.

# laminate-constitutive-model

[![tests](https://github.com/samhughes-fea/laminate-constitutive-model/actions/workflows/tests.yml/badge.svg)](https://github.com/samhughes-fea/laminate-constitutive-model/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/samhughes-fea/laminate-constitutive-model/blob/main/LICENSE)

`laminate-constitutive-model` is the ply and laminate behaviour of composites, on its
own: no mesh, no solver and no geometry. It needs NumPy alone.

- 3D orthotropic materials and a material library, with temperature-dependent properties
- plies and laminates: Q, Q̄ and the stiffness invariants, ABD, offsets
- section stiffness for four through-thickness theories: CLPT, FSDT, refined zig-zag
  (RZT) and layerwise, written as generalised section matrices
- hygrothermal (thermal and moisture) expansion and resultants
- failure criteria: maximum stress, Tsai-Hill, Tsai-Wu, Hashin and a delamination
  onset check, with a registry for your own
- Pagano's exact 3D solution for a simply supported cross-ply plate, as a benchmark

It was split out of composite-laminate-model, whose finite elements, solver, meshing
and blade models are now [shell-fem](https://github.com/samhughes-fea/shell-fem).

```
.
├── pyproject.toml                          package metadata (version from __init__)
├── src/
│   ├── laminate_constitutive_model/
│   │   ├── __init__.py                     public API and __version__
│   │   ├── laminate_materials_3d.py        ElasticProperties3D, StrengthProperties3D, 3D Bond rotations
│   │   ├── laminate_materials.py           MaterialCard, MaterialRepository / MaterialLibrary
│   │   ├── laminate_hygrothermal.py        HygrothermalProperties/State, HygrothermalEngine (N_ht, M_ht)
│   │   ├── laminate_mechanics.py           Ply (Q, Q_s, Q_bar, Q_bar_s), Laminate, Reuter transforms
│   │   ├── laminate_temperature.py         TemperatureDependentMaterial
│   │   ├── laminate_kinematics.py          integrate_abd; CLPT, FSDT, RZT, layerwise section matrices
│   │   ├── laminate_failure_criteria.py    Max stress, Tsai-Hill, Tsai-Wu, Hashin, delamination; registry
│   │   └── laminate_elasticity_3d.py       PaganoPlate: exact 3D cross-ply solution
│   └── composite_laminate_model/           the old import name, re-exporting the above (deprecated)
├── examples/                               worked examples with figures (see examples/README.md)
└── tests/                                  pytest suite
```

## Install

The package is not on PyPI yet. From a clone:

```bash
pip install -e .             # NumPy only
pip install -e ".[examples]" # + matplotlib for the examples
pip install -e ".[dev]"      # + pytest, build and twine
pytest                       # run the test suite
```

Python 3.11 or newer.

### Related packages

| Package | Holds | Depends on |
| --- | --- | --- |
| **laminate-constitutive-model** (this one) | materials, plies, laminates, ABD and the kinematic theories' section stiffness, hygrothermal resultants, failure criteria | numpy |
| [implicit-geometry-kernel](https://github.com/samhughes-fea/implicit-geometry-kernel) | blade and section geometry: airfoils, stations, 40 structural layouts, wall midlines and cells | numpy, shapely, numba |
| [shell-fem](https://github.com/samhughes-fea/shell-fem) | shell and solid elements, solver, stress recovery, Gmsh, CalculiX and Nastran kernels, blade shell models | this package and the geometry kernel |
| [anisotropic-section-model](https://github.com/samhughes-fea/anisotropic-section-model) | thin-walled beam-section theory: 6 × 6 and 7 × 7 section stiffness, restrained torsion, GBT | this package |

This package imports nothing from the others (`tests/test_import_boundary.py` checks
it). The old name `composite_laminate_model` still imports, with a deprecation
warning, and re-exports this package and its modules
(`composite_laminate_model.laminate_mechanics`, ...); it will be removed in the next
minor release. Finite-element names (`FEAMesh`, `BladeBuilder`, ...) moved to
`shell_fem`.

## Quick start

```python
import numpy as np
from laminate_constitutive_model import HashinCriterion, Laminate, create_default_library, integrate_abd

card = create_default_library().get("T300/5208")
laminate = Laminate.from_angles(card, [0, 45, -45, 90, 90, -45, 45, 0], ply_thickness=0.25e-3)

ABD = integrate_abd(laminate)                     # 6 x 6 [A B; B D]
N = np.array([200e3, 0.0, 0.0])                   # Nx = 200 kN/m
strain = np.linalg.solve(ABD[:3, :3], N)          # symmetric laminate: no curvature
for k, ply in enumerate(laminate.plies):
    stress = ply.Q_bar @ strain                   # global axes [σx, σy, τxy]
    print(k, ply.angle, stress / 1e6)
```

## Conventions

| Item | Convention |
|---|---|
| Units | Any consistent set. The default material library is SI (Pa, m, 1/K). |
| Material axes | 1 = fibre, 2 = in-plane transverse, 3 = thickness. 3D Voigt order is [11, 22, 33, 23, 13, 12]. |
| Ply angle | Degrees, counter-clockwise about the element normal from the element's material axis (in shell-fem, the element's `material_axis` projected onto the element plane). |
| Stacking | Plies are listed bottom to top. z is measured from the reference surface, which is the mid-plane unless the laminate has an `offset`. |
| Shear vectors | [γyz, γxz] in global axes and [γ23, γ13] in material axes, so Q_s = diag(G23, G13). |

## Theory notes

- **Q̄ transform.** `Q̄ = T⁻¹ Q R T R⁻¹`, where R is the Reuter matrix. The tests check it against the Tsai–Pagano invariant form at every angle.
- **CLPT.** The section has no transverse shear flexibility; shell-fem's C0 elements impose it with a penalty shear stiffness (1000 × Σ Q̄_s h).
- **FSDT.** A_s = k Σ Q̄_s h with k = 5/6.
- **Refined zig-zag (RZT).** u(z) = u0 + z·β + φ(z)·ψ, with the zig-zag function φ built from the ply shear moduli (Tessler, Di Sciuva and Gherlone). In shell-fem the amplitudes ψ are extra nodal DOFs. `RefinedZigZagModel(condensed=True)` ties ψ to the mean shear instead (6 DOFs per node, shear stiffness A_s = hG).
- **Layerwise.** Each internal interface carries its own in-plane displacement (hat functions in z), optionally with sub-layers inside each ply. `LayerwiseModel(condensed=True)` condenses them out into a 6-DOF section.
- **Generalised sections.** Every theory is written as ε(z) = E(z)·e and γ(z) = G(z)·s, with section matrices ∫EᵀQ̄E dz and ∫GᵀQ̄sG dz, so every theory gives its section stiffness the same way and shell-fem's elements share one formulation.

## Verification

The test suite includes:

- the Q̄ transform against the Tsai–Pagano invariant form at every angle;
- ABD against the closed form for one orthotropic ply and the coupling terms of
  cross-ply and symmetric stacks;
- the RZT and layerwise sections: zig-zag functions that vanish on the faces, shear
  stress continuous across interfaces, layerwise condensing to RZT and projecting
  back to the CLPT ABD;
- hygrothermal expansion coefficients and resultants against numerical integration;
- every failure criterion giving an index of 1 at each uniaxial strength, the Hoffman
  interaction coefficient, and Hashin's modes and 2D matrix-compression formula.

The finite-element checks of these laws (Navier, Pagano, free thermal warping,
first-ply failure, CalculiX) live with the elements in
[shell-fem](https://github.com/samhughes-fea/shell-fem).

## License

MIT; see [LICENSE](LICENSE).

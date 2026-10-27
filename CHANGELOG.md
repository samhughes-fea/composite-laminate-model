# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `blade_geometry_model` now holds a fork, with history, of blade-structure's
  implicit section geometry: `section_geometry` (signed-distance solids with CSG,
  40 structural layouts, wall midlines and cell-wall loops) with `section_contract`,
  `coordinate_frames`, `area_2d` and `shoelace_integration`, and their 785 tests.
  `src/blade_geometry_model/FORK.md` records the fork point and the files that stayed
  in blade-structure.
- `BladeGeometry` lofts one of those layouts between `Station`s (chord, twist, pitch
  axis, offset and airfoil interpolated linearly) and `section_at(z)` returns a
  `SectionCut`: wall midlines with thickness and region labels, cell loops and any
  dropped webs, in the `BladeBuilder` blade frame. Wall thicknesses may vary along the
  span, webs sit at chord fractions or at fixed distances from the leading edge, and
  walls thinner than the geometry's manufacturing floors are built exactly.
- `ThinWalledSection.from_geometry(cut, laminates)` builds a section from a cut, with
  laminates bound by label, region or kind and spar caps merged into the skin they are
  bonded to. On the example 20 section it matches `from_blade_mesh` to within 1% in EA,
  EI and GJ.

### Changed

- Python 3.11 or newer is required (the forked geometry needs it; 3.9 and 3.10 are
  past end of life). Shapely and Numba are now dependencies.
- pytest imports test modules by path (`--import-mode=importlib`) and also collects
  the geometry's tests under `src/blade_geometry_model`.
- The code is split into three packages, each depending only on the ones before it:
  `blade_geometry_model` (`Airfoil`, `Station`, `SkinRegion`, `ShearWeb`),
  `composite_laminate_model` (laminate mechanics, shell and solid FE, solver,
  kernels, `BladeBuilder`, `beam_section_properties`) and
  `anisotropic_section_model` (`ThinWalledSection`, `RestrainedTorsionBeam`,
  `GeneralizedBeam`, `TimoshenkoBeam`). The section modules drop their `laminate_`
  prefix (`laminate_gbt` is now `anisotropic_section_model.gbt`, and so on).
- `laminate_modelling_package` still imports and re-exports the same public names,
  and its old submodules are aliases of the new ones. It will be removed after the
  first release.
- `tests/test_package_split.py` checks every import against those boundaries.

## [0.1.0] - not yet released

First release of `composite-laminate-fea` (import name `laminate_modelling_package`).

### Added

- 3D orthotropic materials, plies and laminates, temperature-dependent properties
  and a material library.
- CLPT, FSDT, refined zig-zag and layerwise kinematics behind one generalised
  section formulation.
- Tri3, Quad4, Quad8, curved Quad8 (default) and Tri6 shells; layered Hex8 and
  Hex20 solids with shell-to-solid coupling; RBE2 and RBE3 links.
- Blade section builder (airfoils, skin regions, spar caps, shear webs, twist,
  taper), mould-line or mid-surface skins, ply books with ply drops, and beam
  section properties extracted from the shell model.
- Pressure fields, line, body and hygrothermal loads; load cases solved with one
  factorisation and load combinations.
- Through-thickness stress recovery, including interlaminar stresses from 3D
  equilibrium; Pagano's exact 3D solution as a benchmark.
- Maximum stress, Tsai-Hill, Tsai-Wu and Hashin failure criteria, a delamination
  onset check, and ply-by-ply stiffness knockdown for progressive damage.
- Native NumPy/SciPy, CalculiX and Nastran solver kernels, a benchmark runner
  (`laminate-benchmarks`), Gmsh meshing, model checks and ParaView export.
- Twenty-six worked examples, each checked against a reference.

### Packaging

- MIT licence, `pyproject.toml` packaging (setuptools), version taken from
  `laminate_modelling_package.__version__`.
- The benchmark runner moved into the package
  (`laminate_modelling_package.run_solver_benchmarks`).
- A release workflow that builds and checks the sdist and wheel; publishing to
  PyPI is manual and off by default (see `RELEASING.md`).

# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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

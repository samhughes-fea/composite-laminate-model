"""laminate_modelling_package - the suite's original import name, kept for one release.

The code now lives in three packages:

- ``blade_geometry_model``: Airfoil, Station, SkinRegion, ShearWeb
- ``composite_laminate_model``: materials, laminates, kinematics, shell and solid
  elements, the solver, kernels, BladeBuilder and beam_section_properties
- ``anisotropic_section_model``: ThinWalledSection, RestrainedTorsionBeam,
  GeneralizedBeam, TimoshenkoBeam

Everything that ``laminate_modelling_package`` exported is re-exported here, and its
old submodules (``laminate_modelling_package.laminate_gbt``,
``laminate_modelling_package.kernels`` and so on) are aliases of the modules that
replaced them, so existing code keeps working unchanged. New code should import from
the three packages.
"""

import importlib as _importlib
import sys as _sys

# old submodule name -> the module that now holds it
_MOVED = {
    "laminate_thin_wall": "anisotropic_section_model.thin_wall",
    "laminate_torsion": "anisotropic_section_model.torsion",
    "laminate_gbt": "anisotropic_section_model.gbt",
    "laminate_beam": "anisotropic_section_model.beam",
}
for _name in (
    "kernels",
    "kernels.base",
    "kernels.calculix",
    "kernels.decks",
    "kernels.nastran",
    "kernels.native",
    "kernels.registry",
    "laminate_benchmark_stress",
    "laminate_blade",
    "laminate_connectors",
    "laminate_curved_shell",
    "laminate_elasticity_3d",
    "laminate_elements",
    "laminate_equilibrium",
    "laminate_failure_criteria",
    "laminate_fea_engine",
    "laminate_gmsh",
    "laminate_hygrothermal",
    "laminate_kernels",
    "laminate_kinematics",
    "laminate_layup",
    "laminate_materials",
    "laminate_materials_3d",
    "laminate_mechanics",
    "laminate_mesh_builders",
    "laminate_model_check",
    "laminate_orientation",
    "laminate_progressive_damage",
    "laminate_results",
    "laminate_solid",
    "laminate_solver_pipeline",
    "laminate_stress_recovery",
    "laminate_temperature",
    "laminate_vtk",
    "run_solver_benchmarks",
):
    _MOVED[_name] = "composite_laminate_model." + _name

for _old, _new in _MOVED.items():
    _module = _importlib.import_module(_new)
    _sys.modules[f"{__name__}.{_old}"] = _module
    if "." not in _old:
        globals()[_old] = _module

from composite_laminate_model import *  # noqa: E402,F401,F403
from composite_laminate_model import __all__ as _laminate_all  # noqa: E402
from anisotropic_section_model import *  # noqa: E402,F401,F403
from anisotropic_section_model import __all__ as _section_all  # noqa: E402
from composite_laminate_model import __version__  # noqa: E402,F401

__all__ = sorted(set(_laminate_all) | set(_section_all))

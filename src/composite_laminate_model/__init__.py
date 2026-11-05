"""Deprecated name of ``laminate_constitutive_model``; removed in its next minor release.

composite-laminate-model was split in two: the ply and laminate behaviour is now
``laminate_constitutive_model`` (this repository) and the finite elements, solver,
meshing and blade models are ``shell_fem``. This alias re-exports the constitutive
package, including its submodules (``composite_laminate_model.laminate_mechanics``),
so code that only uses laminates keeps working. Shell FE names (``FEAMesh``,
``BladeBuilder``, ...) are in ``shell_fem``.
"""

import importlib as _importlib
import sys as _sys
import warnings as _warnings

import laminate_constitutive_model as _target
from laminate_constitutive_model import *  # noqa: F401,F403
from laminate_constitutive_model import __all__, __version__  # noqa: F401

_warnings.warn(
    "composite_laminate_model is now laminate_constitutive_model (and shell_fem for the finite "
    "elements); import from those instead",
    DeprecationWarning,
    stacklevel=2,
)

for _name in ("laminate_elasticity_3d", "laminate_failure_criteria", "laminate_hygrothermal",
              "laminate_kinematics", "laminate_materials", "laminate_materials_3d", "laminate_mechanics",
              "laminate_temperature"):
    _sys.modules[f"{__name__}.{_name}"] = _importlib.import_module(f"{_target.__name__}.{_name}")

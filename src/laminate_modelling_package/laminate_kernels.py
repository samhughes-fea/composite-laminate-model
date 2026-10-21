"""Compatibility facade for ``laminate_modelling_package.kernels``.

The kernels now live in a subpackage with one module per backend; this module
re-exports them so existing imports keep working.
"""

from .kernels import *  # noqa: F401,F403
from .kernels import __all__  # noqa: F401

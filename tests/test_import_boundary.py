"""The laminate model's boundaries with the packages around it.

    implicit_geometry_kernel  <-  composite_laminate_model  <-  anisotropic_section_model

The laminate model may import the geometry kernel, never the section model, which
depends on it. Every import is checked, including ones inside functions.
"""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

PACKAGE = "composite_laminate_model"
ROOT = Path(__file__).resolve().parents[1] / "src" / PACKAGE
FORBIDDEN = {"anisotropic_section_model", "laminate_modelling_package"}


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            yield node.module, node.lineno


def test_never_imports_the_section_model():
    found = [f"{p.relative_to(ROOT.parent)}:{line} imports {target}"
             for p in sorted(ROOT.rglob("*.py")) for target, line in _imports(p)
             if target.split(".")[0] in FORBIDDEN]
    assert found == []


def test_imports_without_the_section_model():
    code = ("import sys, composite_laminate_model; "
            "print(sorted({m.split('.')[0] for m in sys.modules} & {'anisotropic_section_model'}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_geometry_types_are_re_exported():
    """BladeBuilder's inputs are the geometry kernel's own classes, re-exported."""
    laminate = importlib.import_module(PACKAGE)
    geometry = importlib.import_module("implicit_geometry_kernel")
    for name in ("Airfoil", "Station", "SkinRegion", "ShearWeb"):
        assert getattr(laminate, name) is getattr(geometry, name)
    assert set(laminate.__all__).isdisjoint(set(geometry.__all__) - {"Airfoil", "Station", "SkinRegion", "ShearWeb"})

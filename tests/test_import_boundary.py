"""The constitutive model's boundaries with the packages around it.

    laminate_constitutive_model  <-  shell_fem, anisotropic_section_model

The constitutive model depends on numpy alone: it never imports the geometry kernel,
the shell FE or the section model, all of which build on it. Every import is
checked, including ones inside functions.
"""

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = "laminate_constitutive_model"
ROOT = SRC / PACKAGE
SUITE = {"implicit_geometry_kernel", "shell_fem", "anisotropic_section_model", "onedfem", "composite_laminate_model"}
ALLOWED = {"numpy", "scipy", "__future__", "abc", "collections", "dataclasses", "enum", "functools", "itertools",
           "math", "typing", "warnings", "copy", "json", "pathlib", "types", "re", "os"}


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            yield node.module, node.lineno


def test_imports_nothing_from_the_suite():
    found = [f"{p.relative_to(SRC)}:{line} imports {target}"
             for p in sorted(ROOT.rglob("*.py")) for target, line in _imports(p)
             if target.split(".")[0] in SUITE]
    assert found == []


def test_imports_only_numpy_and_the_standard_library():
    found = [f"{p.relative_to(SRC)}:{line} imports {target}"
             for p in sorted(ROOT.rglob("*.py")) for target, line in _imports(p)
             if target.split(".")[0] not in ALLOWED]
    assert found == []


def test_old_name_still_imports_the_laminates():
    code = ("import warnings; warnings.simplefilter('error', DeprecationWarning)\n"
            "try:\n    import composite_laminate_model\nexcept DeprecationWarning:\n    print('warned')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(SRC)})
    assert out.stdout.strip() == "warned"
    import composite_laminate_model as old
    import laminate_constitutive_model as new
    from composite_laminate_model.laminate_kinematics import integrate_abd
    from composite_laminate_model.laminate_mechanics import Laminate

    assert old.Laminate is new.Laminate is Laminate and integrate_abd is new.integrate_abd

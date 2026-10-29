# Releasing

Nothing is published to PyPI automatically. A release is three deliberate steps.

## 1. Prepare

1. Set `__version__` in `src/composite_laminate_model/__init__.py`;
   `pyproject.toml` reads it from there. This package depends on
   [implicit-geometry-kernel](https://github.com/samhughes-fea/implicit-geometry-kernel),
   and [anisotropic-section-model](https://github.com/samhughes-fea/anisotropic-section-model)
   depends on it, so release in that order: geometry kernel, laminate model, section
   model.
2. Move the `Unreleased` notes in `CHANGELOG.md` under the new version and date.
3. Check locally:

   ```bash
   pip install -e ".[dev]"
   pytest
   python -m build
   python -m twine check --strict dist/*
   ```

4. Merge to `main`.

## 2. Tag

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `release` workflow builds the sdist and wheel, checks that the tag matches
`__version__`, installs the wheel in a clean environment, runs the benchmark from
it, and creates a GitHub release with both files attached. It does not upload
anything to PyPI.

## 3. Publish to PyPI (when ready)

One-time setup:

1. On PyPI, add a trusted publisher for the project `composite-laminate-model`:
   owner `samhughes-fea`, repository `composite-laminate-model`, workflow
   `release.yml`, environment `pypi`.
2. In the GitHub repository settings, create an environment called `pypi`
   (optionally with required reviewers, so every upload needs an approval).

Then, for each release: Actions → release → Run workflow, choose the tag (for
example `v0.1.0`) as the ref and tick "Publish to PyPI". The `pypi` job runs only
for a manual run from a `v*` tag with that box ticked.

PyPI versions cannot be replaced once uploaded, so to try the process first, point
a second trusted publisher at TestPyPI and upload there.

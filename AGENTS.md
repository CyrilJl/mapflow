# Repository Guidelines

## Project Structure & Module Organization
- `mapflow/` contains the library code (`_classic.py`, `_quiver.py`, `_misc.py`) and package entry points.
- `tests/` holds pytest suites (`test_animate.py`, `test_plot.py`).
- `docs/` is the Sphinx documentation project; `_static/` stores docs/assets and branding.
- `scripts/` contains one-off utilities such as `logo.py`.

## Build, Test, and Development Commands
- `python -m pip install -e .` installs the package in editable mode for local development.
- `python -m pytest` runs the full test suite.
- `python -m build` produces sdist and wheel artifacts (uses `hatchling` from `pyproject.toml`).
- `cd docs && make html` (or `docs\make.bat html` on Windows) builds the docs locally.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and `snake_case` for functions/variables.
- Modules are private by convention (`_classic.py`, `_quiver.py`).
- Ruff is configured with a 120-character line length; keep formatting aligned with that.
- Prefer explicit parameter names in public APIs (`animate`, `plot_da`) for clarity.

## Testing Guidelines
- Tests use `pytest`; files follow `test_*.py` and functions `test_*` patterns.
- Animation tests rely on `ffprobe`/`ffmpeg` being available on PATH.
- Favor small datasets and short durations to keep test runtime reasonable.

## Commit & Pull Request Guidelines
- Recent commits are short and imperative, sometimes using Conventional Commit prefixes like `feat:` and `chore:`. Keep messages concise and descriptive.
- PRs should include a clear summary, test results (`python -m pytest`), and linked issues if applicable.
- For visual or plotting changes, include a screenshot or GIF.

## Configuration Tips
- Dependencies include `xarray`, `matplotlib`, `geopandas`, and `tqdm`; ensure geospatial libraries are installed for plotting tests.
- For CI parity, avoid hardcoding paths and prefer temporary directories.

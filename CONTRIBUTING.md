# Contributing to mapflow

Thank you for helping improve mapflow. Bug reports, documentation fixes, tests, and focused feature proposals are
welcome.

## Development setup

Python 3.10 or newer and FFmpeg are required to run the complete test suite.

```bash
git clone https://github.com/CyrilJl/mapflow.git
cd mapflow
uv sync --group dev
```

Use a small, focused branch for each change. Public API changes should include docstrings, documentation, and tests.
Plotting changes should include a screenshot or short video in the pull request.

## Checks

Run the same checks enforced in CI before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
uv run python -m build
uv run twine check dist/*
```

Tests must not depend on live network resources. Use small deterministic arrays and pytest temporary directories so
the suite stays reliable and fast.

## Pull requests

Explain why the change is useful, summarize user-visible behavior, and report the commands you ran. Link related
issues where applicable. Maintainers may ask for a changelog entry when a change affects users.

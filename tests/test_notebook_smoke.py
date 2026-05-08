"""Smoke tests for all notebook scripts.

For each notebook .py file this test:
1. Verifies the file has no syntax errors (py_compile).
2. Verifies that the imports-only cell executes without error, confirming
   all internal and third-party dependencies can be resolved.

Notebooks that require external services (GCS, ADC) are expected to guard
themselves with env-var checks *after* the import block, so only the import
block is executed here.
"""

# ruff: noqa:S102
# pylint: disable=W0122,W0718

from __future__ import annotations

import py_compile
import re
import sys
import types
from pathlib import Path

import pytest

NOTEBOOKS_ROOT = Path(__file__).parent.parent / "notebooks"
SRC_ROOT = Path(__file__).parent.parent / "src"

# Files that are not executable notebooks
_EXCLUDED_NAMES = {"__init__.py"}
# Subdirectory that contains throwaway scratch work
_EXCLUDED_DIRS = {"temp", "__pycache__"}


def _collect_notebook_files() -> list[Path]:
    """Return all notebook .py files, excluding helpers and temp."""
    files = []
    for path in sorted(NOTEBOOKS_ROOT.rglob("*.py")):
        if path.name in _EXCLUDED_NAMES:
            continue
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def _extract_import_cell(source: str) -> str:
    """Return the source code of the first '# %%' cell that contains only imports.

    Walks cells in order and collects consecutive import-only cells (cells whose
    non-blank, non-comment lines are all ``import`` or ``from … import`` statements).
    Stops at the first cell that has other executable content.
    """
    # Split on cell markers; keep leading docstring / pragma comments too
    cells = re.split(r"^# %%[^\n]*\n", source, flags=re.MULTILINE)

    import_lines: list[str] = []
    for cell in cells:
        executable = [
            line
            for line in cell.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not executable:
            continue
        all_imports = all(
            re.match(r"^\s*(import |from \S+ import)", line) for line in executable
        )
        if all_imports:
            import_lines.extend(executable)
        else:
            break

    return "\n".join(import_lines)


NOTEBOOK_FILES = _collect_notebook_files()


@pytest.mark.parametrize(
    "notebook_path", NOTEBOOK_FILES, ids=lambda p: str(p.relative_to(NOTEBOOKS_ROOT))
)
def test_notebook_syntax(notebook_path: Path) -> None:
    """Notebook compiles without syntax errors."""
    try:
        py_compile.compile(str(notebook_path), doraise=True)
    except py_compile.PyCompileError as exc:
        pytest.fail(f"Syntax error in {notebook_path.name}: {exc}")


@pytest.mark.parametrize(
    "notebook_path", NOTEBOOK_FILES, ids=lambda p: str(p.relative_to(NOTEBOOKS_ROOT))
)
def test_notebook_imports(notebook_path: Path) -> None:
    """All imports in the first import cell resolve without error."""
    source = notebook_path.read_text(encoding="utf-8")
    import_src = _extract_import_cell(source)

    if not import_src:
        pytest.skip("No import-only cell found; skipping import smoke test.")

    # Execute imports in a fresh throw-away module so they don't pollute globals
    module = types.ModuleType(f"_smoke_{notebook_path.stem}")
    module.__file__ = str(notebook_path)
    # Ensure src/ is on the path (mirrors pytest.ini pythonpath = src)
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    try:
        exec(compile(import_src, str(notebook_path), "exec"), module.__dict__)
    except ImportError as exc:
        pytest.fail(f"Import error in {notebook_path.name}: {exc}")
    except Exception as exc:
        pytest.fail(
            f"Unexpected error executing imports in {notebook_path.name}: {exc}"
        )

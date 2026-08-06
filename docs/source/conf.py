"""Sphinx configuration.

The version is read back out of ``pyproject.toml`` rather than duplicated here, so the
static version string in that file stays the single source of truth.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

_pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
project = "RBFENetworkMap"
author = "Zeke Piskulich"
release = _pyproject["project"]["version"]
version = release
copyright = f"2026, {author}"  # noqa: A001 - Sphinx requires this name


def _available(name: str) -> bool:
    """Whether a module can be located without importing it."""
    return importlib.util.find_spec(name) is not None


_optional = {"numpydoc": "numpydoc", "sphinx_copybutton": "sphinx_copybutton"}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "sphinx_copybutton",
]
extensions = [e for e in extensions if _available(_optional.get(e, e.split(".")[0]))]

# Heavy scientific dependencies are mocked so the docs build in a minimal environment,
# which is also what keeps the CI docs job fast.
autodoc_mock_imports = ["rdkit", "scipy", "kartograf", "gufe", "rdk_amber", "yaml"]

autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": True, "member-order": "bysource"}
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path: list[str] = []
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme" if _available("sphinx_rtd_theme") else "alabaster"
html_static_path: list[str] = []

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

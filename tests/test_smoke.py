"""Packaging smoke tests.

Every module imports, and every declared console script resolves. This is the cheapest
possible guard against the class of bug where a rename leaves an entry point pointing at
a function that no longer exists -- something no unit test catches, because unit tests
import the function directly rather than through the entry point.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import rbfenetmap


def _module_names() -> list[str]:
    """Every importable module in the package."""
    return [name for _, name, _ in pkgutil.walk_packages(rbfenetmap.__path__, prefix="rbfenetmap.")]


@pytest.mark.parametrize("module_name", _module_names())
def test_every_module_imports(module_name):
    importlib.import_module(module_name)


def test_version_is_exposed():
    assert isinstance(rbfenetmap.__version__, str)
    assert rbfenetmap.__version__


def test_console_scripts_resolve():
    from importlib.metadata import entry_points

    try:
        scripts = [ep for ep in entry_points(group="console_scripts") if ep.module.startswith("rbfenetmap")]
    except TypeError:  # pragma: no cover - older importlib.metadata
        scripts = []
    if not scripts:
        pytest.skip("package is not installed; entry points unavailable")
    for entry in scripts:
        assert callable(entry.load())


def test_cli_main_is_importable_and_callable():
    # The entry point target, checked directly so this works from a source tree too.
    from rbfenetmap.cli.main import main

    assert callable(main)


def test_plugin_targets_all_resolve():
    """Every built-in plugin's ``module:Class`` target must actually exist.

    A registry entry is just a string, so a typo or a rename is invisible until someone
    selects that plugin. Optional backends are skipped -- their availability is a property
    of the environment, not of the target string.
    """
    from rbfenetmap.plugins.exporters import BUILTIN_EXPORTERS
    from rbfenetmap.plugins.mappers import BUILTIN_MAPPERS
    from rbfenetmap.plugins.planners import BUILTIN_PLANNERS
    from rbfenetmap.plugins.scorers import BUILTIN_SCORERS

    for table in (BUILTIN_MAPPERS, BUILTIN_SCORERS, BUILTIN_PLANNERS, BUILTIN_EXPORTERS):
        for name, spec in table.items():
            if not spec.available:
                continue
            module_name, _, class_name = spec.target.partition(":")
            module = importlib.import_module(module_name)
            assert hasattr(module, class_name), f"{name}: {spec.target} does not resolve"

"""Lazy plugin registry.

Adapted from ``pharmaforge.core.pluginregistry``, keeping that package's conventions so
the two are recognisably the same mechanism.

The design point worth preserving: a :class:`PluginSpec` describes a plugin without
importing it. Registration is pure metadata, and the implementation module is imported
only when :meth:`PluginRegistry.create` is called. That is what lets ``rbfenet plugins``
list every backend -- including ones whose dependencies are absent -- without importing
RDKit, kartograf, or anything else, and what lets the whole test suite run with no
optional dependency installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any

from rbfenetmap.core.exceptions import PluginError

__all__ = ("PluginRegistry", "PluginSpec")


@dataclass(frozen=True)
class PluginSpec:
    """Describe a plugin without importing its implementation.

    Parameters
    ----------
    name : str
        Unique name within a given plugin kind.
    kind : str
        The plugin category: ``"mapper"``, ``"scorer"``, ``"planner"``, or
        ``"exporter"``.
    target : str
        Import target in ``"package.module:ClassName"`` form.
    description : str, optional
        Short human-readable summary.
    requires : tuple[str, ...], optional
        Top-level modules the plugin's backend needs. Probed with
        :func:`importlib.util.find_spec`, so availability can be reported without
        importing anything.
    """

    name: str
    kind: str
    target: str
    description: str = ""
    requires: tuple[str, ...] = ()

    @property
    def missing_requirements(self) -> tuple[str, ...]:
        """Required modules that cannot be located in this environment."""
        missing: list[str] = []
        for module in self.requires:
            try:
                if find_spec(module) is None:
                    missing.append(module)
            except (ImportError, ValueError):
                missing.append(module)
        return tuple(missing)

    @property
    def available(self) -> bool:
        """Whether every required module can be located."""
        return not self.missing_requirements


class PluginRegistry:
    """Store plugin metadata and construct plugins on demand.

    The registry tracks *registered* plugins, which is not the same as *active* ones.
    Registration says a plugin exists; activation says the current configuration intends
    to use it. Keeping the two apart lets a caller enumerate everything installed while
    still restricting a given run to a chosen subset.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._plugins: dict[tuple[str, str], PluginSpec] = {}
        self._active: set[tuple[str, str]] = set()

    def register(self, spec: PluginSpec) -> None:
        """Register *spec*.

        Raises
        ------
        PluginError
            If a plugin with the same ``(kind, name)`` is already registered. Silently
            overwriting would make plugin behaviour depend on import order.
        """
        key = (spec.kind, spec.name)
        if key in self._plugins:
            existing = self._plugins[key]
            raise PluginError(
                f"A {spec.kind} plugin named {spec.name!r} is already registered "
                f"(target {existing.target!r}); refusing to overwrite it with {spec.target!r}."
            )
        self._plugins[key] = spec

    def get_spec(self, name: str, kind: str) -> PluginSpec:
        """Return the spec for ``(kind, name)``.

        Raises
        ------
        PluginError
            If no such plugin is registered. The message lists the registered names for
            that kind, since the usual cause is a typo.
        """
        try:
            return self._plugins[(kind, name)]
        except KeyError:
            known = sorted(n for k, n in self._plugins if k == kind)
            raise PluginError(f"Unknown {kind} plugin {name!r}. Registered: {known or 'none'}.") from None

    def activate(self, name: str, kind: str) -> None:
        """Mark ``(kind, name)`` active, registering nothing new."""
        self.get_spec(name, kind)
        self._active.add((kind, name))

    def deactivate(self, name: str, kind: str) -> None:
        """Mark ``(kind, name)`` inactive."""
        self._active.discard((kind, name))

    def is_active(self, name: str, kind: str) -> bool:
        """Whether ``(kind, name)`` is active."""
        return (kind, name) in self._active

    def create(self, name: str, kind: str, **kwargs: Any) -> Any:
        """Import and instantiate a plugin.

        This is the only method that imports anything.

        Raises
        ------
        PluginError
            If the plugin is unknown, its module or class cannot be imported, or its
            declared requirements are missing. A missing requirement is reported before
            the import is attempted, so the user sees ``"needs kartograf"`` rather than a
            raw :class:`ModuleNotFoundError`.
        """
        spec = self.get_spec(name, kind)
        missing = spec.missing_requirements
        if missing:
            raise PluginError(
                f"{kind} plugin {name!r} requires module(s) {list(missing)}, which are not installed. "
                f"Install them, or choose another {kind} (see `rbfenet plugins --all`)."
            )
        module_name, _, class_name = spec.target.partition(":")
        if not class_name:
            raise PluginError(f"Malformed plugin target {spec.target!r}; expected 'package.module:ClassName'.")
        try:
            module = import_module(module_name)
        except ImportError as exc:
            raise PluginError(f"Could not import module {module_name!r} for {kind} plugin {name!r}: {exc}") from exc
        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise PluginError(f"Module {module_name!r} has no attribute {class_name!r}.") from exc
        return cls(**kwargs)

    def list_plugins(self, kind: str | None = None, *, active_only: bool = False) -> tuple[PluginSpec, ...]:
        """Return registered specs, optionally filtered by kind or activity."""
        specs = [
            spec
            for (spec_kind, spec_name), spec in sorted(self._plugins.items())
            if (kind is None or spec_kind == kind) and (not active_only or (spec_kind, spec_name) in self._active)
        ]
        return tuple(specs)

    def __contains__(self, key: tuple[str, str]) -> bool:
        """Whether ``(kind, name)`` is registered."""
        return key in self._plugins

    def __len__(self) -> int:
        """Number of registered plugins."""
        return len(self._plugins)

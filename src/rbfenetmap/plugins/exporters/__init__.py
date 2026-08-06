"""Built-in exporter plugins and their registry.

Exporters are the package's hook into other programs: each one adapts a planned network
to a downstream consumer without that consumer's concerns reaching back into the core.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from rbfenetmap.core.exceptions import PluginError
from rbfenetmap.core.pluginregistry import PluginRegistry, PluginSpec

__all__ = (
    "BUILTIN_EXPORTERS",
    "EXPORTER_PROFILES",
    "available_exporters",
    "create_exporter",
    "create_exporter_registry",
    "list_active_exporters",
    "register_exporters",
    "require_exporters",
)

_KIND = "exporter"

_MODULE_MAP = {
    "JSONExporter": "rbfenetmap.plugins.exporters.basic_exporters",
    "EdgeListExporter": "rbfenetmap.plugins.exporters.basic_exporters",
    "GraphMLExporter": "rbfenetmap.plugins.exporters.basic_exporters",
    "AmberExporter": "rbfenetmap.plugins.exporters.amber_exporter",
    "HTMLGalleryExporter": "rbfenetmap.plugins.exporters.html_exporter",
}

BUILTIN_EXPORTERS: dict[str, PluginSpec] = {
    "json": PluginSpec(
        name="json",
        kind=_KIND,
        target="rbfenetmap.plugins.exporters.basic_exporters:JSONExporter",
        description="Full round-trippable network, including rejected candidates.",
        requires=(),
    ),
    "edgelist": PluginSpec(
        name="edgelist",
        kind=_KIND,
        target="rbfenetmap.plugins.exporters.basic_exporters:EdgeListExporter",
        description="Plain 'source target cost' text, for shell and workflow use.",
        requires=(),
    ),
    "graphml": PluginSpec(
        name="graphml",
        kind=_KIND,
        target="rbfenetmap.plugins.exporters.basic_exporters:GraphMLExporter",
        description="GraphML for Cytoscape, Gephi, and similar viewers.",
        requires=("networkx",),
    ),
    "amber": PluginSpec(
        name="amber",
        kind=_KIND,
        target="rbfenetmap.plugins.exporters.amber_exporter:AmberExporter",
        description="amberstudio/guimapper edges.dat plus atommap_*.runconfig files.",
        requires=("yaml",),
    ),
    "html": PluginSpec(
        name="html",
        kind=_KIND,
        target="rbfenetmap.plugins.exporters.html_exporter:HTMLGalleryExporter",
        description="Self-contained HTML report with depictions and rejections.",
        requires=("rdkit",),
    ),
}

EXPORTER_PROFILES: dict[str, tuple[str, ...]] = {
    "all": tuple(BUILTIN_EXPORTERS),
    "examples": ("json", "edgelist", "html"),
}


def available_exporters() -> dict[str, PluginSpec]:
    """Return the built-in exporters whose requirements are importable."""
    return {name: spec for name, spec in BUILTIN_EXPORTERS.items() if spec.available}


def register_exporters(registry: PluginRegistry, names: tuple[str, ...] | None = None) -> PluginRegistry:
    """Register the named exporters (default: all built-ins) into *registry*."""
    for name in names or tuple(BUILTIN_EXPORTERS):
        try:
            registry.register(BUILTIN_EXPORTERS[name])
        except KeyError:
            raise PluginError(f"Unknown built-in exporter {name!r}. Known: {sorted(BUILTIN_EXPORTERS)}.") from None
    return registry


def create_exporter_registry(profile: str = "all") -> PluginRegistry:
    """Return a registry with the exporters of *profile* registered and activated."""
    if profile not in EXPORTER_PROFILES:
        raise PluginError(f"Unknown exporter profile {profile!r}. Known: {sorted(EXPORTER_PROFILES)}.")
    registry = register_exporters(PluginRegistry(), EXPORTER_PROFILES[profile])
    for name in EXPORTER_PROFILES[profile]:
        registry.activate(name, _KIND)
    return registry


def create_exporter(name: str, profile: str = "all", **kwargs: Any) -> Any:
    """Instantiate the exporter *name*."""
    return create_exporter_registry(profile).create(name, _KIND, **kwargs)


def list_active_exporters(profile: str = "all") -> list[str]:
    """Return the names of the exporters in *profile* that can be created."""
    registry = create_exporter_registry(profile)
    return sorted(spec.name for spec in registry.list_plugins(_KIND, active_only=True) if spec.available)


def require_exporters(names: tuple[str, ...], profile: str = "all") -> None:
    """Raise unless every exporter in *names* is available."""
    available = available_exporters()
    missing = {n: BUILTIN_EXPORTERS[n].missing_requirements for n in names if n not in available}
    if missing:
        detail = "; ".join(f"{n} needs {list(mods)}" for n, mods in sorted(missing.items()))
        raise PluginError(f"Required exporter(s) unavailable: {detail}.")


def __getattr__(name: str) -> Any:
    """Import exporter classes lazily (PEP 562)."""
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_MODULE_MAP[name]), name)

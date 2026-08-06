"""Built-in mapper plugins and their registry.

Follows the ``pharmaforge.plugins.calculators`` template: a table of
:class:`~rbfenetmap.core.pluginregistry.PluginSpec` metadata, availability probing that
never imports a backend, factory helpers, and a PEP 562 ``__getattr__`` so
``from rbfenetmap.plugins.mappers import KartografMapper`` works without importing
kartograf for everyone else.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from rbfenetmap.core.exceptions import PluginError
from rbfenetmap.core.pluginregistry import PluginRegistry, PluginSpec

__all__ = (
    "BUILTIN_MAPPERS",
    "MAPPER_PROFILES",
    "available_mappers",
    "create_mapper",
    "create_mapper_registry",
    "list_active_mappers",
    "register_mappers",
    "require_mappers",
)

_KIND = "mapper"

_MODULE_MAP = {
    "MCSSMapper": "rbfenetmap.plugins.mappers.mcss_mapper",
    "MCSSExtendedMapper": "rbfenetmap.plugins.mappers.mcss_mapper",
    "MCSSExtended2Mapper": "rbfenetmap.plugins.mappers.mcss_mapper",
    "CartographMapper": "rbfenetmap.plugins.mappers.cartograph_mapper",
    "KartografMapper": "rbfenetmap.plugins.mappers.kartograf_mapper",
    "IdentityMapper": "rbfenetmap.plugins.mappers.identity_mapper",
}

BUILTIN_MAPPERS: dict[str, PluginSpec] = {
    "mcss": PluginSpec(
        name="mcss",
        kind=_KIND,
        target="rbfenetmap.plugins.mappers.mcss_mapper:MCSSMapper",
        description="Maximum common substructure, no property-based core pruning.",
        requires=("rdkit",),
    ),
    "mcss-e": PluginSpec(
        name="mcss-e",
        kind=_KIND,
        target="rbfenetmap.plugins.mappers.mcss_mapper:MCSSExtendedMapper",
        description="MCS, additionally demoting pairs whose connectivity differs.",
        requires=("rdkit",),
    ),
    "mcss-e2": PluginSpec(
        name="mcss-e2",
        kind=_KIND,
        target="rbfenetmap.plugins.mappers.mcss_mapper:MCSSExtended2Mapper",
        description="MCS, demoting pairs differing in element or connectivity.",
        requires=("rdkit",),
    ),
    "cartograph": PluginSpec(
        name="cartograph",
        kind=_KIND,
        target="rbfenetmap.plugins.mappers.cartograph_mapper:CartographMapper",
        description="Geometry-based: O3A shape alignment plus Hungarian assignment.",
        requires=("rdkit", "numpy", "scipy"),
    ),
    "kartograf": PluginSpec(
        name="kartograf",
        kind=_KIND,
        target="rbfenetmap.plugins.mappers.kartograf_mapper:KartografMapper",
        description="Adapter for the external kartograf geometry mapper.",
        requires=("kartograf", "gufe"),
    ),
    "identity": PluginSpec(
        name="identity",
        kind=_KIND,
        target="rbfenetmap.plugins.mappers.identity_mapper:IdentityMapper",
        description="Pairs atom i with atom i; for pre-aligned inputs and tests.",
        requires=(),
    ),
}

MAPPER_PROFILES: dict[str, tuple[str, ...]] = {
    "all": tuple(BUILTIN_MAPPERS),
    "core": ("mcss", "mcss-e", "mcss-e2", "cartograph", "identity"),
    "examples": ("mcss-e2", "cartograph"),
}


def available_mappers() -> dict[str, PluginSpec]:
    """Return the built-in mappers whose requirements are importable.

    Probes with :func:`importlib.util.find_spec`, so nothing is imported.
    """
    return {name: spec for name, spec in BUILTIN_MAPPERS.items() if spec.available}


def register_mappers(registry: PluginRegistry, names: tuple[str, ...] | None = None) -> PluginRegistry:
    """Register the named mappers (default: all built-ins) into *registry*."""
    for name in names or tuple(BUILTIN_MAPPERS):
        try:
            registry.register(BUILTIN_MAPPERS[name])
        except KeyError:
            raise PluginError(f"Unknown built-in mapper {name!r}. Known: {sorted(BUILTIN_MAPPERS)}.") from None
    return registry


def create_mapper_registry(profile: str = "all") -> PluginRegistry:
    """Return a registry with the mappers of *profile* registered and activated."""
    if profile not in MAPPER_PROFILES:
        raise PluginError(f"Unknown mapper profile {profile!r}. Known: {sorted(MAPPER_PROFILES)}.")
    registry = register_mappers(PluginRegistry(), MAPPER_PROFILES[profile])
    for name in MAPPER_PROFILES[profile]:
        registry.activate(name, _KIND)
    return registry


def create_mapper(name: str, profile: str = "all", **kwargs: Any) -> Any:
    """Instantiate the mapper *name*.

    Raises
    ------
    rbfenetmap.core.exceptions.PluginError
        If the mapper is unknown or its backend is not installed.
    """
    return create_mapper_registry(profile).create(name, _KIND, **kwargs)


def list_active_mappers(profile: str = "all") -> list[str]:
    """Return the names of the mappers in *profile* that can actually be created."""
    registry = create_mapper_registry(profile)
    return sorted(spec.name for spec in registry.list_plugins(_KIND, active_only=True) if spec.available)


def require_mappers(names: tuple[str, ...], profile: str = "all") -> None:
    """Raise unless every mapper in *names* is available.

    Raises
    ------
    rbfenetmap.core.exceptions.PluginError
        Naming the unavailable mappers and the modules each is missing, so the user
        learns what to install rather than merely that something is wrong.
    """
    available = available_mappers()
    missing = {n: BUILTIN_MAPPERS[n].missing_requirements for n in names if n not in available}
    if missing:
        detail = "; ".join(f"{n} needs {list(mods)}" for n, mods in sorted(missing.items()))
        raise PluginError(f"Required mapper(s) unavailable: {detail}.")


def __getattr__(name: str) -> Any:
    """Import mapper classes lazily (PEP 562)."""
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_MODULE_MAP[name]), name)

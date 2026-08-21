"""Built-in intermediate generators and their registry.

The fifth plugin kind. Same shape as :mod:`rbfenetmap.plugins.mappers` -- a table of
:class:`~rbfenetmap.core.pluginregistry.PluginSpec` metadata, availability probed without
importing anything, and a PEP 562 ``__getattr__`` so an implementation class can be
imported by name without pulling its backend in for everyone else.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from rbfenetmap.core.exceptions import PluginError
from rbfenetmap.core.pluginregistry import PluginRegistry, PluginSpec

__all__ = (
    "BUILTIN_INTERMEDIATES",
    "INTERMEDIATE_PROFILES",
    "available_intermediates",
    "create_intermediate",
    "create_intermediate_registry",
    "list_active_intermediates",
    "register_intermediates",
    "require_intermediates",
)

_KIND = "intermediate"

_MODULE_MAP = {"FragmentSwapGenerator": "rbfenetmap.plugins.intermediates.fragment_swap"}

BUILTIN_INTERMEDIATES: dict[str, PluginSpec] = {
    "fragment-swap": PluginSpec(
        name="fragment-swap",
        kind=_KIND,
        target="rbfenetmap.plugins.intermediates.fragment_swap:FragmentSwapGenerator",
        description="Hybrids swapping one substituent at a time between the two parents.",
        requires=("rdkit", "numpy"),
    )
}

INTERMEDIATE_PROFILES: dict[str, tuple[str, ...]] = {"all": tuple(BUILTIN_INTERMEDIATES), "core": ("fragment-swap",)}


def available_intermediates() -> dict[str, PluginSpec]:
    """Return the built-in generators whose requirements are importable."""
    return {name: spec for name, spec in BUILTIN_INTERMEDIATES.items() if spec.available}


def register_intermediates(registry: PluginRegistry, names: tuple[str, ...] | None = None) -> PluginRegistry:
    """Register the named generators (default: all built-ins) into *registry*."""
    for name in names or tuple(BUILTIN_INTERMEDIATES):
        try:
            registry.register(BUILTIN_INTERMEDIATES[name])
        except KeyError:
            raise PluginError(
                f"Unknown built-in intermediate generator {name!r}. Known: {sorted(BUILTIN_INTERMEDIATES)}."
            ) from None
    return registry


def create_intermediate_registry(profile: str = "all") -> PluginRegistry:
    """Return a registry with the generators of *profile* registered and activated."""
    if profile not in INTERMEDIATE_PROFILES:
        raise PluginError(f"Unknown intermediate profile {profile!r}. Known: {sorted(INTERMEDIATE_PROFILES)}.")
    registry = register_intermediates(PluginRegistry(), INTERMEDIATE_PROFILES[profile])
    for name in INTERMEDIATE_PROFILES[profile]:
        registry.activate(name, _KIND)
    return registry


def create_intermediate(name: str, profile: str = "all", **kwargs: Any) -> Any:
    """Instantiate the intermediate generator *name*.

    Raises
    ------
    rbfenetmap.core.exceptions.PluginError
        If the generator is unknown or its backend is not installed.
    """
    return create_intermediate_registry(profile).create(name, _KIND, **kwargs)


def list_active_intermediates(profile: str = "all") -> list[str]:
    """Return the names of the generators in *profile* that can actually be created."""
    registry = create_intermediate_registry(profile)
    return sorted(spec.name for spec in registry.list_plugins(_KIND, active_only=True) if spec.available)


def require_intermediates(names: tuple[str, ...], profile: str = "all") -> None:
    """Raise unless every generator in *names* is available.

    Raises
    ------
    rbfenetmap.core.exceptions.PluginError
        Naming the unavailable generators and the modules each is missing.
    """
    available = available_intermediates()
    missing = {n: BUILTIN_INTERMEDIATES[n].missing_requirements for n in names if n not in available}
    if missing:
        detail = "; ".join(f"{n} needs {list(mods)}" for n, mods in sorted(missing.items()))
        raise PluginError(f"Required intermediate generator(s) unavailable: {detail}.")


def __getattr__(name: str) -> Any:
    """Import generator classes lazily (PEP 562)."""
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_MODULE_MAP[name]), name)

"""Built-in scorer plugins and their registry."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from rbfenetmap.core.exceptions import PluginError
from rbfenetmap.core.pluginregistry import PluginRegistry, PluginSpec

__all__ = (
    "BUILTIN_SCORERS",
    "SCORER_PROFILES",
    "available_scorers",
    "create_scorer",
    "create_scorer_registry",
    "list_active_scorers",
    "register_scorers",
    "require_scorers",
)

_KIND = "scorer"

_MODULE_MAP = {
    "LinearScorer": "rbfenetmap.plugins.scorers.linear_scorer",
    "LomapLikeScorer": "rbfenetmap.plugins.scorers.lomaplike_scorer",
    "SoftcoreSizeScorer": "rbfenetmap.plugins.scorers.softcore_size_scorer",
}

BUILTIN_SCORERS: dict[str, PluginSpec] = {
    "linear": PluginSpec(
        name="linear",
        kind=_KIND,
        target="rbfenetmap.plugins.scorers.linear_scorer:LinearScorer",
        description="Weighted sum of normalised descriptors; the tunable default.",
        requires=(),
    ),
    "lomaplike": PluginSpec(
        name="lomaplike",
        kind=_KIND,
        target="rbfenetmap.plugins.scorers.lomaplike_scorer:LomapLikeScorer",
        description="Multiplicative similarity converted to a cost by -log.",
        requires=(),
    ),
    "softcore-size": PluginSpec(
        name="softcore-size",
        kind=_KIND,
        target="rbfenetmap.plugins.scorers.softcore_size_scorer:SoftcoreSizeScorer",
        description="Baseline: cost equals the larger soft-core heavy-atom count.",
        requires=(),
    ),
}

SCORER_PROFILES: dict[str, tuple[str, ...]] = {"all": tuple(BUILTIN_SCORERS), "examples": ("linear", "softcore-size")}


def available_scorers() -> dict[str, PluginSpec]:
    """Return the built-in scorers whose requirements are importable."""
    return {name: spec for name, spec in BUILTIN_SCORERS.items() if spec.available}


def register_scorers(registry: PluginRegistry, names: tuple[str, ...] | None = None) -> PluginRegistry:
    """Register the named scorers (default: all built-ins) into *registry*."""
    for name in names or tuple(BUILTIN_SCORERS):
        try:
            registry.register(BUILTIN_SCORERS[name])
        except KeyError:
            raise PluginError(f"Unknown built-in scorer {name!r}. Known: {sorted(BUILTIN_SCORERS)}.") from None
    return registry


def create_scorer_registry(profile: str = "all") -> PluginRegistry:
    """Return a registry with the scorers of *profile* registered and activated."""
    if profile not in SCORER_PROFILES:
        raise PluginError(f"Unknown scorer profile {profile!r}. Known: {sorted(SCORER_PROFILES)}.")
    registry = register_scorers(PluginRegistry(), SCORER_PROFILES[profile])
    for name in SCORER_PROFILES[profile]:
        registry.activate(name, _KIND)
    return registry


def create_scorer(name: str, profile: str = "all", **kwargs: Any) -> Any:
    """Instantiate the scorer *name*."""
    return create_scorer_registry(profile).create(name, _KIND, **kwargs)


def list_active_scorers(profile: str = "all") -> list[str]:
    """Return the names of the scorers in *profile* that can be created."""
    registry = create_scorer_registry(profile)
    return sorted(spec.name for spec in registry.list_plugins(_KIND, active_only=True) if spec.available)


def require_scorers(names: tuple[str, ...], profile: str = "all") -> None:
    """Raise unless every scorer in *names* is available."""
    available = available_scorers()
    missing = {n: BUILTIN_SCORERS[n].missing_requirements for n in names if n not in available}
    if missing:
        detail = "; ".join(f"{n} needs {list(mods)}" for n, mods in sorted(missing.items()))
        raise PluginError(f"Required scorer(s) unavailable: {detail}.")


def __getattr__(name: str) -> Any:
    """Import scorer classes lazily (PEP 562)."""
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_MODULE_MAP[name]), name)

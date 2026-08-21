"""Built-in network planner plugins and their registry."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from rbfenetmap.core.exceptions import PluginError
from rbfenetmap.core.pluginregistry import PluginRegistry, PluginSpec

__all__ = (
    "BUILTIN_PLANNERS",
    "PLANNER_PROFILES",
    "available_planners",
    "create_planner",
    "create_planner_registry",
    "list_active_planners",
    "register_planners",
    "require_planners",
)

_KIND = "planner"

_MODULE_MAP = {
    "MSTRedundancyPlanner": "rbfenetmap.plugins.planners.mst_planner",
    "StarPlanner": "rbfenetmap.plugins.planners.simple_planners",
    "ExplicitPlanner": "rbfenetmap.plugins.planners.simple_planners",
    "CompletePlanner": "rbfenetmap.plugins.planners.simple_planners",
    "OptimalDesignPlanner": "rbfenetmap.plugins.planners.optimal_planner",
}

BUILTIN_PLANNERS: dict[str, PluginSpec] = {
    "mst": PluginSpec(
        name="mst",
        kind=_KIND,
        target="rbfenetmap.plugins.planners.mst_planner:MSTRedundancyPlanner",
        description="Minimum spanning tree plus degree and cycle-closure redundancy.",
        requires=("networkx",),
    ),
    "star": PluginSpec(
        name="star",
        kind=_KIND,
        target="rbfenetmap.plugins.planners.simple_planners:StarPlanner",
        description="Connect every ligand to one hub.",
        requires=(),
    ),
    "explicit": PluginSpec(
        name="explicit",
        kind=_KIND,
        target="rbfenetmap.plugins.planners.simple_planners:ExplicitPlanner",
        description="Select exactly the edges the user names.",
        requires=(),
    ),
    "complete": PluginSpec(
        name="complete",
        kind=_KIND,
        target="rbfenetmap.plugins.planners.simple_planners:CompletePlanner",
        description="Select every feasible candidate.",
        requires=(),
    ),
    "optimal": PluginSpec(
        name="optimal",
        kind=_KIND,
        target="rbfenetmap.plugins.planners.optimal_planner:OptimalDesignPlanner",
        description="A- or D-optimal statistical design over the Fisher information (graph Laplacian).",
        requires=("networkx", "numpy"),
    ),
}

PLANNER_PROFILES: dict[str, tuple[str, ...]] = {"all": tuple(BUILTIN_PLANNERS), "examples": ("mst", "star")}


def available_planners() -> dict[str, PluginSpec]:
    """Return the built-in planners whose requirements are importable."""
    return {name: spec for name, spec in BUILTIN_PLANNERS.items() if spec.available}


def register_planners(registry: PluginRegistry, names: tuple[str, ...] | None = None) -> PluginRegistry:
    """Register the named planners (default: all built-ins) into *registry*."""
    for name in names or tuple(BUILTIN_PLANNERS):
        try:
            registry.register(BUILTIN_PLANNERS[name])
        except KeyError:
            raise PluginError(f"Unknown built-in planner {name!r}. Known: {sorted(BUILTIN_PLANNERS)}.") from None
    return registry


def create_planner_registry(profile: str = "all") -> PluginRegistry:
    """Return a registry with the planners of *profile* registered and activated."""
    if profile not in PLANNER_PROFILES:
        raise PluginError(f"Unknown planner profile {profile!r}. Known: {sorted(PLANNER_PROFILES)}.")
    registry = register_planners(PluginRegistry(), PLANNER_PROFILES[profile])
    for name in PLANNER_PROFILES[profile]:
        registry.activate(name, _KIND)
    return registry


def create_planner(name: str, profile: str = "all", **kwargs: Any) -> Any:
    """Instantiate the planner *name*."""
    return create_planner_registry(profile).create(name, _KIND, **kwargs)


def list_active_planners(profile: str = "all") -> list[str]:
    """Return the names of the planners in *profile* that can be created."""
    registry = create_planner_registry(profile)
    return sorted(spec.name for spec in registry.list_plugins(_KIND, active_only=True) if spec.available)


def require_planners(names: tuple[str, ...], profile: str = "all") -> None:
    """Raise unless every planner in *names* is available."""
    available = available_planners()
    missing = {n: BUILTIN_PLANNERS[n].missing_requirements for n in names if n not in available}
    if missing:
        detail = "; ".join(f"{n} needs {list(mods)}" for n, mods in sorted(missing.items()))
        raise PluginError(f"Required planner(s) unavailable: {detail}.")


def __getattr__(name: str) -> Any:
    """Import planner classes lazily (PEP 562)."""
    if name not in _MODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_MODULE_MAP[name]), name)

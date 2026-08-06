"""Plugin registry semantics.

The property that matters most: registration imports nothing. It is what lets
``rbfenet plugins`` report on a backend that is not installed, and what keeps this whole
test module runnable in a bare environment.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rbfenetmap.core.exceptions import PluginError
from rbfenetmap.core.pluginregistry import PluginRegistry, PluginSpec

SPEC = PluginSpec(name="thing", kind="mapper", target="some.module:Thing", description="A thing.")


class TestPluginSpec:
    def test_availability_probes_without_importing(self):
        spec = PluginSpec(name="x", kind="mapper", target="m:X", requires=("json",))
        assert spec.available
        assert spec.missing_requirements == ()

    def test_missing_requirements_are_named(self):
        spec = PluginSpec(name="x", kind="mapper", target="m:X", requires=("definitely_not_a_real_module_xyz",))
        assert not spec.available
        assert spec.missing_requirements == ("definitely_not_a_real_module_xyz",)


class TestPluginRegistry:
    def test_register_and_retrieve(self):
        registry = PluginRegistry()
        registry.register(SPEC)
        assert registry.get_spec("thing", "mapper") is SPEC
        assert ("mapper", "thing") in registry
        assert len(registry) == 1

    def test_duplicate_registration_raises(self):
        # Overwriting silently would make behaviour depend on import order.
        registry = PluginRegistry()
        registry.register(SPEC)
        with pytest.raises(PluginError, match="already registered"):
            registry.register(SPEC)

    def test_same_name_different_kind_is_allowed(self):
        registry = PluginRegistry()
        registry.register(SPEC)
        registry.register(PluginSpec(name="thing", kind="scorer", target="other:Thing"))
        assert len(registry) == 2

    def test_unknown_plugin_lists_what_is_registered(self):
        registry = PluginRegistry()
        registry.register(SPEC)
        with pytest.raises(PluginError, match=r"Unknown mapper plugin 'nope'.*\['thing'\]"):
            registry.get_spec("nope", "mapper")

    def test_activation_is_separate_from_registration(self):
        registry = PluginRegistry()
        registry.register(SPEC)
        assert not registry.is_active("thing", "mapper")
        registry.activate("thing", "mapper")
        assert registry.is_active("thing", "mapper")
        assert registry.list_plugins("mapper", active_only=True) == (SPEC,)
        registry.deactivate("thing", "mapper")
        assert registry.list_plugins("mapper", active_only=True) == ()

    def test_create_imports_lazily(self, monkeypatch):
        registry = PluginRegistry()
        registry.register(SPEC)
        imported: list[str] = []

        def fake_import(name):
            """Record the import and hand back a stub module."""
            imported.append(name)
            return SimpleNamespace(Thing=lambda **kw: ("built", kw))

        monkeypatch.setattr("rbfenetmap.core.pluginregistry.import_module", fake_import)
        assert imported == [], "registration must not import anything"
        assert registry.create("thing", "mapper", flag=1) == ("built", {"flag": 1})
        assert imported == ["some.module"]

    def test_create_reports_missing_requirements_before_importing(self):
        registry = PluginRegistry()
        registry.register(PluginSpec(name="x", kind="mapper", target="m:X", requires=("definitely_not_real_xyz",)))
        with pytest.raises(PluginError, match=r"requires module\(s\) \['definitely_not_real_xyz'\]"):
            registry.create("x", "mapper")

    def test_malformed_target_is_reported(self):
        registry = PluginRegistry()
        registry.register(PluginSpec(name="x", kind="mapper", target="no_colon_here"))
        with pytest.raises(PluginError, match="Malformed plugin target"):
            registry.create("x", "mapper")


class TestBuiltinRegistries:
    @pytest.mark.parametrize(
        ("module", "kind"),
        [
            ("rbfenetmap.plugins.mappers", "mapper"),
            ("rbfenetmap.plugins.scorers", "scorer"),
            ("rbfenetmap.plugins.planners", "planner"),
            ("rbfenetmap.plugins.exporters", "exporter"),
        ],
    )
    def test_every_kind_follows_the_same_shape(self, module, kind):
        import importlib

        mod = importlib.import_module(module)
        plural = f"{kind}s"
        assert hasattr(mod, f"BUILTIN_{plural.upper()}")
        assert hasattr(mod, f"available_{plural}")
        assert hasattr(mod, f"create_{kind}")
        assert hasattr(mod, f"require_{plural}")
        for name, spec in getattr(mod, f"BUILTIN_{plural.upper()}").items():
            assert spec.kind == kind
            assert spec.name == name
            assert ":" in spec.target
            assert spec.description

    def test_kartograf_is_declared_but_optional(self):
        from rbfenetmap.plugins.mappers import BUILTIN_MAPPERS

        spec = BUILTIN_MAPPERS["kartograf"]
        assert spec.requires == ("kartograf", "gufe")

    def test_require_names_the_missing_modules(self):
        from rbfenetmap.plugins.mappers import available_mappers, require_mappers

        if "kartograf" in available_mappers():  # pragma: no cover - depends on the env
            pytest.skip("kartograf is installed")
        with pytest.raises(PluginError, match="needs"):
            require_mappers(("kartograf",))

    def test_lazy_attribute_access(self):
        from rbfenetmap.plugins import scorers

        assert scorers.LinearScorer.__name__ == "LinearScorer"
        with pytest.raises(AttributeError):
            _ = scorers.NoSuchScorer

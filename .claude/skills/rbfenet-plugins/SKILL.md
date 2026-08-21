---
name: rbfenet-plugins
description: Adding or modifying a mapper, scorer, planner, exporter, or intermediate generator in rbfenetmap — the PluginSpec registry, the five ABCs in core/meta, lazy imports, and optional-dependency handling. Use when extending the pipeline or when a plugin is unavailable or fails to register.
---

# Adding a plugin

Five kinds — `mapper`, `scorer`, `planner`, `exporter`, `intermediate` — each an ABC in
`core/meta/` with implementations under `plugins/<kind>s/`. The registry
(`core/pluginregistry.py`) is adapted
from `pharmaforge.core.pluginregistry`; keep the two recognisably the same mechanism.

## The rule everything else follows from

**Registration is metadata; nothing is imported until `create()` is called.** `PluginSpec`
names an import target as a `"package.module:ClassName"` string and lists `requires` as
top-level module names, probed with `find_spec`. This is what lets `rbfenet plugins --all`
report on backends that are not installed, and lets the whole test suite run with no optional
dependency present.

So: never import a backend at module import time, and never import a plugin implementation
from `core/`.

## Checklist

In `plugins/<kind>s/__init__.py`:

1. Add the class to `_MODULE_MAP` (`"ClassName" -> "module.path"`), which feeds the PEP 562
   `__getattr__` so `from rbfenetmap.plugins.mappers import KartografMapper` works without
   importing kartograf for everyone else.
2. Add a `PluginSpec` to `BUILTIN_<KIND>S` with `name`, `kind=_KIND`, `target`,
   `description`, and `requires`. The description is user-facing in `rbfenet plugins`.
3. Add the name to the relevant `<KIND>_PROFILES` entries. `"all"` is derived from the table;
   curated profiles like `"examples"` are not.

Then the implementation module: subclass the ABC, set `name: ClassVar[str]` to the spec name,
import the backend at module top level (this module is only imported on demand).

If it needs a new third-party package, add an extra in `pyproject.toml` and fold it into
`all`. `test` deliberately installs only `amber` and `viz`, which is why CI runs
`-m "not optional_dep and not slow"`.

## Failure behaviour

- Duplicate `(kind, name)` raises `PluginError` rather than overwriting — silent overwrite
  would make behaviour depend on import order.
- Unknown name raises `PluginError` listing the registered names, since the cause is usually
  a typo.
- `require_*` names the missing modules per plugin, so the user learns what to install.

Keep that shape: an error message here is the whole diagnostic the user gets.

## Testing

`tests/test_plugins.py` covers registry mechanics. `tests/test_smoke.py` imports every module
and will catch an eager backend import. A test that needs an uninstalled backend gets
`@pytest.mark.optional_dep`; anything genuinely slow gets `@pytest.mark.slow`. Prefer the
`Dummy*` plugins in `conftest.py` for everything downstream of the plugin boundary.

## Intermediate generators specifically

The one kind whose output changes the *ligand set*, not the network over it. It proposes
molecules and nothing else: `ProposedMolecule.mol` carries **no conformer** (posing is
centralised in `core/posing.py`, and a `ProposedMolecule` strips any conformer it is
handed), `ProposedLink.hint` is advisory and must never reach an `EdgeScore.total`, and
`IntermediateProposal.rejection` is a plain `str` — `RejectionReason` is the vocabulary of
*edge* feasibility and is not reused here.

A generator that knows where its atoms belong says so with a complete `parent_atom_map`,
which spares the poser a `GetSubstructMatch` whose symmetry it would have to resolve by
guessing. Posing failures are data: `pose_intermediate` returns a `PoseResult` carrying a
`PoseRejection`, never an exception.

## Exporters specifically

An exporter adapts a network to a downstream consumer without that consumer's concerns
reaching back into the core — keep format-specific logic entirely inside the exporter.
`export()` returns the paths it wrote. The Amber exporter splits `rbfe/` and `cbfe/`
subdirectories because amberstudio's `BuildEdges` takes `alchemical_mode` per invocation
rather than per edge.

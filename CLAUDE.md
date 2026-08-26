# CLAUDE.md

`rbfe-network-map` plans Relative Binding Free Energy perturbation networks from RDKit
molecules. The four-stage pipeline (map → repair → score → plan), the CLI surface, and the
counterpoised-edge story are in [README.md](README.md); the algorithms are in the module
docstrings, which are current and worth reading. This file covers only what those don't say.

## Commands

```bash
pip install -e ".[all]"
ruff check src tests examples && ruff format --check src tests examples
pytest -ra -m "not optional_dep and not slow"      # exactly what CI gates on
pytest -ra                                          # everything; needs the optional extras
sphinx-build -W -b html docs/source docs/build/html
```

## Layout

| Path | |
|---|---|
| `core/` | The algorithms. Depends only on rdkit/networkx/numpy/scipy, and never on `plugins/`. |
| `core/meta/` | The five plugin ABCs, importable with no optional dependency present. |
| `plugins/{mappers,scorers,planners,exporters,intermediates}/` | Implementations plus their `PluginSpec` tables. |
| `io/`, `viz/`, `cli/` | Loaders and network JSON, depictions, argparse. |
| `gui/` | Optional local browser GUI. Stdlib only; derives its form from `cli/`. |
| `tests/conftest.py` | `Dummy*` plugins and the co-posed ligand factories — read it before writing a test. |

## Gotchas

**A rejected edge is not an error.** Infeasibility is a `RejectionReason` recorded on the
edge's `EdgeScore` and *kept* in the network. The exceptions in `core/exceptions.py` are for
impossible requests. Never turn one into the other: the retained rejections are what explain
a sparse or disconnected network to the user.

**Geometry-touching tests must co-pose their ligands.** Ligands are assumed supplied in a
shared binding-site frame; independently embedded conformers share none, so `core_rmsd`
correctly rejects every edge between them. Use `make_coposed` from `tests/conftest.py`.
`make_ligand` alone is only safe for tests that never reach geometry.

**Plugin backends must stay unimported at module import time.** Registration is metadata
probed with `find_spec`. That is what lets `rbfenet plugins --all` report on kartograf
without it installed, and lets the full suite run with no optional dependency present.

**ruff is configured non-default and pinned in CI**: `line-length = 120`,
`skip-magic-trailing-comma = true`, ruff `0.15.12`. A different local ruff will reformat
files CI is happy with.

**Docs build with `-W`.** A new public object with a malformed numpydoc docstring fails CI.

**A release tag must equal the `pyproject.toml` version** or the release workflow fails
before building — deliberately, since PyPI will not let a wrong version be replaced. Version
bumps land as their own commit.

**`--consistency graph` is a post-selection pass, not a feasibility knob.** It intersects
each ligand's core across its selected RBFE edges, re-repairs to a fixed point, and re-costs
the reduced edges (`core/consistency.py`). It never re-selects, and it *raises* when the
reduced core leaves a selected edge infeasible rather than reverting that edge.
**A synthetic ligand is an ordinary vertex.** An intermediate generator proposes molecules
and nothing else; the in-place `core_rmsd` gate on its `A~M` and `M~B` sub-edges is the
*only* thing that certifies its pose. Don't add a second check, and don't trust a
generator's own opinion of its output — a `ProposedLink.hint` is advisory and must never
reach an `EdgeScore.total`. A badly posed intermediate is meant to come back as an ordinary
`core_geometry_mismatch` and be dropped whole, molecules included.

**An invented ligand needs a structure on disk.** `edges.dat` names residues and
amberstudio needs a topology for each; the user has never seen the synthetic ones. The
Amber exporter writes `ligands/<name>.sdf` for every ligand plus an `intermediates.txt`
manifest. If you touch that exporter, keep the invariant — every name in `edges.dat` has a
file in `ligands/` — and the test that asserts it.

**The GUI has no option list of its own, and must not grow one.** `gui/schema.py`
derives every form field by walking the CLI's own argument groups, and the filled form is
serialized back to flags that `cli/_args.build_network_options` parses -- so a knob added to
the CLI appears in the GUI for free, and the GUI can always print the exact `rbfenet plan`
line that produced what it is showing. `tests/test_gui_schema.py` asserts set equality
between the parser's destinations and the schema's, so a new flag fails the suite until it
is classified as a knob, as output plumbing, or in `KNOB_EXCLUSIONS` *with a reason*.
Adding a hand-maintained field there would silently defeat all of it.

**`examples/data/benzamides.sdf` is gitignored**; regenerate it with
`python examples/data/make_conformers.py`.

**The Tyk2 variant matrix is a checked-in docs asset, not output.**
`docs/source/_static/tyk2-matrix/` holds 21 self-contained planning reports (67 MB) that
`html_static_path` publishes verbatim. `.gitignore` ignores `network.html` at any depth, so
that tree only stays in git because of an explicit negation — if you move or rename the
directory, move the negation with it or the reports silently vanish from the next commit.
Regenerate the whole tree, index included, with
`python examples/06_variant_matrix.py docs/source/_static/tyk2-matrix`; planning is
deterministic, so a regenerated report that differs is a real behaviour change, and the page
is the cheapest regression signal in the repo. `examples/data/tyk2/matrix-pre-repair.json` is a
frozen pre-#42 baseline that cannot be regenerated — don't "refresh" it.

## Working agreements

Non-trivial work goes issue → `feature/<topic>` or `fix/<topic>` branch → PR. Commit subjects
are plain imperative sentences ("Pair interchangeable hydrogens by geometry"), not
Conventional Commits. Modules use `from __future__ import annotations` and a tuple `__all__`;
public objects carry numpydoc docstrings.

## Deeper guides

Load the skill when a task lands in its area:

- `rbfenet-softcore` — the soft-core repair: invariants, closure rules, how to test one.
- `rbfenet-planning` — edge selection, connectivity and cycle budgets, CBFE eligibility.
- `rbfenet-plugins` — adding a mapper, scorer, planner, exporter, or intermediate generator.

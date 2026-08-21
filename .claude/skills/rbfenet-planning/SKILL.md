---
name: rbfenet-planning
description: Working on network selection — the mst planner, connectivity and cycle budgets, edges-per-ligand, CBFE bridge eligibility, adaptive pair evaluation, or a network whose shape is wrong. Use when the question is which edges get selected rather than whether an edge is feasible.
---

# Network selection

`plugins/planners/mst_planner.py`, `core/cbfe.py`, and `core/pipeline.py` carry the detail.
This is the model to hold while reading them.

## Feasibility is not selection

Two separate stages, and conflating them is the most common mistake in this repo.

- **Feasibility** decides which candidates exist: mapping, soft-core repair,
  `max_softcore_atoms`, charge policy, ring policy. Output is a candidate pool with
  rejections retained.
- **Selection** picks edges from that pool: MST, then redundancy passes for degree and
  cycles.

So a knob like `--max-softcore-atoms` changes the *pool*, not the *shape*. Loosening it
usually produces an identical network, because the planner's Kruskal pass compares edges
individually and will still prefer two cheap hops to one dearer direct edge. **If someone
asks for a differently shaped network — clusters instead of chains, say — the answer is a
selection-level objective, not the soft-core budget and not scorer weights.**

## Guarantees to preserve

- Spanning **iff** the feasible candidate graph is connected. The MST is built first; the
  redundancy pass only ever adds.
- Conflicting budgets are a hard error up front, never a silent trim. `n_edges` below
  `n_ligands - 1` with connectivity required raises `NetworkPlanError`.
- An `edges_per_ligand` shortfall is best-effort: warned and recorded, not forced.
- A forced edge bypasses scoring but not feasibility.
- `NetworkPlanError` carries the `rejected` candidates, because the *pattern* of rejections
  diagnoses a run better than any single one. `_describe_disconnection` turns that into the
  "candidates that would have bridged these components" message — keep it informative.

## CBFE eligibility is a gate, not a price

A counterpoised edge needs no mapping, cannot be infeasible, and exists between every pair.
That makes cost the wrong mechanism to control it, so `--cbfe {off,bridge,cycles,all}`
gates *eligibility first* and only then lets cost choose among what the mode allows.

Two consequences worth protecting with tests:

- Degree padding never spends a CBFE edge — a shortfall is reported, not quietly bought.
- Cycle closure prefers an RBFE edge even when it is dearer.

The default base cost (8.0) is the linear scorer's charge-change ceiling, so a CBFE edge
never wins on price — only on being available where nothing else is. If you change scorer
scales, re-check that relationship.

Bridges are ranked by similarity *and* how well connected each endpoint is inside its own
subnetwork (`component_centrality`, `bridge_rank_key`).

## Adaptive pair evaluation

`evaluate_pairs_adaptively` exists because a complete scored pair matrix costs O(n²)
mappings. It fingerprint-ranks pairs, starts from each ligand's nearest neighbours, then
expands in batches only while connectivity, degree, or cycle targets remain unmet. It
therefore does **not** produce a complete pair matrix — anything that needs one (a full cost
report, a `complete` plan) must use the default eager mode.

## Checking a change

Selection is where regressions hide, because the output is still a plausible network. Compare
edge sets, degree sequences, and diameter before and after on
`examples/data/benzamides.sdf` — not just "did it run". `tests/test_planners.py` and
`tests/test_cbfe.py` use `make_transformation` from `conftest.py`, which carries no chemistry
at all: the planner only reads endpoints, feasibility, cost, and kind.

## Selection is not the only thing that edits a network

Two post-selection passes run *after* the planner and must not be confused with it.

- `core/consistency.py` — `--consistency graph` intersects each ligand's core across its
  selected RBFE edges and re-repairs to a fixed point. It rewrites mappings and costs; it
  never changes which edges were chosen.
- `core/surgery.py` and `core/diagnostics.py` — appending, deleting, merging, and
  LMI-driven replanning. Replanning is expressed as bans plus forced edges and then runs
  *this* planner, deliberately: a second selection strategy alongside the real one would
  drift from it. If you are tempted to add edge-patching logic there, express it as a
  constraint and let the planner resolve it.

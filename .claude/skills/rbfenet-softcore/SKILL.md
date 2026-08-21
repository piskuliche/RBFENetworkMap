---
name: rbfenet-softcore
description: Working on the soft-core connectivity repair — core/softcore.py, core/molgraph.py, core/coreprune.py, the Steiner search, the closure rules, or any softcore_* rejection reason. Use when an edge is repaired, rejected, or fragmented unexpectedly.
---

# Soft-core repair

`core/softcore.py`'s module docstring states the invariant and the three closure rules in
full — read it first. This file is what the docstring doesn't cover.

## The invariant, restated as an acceptance test

At most one connected soft-core region per side, each attached to the common core through
**exactly one bond**. Both halves are enforced, and they fail differently:

| Symptom | Reason |
|---|---|
| Pieces never join | `SOFTCORE_FRAGMENTED` |
| Region joined but hangs off the core by ≥2 bonds | `SOFTCORE_MULTIPLE_ATTACHMENTS` |
| Joined and singly attached, but too big | `SOFTCORE_TOO_LARGE` / `SOFTCORE_FRACTION` |
| Loop exhausted its iteration cap | `REPAIR_DID_NOT_CONVERGE` |

The attachment half is the one people forget. Bridging two regions through the core, and any
ring path with two or more core attachment bonds, are rejected even though the result is a
single connected region.

## Where the pieces live

- `core/softcore.py` — `precheck_mapping`, `joint_closure`, `repair_softcore_connectivity`,
  and `RepairContext` (which carries both molecules' graphs, the mapping, and the policy).
- `core/molgraph.py` — the graph layer: `node_weighted_steiner`, `ring_systems`,
  `hydrogen_parents`, `acyclic_branches`. Pure networkx, no chemistry policy.
- `core/coreprune.py` — property-based core demotion that runs *before* repair, in the
  `mcss-e` / `mcss-e2` mappers.

## Things that will trip you

**Don't make hydrogen-follows-parent symmetric.** A soft-core hydrogen on a common-core
parent is a legitimate one-atom region — that is `R–H → R–CH₃`, the most common
transformation in the field. Making the rule two-way demotes the parent, then its ring, and
destroys the edge. The asymmetry is the design.

**The iteration cap defaults to `n_atoms_1 + n_atoms_2`** (`SoftcorePolicy.max_iterations`
overrides). That bound is not arbitrary — each iteration demotes at least one atom, so
hitting it means the loop is oscillating rather than merely working hard.

**The repair is joint, not per-side.** `mapped-partner` couples the molecules, so fixing
side 1 can fragment side 2. Anything that iterates one side to convergence and then the other
is wrong; run both to a fixpoint together.

**Atom cost is the closure it triggers, not a per-element weight.** That is deliberate — it
is what makes the solver route around rings with no hand-tuned chemistry table. Resist
adding element weights to fix a bad edge; the fix is nearly always in the closure rules or
the terminal set.

**Two fragments are solved exactly; three or more are not.** `node_weighted_steiner`
runs Dijkstra on a node-split digraph for two terminals, and `_greedy_merge_steiner`
(iterative cheapest merge) for three or more, which is NP-hard. It returns
`(nodes, approximate)`, and that flag reaches the repair trace — an approximate bridge is
valid but not guaranteed reproducible across networkx versions. A suboptimal bridge is not
a bug.

**Do not reintroduce exhaustive subset search here.** An earlier version enumerated
candidate subsets for "small" instances, bounded at 25 nodes; `C(25, 12)` is 5.2 million,
so real ligands (20–32 candidates, 3–4 fragments) hung instead of solving. The docstring
records this — exponential search is not viable even at molecular size.

## Testing a specific fragmentation

Asking a real mapper for a soft-core that falls into exactly three pieces means hunting for a
molecule pair that happens to produce one. Don't. `DummyMapper(contract=...)` in
`tests/conftest.py` takes a hand-authored `{"sc1", "sc2", "cc1", "cc2"}` dict and returns it
verbatim, which is the only practical way to drive the repair into a chosen state. Pair it
with `make_coposed` when the test also reaches geometry.

`rbfenet inspect --edge "<src>~<dst>" --show-repair-trace --show-masks` renders the
per-iteration trace for a planned network — the fastest way to see what the repair actually
did before changing anything.

## Amber contract

`AtomMapping.to_contract()` must keep reproducing `amberstudio` `BuildEdges`' exact
`{"sc1", "sc2", "cc1", "cc2"}` dictionary; a shim depends on it. `io/amber_masks.py` turns
that into `scmask1`/`scmask2`. Changing the contract shape is a cross-repo break.

# RBFENetworkMap

Plan Relative Binding Free Energy perturbation networks from RDKit molecules.

Documentation: <https://piskuliche.github.io/RBFENetworkMap/>

Give it a series of ligands; it returns a scored, tunable network of alchemical
transformations, each carrying a common-core / soft-core partition that satisfies the
constraints the package is built around: **a transformation has at most one connected
soft-core region per side, attached to the common core through exactly one bond.**

```bash
pip install -e ".[all]"

python examples/data/make_conformers.py            # regenerate the example ligands

rbfenet plan --ligands examples/data/benzamides.sdf \
             --edges-per-ligand 2 --min-cycle-coverage 1.0 \
             --max-softcore-atoms 12 --show-rejected \
             --out network.json --export amber html --export-dir ./out
```

```
Planned 11 edge(s) over 9 ligand(s) -> network.json
edge            kind  cost   soft-core  core  repaired
--------------  ----  -----  ---------  ----  --------
bza_H~bza_F     rbfe  0.240  1/1        15
bza_H~bza_Me    rbfe  0.305  1/4        15
...
bza_CF3~bza_Et  rbfe  1.377  4/7        15    yes
```

## What it does

Four stages, each pluggable:

1. **Map** — propose an atom correspondence (`mcss`, `mcss-e`, `mcss-e2`, `cartograph`,
   `kartograf`, `identity`).
2. **Repair** — the interesting part. A mapper may leave the soft-core in several
   disconnected pieces, which no single alchemical transformation can run. The repair
   demotes common-core atoms until the pieces join up, or rejects the edge.
3. **Score** — reduce descriptors to a cost (`linear`, `lomaplike`, `softcore-size`).
4. **Plan** — select the final edge set (`mst`, `star`, `explicit`, `complete`).

## The soft-core repair

Choosing which atoms to demote is a **node-weighted Steiner tree** problem: the soft-core
fragments are the terminals, the bond graph is the network, and the cost of recruiting an
atom is how much soft-core that recruitment ultimately drags in. Three closure rules run
to a fixpoint after each recruitment:

- **whole-ring** — a ring is never left half soft-core; fused systems cascade on their own.
- **hydrogen-follows-parent** — one-way, deliberately. A soft-core hydrogen on a
  common-core parent stays put, because that is exactly `R–H → R–CH₃`, the most common
  transformation in the field.
- **mapped-partner** — demoting an atom demotes what it maps to. This couples the two
  molecules, so fixing side 1 can fragment side 2, which the loop then fixes in turn.

Measuring the cost of an atom as the *closure it triggers* is what makes the search behave
chemically with no hand-tuned per-element weights: a hydrogen costs about 2, an aromatic
carbon costs its whole fused system plus every attached hydrogen plus all their partners,
so the solver routes around rings unless there is no alternative.

Every repair is auditable:

```bash
rbfenet inspect --network network.json --edge "bza_CF3~bza_Et" --show-repair-trace --show-masks
```
```
regions       (3, 3) -> (1, 1)
repair trace
  initial: 3 soft-core region(s) on side 1, 3 on side 2
  iter 1 side 1: bridged 3 regions by demoting 1 atom(s) [1]
  iter 1 side 2: bridged 3 regions by demoting 1 atom(s) [1]
  final: 1/1 region(s), soft-core 7/4 atom(s)
amber masks
  scmask1    :SRC&@C2,F1,F3,F4
  scmask2    :DST&@C1,C2,H12,H13,H14,H15,H16
```

Every final soft-core region must also attach to the common core through exactly one
bond. Bridging regions and ring paths with two or more common-core attachment bonds are
rejected as ``softcore_multiple_attachments``.

An edge that cannot be repaired within budget is **rejected, not mutated** — and the
rejection is kept, because it is what explains a sparse or disconnected network:

```
The feasible candidate graph is disconnected: 2 components.
  component 1 (7): ['bza_CF3', 'bza_Cl', 'bza_Et', ...]
  component 2 (2): ['bza_Ph', 'bza_cPr']
  Rejected candidates that would have bridged these components:
    bza_H~bza_cPr (components 1/2): core_geometry_mismatch
```

## Tuning the network

| knob | effect |
|---|---|
| `--compat v0.4` | Pin every algorithmic knob to what a released version used, so a network stays reproducible after a default moves. Ligand intent, `--align`, and `--jobs`/`--progress` are not pinned and may be combined with it; naming a pinned knob is refused as a contradiction. |
| `--n-edges` | Cap on total edges. Below `n_ligands - 1` with connectivity required is a **hard error**, never a silent trim. |
| `--edges-per-ligand` | Target minimum degree. Best-effort; shortfalls are warned and recorded. |
| `--min-cycle-coverage` | Fraction of ligands on a cycle. Cycles make free energies checkable against themselves. |
| `--selection-objective connectivity_then_cycles` | After the spanning network is built, prioritize putting as many ligands as possible onto at least one cycle before chasing uniform extra degree. |
| `--max-cycle-size` | During cycle coverage, ignore candidate additions that would only make larger cycles than this. |
| `--pair-evaluation adaptive` | Fingerprint-rank all pairs and run expensive mappings in batches until connectivity and redundancy targets are met. |
| `--cbfe {off,bridge,cycles,all}` | Use counterpoised edges, which need no atom mapping and so are available between *any* two ligands. See below. |
| `--intermediates {off,bridge,gaps}` | Invent a bridging ligand for pairs no mapping can relate, turning one impossible edge into two possible ones. The invented molecule is posed against its parents and its sub-edges face the same feasibility checks as any other edge; a proposal whose sub-edges do not survive is dropped whole, molecules included. Budgeted out of `--n-edges`, never on top of it. See below. |
| `--progress` / `--no-progress` | Show or suppress pair-mapping progress. Interactive CLI runs show it automatically. |
| `--forced-edge` / `--banned-edge` | Absolute. A forced edge bypasses scoring but not feasibility. |
| `--max-softcore-atoms` | A *feasibility* knob: it changes the candidate pool, not the selection. |
| `--charge-change-policy` | `allow` / `penalize` / `reject`. |
| `--ring-policy none` | Permit half-broken rings, for deliberate ring-opening work. |

Selection guarantees a spanning network **iff** the feasible candidate graph is
connected: the MST is built first, the redundancy pass only ever adds, and conflicting
budgets are rejected up front rather than by trimming the tree.

## Counterpoised (CBFE) edges

A CBFE edge is two absolute calculations run simultaneously in opposite directions — one
ligand decoupling as the other couples. It gives the same relative quantity an RBFE edge
does, but neither molecule is morphed into the other, so there is **no common core and no
atom mapping**. It cannot be infeasible, and it exists between every pair of ligands —
including the ones an MCS search cannot relate at all.

That makes it the fix for the usual failure on a real series: a candidate pool that comes
back in several disconnected pieces with no relative edge able to cross between them.

```bash
rbfenet plan --ligands ligands.sdf --cbfe bridge --out network.json
```

| mode | effect |
|---|---|
| `off` (default) | Never. Every edge is RBFE. |
| `bridge` | Only to join subnetworks the feasible RBFE pool leaves disconnected — turning a hard connectivity failure into a planned network. Bridges are picked to maximize similarity *and* how well connected each endpoint is inside its own subnetwork. |
| `cycles` | Everything `bridge` does, plus putting a ligand on a cycle when no RBFE candidate can. |
| `all` | The whole network is counterpoised. Mapping is skipped entirely, so it returns in milliseconds where mapping takes minutes. |

Cost is `--cbfe-base-cost + --cbfe-atom-weight * (n_heavy_1 + n_heavy_2)`, on the scorer's
own scale. The default base of 8.0 is the linear scorer's charge-change ceiling — about
the most expensive thing that can happen to a still-feasible RBFE edge — so CBFE never
wins on price, only on being available where nothing else is.

**Eligibility is a gate applied before cost competition.** Cost picks among the edges a
mode makes eligible; it never lets a CBFE edge displace a feasible RBFE edge inside an
already connected component. Two consequences: degree padding never spends a CBFE edge
(an `edges_per_ligand` shortfall is still reported, not quietly bought), and cycle closure
prefers an RBFE edge even when it is dearer.

Every edge is marked in the JSON (`"kind"`), the GraphML and edge-list exports, and the
HTML report, where counterpoised edges are drawn violet and their cards report both ligands
as fully decoupled. The Amber export writes `rbfe/` and `cbfe/` subdirectories, because
amberstudio's `BuildEdges` takes `alchemical_mode` per invocation rather than per edge.

For a "connect everyone once, then put as many ligands as possible on at least one short
cycle" workflow:

```bash
rbfenet plan --ligands ligands.sdf \
             --edges-per-ligand 1 --min-cycle-coverage 1.0 \
             --selection-objective connectivity_then_cycles \
             --pair-evaluation adaptive \
             --max-cycle-size 4 \
             --out network.json
```

Adaptive evaluation starts from each ligand's three nearest fingerprint neighbours,
maps additional component-bridging pairs until every ligand is connected if possible,
then expands only while degree or cycle targets remain unmet. Tune its granularity with
`--adaptive-initial-neighbors` and `--adaptive-batch-size`; keep the default eager mode
when a complete scored pair matrix is required. Interactive runs show completed mappings,
elapsed time, throughput, and estimated remaining time; pass `--progress` to retain this
display when stderr is redirected to a log.

## Intermediate ligands

Some pairs cannot be related by any mapping. `--intermediates` lets the pipeline *invent* a
molecule that sits between them, turning one impossible edge into two possible ones.

```bash
rbfenet plan --ligands ligands.sdf --intermediates bridge --export amber --out network.json
```

**Be clear about what this buys.** IMERGE measured paths through an intermediate converging
roughly **20% more slowly** than the direct path: two calculations replace one, and neither
is free. The win is not speed and it is not accuracy on a pair that already works. It is
the pairs where **the direct path does not converge at all**, where the choice is not "one
calculation or two" but "two calculations or no comparison". Hence `off` by default, and
`bridge` — only gaps that actually disconnect the network — before `gaps`.

The default generator is `pairmap`, after
[Furui *et al.*, *JCIM* 2025, 65, 705–721](https://doi.org/10.1021/acs.jcim.4c01634)
(reference implementation [ohuelab/PairMap](https://github.com/ohuelab/PairMap), CC-BY;
this implementation is re-derived from the paper, not adapted from that code). It emits a
small *subnetwork* rather than a chain — the cheapest path from one parent to the other,
plus what it takes to put each of its links in a short cycle. The smallest shape is
`A–M1–B–M2–A`: two independent routes across a pair that had no direct edge at all, whose
closure error is a real consistency check. `fragment-swap` is the simple alternative: one
hybrid per differing position, no search.

Everything a generator proposes is judged by the machinery that judges every other edge.
The molecule is posed against its parents, and its `A~M` and `M~B` sub-edges go through the
same mapper, scorer, and soft-core policy — a badly posed intermediate comes back as an
ordinary `core_geometry_mismatch` and the whole proposal is dropped, molecules included, so
there are never orphan synthetic vertices.

An invented ligand is a molecule nobody has seen and nobody has parameterised, and the
outputs say so. The Amber export writes `ligands/<name>.sdf` for every ligand plus an
`intermediates.txt` manifest of `<name> <parent> <parent> <generator>`, so every name in
`edges.dat` has a structure beside it; `--validate-exporter amber` warns about the count
before the mapping run. The HTML report draws them with a dashed outline and a `SYN` badge,
counts them separately from the real ligands, and lists each one's parents, generator and
pose RMSD. `network.intermediates` records every gap attempted, bridged or not.

## Python API

```python
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.core.options import NetworkOptions, SoftcorePolicy
from rbfenetmap.io.loaders import load_ligands

ligands = load_ligands(["examples/data/benzamides.sdf"])
network = build_network(
    ligands,
    mapper="mcss-e2",
    network_options=NetworkOptions(
        edges_per_ligand=2, softcore=SoftcorePolicy(max_softcore_atoms=12)
    ),
)
network.validate()
```

## Hooks into other programs

Exporters adapt a planned network to a downstream consumer without that consumer's
concerns reaching back into the core:

- `amber` — `edges.dat` plus one `atommap_<src>~<dst>.runconfig` per edge, in the layout
  `amberstudio`'s `BuildEdges` produces and `guimapper` edits, plus `ligands/<name>.sdf` for
  every ligand so no name in `edges.dat` lacks a structure, and `intermediates.txt` when any
  were invented.
- `json` — the round-trippable native format, including rejected candidates.
- `edgelist`, `graphml`, `html` (a self-contained report with depictions).

Adding your own is a `PluginSpec` plus a class implementing one of the five ABCs in
`rbfenetmap.core.meta`. Registration is metadata only — nothing is imported until the
plugin is actually used, which is why `rbfenet plugins --all` can report on backends that
are not installed.

## Relationship to amberstudio

The algorithms are ported and generalised from `amberstudio.worknodes` rather than
imported, so the core depends only on rdkit, networkx, numpy, and scipy — no ParmEd, no
worknode framework. `AtomMapping.to_contract()` reproduces `BuildEdges`' `{"sc1", "sc2",
"cc1", "cc2"}` dictionary exactly, so `BuildEdges(mapping_method=...)` can call this
package through a small shim that absorbs its two unused `parmed.Structure` positionals.

Two defects in the original are fixed in the port, and both affect `amberstudio` today:

- `_partition_across_atom` blocks only the central atom, so for a **ring** atom every
  "branch" wraps around the ring and they overlap almost entirely; the downstream
  heuristic then demotes nearly the whole molecule. The port partitions across acyclic
  bonds only. Affects `mcss-e` and `mcss-e2`.
- `_find_mcs` calls the singular `GetSubstructMatch` on each molecule independently and
  zips the results, giving an **arbitrary correspondence for any symmetric substructure**.
  The port enumerates embeddings and selects by fewest soft-core fragments, then RMSD.
  On the example series this alone moved 20/36 feasible candidates to 34/36.

## Notes

Ligands must be **co-posed** — supplied in a common binding-site frame. The `core_rmsd`
descriptor measures in-place deviation without superposition, precisely so it detects
mappings that pair atoms occupying different parts of the pocket. Independently embedded
conformers share no frame and every edge between them is correctly rejected for geometry.
`examples/data/make_conformers.py` shows the constrained-embedding pattern.

Structures prepared *separately* — set up individually for ABFE runs, say, and written to
mol2 from their own Amber topologies — are a different case: their conformers are real bound
poses and only the frames disagree. `rbfenet plan --align` rigidly superposes them into a
common frame first, reports the per-ligand fit, and writes the moved structures out with
`--write-aligned` so you can check them. It recovers a common frame, not a common
conformation, so expect some residual `core_rmsd` to survive. See
[Aligning ligands](https://piskuliche.github.io/RBFENetworkMap/quickstart.html#aligning-ligands).

## Development

```bash
ruff check src tests && ruff format --check src tests
pytest -ra
pytest -ra -m "not optional_dep and not slow"     # what CI runs
```

## License

MIT. See [LICENSE](LICENSE).

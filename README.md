# RBFENetworkMap

Plan Relative Binding Free Energy perturbation networks from RDKit molecules.

Give it a series of ligands; it returns a scored, tunable network of alchemical
transformations, each carrying a common-core / soft-core partition that satisfies the
constraint the package is built around: **a transformation has at most one connected
soft-core region per side.**

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
edge            cost   soft-core  core  repaired
--------------  -----  ---------  ----  --------
bza_H~bza_F     0.240  1/1        15
bza_H~bza_Me    0.305  1/4        15
...
bza_CF3~bza_Et  1.377  4/7        15    yes
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
| `--n-edges` | Cap on total edges. Below `n_ligands - 1` with connectivity required is a **hard error**, never a silent trim. |
| `--edges-per-ligand` | Target minimum degree. Best-effort; shortfalls are warned and recorded. |
| `--min-cycle-coverage` | Fraction of ligands on a cycle. Cycles make free energies checkable against themselves. |
| `--selection-objective connectivity_then_cycles` | After the spanning network is built, prioritize putting as many ligands as possible onto at least one cycle before chasing uniform extra degree. |
| `--max-cycle-size` | During cycle coverage, ignore candidate additions that would only make larger cycles than this. |
| `--forced-edge` / `--banned-edge` | Absolute. A forced edge bypasses scoring but not feasibility. |
| `--max-softcore-atoms` | A *feasibility* knob: it changes the candidate pool, not the selection. |
| `--charge-change-policy` | `allow` / `penalize` / `reject`. |
| `--ring-policy none` | Permit half-broken rings, for deliberate ring-opening work. |

Selection guarantees a spanning network **iff** the feasible candidate graph is
connected: the MST is built first, the redundancy pass only ever adds, and conflicting
budgets are rejected up front rather than by trimming the tree.

For a "connect everyone once, then put as many ligands as possible on at least one short
cycle" workflow:

```bash
rbfenet plan --ligands ligands.sdf \
             --edges-per-ligand 1 --min-cycle-coverage 1.0 \
             --selection-objective connectivity_then_cycles \
             --max-cycle-size 4 \
             --out network.json
```

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
  `amberstudio`'s `BuildEdges` produces and `guimapper` edits.
- `json` — the round-trippable native format, including rejected candidates.
- `edgelist`, `graphml`, `html` (a self-contained report with depictions).

Adding your own is a `PluginSpec` plus a class implementing one of the four ABCs in
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

## Development

```bash
ruff check src tests && ruff format --check src tests
pytest -ra
pytest -ra -m "not optional_dep and not slow"     # what CI runs
```

Network selection
=================

The default planner works in two stages, and the order is what makes the connectivity
guarantee hold.

1. **Minimum spanning tree**, seeded so that forced edges are already in it. This spans
   every ligand whenever the feasible candidate graph is connected.
2. **A purely additive redundancy pass** that raises degrees and closes cycles without
   ever removing a tree edge.

Because the second stage only adds, connectivity established in the first cannot be lost.

Adaptive pair evaluation
------------------------

Mapping dominates runtime for an all-pairs pool. With ``pair_evaluation="adaptive"``,
the pipeline ranks pairs using inexpensive Morgan fingerprint similarity and evaluates
only an initial nearest-neighbour graph. Failed mappings split the feasible graph into
components; the next batch is drawn from pairs crossing those components, so work is
directed toward the primary connectivity objective. After connectivity, expansion
favours deficient ligands and short cycle closures and stops when the requested targets
are met. If no connected network exists, every possible component bridge is evaluated
before failure is reported.

Two redundancy objectives are available:

``uniform_redundancy``
   The historical behaviour. Raise degree targets first, then close cycles.

``connectivity_then_cycles``
   After the spanning network is built, first add edges that place as many ligands as
   possible onto at least one cycle. Only after that does the planner spend any remaining
   budget raising degrees above one.

The guarantee, stated exactly
-----------------------------

   The planner returns a network spanning every ligand **if and only if** the feasible
   candidate graph is connected.

If the pool is disconnected and ``require_connected`` is set, the planner raises with an
actionable message: the components, *and* the best rejected candidate that would have
bridged each gap along with its reason.

.. code-block:: text

   The feasible candidate graph is disconnected: 2 components.
     component 1 (7): ['bza_CF3', 'bza_Cl', 'bza_Et', ...]
     component 2 (2): ['bza_Ph', 'bza_cPr']
     Rejected candidates that would have bridged these components:
       bza_H~bza_cPr (components 1/2): core_geometry_mismatch

"Disconnected" on its own tells a user nothing they can act on. Which edge was rejected,
and why, tells them exactly which knob to loosen.

Cycles
------

``min_cycle_coverage`` is the knob that buys statistical confidence rather than raw
coverage: a cycle lets the network's free energies be checked against themselves.

A node lies on a cycle exactly when it belongs to a biconnected component of **two or more
edges**. Using :func:`networkx.biconnected_component_edges` matters here --
``biconnected_components`` alone gives the wrong answer, because a bridge is a biconnected
component of a single edge and its endpoints would be counted as covered when they are not.

When ``selection_objective="connectivity_then_cycles"``, candidate additions are ranked
by how many *new* ligands they place on a cycle, then by cycle length, then by cost.
``max_cycle_size`` can be used to ignore long loops and prefer triangles or 4-cycles.

Counterpoised (CBFE) edges
--------------------------

A counterpoised binding free energy runs two absolute calculations simultaneously in
opposite directions: one ligand decouples from the site as the other couples into it. It
yields the same relative quantity an RBFE edge does, but **neither molecule is morphed into
the other**, so there is no common core to find and no soft-core region to repair. A CBFE
edge therefore cannot be infeasible, and it exists between *every* pair of ligands --
including the pairs an MCS search cannot relate at all.

That is what makes it useful here. The guarantee above is conditional on the feasible pool
being connected, and on a real ligand series it often is not. With ``cbfe_mode`` set, the
condition is discharged: the pool can no longer be too sparse to span.

``off`` (default)
   Never. Every edge is RBFE, and the behaviour is exactly as described above.

``bridge``
   Only to join subnetworks the feasible RBFE pool leaves disconnected. This is the mode
   that turns the hard connectivity failure into a planned network.

``cycles``
   Everything ``bridge`` does, and additionally to put a ligand on a cycle when no RBFE
   candidate can.

``all``
   The whole network is counterpoised. Mapping is skipped entirely -- no mapper is even
   resolved -- which on a large series is the difference between minutes and milliseconds.

The modes form a strict ladder, so raising the setting only ever adds possibilities.

Eligibility is a gate, not a price
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the part most easily misread. ``cbfe_base_cost`` and ``cbfe_atom_weight`` put a
CBFE edge on the same scale as the scorer's totals::

   cost = cbfe_base_cost + cbfe_atom_weight * (n_heavy_1 + n_heavy_2)

The default base, ``8.0``, is the linear scorer's charge-change ceiling: a CBFE edge is
priced at roughly the most expensive thing that can happen to a still-feasible RBFE edge.
Realistic totals land in ~[9, 13] against ~0.3 for a good relative edge and ~5-6 for a bad
one.

But cost only decides *which* CBFE edge is chosen among the ones the mode makes eligible,
and orders RBFE against CBFE inside cycle closure. It never lets a CBFE edge outbid a
feasible RBFE edge inside an already connected component: under ``bridge``, a CBFE edge
that does not join two components is not in the pool at any price.

Two consequences worth stating:

- **Degree raising never spends a CBFE edge**, in any mode below ``all``. An extra edge on
  an already connected, already cycled ligand is a refinement, and two absolute
  calculations is not a trade anyone would make for one. A shortfall in
  ``edges_per_ligand`` is still reported rather than quietly bought.
- **Cycle closure prefers RBFE even when it is dearer.** Candidates are ranked by new
  coverage, then by kind, then by cycle length and cost -- so an RBFE five-cycle beats a
  counterpoised four-cycle.

How bridges are chosen
~~~~~~~~~~~~~~~~~~~~~~

With *c* components, exactly *c - 1* bridges are needed, and choosing which is a maximum
merit spanning selection over the component quotient graph::

   merit = tanimoto + 0.5 * mean(centrality of the two endpoints)

Similarity because a bridge between chemically close ligands is the one most likely to give
a trustworthy number even run as two absolute calculations. Centrality -- degree within a
ligand's *own* subnetwork, normalised to its most-connected member -- because a bridge
landing on a hub propagates through the subnetwork and participates in cycles, whereas one
landing on a leaf leaves a dangling path nothing checks. A singleton component scores 1.0
rather than 0.0: it has exactly one way into the network, and ranking its only option last
would be backwards.

Interaction with adaptive evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adaptive evaluation does **not** stop earlier when CBFE is enabled, and that is deliberate.
Its connectivity phase keys on the feasible *RBFE* graph, so it still exhausts every
cross-component pair before giving up -- the only way to know RBFE cannot reach across a
gap is to try, and ``bridge`` means "only where RBFE cannot reach".

Conversely, the intermediate plans it uses to decide whether to keep expanding are probed
with CBFE switched off. Left on, a ``cycles``-mode probe would satisfy
``min_cycle_coverage`` with counterpoised edges, report no unmet constraints, and halt the
search -- masking the shortfall and short-circuiting the very RBFE expansion the loop
exists to drive.

Downstream
~~~~~~~~~~

Every edge carries its kind through the JSON (``"kind": "rbfe" | "cbfe"``), the GraphML and
edge-list exports, and the HTML report, where counterpoised edges are drawn violet and their
cards report both ligands as fully decoupled rather than showing a common core of zero.
The kind is never conveyed by colour alone: every counterpoised edge also carries ``CBFE``
in its tooltip and a badge on its card.

The Amber export writes ``rbfe/`` and ``cbfe/`` subdirectories, because amberstudio's
``BuildEdges`` takes ``alchemical_mode`` per *invocation* rather than per edge: a mixed
network is two ``BuildEdges`` runs, and the layout mirrors that. Each carries an
``edges.txt`` in amberstudio's ``<src>~<dst>`` form. A CBFE edge needs nothing else --
amberstudio builds its masks from the residue roles, since there is no mapping to convey --
so ``cbfe/`` holds only the edge list. An all-RBFE network keeps the historical flat layout.

Core-based clustering
---------------------

A congeneric series planned by minimum spanning tree comes back as a **chain**. Kruskal
compares edges one at a time, and two cheap hops always beat one dearer direct edge: on the
benzamides, ``bza_H~bza_Et`` costs 0.626 and is perfectly feasible, but ``H-Me`` (0.305)
plus ``Me-Et`` (0.470) wins, and the result is a diameter-4 path.

Loosening ``max_softcore_atoms`` does not help, and it is worth being explicit about why,
because it is the first thing everyone reaches for. That knob decides which edges are
*feasible*; it never decides which are *chosen*. Between 6 and 20 the benzamide network is
byte-identical -- same eight edges, same degree sequence, same diameter -- while the
feasible pool sits unchanged at 33 of 36 pairs. The direct edges are there the whole time.
Selection simply never wants them.

What does change the shape is grouping ligands by the core they actually share, cycling
each group internally, and joining the groups with single bridging edges.

``core_clusters`` is a ladder, the same shape as ``cbfe_mode``:

``off`` (default)
   No partition is computed and selection is exactly what it was before the option
   existed. Costs nothing -- the clustering code is never entered.

``report``
   The partition is computed and recorded on :attr:`~rbfenetmap.core.models.Network.clusters`,
   and **selection is unchanged**. The network is identical to ``off``. This is how to see
   how a series clusters without changing what you get, and the count of edges crossing a
   cluster boundary is the diagnostic: on the benzamides it is 7 of 12.

``plan``
   The partition drives selection.

Because ``report`` attaches the partition *after* planning rather than threading it through
the planner, "report changes nothing" holds by construction rather than by review. It is
also pinned by test, across both example sets and a spread of soft-core budgets.

How the partition is found
~~~~~~~~~~~~~~~~~~~~~~~~~~

Clusters merge agglomeratively, seeded by the pairwise-feasible candidate graph, and gated
on the **N-way MCS** of the merged set -- two clusters join only if *everything* in the
union still shares enough core. :func:`~rbfenetmap.core.mcs.mcs_query_many` computes that
directly; intersecting pairwise MCS results would not merely be slower but meaningless,
since each pairwise substructure is its own query molecule with its own atom indexing.

Seeding on the feasible graph bounds the work to cluster pairs that are adjacent in it, and
guarantees a cluster is internally mappable -- a cluster whose members cannot be mapped to
one another would have no sub-network to build. Merging the *best available* pair each
round, rather than the first admissible one, makes the partition independent of iteration
order.

Sizing that shared core has one trap worth recording. Counting atoms of the MCS query
molecule is wrong: ``atomCompare=CompareAny``, which this package uses everywhere, emits
generic query atoms whose ``GetAtomicNum()`` is 0, and those read as heavy. It reported a
four-benzamide core of 10 heavy atoms when the smallest member has 9 atoms in total --
impossible, since a shared substructure has to embed in every member.
:func:`~rbfenetmap.core.clustering.cluster_core_size` counts the match in each *real*
molecule and takes the minimum across members; the minimum because a generic atom can match
a hydrogen in one molecule and a heavy atom in another, and under-counting makes the gate
decline to merge rather than merge on structure that is not there.

That invariant -- the core never exceeds the smallest member -- also explains the one
surprising failure mode: a ``min_core_atoms`` above a cluster's smallest ligand shatters it
into singletons. ``min_core_fraction`` is the scale-free companion for series of mixed size.

How clusters shape the network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By **subtraction**. The cross-cluster relative edges are removed from the candidate graph,
and every existing pass then runs unchanged on what is left: Kruskal cannot chain from one
scaffold into the next, cycle closure has only intra-cluster edges to reach for, and the
clusters are joined by exactly the bridges chosen at step (2) of the planner. Since those
bridges are a spanning forest, no cycle ever crosses a boundary.

One consequence is worth stating, because the opposite is the intuitive guess:
:func:`~rbfenetmap.core.cbfe.select_cbfe_bridges` is **not** given the partition. It uses
the graph's own connected components, as it always has. Removing the edges is what enforces
the partition, so what remains to be joined is precisely whatever that left disconnected.
Handing it the clusters instead would insist on ``c - 1`` bridges even where
``inter_cluster="prefer_rbfe"`` had already joined two of them, and would miss a cluster
that a banned edge had split in half.

``inter_cluster`` chooses what the bridges are made of. ``"cbfe"`` (default) always spends a
counterpoised edge -- the clusters are separate *because* the relative edge across the
boundary would be a bad one. ``"prefer_rbfe"`` restores a **spanning** set of feasible
cross-cluster relative edges, at most ``c - 1``; keeping the cheapest per cluster *pair*
would restore up to ``C(c, 2)`` of them, which on ten clusters is 45 edges and most of the
way back to the unclustered network.

Degradation
~~~~~~~~~~~

A homogeneous series yields one cluster, and ``plan`` then reduces to exactly the ordinary
network with no bridges -- trying it can never be worse than leaving it off. A cluster below
``min_cluster_size`` cannot hold a cycle; it still forms and is still bridged, and the
shortfall lands on ``unmet_constraints`` alongside ``edges_per_ligand`` and
``min_cycle_coverage``.

Two combinations are refused at construction rather than discovered mid-run: clustering with
``cbfe_mode="all"``, which skips mapping entirely and so has no cores to cluster on, and
``plan`` with ``inter_cluster="cbfe"`` under ``cbfe_mode="off"``, which would have nothing
to bridge with. Cluster-driven selection also forces eager pair evaluation, because adaptive
evaluation maps a deliberately partial pool and the clusters would form, split, and re-form
as batches arrived.

Knob precedence
---------------

.. list-table::
   :header-rows: 1
   :widths: 8 25 67

   * - #
     - Knob
     - Semantics
   * - 1
     - ``banned_edges``
     - Absolute. Overlap with ``forced_edges`` raises at construction.
   * - 2
     - ``forced_edges``
     - Absolute, subject to feasibility. Bypasses *scoring* but not feasibility; an
       infeasible forced edge raises rather than being silently dropped.
   * - 3
     - ``require_connected``
     - The spanning tree is never trimmed below spanning.
   * - 4
     - ``n_edges``
     - Caps the redundancy pass. ``n_edges < n_ligands - 1`` with connectivity required is
       a **hard error** -- see below.
   * - 5
     - ``edges_per_ligand``, ``min_cycle_coverage``
     - Best-effort. Shortfalls warn and land on ``unmet_constraints``; never raise.
   * - 6
     - ``hub``
     - Pre-seeds the hub's edges before the spanning tree.
   * - 7
     - ``max_softcore_atoms``
     - A **feasibility** knob applied during repair: it changes the candidate pool, not
       the selection. Tightening it can disconnect the pool, which then errors at (3).
   * - 8
     - ``cbfe_mode``
     - Widens the pool rather than steering selection. Applied *before* cost competition,
       so it can rescue (3) without ever displacing a feasible RBFE edge.
   * - 9
     - ``core_clusters``
     - ``"report"`` does not participate: it records the partition and leaves selection
       alone. ``"plan"`` *narrows* the pool, by removing the cross-cluster relative edges
       before any of the above runs. It is the only knob here that can make (3) harder to
       satisfy, which is why it requires something to bridge with.

Why the ``n_edges`` conflict is a hard error
--------------------------------------------

Asking for a connected 12-ligand network in 8 edges is impossible, and both silent
resolutions are wrong. Trimming the spanning tree would produce a disconnected network the
user explicitly forbade; quietly raising ``n_edges`` would ignore a budget the user
explicitly set. Only the user can say which they meant, so the request is refused with the
arithmetic spelled out:

.. code-block:: text

   n_edges=8 cannot connect 12 ligands; a spanning network needs at least 11 edges.
   Raise n_edges to >= 11, or pass require_connected=False.

Other planners
--------------

``star``
   Every ligand joined to one hub. The hub defaults to the most central compound -- the
   one with the most feasible partners, tie-broken on total cost.

``explicit``
   Exactly the edges named. Refuses to silently omit an infeasible one.

``complete``
   Every feasible candidate. Maximum redundancy at maximum cost; useful for small series
   and for benchmarking a sparser network against the full measurement.

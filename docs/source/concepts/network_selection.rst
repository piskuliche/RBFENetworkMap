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

Clustered planning
------------------

``cluster_by`` partitions the ligands and plans each cluster as its own subnetwork, joined
to the others by a few deliberately chosen edges. It is a knob on the default planner, not a
planner of its own, because a user who partitions their ligands still wants cycle coverage,
degree targets and CBFE bridging -- and a separate planner would have to reimplement all
three to offer any of them.

Why a partition is cheaper
~~~~~~~~~~~~~~~~~~~~~~~~~~

The precision floor of an RBFE network goes as ``k_min ~ n ln n`` (Pitman *et al.*, *JCIM*
2023, 63, 1776-1793); below it, precision degrades *worse* as the set grows. That floor is
superlinear, and superlinear costs are exactly the ones a partition beats::

   sum_i n_i ln n_i  <  n ln n

with equality only for a single cluster. One hundred ligands in five balanced clusters need
roughly 190 edges rather than 460, a 59% saving at maintained per-cluster precision. Even a
badly imbalanced split saves 30-50%, because the dominant term is the largest cluster and it
is still smaller than the whole set.

The clusterers
~~~~~~~~~~~~~~

``none`` (default)
   One cluster. Exactly the behaviour described above.

``charge``
   Net formal charge classes. The one clusterer with no threshold in it -- charge is a
   property of the molecule rather than of a similarity measure -- and it isolates the
   transformation the scorer already penalises hardest.

``scaffold``
   The Bemis-Murcko framework, which is close to what a medicinal chemist means by "series".
   Acyclic ligands share the empty scaffold rather than each becoming a singleton.

``fingerprint``
   Average-linkage hierarchical clustering on ``1 - Tanimoto``, cut at a distance of 0.6 --
   one minus the package's own ``prefilter_min_tanimoto``, so there is a single notion of
   "similar enough" to reason about. scipy's :func:`~scipy.cluster.hierarchy.linkage` and
   :func:`~scipy.cluster.hierarchy.fcluster` do the work; scipy is already a dependency and
   the density methods used elsewhere in the field (HDBSCAN, DBSCAN) would return a noise
   label this package has no use for, since every ligand must land in a cluster to be
   planned at all.

Clustering is a **selection-level objective, not a feasibility one.** Nothing here consults
the soft-core budget, a mapping, or a rejection. A cross-cluster edge is as feasible as it
ever was; the point is that buying many of them is a worse use of the budget than buying
edges inside a cluster, because the within-cluster edges are the ones a cycle can check.

Why two bridges
~~~~~~~~~~~~~~~

``cluster_bridges`` defaults to **2**, and the second one is the whole point. Two edges
between the same two clusters put the crossing itself on a cycle, since the paths inside
each cluster close the loop. Cross-cluster edges are the least similar and therefore the
least trustworthy edges in the network, so applying the every-edge-in-a-cycle invariant
precisely there buys more per edge than anywhere else. ``cluster_bridges=1`` gives the
minimal spanning join and leaves each crossing unchecked by anything.

The crossings are chosen by the same maximum-merit sweep the CBFE bridges use --
:func:`~rbfenetmap.core.cbfe.select_bridges`, with the clusterer's partition passed in
where the connected components would otherwise go. Connected components are only one
interesting partition of a ligand set, and joining the groups of any other is the same
problem with the same ranking.

Where it acts
~~~~~~~~~~~~~

At one point, before anything is selected: the cross-cluster edges are pruned from the
candidate graph down to the chosen crossings, and every stage downstream then runs unchanged
and simply cannot spend on a crossing. The kept crossings are added to the selection
outright rather than left to the spanning pass, because Kruskal would take one of the two
and discard the other as redundant -- and "redundant" is exactly what makes the second one
worth having.

Pruning is an optimisation, and an optimisation that changes the answer would be a bug. If
a cluster's members reach each other only *through* another cluster, removing the crossings
would disconnect a network that was connected, so the cheapest necessary crossings are
restored and the restoration is reported on ``unmet_constraints`` -- the user asked for a
partition and did not entirely get one, which is a fact about their ligand set worth seeing.

Clustering composes with CBFE rather than competing with it: inter-cluster edges are exactly
where a counterpoised edge belongs, and a cluster the RBFE pool never connected internally
is still bridged by ``cbfe_mode``.

Reproducing a released behaviour
--------------------------------

Every knob added after v0.4.0 defaults to what v0.4.0 did, so an existing command keeps
planning the network it always planned. That is a promise about *defaults*, and defaults
move: a later release may well decide that ``n_edges`` should follow ``n ln n``, or that
cycle coverage should be measured over edges rather than nodes. When that happens, every
network planned before it becomes irreproducible unless the old behaviour is still
reachable by name.

``--compat`` is that name::

   rbfenet plan --ligands ligands.sdf --compat v0.4 --out network.json

It pins every algorithmic knob -- mapper, scorer, soft-core policy, planner, and the whole
of selection -- to the values the named release used. Those values are written out
literally in :data:`rbfenetmap.cli._args.COMPAT_CLI_PINS` and in
:meth:`rbfenetmap.core.options.NetworkOptions.preset`, rather than being read back from the
current defaults. That is the entire mechanism: a table derived from the defaults would
move when they moved and would silently stop reproducing the version it names.

Versioned, not a boolean
~~~~~~~~~~~~~~~~~~~~~~~~

There is deliberately no ``--legacy``. A boolean stops meaning anything the moment there
are two past behaviours to choose between, and the flag has to keep being unambiguous
several releases from now.

What is pinned, and what is not
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pinned: the *algorithmic* surface -- the knobs whose meaning or default may change between
releases.

Not pinned, and usable alongside it:

- **ligand intent** -- ``--hub``, ``--forced-edge``, ``--banned-edge``, ``--explicit-edge``.
  Banning an edge is a statement about one ligand set, not about a version's behaviour.
- **input preparation** -- ``--ligands``, ``--align`` and friends. That is which molecules
  go in, not how they are planned.
- **operational flags** -- ``--jobs``, ``--progress``, ``--out``, ``--export``. None of
  them can change which network comes out.

This is what makes the flag practical rather than merely principled: pinning ligand intent
would leave ``--compat`` unable to plan a real series.

Naming a pinned knob is a contradiction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--compat v0.4 --edges-per-ligand 3`` asks for v0.4's behaviour and for something other
than v0.4's behaviour, so it is refused with the offending flag named. This follows the
same rule as the ``n_edges`` conflict below: both resolutions are defensible, so neither is
chosen on the user's behalf.

Naming the knob is the contradiction, not disagreeing with it. ``--compat v0.4
--edges-per-ligand 2`` is refused too, even though 2 is what v0.4 used. Accepting it
because it happens to match today would make the rule depend on the current default, so
the same command would start failing the day that default moved -- which is precisely the
surprise ``--compat`` exists to prevent.

The level is recorded in the network JSON as ``options.compat``, so a planned network
states which behaviour produced it. A network planned without the flag writes no such key
at all, leaving its output byte-for-byte what it was before the flag existed.

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
     - ``cluster_by``, ``cluster_bridges``
     - Narrows the pool, before everything above except (1) and (2): cross-cluster
       candidates are pruned to ``cluster_bridges`` per joined cluster pair. Never at the
       expense of (3) -- a crossing needed to span the ligands is restored and reported.

``--compat`` is not in this table. It is a *constructor*: it writes the values the rest of
the table then operates on, so it is applied before precedence rather than competing inside
it.

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

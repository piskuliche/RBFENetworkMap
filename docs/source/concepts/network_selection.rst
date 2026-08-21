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

Statistical optimal design
--------------------------

Everything above selects on **cost**: cheapest spanning tree, then cheap redundancy. That
answers "what is the cheapest network that connects everything and closes enough cycles?".
A different question is worth asking -- "which network, at this budget, gives the most
*precise* free energies?" -- and it has a classical answer.

The fact that makes it tractable: **the Fisher information matrix of a network of relative
measurements is the weighted graph Laplacian.**

.. math::

   F_{ii} = \sum_{k \ne i} \sigma_{ik}^{-2}, \qquad
   F_{ij} = -\sigma_{ij}^{-2}, \qquad
   C = F^{+}

DiffNet, HiMap, Yang's MLE and cinnabar's network analysis are the same object, so one
implementation (:mod:`rbfenetmap.core.design`) serves selection, sample allocation, and
analysis.

Two criteria, and when to use which
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--design a_optimal``
   Minimise :math:`\operatorname{tr} C`, the summed variance of the estimated free
   energies. Use this when each ligand's own number is what matters.

``--design d_optimal``
   Minimise :math:`\ln \det C`, the volume of their joint confidence ellipsoid.

The choice is not arbitrary. The pseudo-determinant of a Laplacian is :math:`n` times its
weighted spanning-tree count (Kirchhoff), so minimising :math:`\ln \det C` *maximises the
spanning-tree count* -- and a network with more spanning trees is a network with more
cycles. Pitman measures 40--80% more cycles at equal edge count, which is why the
recommendation is: **D-optimal when a cycle-closure correction will be applied downstream,
A-optimal otherwise.**

Both need a planner that optimises them::

   rbfenet plan --ligands ligands.sdf --scorer variance --planner optimal --design d_optimal

``--design`` under any other planner is **refused, not ignored**. A criterion is an
objective, and there is nothing a planner can do with one halfway; a flag that silently did
nothing is the ``--consistency graph`` failure mode this package already has one instance
of and does not want a second. For the same reason, ``--planner optimal`` without
``--design`` is refused rather than defaulting to a criterion: the two answer different
questions and neither is a safe guess.

The cost scale has to mean something
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`F` is built from :math:`1/\sigma^2`, so the scorer's totals have to *be* standard
deviations rather than a ranking. That is what the ``variance`` scorer supplies -- NetBFE's
equation 19, a predicted per-edge standard deviation in kcal/mol::

   s_ij = 1.0 + 1.0 * sqrt(max(h_ij, h_ji)) + 0.5 * sqrt(max(H_ij, H_ji))

with *h* the transforming (soft-core) heavy-atom count and *H* the total. The square roots
are the point: sampling error grows with the square root of the decoupled degrees of
freedom, so doubling the soft-core does not double the noise. The intercept is the
irreducible part, and it also keeps :math:`1/\sigma^2` finite.

The design planner runs under any scorer. Under any *other* scorer the numbers it minimises
are internally consistent but are not variances, and none of the published payoffs apply.

Why the singular matrix is not a problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`F` is always singular for an RBFE-only network -- no relative measurement pins the
absolute offset, so the all-ones vector is in the null space. Restraining the mean
NetBFE-style, :math:`F^*(\omega) = F + \omega m^{-2} \mathbb{1}\mathbb{1}^T`, and taking
:math:`\omega \to \infty` through the bordered system converges on the Moore-Penrose
pseudo-inverse. **The optimal design does not depend on** :math:`\omega`, which is what
makes the problem well posed: the regulariser fixes the unidentifiable offset and never
trades against the criterion.

How the edges are chosen
~~~~~~~~~~~~~~~~~~~~~~~~

Choosing the best *k*-subset of :math:`\binom{n}{2}` edges is combinatorial. The planner
ships Xu's Appendix-H heuristic:

1. the cheapest spanning tree, so connectivity is established before anything competes for
   the budget;
2. a candidate pool capped at ``design_candidate_factor * n`` edges (default 3n) -- the tree
   plus the cheapest remaining candidates;
3. greedy descent on the criterion within that pool, up to the edge budget.

Published at **1.10 ± 0.03x** of the true optimum. Measured here against exhaustive
enumeration on small complete graphs, the worst ratio over 40 randomised instances is 1.03x
(A-optimal) and 1.08x (D-optimal).

``--design-refine`` adds a **Fedorov exchange** pass: repeatedly swap the in-design edge
whose removal costs least for the candidate whose addition helps most, until no swap
improves the criterion. It brings the same worst case to 1.005x, at far more criterion
evaluations. Off by default. It is implemented in numpy rather than through HiMap's route,
which pins ``rpy2==3.4.5`` and ``scikit-learn==0.23.2`` and requires an R installation.

One deviation from Appendix H is worth knowing about. Its first stage is the cheapest
**2-edge-connected** spanning subgraph, not a spanning tree. Forcing that costs more than it
buys: a bridge cover chosen by *cost* spends budget the criterion would rather spend
elsewhere, and on the same instances it pushes the D-optimal result to 1.33x where letting
the criterion spend that budget itself stays at 1.08x. The criterion closes the bridges
worth closing on its own; any that survive are reported on ``unmet_constraints`` rather than
bought out.

The edge budget
~~~~~~~~~~~~~~~

With ``n_edges`` unset, the design planner uses Pitman's floor,
:math:`k_{\min} = \operatorname{round}(n \ln n)`, instead of "as many as redundancy
wants". Below that bound precision degrades **worse as n grows**, so a design planner that
ignored it would be optimising inside a budget already known to be too small. At *n* = 40
that is 148 edges, against the ~40 that ``edges_per_ligand=2`` buys.

This is a property of this planner, not a change to the package default. ``mst`` is
untouched and ``--compat v0.4`` is unaffected.

``edges_per_ligand`` and ``min_cycle_coverage`` are **not** enforced here. Spending budget
to hit a degree target would work directly against the objective the user asked for, so a
shortfall is recorded on ``unmet_constraints`` and left alone.

Per-edge sample allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~

The same matrix answers a second question: given a fixed simulation budget, how should it be
split across the selected edges? Model the variance as falling with :math:`1/t`, so an edge
given :math:`t_e` nanoseconds contributes weight :math:`t_e / \sigma_e^2`. Minimising
:math:`\operatorname{tr} C` subject to :math:`\sum t_e = T` is convex, and at the optimum

   every edge returns the same variance reduction per nanosecond.

Any other split has an edge worth moving time to. Set ``--design-total-ns`` and the Amber
exporter writes the result into each per-edge ``.runconfig``, alongside the atom map it
already carries:

.. code-block:: yaml

   sample_allocation:
     lambda_windows: 19
     simulation_ns: 12.4
     predicted_sigma_kcal: 2.31

The window count is the edge's share mapped onto ``--design-lambda-min`` /
``--design-lambda-max``. An edge allocated near-zero time is not a bug -- it is the
allocation reporting that an edge the planner selected turned out redundant given how the
rest was funded.

This is the **static** first pass. Published payoff is roughly a twofold variance reduction
at equal cost; the iterative refit that converges in five rounds needs measured variances
and therefore a round trip through the MD engine, which is out of scope here.

.. warning::

   **Optimal design buys precision. It does not promise accuracy.**

   Over five TYK2 iterations NetBFE's :math:`\operatorname{tr} C` fell monotonically from
   1.08 to 0.78 while the RMSE against experiment *rose* from 0.84 to 0.91. The criterion
   measures how reproducible the numbers are, not how right they are -- it knows nothing
   about the force field, the poses, or the protonation states. A network that halves its
   predicted variance can be no closer to experiment than the one before it, and this is
   the observed case, not a hypothetical one.

   Read a falling :math:`\operatorname{tr} C` as "the statistics are no longer the
   bottleneck", and take that as a cue to look at the model rather than as evidence the
   answers improved.

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
     - ``design``
     - Replaces the *objective* of (5) rather than competing with it, and only under the
       ``optimal`` planner. (1) to (4) still bind: banned edges stay out, forced edges stay
       in, the network still spans, and ``n_edges`` still caps. What changes is what fills
       the remaining budget -- the criterion instead of cost and degree targets -- so
       ``edges_per_ligand`` and ``min_cycle_coverage`` drop to reporting only.
   * - 10
     - ``design_total_ns``
     - Not a selection knob at all. Applied after planning, at export, over whatever edges
       were chosen; it changes how the network is run, never which network it is.

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

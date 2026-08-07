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

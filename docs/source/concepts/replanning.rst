Surgery and replanning
======================

Nobody plans once. A campaign gains compounds in batches, loses edges when a run will not
converge, and grows by joining a new series onto one already running. Re-planning from
scratch each time is the wrong answer to all three: it throws away mappings that were
already computed, and -- worse -- it reshuffles edges that are already set up, queued, or
finished, so the network handed back is not the network being run.

:mod:`rbfenetmap.core.surgery` edits instead. :class:`~rbfenetmap.core.models.Network` is
frozen, so every operation returns a new one and leaves its input alone. Edges that were
already there keep their identity, their mappings, and their costs; only the requested
change and its consequences are new.

.. code-block:: python

   from rbfenetmap.core.surgery import append_ligand, concatenate_networks, delete_edge

   bigger = append_ligand(network, new_ligand, n_edges=2)
   smaller = delete_edge(network, "lig_a~lig_b")
   joined = concatenate_networks(series_1, series_2, n_bridges=2)

Appending a ligand
------------------

``append_ligand`` maps only the new ligand's pairs -- *n* mappings for an *n*-ligand
network, rather than the *n(n+1)/2* a re-plan would cost -- and attaches it by its cheapest
two edges. Two, not one, because a single edge leaves the new ligand hanging off a bridge
where its free energy is checked by nothing; the second edge is what puts it on a cycle.

Fewer feasible partners than asked for is best-effort and lands on ``unmet_constraints``,
exactly as an ``edges_per_ligand`` shortfall does in the planner. *No* feasible partner is
not a shortfall but a failure, because the result would be a disconnected network nobody
asked for -- unless ``cbfe_mode`` permits a counterpoised edge, in which case one is spent
and the substitution is recorded.

Deleting an edge
----------------

``delete_edge`` refuses a bridge by default, and says which::

   ValueError: Edge lig_c~lig_d is a bridge: it is the only link between
   ['lig_a', 'lig_b', 'lig_c'] and ['lig_d', 'lig_e']. Deleting it would leave the
   network disconnected, so the two halves' free energies could no longer be compared.
   Attach another edge across the gap first, or pass must_stay_connected=False if a
   disconnected network is intended.

"That would disconnect the network" on its own does not say what to add instead. Naming the
two sides does.

Merging and concatenating
-------------------------

``merge_networks`` joins two networks that **share ligands**; ``concatenate_networks`` joins
two that share none. The distinction is not bookkeeping. Free energies from two networks are
on the same scale only through a path that joins them: a shared ligand *is* that path, so a
merge needs no new edges, while a concatenation must buy them. Calling either with the other
one's input is refused with a pointer to its sibling.

A shared ligand name must denote the same molecule **in the same atom order**. Every mapping
addresses atoms positionally, so two chemically identical molecules read in different orders
make the two networks' mappings mean different things, and comparing canonical SMILES would
happily merge them into a network whose edges point at the wrong atoms.

``concatenate_networks`` evaluates every cross pair -- ``len(a) * len(b)`` mappings, the
honest cost of finding the best join rather than a plausible one -- and its second bridge
avoids the endpoints the first used. Two bridges sharing an endpoint make that ligand a
single point of failure for the whole join, which is most of what the second bridge was
bought to avoid.

Cyclizing a component
---------------------

``cyclize_around_component`` adds edges until every ligand in a named set lies on a cycle. A
ligand on no cycle has a free energy nothing checks: every path to it runs through a bridge,
so an error on that bridge moves the ligand's number and shows up nowhere.

It exists as its own operation rather than as a re-plan because after an append or a
deletion, one or two ligands are in exactly that state and the rest of the network is fine.
By default it draws only on candidates the network already carries -- the original pool
holds far more feasible pairs than the plan selected, so this usually costs nothing. Pass a
mapper to have it evaluate pairs that were never scored.

Konnektor declares the same operation and raises ``NotImplementedError`` for it.

.. _lmi-replanning:

Diagnostics-driven replanning
-----------------------------

An RBFE campaign is a loop -- plan, run, analyse, replan --  and
:mod:`rbfenetmap.core.diagnostics` is the return leg.

``edgembar`` fits the whole network at once under the constraint that every cycle closes.
Each edge's Lagrange multiplier records how hard that constraint had to pull on it, and a
large **Lagrange Multiplier Index (LMI)** means the edge disagrees with the consensus its
cycles impose -- the signature of a poorly converged or badly set up transformation. The
CBFE paper does exactly this by hand on BACE1 and BRD4: inspect the worst edges, drop them,
re-run.

.. code-block:: bash

   rbfenet plan   --ligands ligands.sdf --out network.json
   # ... run the edges, analyse them, write out the per-edge LMIs ...
   rbfenet replan --network network.json --lmi lmi.json --lmi-quantile 0.9 \
                  --out replanned.json

Pruning is expressed as a **ban**, and the replan is the ordinary planner:

.. code-block:: text

   selected edges + LMIs  ->  ban the worst  ->  MSTRedundancyPlanner  ->  new network
                                                 (same candidate pool)

That is the whole design. Patching the holes by hand would need a second selection strategy
running alongside the real one and free to drift from it; banning and re-planning means the
replanned network satisfies exactly the same guarantees the first one did -- spanning,
degree targets, cycle coverage, the CBFE eligibility ladder -- over a smaller pool. **Nothing
new is mapped**: the replacements come from candidates the original run already scored and
did not select. A pruned edge has to be *banned* rather than merely dropped, because a
replan over an unmodified pool would simply select it again; it was, after all, the cheapest
edge there.

The surviving edges are held in place, as forced edges, unless ``--reselect`` is given. That
default is the difference between a replan you can act on and one you cannot: those edges are
set up, queued, or already finished, and a selection pass free to move them hands back a
network that is not the one being run. Before anything has been submitted, ``--reselect``
gives a clean re-selection over the pruned pool.

A forced edge is never pruned. Pinning an edge is an assertion that it must be in the
network, and a diagnostic does not override it -- though "your forced edge is the worst edge
in the network" is logged, because it is worth reading.

.. warning::

   **LMI pruning substantially reduces cycle-closure error and leaves MUE and RMSE against
   experiment essentially unchanged.**

   That is not a disappointing result; it is the correct reading of the quantity.
   Hysteresis is a **sampling** diagnostic, not an accuracy predictor. A network can close
   every cycle perfectly and still sit a kcal/mol from experiment, because a systematic
   error -- a force-field term, a protonation state, a missing water -- moves every edge
   around a cycle the same way and cancels in exactly the place hysteresis would have shown
   it.

   Use this to find edges that are internally inconsistent and worth re-running or
   replacing. Do not use it, and do not report it, as a route to better agreement with
   experiment.

:func:`~rbfenetmap.core.diagnostics.cycle_closure_errors` is provided so that claim is
checkable on your own data rather than merely asserted: sum your computed ΔΔG around each
basis cycle before and after, and compare the change there against the change in MUE.

The ingestion format
~~~~~~~~~~~~~~~~~~~~

:func:`~rbfenetmap.core.diagnostics.load_edge_lmi` reads a small JSON document of this
package's own:

.. code-block:: json

   {
     "lig_a~lig_b": 0.31,
     "lig_b~lig_c": 2.75,
     "lig_a~lig_c": 0.44
   }

A ``{"edges": {...}}`` wrapper is also accepted, so a file may carry other analysis output
alongside. Keys are read as unordered pairs -- an LMI is a property of the transformation,
and the transformation is undirected -- and the two orientations of one edge may not
disagree.

.. note::

   **This is not edgembar's on-disk format.** Wiring the reader to a real ``edgembar``
   network analysis is a follow-up, and it needs a real output file to develop against;
   writing a parser for a format that could not be checked would be guesswork presented as
   an integration. Extracting the multipliers from your analysis and writing the JSON above
   is a short script today, and everything downstream of it is implemented and tested.

A selected edge with no LMI value is an error rather than an exemption. Treating a missing
diagnostic as a good one would quietly spare exactly the edges the analysis could not
produce a number for, which are not the edges anyone means to trust by default;
``--allow-missing-lmi`` waives it deliberately.

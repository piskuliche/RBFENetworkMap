Scoring
=======

Descriptors are computed once, centrally, by
:func:`rbfenetmap.core.descriptors.compute_descriptors`. A scorer receives a plain
``Mapping[str, float]`` and nothing else -- no molecules, no mapping object, no RDKit.

That narrow interface buys three things: re-scoring under new weights costs nothing
because no mapping is recomputed; a scorer is testable against hand-written dictionaries
with no chemistry involved; and a third-party scorer cannot reach past its inputs and grow
a dependency on how the mapping was produced.

Reject versus bad score
-----------------------

The boundary is hard and deliberate.

**Rejection is structural** and originates only in the mapper, the repair, or validation
-- never from a weighted sum crossing a threshold. An infeasible edge gets
``EdgeScore(total=inf, feasible=False)`` and is excluded from the graph the planner sees.

**A bad score is a large finite cost** and stays in the candidate pool, where the planner
can still use it if the alternative is a disconnected network.

A scorer must never invent a rejection. ``rejections`` is passed in so it can be
*propagated*, not added to.

The linear scorer
-----------------

Each descriptor is normalised so ``1.0`` means roughly "one typical unit of badness", then
multiplied by a weight and clipped. The normalisation is what makes the weights
interpretable: a weight of ``4.0`` on ``charge_delta`` against ``1.0`` on
``softcore_atoms`` says a unit charge change costs about as much as four soft-core-sized
problems -- a statement a chemist can argue with.

.. list-table::
   :header-rows: 1

   * - Term
     - Descriptor
     - Weight
   * - ``softcore_atoms``
     - ``n_softcore_max_heavy / 8``
     - 1.00
   * - ``charge_delta``
     - ``charge_delta`` (cap 2)
     - 4.00
   * - ``mcs_deficit``
     - ``1 - mcs_fraction``
     - 2.00
   * - ``ring_delta``
     - ``ring_delta`` (cap 3)
     - 1.00
   * - ``core_rmsd``
     - ``core_rmsd`` in angstroms (cap 3)
     - 1.00
   * - ``repair_cost``
     - ``n_demoted_atoms / 6``
     - 0.75
   * - ``ring_atoms_in_softcore``
     - ``n_ring_atoms_in_softcore / 6``
     - 0.50
   * - ``softcore_asymmetry``, ``heavy_atom_delta``
     - ``... / 8``
     - 0.25
   * - ``rotatable_delta``
     - ``rotatable_delta / 3``
     - 0.20
   * - ``logp_delta``
     - ``|logp_delta| / 2`` (cap 3)
     - 0.10

Override with ``--weights key=value`` (repeatable) or ``--weights-file``. An unknown key
**raises**: a silently ignored typo would make a tuning run look effective while it
quietly used the defaults.

Other scorers
-------------

``lomaplike``
   Multiplicative similarity in ``(0, 1]``, converted to a cost by ``-log``. Composition
   differs meaningfully from a weighted sum: any single factor near zero sinks the whole
   edge regardless of the rest. That is the right shape when the penalties are independent
   reasons the edge will not converge, rather than competing preferences to balance.

``softcore-size``
   Cost equals the larger soft-core heavy-atom count. The honest baseline any richer
   scorer should be shown to beat, and -- because its costs are whole numbers -- what makes
   planner tests verifiable by hand.

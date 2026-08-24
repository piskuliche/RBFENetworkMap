The Tyk2 variant matrix
=======================

The other guides describe what each knob *does*. This one shows what they cost, by
planning one real series 21 different ways and putting the results side by side.

`Open the matrix <_static/tyk2-matrix/index.html>`_ -- every row is one ``rbfenet plan``
invocation over the same 16 ligands, and links to that variant's own full report.

The series is the 16-ligand Tyk2 set (``ejm31``--``ejm55``, ``jmc23``--``jmc30``),
shipped in ``examples/data/tyk2/`` and co-posed in a shared binding-site frame. All
**120 of 120** pairs map and repair, so the candidate pool is a single connected
component and no variant is working around an infeasible edge.

Reproduce the whole thing, reports included:

.. code-block:: bash

   python examples/06_variant_matrix.py docs/source/_static/tyk2-matrix

It takes about 45 s. The planning is deterministic: every report regenerates
byte-identically, which is what makes the published page checkable rather than merely
illustrative.

.. note::

   Costs in the table are in **scorer units and are comparable only within one scorer**.
   The two optimal-design rows use the ``variance`` scorer (kcal/mol predicted σ) and
   are not on the same scale as the ``linear``-scorer rows. GPU hours are comparable
   throughout.

Which topologies are worth running
----------------------------------

**Run** ``--cycle-coverage-mode edge``. It is the best value change to the default by a
wide margin: same 20 edges, one edge swapped, and it takes bridges 3 → 0 and edge cycle
coverage 85% → 100% for +1.79 scorer cost and *zero* additional GPU hours. The default
``mst`` network leaves three edges on no cycle, so those three transformations have no
independent check, and its Monte-Carlo robustness is the worst of any connected variant
(77% of trials stay connected at a 5% per-edge failure rate).

**Then decide how far above the** ``n·ln(n)`` **= 45 floor you want to be.** The
default's 20 edges is 25 short.

======================================  =====  =====  ========  =======  ==========
variant                                 edges  GPU h  diameter  bridges  robustness
======================================  =====  =====  ========  =======  ==========
``mst`` default                         20     79     7         3        0.77
``--cycle-coverage-mode edge``          20     79     5         0        0.93
``--edges-per-ligand 3``                26     103    6         0        0.98
``--max-diameter 4``                    24     95     4         0        0.98
``redundant-mst --n-redundancy 3``      45     179    4         0        1.00
======================================  =====  =====  ========  =======  ==========

``--max-diameter 4`` is the efficient one: 24 edges buys diameter 4 and full cycle
coverage for +16 GPU h over the default. A diameter of 7 in the default network means
seven chained perturbations between the two most distant ligands, which is where
accumulated error lives -- LOMAP caps at 6 and FEP+ works below 5, so the default is
outside both conventions.

**Skip** ``star``. Both hub heuristics pick ``ejm46`` (it is simultaneously tied for
most partners at 15 and the strict minimum-total-cost hub at 22.70 against 23.55 next
best), so ``--hub-selection min_total_cost`` is genuinely evaluated but changes nothing
on this set. The result has 15 bridges and zero cycles -- no consistency check anywhere.
Cheapest at 60 GPU h, and not worth it.

``complete`` (120 edges, 476 GPU h) is a reference, not a proposal, as is ``rmst-n3`` at
45 edges unless you specifically want to sit exactly on the floor.

Optimal design buys precision, not accuracy
-------------------------------------------

The A- and D-optimal rows at 44 edges land at 175 GPU h with diameter 3 and 95% edge
cycle coverage. The two designs are near-indistinguishable here (cost 221.04 vs 220.40,
201 vs 195 short cycles).

.. warning::

   Optimal design minimises *predicted variance*, which is not the same as being right.
   In NetBFE, ``tr(C)`` fell monotonically (1.08 → 0.78) as the design was optimised
   while RMSE against experiment *rose* (0.84 → 0.91). That result was measured on 16
   TYK2 inhibitors -- almost certainly this same ``ejm*``/``jmc*`` series -- so it
   applies here about as directly as such a warning can.

   NetBFE's payoff also comes from *iterating*: re-optimising allocation each round from
   variances measured in the previous one. These runs are the static first pass; there
   is no MD round trip behind them.

   Li, P.; Li, Z.; Wang, Y.; Dou, H.; Radak, B. K.; Allen, B. K.; Sherman, W.; Xu, H.
   *J. Chem. Theory Comput.* **2022**, *18* (2), 650-663.
   `doi:10.1021/acs.jctc.1c00703 <https://doi.org/10.1021/acs.jctc.1c00703>`_

What soft-core consistency changes for an Amber setup
-----------------------------------------------------

This is the most consequential knob in the matrix, and it is invisible in the topology.

Under the default ``--consistency pairwise``, **8 of the 16 ligands carry more than one
distinct soft-core mask** across the edges they participate in. Each such ligand needs a
different ``scmask`` per edge -- the same molecule parameterised differently depending on
which transformation it is in.

``--consistency component`` and ``--consistency graph`` both take that to **0**, on the
*identical* 20-edge topology, for +4.75 scorer cost (20.00 → 24.75) and no change in GPU
hours. One soft-core per ligand, one ``scmask``, reusable across every edge it appears
in.

The two modes produce byte-identical networks here because the candidate pool is a
single connected component -- there is only one component for ``component`` mode to work
over. Use either; ``graph`` states the intent more clearly. See
:doc:`concepts/softcore_repair` for what the repair itself does.

What ``--cbfe`` costs, and why the planner never reaches for it
---------------------------------------------------------------

``--cbfe all`` costs **3.20× per edge**: 23 counterpoised edges at 12.70 GPU h each
against 3.97 for the 20 RBFE edges. Whole-network 79 → 292 GPU h, a 3.68× increase. It
plans in 0.5 s because it needs no atom mapping at all.

More useful: **CBFE edges are never competitive on this series.** ``--cbfe bridge`` and
``--cbfe cycles`` both return the default network unchanged. That is not an inert flag --
``cycles`` mode does stock the CBFE pool, but with node cycle coverage already at 1.0
nothing ever spends it. Forcing the issue with ``--cbfe cycles --cycle-coverage-mode
edge`` *still* selects zero CBFE edges. With every pair feasible there is always a
cheaper RBFE edge to close any cycle.

On a set like this, counterpoised edges are a deliberate methodological choice, not
something the planner will reach for to fix topology. If you want them, ``--cbfe all``
is the honest way to ask.

What the stranded-core repair changed
-------------------------------------

Four pairs (``ejm42~ejm44``, ``ejm42~ejm55``, ``ejm43~ejm55``, ``ejm44~ejm55``) used to
be refused for ``softcore_multiple_attachments``: the MCS kept a terminal methyl in the
common core while the atom joining it to the rest of the core went soft, making the
soft-core a bridge rather than a substituent. The repair now absorbs the stranded
fragment.

It changes less than it sounds. Those edges cost 1.544-2.354 against a selected range of
0.273-1.983, because the demotion is charged through the scorer's ``repair_cost`` term,
so **the default network is byte-identical to before**. Only the denser variants move:
``--edges-per-ligand 4`` now reaches degree four with 32 edges rather than 33 and costs
less (38.94 → 38.03), ``complete`` gains the four, and the optimal designs swap a few for
marginally better totals.

The matrix page carries the full before/after table. It is built from
``examples/data/tyk2/matrix-pre-repair.json``, a frozen record of the same 21 variants
planned before the fix -- it cannot be regenerated from a current checkout, which is why
it is kept.

A known gap in budget allocation
--------------------------------

``--design-total-ns 400`` allocates the budget correctly in aggregate: the 44 per-edge
``simulation_ns`` values sum to exactly 400.0, with λ windows spread 12-24. But the
allocation has **no lower floor**, and two edges are assigned exactly 0.0 ns
(``ejm31~ejm42`` and ``ejm31~ejm46``). Next smallest is 5.16 ns, median 7.85, max 36.02.

Zero is a contradiction rather than a rounding artifact: both edges are selected, appear
in ``edges.dat``, and have a runconfig telling you to run them for no time at all. An
edge the design considers uninformative should be dropped from the network, not kept at
an unrunnable length.

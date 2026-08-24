Command line
============

.. code-block:: text

   rbfenet plan      Map, score, and select a network.
   rbfenet score     Score candidate edges without selecting a network.
   rbfenet map       Compute mappings for specific pairs.
   rbfenet export    Export an already-planned network.
   rbfenet replan    Prune high-LMI edges from a planned network and replan the gaps.
   rbfenet report    Render a self-contained HTML report.
   rbfenet plugins   List plugins and their availability.
   rbfenet inspect   Show everything known about one edge.
   rbfenet diagnose  Report network-level metrics for a planned network.

Exit codes: ``0`` success, ``1`` a package-level failure (unsatisfiable constraints, a
missing plugin, unreadable input), ``2`` an argparse usage error. Package errors print a
single message rather than a traceback; ``-v`` restores the traceback.

plan
----

.. code-block:: bash

   rbfenet plan --ligands ligands.sdf \
                --mapper mcss-e2 --scorer linear --planner mst \
                --edges-per-ligand 2 --min-cycle-coverage 1.0 \
                --max-softcore-atoms 12 --show-rejected \
                --out network.json --export amber html --export-dir ./out

To connect every ligand first, then prioritize getting as many ligands as possible onto
at least one short cycle:

.. code-block:: bash

   rbfenet plan --ligands ligands.sdf \
                --edges-per-ligand 1 --min-cycle-coverage 1.0 \
                --selection-objective connectivity_then_cycles \
                --pair-evaluation adaptive \
                --max-cycle-size 4 \
                --out network.json

To rescue a series whose feasible pool comes back in disconnected pieces, join them with
counterpoised edges instead of loosening the soft-core budget:

.. code-block:: bash

   rbfenet plan --ligands ligands.sdf \
                --cbfe bridge \
                --out network.json --export html amber --export-dir ./out

``--cbfe`` accepts ``off`` (default), ``bridge``, ``cycles``, and ``all``; the price of a
counterpoised edge is set by ``--cbfe-base-cost`` and ``--cbfe-atom-weight``. See
:doc:`concepts/network_selection` for what each mode may and may not spend an edge on.
``--cbfe all`` skips mapping altogether, so ``--mapper`` is ignored there.

For a set large enough that ``n ln n`` edges is more than anyone will run, plan it as
clusters instead:
To select edges by a statistical criterion rather than by cost, use the ``optimal`` planner
with the ``variance`` scorer -- the one scorer whose totals are predicted standard
deviations in kcal/mol, which is the scale the criterion is built on:
To try inventing a bridging ligand *before* falling back to counterpoised edges, add
``--intermediates``:

.. code-block:: bash

   rbfenet plan --ligands ligands.sdf \
                --cluster-by scaffold --cluster-bridges 2 \
                --out network.json

``--cluster-by`` accepts ``none`` (default), ``charge``, ``scaffold``, and ``fingerprint``.
Each cluster is planned as its own subnetwork and joined to the others by
``--cluster-bridges`` edges per joined cluster pair; the default of two puts each crossing
on a cycle. See :doc:`concepts/network_selection` for why the saving is real and what
happens when a partition would disconnect the network.
                --scorer variance --planner optimal --design d_optimal \
                --design-total-ns 500 \
                --out network.json --export amber --export-dir ./out

``--design`` accepts ``none`` (default), ``a_optimal``, and ``d_optimal``. Prefer
``d_optimal`` when a cycle-closure correction will be applied downstream -- it yields a
markedly more cyclic network at the same edge count -- and ``a_optimal`` otherwise. Naming
it alongside any planner but ``optimal`` is refused rather than ignored, and ``--planner
optimal`` without it is refused too: the two criteria answer different questions and
neither is a safe default. With ``--n-edges`` unset the design planner uses Pitman's floor,
``round(n ln n)``.

``--design-total-ns`` additionally splits a simulation budget A-optimally across the
selected edges and writes it into each Amber ``.runconfig`` as a lambda-window and
nanosecond allocation, bounded by ``--design-lambda-min`` / ``--design-lambda-max``.
``--design-refine`` adds a Fedorov exchange pass. See
:doc:`concepts/network_selection` -- including the warning that optimal design buys
precision and does not promise accuracy.
                --intermediates bridge --cbfe bridge \
                --out network.json

The two compose without any precedence rule: generation runs before selection, so a gap
an invented ligand closed is no longer a gap when CBFE eligibility is evaluated, and one
it could not close is still rescued by ``--cbfe bridge``. ``--intermediates gaps``
additionally offers infeasible pairs *inside* a connected component.
``--max-intermediates``, ``--max-intermediate-gaps`` and ``--intermediates-per-gap`` bound
the work; ``--intermediate-generator`` chooses the plugin, defaulting to ``pairmap``. The
subnetwork search is tuned with ``--intermediate-min-link-score``,
``--intermediate-max-dist``, ``--intermediate-max-cycle``,
``--intermediate-max-subgraph-dist`` and ``--intermediate-beta``, whose names and defaults
are the paper's. Every attempt is recorded in the network JSON, and every invented ligand
carries the parents, generator and pose RMSD it was built from.

An invented ligand is a residue nobody has parameterised, so ``--export amber`` writes
``ligands/<name>.sdf`` for every ligand and an ``intermediates.txt`` manifest, and
``--validate-exporter amber`` warns about the count before the mapping run rather than
after. See :doc:`concepts/intermediates` for the generator and
:doc:`concepts/network_selection` for the stage.

``--pair-evaluation adaptive`` fingerprint-ranks the all-pairs pool and maps it in
batches. It first evaluates each ligand's nearest neighbours, then prioritizes pairs
that bridge currently disconnected feasible components. Once connected, it evaluates
more pairs only while requested degree or cycle coverage remains unmet. Connectivity is
not declared impossible until all remaining component-bridging pairs have been tried.
The initial breadth and subsequent batch size are controlled by
``--adaptive-initial-neighbors`` and ``--adaptive-batch-size``.

Pair mapping progress is displayed automatically when stderr is an interactive terminal.
Use ``--progress`` to force it in a redirected job log or ``--no-progress`` to suppress
it. Adaptive progress reports the current batch and uses the complete candidate pool as
its denominator; ``(stopped early)`` means the requested network was satisfied without
mapping every pair.

``--validate-exporter amber`` checks that exporter's format constraints *before* the
expensive mapping stage, so a problem knowable from the inputs alone does not cost a full
planning run to discover.

To bound how far apart two ligands can be in the network, and to require that every
selected edge -- not merely every ligand -- lies on a cycle:

.. code-block:: bash

   rbfenet plan --ligands ligands.sdf \
                --max-diameter 5 \
                --cycle-coverage-mode edge \
                --out network.json

Both are best-effort and both default to what the planner already did: ``--max-diameter``
is unset and ``--cycle-coverage-mode`` is ``node``. A target the candidate pool cannot
deliver is warned about and recorded on ``unmet_constraints``, never raised.

For a network built from overlaid spanning trees rather than a tree plus greedy
redundancy:

.. code-block:: bash

   rbfenet plan --ligands ligands.sdf --planner redundant-mst --n-redundancy 2 \
                --out network.json

``--cost-units gpu_hours`` restates the reported cost as estimated machine time and a
dollar figure. It is a display unit: it cannot change which edges are chosen, and it is
usable alongside ``--compat``.

To generate a browsable HTML report alongside the JSON network:

.. code-block:: bash

   rbfenet plan --ligands ligands.sdf --out network.json \
                --export html --export-dir ./out

Ligand alignment
----------------

``--align``, ``--align-reference``, ``--align-min-atoms``, and ``--write-aligned`` belong
to the shared ligand-input group, so ``plan``, ``score``, and ``map`` all accept them with
the same meaning.

.. code-block:: bash

   rbfenet plan --ligands prepared_from_abfe/ --align \
                --write-aligned aligned/ --out network.json

Bare ``--align`` selects the maximum-common-substructure method; ``--align o3a`` selects
Open3DAlign for sets with no substructure large enough to fit on. The per-ligand report is
written to stderr, so ``rbfenet score --align --format json`` still produces a parseable
document on stdout.

If a run rejects every candidate for ``core_geometry_mismatch`` and ``--align`` was not
given, the failure message ends with a note suggesting it. See :ref:`aligning-ligands` for
how to read the report, and for what alignment does and does not change.

score
-----

Rank candidates without committing to a selection. ``--explain`` adds one column per cost
contribution, which is how you find out *why* an edge scores as it does.

.. code-block:: bash

   rbfenet score --ligands ligands.sdf --explain --top 20 --show-rejected

replan
------

The return leg of the plan-run-diagnose-replan loop. Give it a planned network and the
per-edge Lagrange Multiplier Indices from the analysis; it bans the worst edges and
re-selects the gaps from the candidate pool the original run already scored, mapping
nothing new.

.. code-block:: bash

   rbfenet replan --network network.json --lmi lmi.json \
                  --lmi-quantile 0.9 --max-pruned 3 \
                  --out replanned.json

Surviving edges are held in place unless ``--reselect`` is given, so the replan changes only
the gaps -- edges that are already set up or running do not move. ``--lmi-threshold`` cuts at
an absolute value instead of a quantile. See :ref:`lmi-replanning` for the LMI file format,
and for what pruning on hysteresis does and does not buy.

inspect
-------

The command that makes the algorithm auditable.

.. code-block:: bash

   rbfenet inspect --network network.json --edge "lig_a~lig_b" \
                   --show-repair-trace --show-descriptors --show-masks

diagnose
--------

Network-level metrics: cost, degree spread, isolated ligands, diameter, short cycle count,
Monte-Carlo failure robustness, and how the edge count compares with the ``n ln n``
precision floor. ``inspect`` is per-edge; this is per-network.

.. code-block:: bash

   rbfenet diagnose --network network.json --cost-units gpu_hours
   rbfenet diagnose --network network.json --format json --seed 0

``--seed`` fixes the robustness estimate, and it defaults to ``0`` so two runs over the
same file always agree. The same table is folded into the HTML report.

report
------

Render a self-contained HTML report with the selected network, a clickable edge index,
and per-transformation views that highlight the soft-core and common-core regions on both
ligands.

.. code-block:: bash

   rbfenet report --network network.json --out network.html --show-indices

plugins
-------

Lists what is installed. Availability is probed with :func:`importlib.util.find_spec`, so
nothing is imported -- which is why ``--all`` can report on backends that are absent.

.. code-block:: bash

   rbfenet plugins --all

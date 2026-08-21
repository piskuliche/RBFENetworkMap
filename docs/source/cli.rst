Command line
============

.. code-block:: text

   rbfenet plan      Map, score, and select a network.
   rbfenet score     Score candidate edges without selecting a network.
   rbfenet map       Compute mappings for specific pairs.
   rbfenet export    Export an already-planned network.
   rbfenet report    Render a self-contained HTML report.
   rbfenet plugins   List plugins and their availability.
   rbfenet inspect   Show everything known about one edge.

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

To select edges by a statistical criterion rather than by cost, use the ``optimal`` planner
with the ``variance`` scorer -- the one scorer whose totals are predicted standard
deviations in kcal/mol, which is the scale the criterion is built on:

.. code-block:: bash

   rbfenet plan --ligands ligands.sdf \
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

inspect
-------

The command that makes the algorithm auditable.

.. code-block:: bash

   rbfenet inspect --network network.json --edge "lig_a~lig_b" \
                   --show-repair-trace --show-descriptors --show-masks

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

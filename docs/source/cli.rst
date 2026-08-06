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
                --max-cycle-size 4 \
                --out network.json

``--validate-exporter amber`` checks that exporter's format constraints *before* the
expensive mapping stage, so a problem knowable from the inputs alone does not cost a full
planning run to discover.

To generate a browsable HTML report alongside the JSON network:

.. code-block:: bash

   rbfenet plan --ligands ligands.sdf --out network.json \
                --export html --export-dir ./out

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

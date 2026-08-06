Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install -e ".[all]"

Core dependencies are only ``rdkit``, ``networkx``, ``numpy``, and ``scipy``. The
``amber``, ``kartograf``, and ``amber-mol2`` extras pull in optional backends.

Example data
------------

No binary files are checked in. Regenerate the example ligands from their SMILES:

.. code-block:: bash

   python examples/data/make_conformers.py

Plan a network
--------------

.. code-block:: bash

   rbfenet plan --ligands examples/data/benzamides.sdf \
                --edges-per-ligand 2 --min-cycle-coverage 1.0 \
                --max-softcore-atoms 12 --show-rejected \
                --out network.json

From Python:

.. code-block:: python

   from rbfenetmap.core.options import NetworkOptions, SoftcorePolicy
   from rbfenetmap.core.pipeline import build_network
   from rbfenetmap.io.loaders import load_ligands

   network = build_network(
       load_ligands(["examples/data/benzamides.sdf"]),
       mapper="mcss-e2",
       network_options=NetworkOptions(
           edges_per_ligand=2, softcore=SoftcorePolicy(max_softcore_atoms=12)
       ),
   )
   network.validate()

Ligands must be co-posed
------------------------

.. warning::

   Ligands must be supplied in a **common binding-site frame**. The ``core_rmsd``
   descriptor measures in-place deviation *without* superposition, precisely so that it
   detects a mapping pairing atoms that occupy different parts of the pocket.

   Independently embedded conformers share no frame, so every candidate edge between them
   is rejected with ``core_geometry_mismatch`` -- correctly, but uselessly.
   ``examples/data/make_conformers.py`` shows the constrained-embedding pattern.

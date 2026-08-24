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

No binary files are checked in. Regenerate the benzamide example ligands from their
SMILES:

.. code-block:: bash

   python examples/data/make_conformers.py

The 16-ligand Tyk2 series used by :doc:`variant_matrix` is checked in directly, as
co-posed mol2 under ``examples/data/tyk2/``, because the comparison it supports is only
meaningful on a real, consistently posed set.

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

   If your structures were prepared *separately* -- set up individually for ABFE runs, say,
   and written to mol2 from their own Amber topologies -- then their conformers are real
   bound poses and only the frames disagree. :ref:`aligning-ligands` recovers the common
   frame rather than asking you to re-prepare them.

.. _aligning-ligands:

Aligning ligands
----------------

``--align`` rigidly superposes the set into a common frame before anything is mapped. It is
off by default, and should stay off for ligands that were prepared together: those are
already co-posed, and aligning them would paper over a genuine pose problem instead of
revealing it.

.. code-block:: bash

   rbfenet plan --ligands prepared_from_abfe/ --align \
                --write-aligned aligned/ --out network.json

Each ligand is fitted onto an already-aligned neighbour through their maximum common
substructure and moved there rigidly. Neighbour rather than root: the walk order is a
maximum spanning tree over fingerprint similarity, so a ligand is fitted onto the one it
most resembles. For a set too diverse for any substructure to bite on, ``--align o3a``
switches to Open3DAlign, which needs no shared substructure but gives no auditable set of
paired atoms in exchange.

The report tells you whether it worked, and goes to stderr so a ``--format json`` run stays
machine-readable::

   ligand   reference  fit atoms  rmsd   note
   -------  ---------  ---------  -----  ---------
   bza_CF3  -          0          -      reference
   bza_Cl   bza_F      10         0.132
   bza_H    bza_Cl     9          0.030

Read both columns. A small ``fit atoms`` count deserves suspicion even when the RMSD looks
good -- three atoms determine a rigid body, but only just -- and a ligand that could not be
fitted at all is reported separately and left in its own frame, where its edges will still
be rejected for geometry.

Then look at the structures, because a plausible table can still hide a flipped ring:

.. code-block:: bash

   pymol aligned/*.sdf   # one object per ligand, not one object with N states

.. note::

   Rigid alignment recovers a common **frame**. It cannot recover a common
   **conformation**. Each independently relaxed structure keeps its own ring puckers,
   exocyclic torsions, and bond-length noise, so a residual ``core_rmsd`` survives alignment
   and should -- expect more of it than a constrained-embedded series would show, and expect
   the default ``--core-rmsd-threshold 2.0`` to be doing real work rather than waving
   everything through. Raise it only after looking at the actual distribution::

      rbfenet score --ligands prepared_from_abfe/ --align --show-rejected --explain

What alignment does *not* touch is the atom mapping. Mappings are index-based, so Amber
masks, exported edge lists, and every depiction are identical with and without ``--align``;
it cannot corrupt a downstream setup. What it does change is every quantity measured from
coordinates -- ``core_rmsd``, the geometry gate, the RMSD terms in both scorers -- and the
molecules embedded in ``network.json``, which carry the aligned frame from then on. Each
ligand records what was done to it under ``metadata["alignment"]``, and that survives the
network round trip.

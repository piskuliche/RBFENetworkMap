Intermediate ligands
====================

Some pairs cannot be related by any mapping. The soft-core would be too large, the cores
too dissimilar, the geometry irreconcilable. The pipeline can *invent* a molecule that sits
between them and turn one impossible edge into two possible ones.

Where the stage sits, which gaps it is offered, and how it interacts with every other knob
is :doc:`network_selection`. This page is about the molecules themselves: who proposes
them, how they are posed, what certifies the pose, and what leaves the package once one
exists.

Read this first
---------------

**An intermediate is slower on a pair that already works.** IMERGE measured paths through
an intermediate converging roughly **20% more slowly** than the direct path. Two
calculations replace one, and each one is not free.

The win is not speed and it is not accuracy on an easy pair. It is the pairs where **the
direct path does not converge at all** -- a transformation the mapper refuses, a soft-core
too large to sample, a core the geometry cannot hold fixed. There, the choice is not
"one calculation or two"; it is "two calculations or no comparison". That is why the
feature is off by default and why ``mode="bridge"`` -- only gaps that actually disconnect
the network -- is the mode to reach for first.

The generators
--------------

``pairmap`` (default)
~~~~~~~~~~~~~~~~~~~~~

:class:`~rbfenetmap.plugins.intermediates.pairmap_generator.PairMapGenerator`, after

   K. Furui, S. Shimizu, Y. Akiyama, S. Kimura, T. Terada and M. Ohue, "PairMap: An
   Intermediate Scaffold-Based Approach to Improve Alchemical Free Energy Calculations for
   Complex Perturbations", *J. Chem. Inf. Model.* **2025**, 65, 705-721,
   `doi:10.1021/acs.jcim.4c01634 <https://doi.org/10.1021/acs.jcim.4c01634>`_. Reference
   implementation: `github.com/ohuelab/PairMap <https://github.com/ohuelab/PairMap>`_,
   CC-BY 4.0.

The implementation here is **re-derived from the paper**, not adapted from that code.

It emits a *subnetwork*, not a chain. Every position on the shared core offers three
groups -- the source parent's, the target parent's, and nothing, which is the MCS itself --
and one group per position is a molecule. Two such molecules are linked with score

.. math:: s = \exp(-\beta \Delta)

where :math:`\Delta` counts the heavy atoms that disappear plus those that appear. That is
the LOMAP similarity, which is why the paper's :math:`\beta` and this package's
``beta = 0.1`` are the same constant.

A path is scored by the harmonic mean of its squared link scores divided by its length,
and that expression collapses exactly to :math:`1 / \sum_i s_i^{-2}`. So the paper's first
two subnetwork requirements -- shortest path, highest link scores -- are **one**
shortest-path search rather than two objectives to trade off. Paths of a single link are
excluded, and the direct link is not in the graph at all: it is the transformation the
pipeline already rejected. Cycles are then closed around the chosen path, one uncovered
link at a time and cheapest first, until every link sits in a cycle of at most
``max_cycle`` -- the remaining two requirements, shortest cycles and minimal redundancy.

The smallest shape it emits is ``A-M1-B-M2-A``: two independent routes across one gap,
whose closure error is a genuine consistency check on a pair that had no direct edge at
all.

**What is not covered.** The paper enumerates atom- and ring-level operations -- change an
element, add or delete an atom, open or close a ring -- so it can pass through
intermediates that live *inside* a substituent. This implementation's operations are whole
substituents. On a pair differing by several R-groups, which is the common hard case, the
two enumerations agree on the useful molecules. On a scaffold hop, or a change buried in
the middle of one substituent, this generator finds the truncation to the shared core and
nothing finer, and will often refuse the gap outright -- saying so in the record rather
than proposing something worse.

``fragment-swap``
~~~~~~~~~~~~~~~~~

:class:`~rbfenetmap.plugins.intermediates.fragment_swap.FragmentSwapGenerator`: one hybrid
per differing position, no search, no subnetwork. It plays the role
:class:`~rbfenetmap.plugins.mappers.identity_mapper.IdentityMapper` plays for mappers --
its output is obvious by inspection, so a surprise downstream of it is unambiguously
downstream.

Posing, and the only thing that certifies it
--------------------------------------------

A generator proposes a molecule with **no conformer**. Posing is centralised in
:func:`rbfenetmap.core.posing.pose_intermediate` for the same reason descriptors are: a
generator that posed its own output would make an intermediate's quality depend on which
plugin invented it.

An intermediate is a hybrid, and every heavy atom of it corresponds to a *specific* atom of
a parent whose pose is already right in the binding-site frame. So the poser seeds a
conformer from the parents' coordinates through the generator's ``parent_atom_map``,
embeds under a ``coordMap``, restores exactness with a restrained minimisation, and fits
rigidly back onto the donor coordinates. Neither ``ConstrainedEmbed`` (which discards a
known-good pose for everything outside the scaffold) nor :mod:`rbfenetmap.core.align`
(which recovers a frame for molecules that already have good conformers) does that.

Failures are data: :class:`~rbfenetmap.core.posing.PoseResult` carries a rejection --
``EMBED_FAILED``, ``POSE_RMSD_EXCEEDED``, ``CHARGE_MISMATCH``, ``STEREO_UNDEFINED`` -- and
never raises.

.. important::

   **You do not have to trust the poser.** A synthetic ligand is an ordinary vertex, and
   the in-place ``core_rmsd`` gate on its ``A~M`` and ``M~B`` sub-edges is a complete check
   on whether it is posed in its parents' frame. A badly posed intermediate comes back as
   an ordinary ``core_geometry_mismatch``, the proposal fails to close the gap, and it is
   dropped whole. The poser's job is to make that check *pass often*, not to make it
   unnecessary.

Provenance
----------

Every invented ligand carries a :class:`~rbfenetmap.core.models.LigandProvenance`: the
generator, the parents, the pose method, and the pose RMSD. It is a field on
:class:`~rbfenetmap.core.models.Ligand` rather than a subclass, because
``networkio._ligand_from_dict`` constructs ``Ligand`` directly and a subclass would silently
downgrade on round-trip. It is omitted entirely from the serialized form when ``None``, so
an all-real network is byte-identical to what it was before the feature existed.

``network.intermediates`` records **every gap attempted**, accepted or not, with the
rejection and the trace. Without it, a network where generation was tried and failed is
indistinguishable from one where it was never switched on -- and those call for entirely
different responses.

What leaves the package
-----------------------

Amber export
~~~~~~~~~~~~

``edges.dat`` names residues, and ``BuildEdges`` needs a parameterised topology for every
one of them. An invented ligand exists only inside the planned network; nobody has seen it,
and nothing on disk describes it. So:

- every ligand is written as ``ligands/<name>.sdf``. The real ones too, because *"every
  name in* ``edges.dat`` *has a structure in* ``ligands/``\ *"* is an invariant a setup
  script can assert, while "every name except the ones you already had" is not;
- invented ligands are listed in ``intermediates.txt`` as ``<name> <parent> <parent>
  <generator>``, so a setup script knows which residues to parameterise first;
- ``rbfenet plan --validate-exporter amber`` warns with the count **before** the mapping
  run, and warns pre-flight when generation is merely enabled.

HTML report and GraphML
~~~~~~~~~~~~~~~~~~~~~~~

An invented vertex is drawn with a dashed outline and its label ends in ``SYN``; every edge
card naming one carries a ``SYN`` badge. Never colour alone -- the same rule the CBFE badge
follows. An *Invented ligands* section lists parents, generator, pose method and pose RMSD;
the RMSD is measured with the same :func:`~rbfenetmap.core.kabsch.core_rmsd` the
feasibility gate uses, so it reads directly against ``core_rmsd_threshold``. The summary
counts real ligands and invented ones separately, because "12 ligands" quietly meaning nine
is the misreading that costs somebody a simulation.

:meth:`~rbfenetmap.core.models.Network.to_networkx` puts ``synthetic`` on every node, so
GraphML carries the distinction too.

Tuning
------

The five search constants keep the paper's names and published defaults, on
:class:`~rbfenetmap.core.intermediates.IntermediateOptions` and on the command line.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Option / flag
     - Default
     - What it bounds
   * - ``min_link_score`` / ``--intermediate-min-link-score``
     - ``0.2``
     - Weakest link worth proposing. At ``beta = 0.1`` this is about sixteen heavy atoms
       changing in one step.
   * - ``max_dist`` / ``--intermediate-max-dist``
     - ``3``
     - Links on the source-to-target path. At least 2: one link *is* the rejected direct
       edge.
   * - ``max_cycle`` / ``--intermediate-max-cycle``
     - ``4``
     - Largest cycle built to give a link a second, independent route.
   * - ``max_subgraph_dist`` / ``--intermediate-max-subgraph-dist``
     - ``4``
     - How far from either parent, in links, a molecule may sit and still join the
       subnetwork.
   * - ``beta`` / ``--intermediate-beta``
     - ``0.1``
     - Decay of the link score per heavy atom changed.

``max_molecules`` (``--intermediates-per-gap``) is the one to reach for first: it is what
stops a heavily decorated pair from spending the whole run on embeddings. The path is paid
for before any cycle is closed, so tightening it costs redundancy before it costs the
bridge.

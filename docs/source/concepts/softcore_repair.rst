Soft-core repair
================

The constraint
--------------

An alchemical transformation grows one singly attached region of a molecule into
another. The final soft-core on each side must therefore be connected and attach to the
common core through exactly one bond. A mapper, however, is free to return a
correspondence whose unmapped atoms fall into several disconnected pieces or form a
bridge between two common-core atoms. Benzene to *para*-xylene is the everyday fragmented
example: two hydrogens on opposite sides of the ring vanish, giving two separate
soft-core regions on one side and two appearing methyls on the other. No single
transformation can run that.

So the partition must either be **repaired** or the edge **rejected**.

The algorithm
-------------

Repair works by *demoting* common-core atoms into the soft-core until the pieces join up.
Choosing which atoms is a **node-weighted Steiner tree** problem: the soft-core fragments
are the terminals, the bond graph is the network, and the cost of recruiting an atom is
how much soft-core that recruitment ultimately drags in.

Three closure rules run to a fixpoint after every recruitment:

**(a) whole-ring**
   A ring is never left half soft-core. Touching one ring atom absorbs the ring. Fused
   systems then cascade on their own, with no special case: absorbing one ring pulls in
   the atoms it shares with its neighbours, which makes those neighbours
   intersected-but-not-contained on the next sweep.

**(b) hydrogen-follows-parent**
   A hydrogen joins the soft-core when its heavy parent does. **Deliberately one-way.** A
   soft-core hydrogen whose parent is common core stays put as a one-atom region -- that
   asymmetry is not an oversight. ``R-H -> R-CH3`` is the single most common
   transformation in the field, and its soft-core on the ``R-H`` side is exactly one
   hydrogen attached to a core carbon. A two-way rule would demote that carbon, then its
   ring, and destroy the edge.

**(c) mapped-partner**
   Demoting an atom demotes whatever it is mapped to. This is the only rule coupling the
   two molecules, and it is why the repair is genuinely *joint*: fixing a fragmentation on
   side 1 can create a new one on side 2, which the loop must then fix in turn.

Why closure size is the cost
----------------------------

The cost of demoting an atom is measured as **the size of the closure that demotion
triggers, across both molecules**. That single number makes the search behave chemically
with no hand-tuned per-element weight table:

* a hydrogen costs about 2 -- itself and its partner;
* a peripheral heavy atom costs a little more;
* an aromatic carbon costs its entire fused ring system, plus every attached hydrogen,
  plus all of their partners on the other side.

So the solver routes around rings whenever an acyclic path exists, and only pays for a
ring when there is no alternative.

Termination
-----------

Both soft-core sets grow monotonically within finite atom sets, and any iteration that
does not return adds at least one atom -- a bridge joining two or more fragments must
contain an atom that is not already soft-core. The loop therefore terminates in at most
``n_atoms_1 + n_atoms_2`` iterations.

Rejection
---------

Checks run in order of severity, so the reported reason is the most fundamental problem
rather than whichever threshold happened to be tightest:

.. list-table::
   :header-rows: 1

   * - Condition
     - Reason
   * - The soft-core covers a whole molecule
     - ``no_common_core``
   * - Fewer than ``min_core_atoms`` heavy atoms remain in the core
     - ``core_too_small``
   * - Heavy soft-core exceeds ``max_softcore_atoms``
     - ``softcore_too_large``
   * - Heavy soft-core exceeds ``max_softcore_fraction`` of a molecule
     - ``softcore_fraction_exceeded``
   * - A soft-core region has more than one common-core attachment bond
     - ``softcore_multiple_attachments``
   * - The loop is exhausted
     - ``repair_did_not_converge``

A rejected edge is **returned unchanged**, never silently mutated, and is retained on
:attr:`~rbfenetmap.core.models.Network.candidates`. Those rejections are what explain a
sparse or disconnected network afterwards.

The ring cascade is real
------------------------

.. note::

   Because the whole-ring rule cascades through fused systems, a repair that begins with a
   two-atom bridge on a polycyclic scaffold can end with thirty soft-core atoms and be
   rejected. This is *correct* -- there genuinely is no single-region transformation there
   -- but it will remove many candidate edges on fused scaffolds.

   Mitigations: the closure-size cost steers the solver away from ring atoms in the first
   place; ``--show-rejected`` and ``rbfenet inspect`` make every rejection visible;
   ``--ring-policy none`` is the escape hatch for deliberate ring-opening work.

Auditing a repair
-----------------

.. code-block:: bash

   rbfenet inspect --network network.json --edge "bza_CF3~bza_Et" --show-repair-trace

.. code-block:: text

   regions       (3, 3) -> (1, 1)
   repair trace
     initial: 3 soft-core region(s) on side 1, 3 on side 2
     iter 1 side 1: bridged 3 regions by demoting 1 atom(s) [1]
     iter 1 side 2: bridged 3 regions by demoting 1 atom(s) [1]
     final: 1/1 region(s), soft-core 4/7 atom(s)

A bridge found by the approximate Steiner solver is marked ``[steiner:approximate]`` in
the trace. It is still valid, but it is not guaranteed reproducible across networkx
versions, and a user comparing two runs deserves to know which is which.

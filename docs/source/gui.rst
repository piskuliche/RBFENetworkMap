Graphical interface
===================

``rbfenet plan`` has some sixty knobs. Choosing among them from the command line means
running it, opening the report, changing a flag, running it again, and holding the previous
answer in your head. ``rbfenet gui`` closes that loop: move a knob, see the network and its
metrics, pin the run, move another knob, compare.

.. code-block:: bash

   rbfenet gui --ligands ligands.sdf

.. code-block:: text

   Loaded 16 ligand(s).
   rbfenet gui listening on http://127.0.0.1:8765/  (ctrl-c to stop)

It is part of the package, not an extra: the server is standard library only, and there is
nothing to install beyond what planning already needs.

The form is the command line
----------------------------

Every control on the page is generated from the CLI's own argument parser, and the filled
form is turned back into flags that ``rbfenet plan`` parses. The GUI holds no list of knobs
of its own.

That is worth stating plainly because of what it buys. A flag added to the CLI appears in
the form with no change to the GUI. Every refusal you see -- an edge budget too small to
span the ligands, a ``--compat`` level contradicting a knob it pins, a soft-core threshold
out of range -- is the CLI's own message, not a second opinion about it. And the page can
always show you the exact command that produced what you are looking at:

.. code-block:: bash

   rbfenet plan --ligands ligands.sdf --planner redundant-mst --n-redundancy 3 --cbfe bridge

Copy that into a job script and you get the network on the screen. Exploring interactively
and running reproducibly are the same activity, rather than two that have to be kept in
agreement by hand.

What it shows you that the command line cannot
----------------------------------------------

**Knobs the chosen planner will ignore are greyed out.** ``star``, ``explicit`` and
``complete`` accept some fourteen network flags and then never read them; only ``--design``
and ``--cbfe`` are refused out loud, by the planner's own support checks. The form marks
the rest rather than letting you tune something that cannot take effect. ``--pair-evaluation
adaptive`` gets the same treatment: it is honoured only by the ``mst`` planner and falls
back to eager evaluation under any other.

**Knobs that cannot change the network are labelled.** ``--design-total-ns`` and the lambda
bounds are read by the Amber exporter when it writes runconfigs. They are worth setting and
they will not move an edge, so the page says so instead of leaving you to wonder why the
diagram did not change.

**Peak mapping memory is estimated as you type.** ``FindMCS`` allocates monotonically and
frees nothing until it returns, at roughly 40 MB per second of ``--mcs-timeout`` per job, so
the defaults of 60 s and 8 jobs are some 20 GB before a single candidate is retained. The
page works that out from the two knobs and warns before you start the run rather than after
the machine has begun swapping.

Why a second run is faster
--------------------------

Mapping is where a plan spends its time, and it is the one stage that does not depend on
which planner runs afterwards. Over the shipped Tyk2 set -- sixteen ligands, a hundred and
twenty pairs -- a full plan takes about 2.1 s, and takes about 2.1 s for every variant in
:doc:`the published matrix <variant_matrix>` whatever the selection knobs; the same run with
``--cbfe all``, which skips mapping entirely, takes 0.5 s.

So the GUI remembers atom correspondences. Moving a selection knob re-runs the repair and
the scorer, which are cheap, while the MCS searches come back from a cache. Failed mappings
are remembered too: a pair no search can relate is exactly the one that costs a full
``--mcs-timeout`` to fail, and its answer cannot change when a planner knob moves.

``--cache-dir`` keeps that across sessions, which makes the *second launch* fast rather than
only the second run:

.. code-block:: bash

   rbfenet gui --ligands ligands.sdf --cache-dir ~/.cache/rbfenet

The cache is keyed on the molecules' atom blocks, the mapper, and the mapping options, so
editing a ligand file or changing ``--mcs-timeout`` correctly misses. Nothing in it can go
stale into a result: everything cached is recomputable, and a damaged cache file is
discarded with a warning rather than raising.

Running it on a cluster
-----------------------

The server binds the loopback interface. To use it against ligands on a remote machine,
forward the port rather than binding a public one:

.. code-block:: bash

   ssh -L 8765:127.0.0.1:8765 user@cluster
   # then, on the cluster:
   rbfenet gui --ligands /scratch/ligands.sdf --no-browser --cache-dir ~/.cache/rbfenet

``--host`` will bind elsewhere and warns when you ask it to. The server plans networks from
any ligand path it is given, runs with your privileges, and has no authentication; it is a
local tool, and it is not written to face a network.

Options
-------

.. code-block:: text

   --ligands PATH...     Ligands to load at startup. Optional; a file can be chosen in the page.
   --name-property PROP  Molecule property to read ligand names from (default: _Name).
   --host HOST           Interface to bind (default: 127.0.0.1).
   --port PORT           Port to bind (default: 8765). 0 picks a free one.
   --no-browser          Do not open a browser window.
   --cache-dir DIR       Persist mapped atom correspondences between sessions.

What it does not do
-------------------

It plans and it compares. It does not export: a run offers its ``network.json`` and its
:doc:`full HTML report <cli>` for download, and everything downstream -- Amber setup,
:doc:`surgery and replanning <concepts/replanning>` -- stays with the CLI and the Python
API. Nor does it expose knobs the CLI has not got, such as the ``FindMCS`` settings on
:class:`~rbfenetmap.core.options.MappingOptions` or the fields of
:class:`~rbfenetmap.core.cost.CostModel`. A knob the GUI had and the command line could not
express would break the one property the whole design rests on.

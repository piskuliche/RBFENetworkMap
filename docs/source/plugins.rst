Plugins
=======

Four plugin kinds, each an abstract base class in :mod:`rbfenetmap.core.meta`:

.. list-table::
   :header-rows: 1

   * - Kind
     - Responsibility
     - Built-ins
   * - mapper
     - Propose an atom correspondence
     - ``mcss``, ``mcss-e``, ``mcss-e2``, ``cartograph``, ``kartograf``, ``identity``
   * - scorer
     - Reduce descriptors to a cost
     - ``linear``, ``lomaplike``, ``softcore-size``, ``variance``
   * - planner
     - Select the final edge set
     - ``mst``, ``star``, ``explicit``, ``complete``, ``optimal``
   * - exporter
     - Serialize for a downstream program
     - ``json``, ``edgelist``, ``graphml``, ``amber``, ``html``

Lazy registration
-----------------

A :class:`~rbfenetmap.core.pluginregistry.PluginSpec` describes a plugin **without
importing it**. Registration is pure metadata; the implementation module is imported only
when :meth:`~rbfenetmap.core.pluginregistry.PluginRegistry.create` is called.

That is what lets ``rbfenet plugins --all`` list every backend -- including ones whose
dependencies are absent -- without importing RDKit or kartograf, and what lets the whole
test suite run with no optional dependency installed.

The ``requires`` tuple is probed with :func:`importlib.util.find_spec`, so a missing
backend is reported as ``needs kartograf, gufe`` rather than as a raw
:class:`ModuleNotFoundError`.

Writing a mapper
----------------

.. code-block:: python

   from typing import ClassVar

   from rbfenetmap.core.meta.mappers import AbstractMapper
   from rbfenetmap.core.models import AtomMapping

   class MyMapper(AbstractMapper):
       name: ClassVar[str] = "mine"

       def map_pair(self, source, target, options) -> AtomMapping:
           core = {...}  # {index_in_source: index_in_target}
           return AtomMapping.from_core_pairs(
               core, n_atoms_1=source.n_atoms, n_atoms_2=target.n_atoms, method=self.name
           )

A mapper does **not** need to produce a connected soft-core. Repairing fragmentation is
:func:`rbfenetmap.core.softcore.repair_softcore_connectivity`'s job, and it runs on every
mapper's output. Duplicating that logic inside a mapper only makes mappers harder to write
and compare.

Registering it
--------------

.. code-block:: python

   from rbfenetmap.core.pluginregistry import PluginRegistry, PluginSpec

   registry = PluginRegistry()
   registry.register(PluginSpec(
       name="mine",
       kind="mapper",
       target="my_package.mappers:MyMapper",
       description="What it does.",
       requires=("some_backend",),
   ))

An instance can also be passed straight to
:func:`~rbfenetmap.core.pipeline.build_network`; registering is only needed to make the
plugin selectable by name from the CLI.

Exporters: the hook into other programs
---------------------------------------

An exporter adapts a planned network to a downstream consumer without that consumer's
concerns reaching back into the core. It may also implement
:meth:`~rbfenetmap.core.meta.exporters.AbstractExporter.validate`, which checks
format-specific constraints the core does not enforce, early and without writing anything.

The ``amber`` exporter is the motivating case: Amber soft-core masks select atoms **by
name**, so a soft-core atom sharing a name with a common-core atom would silently widen
the mask and produce wrong free energies with no error at run time. ``validate`` catches
that before any expensive work happens.

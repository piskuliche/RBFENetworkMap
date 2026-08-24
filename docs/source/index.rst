RBFENetworkMap
==============

Plan Relative Binding Free Energy perturbation networks from RDKit molecules.

Given a series of ligands, the package returns a scored, tunable network of alchemical
transformations. Every transformation carries a common-core / soft-core partition that
satisfies the constraint the whole package is organised around: **a transformation has at
most one connected soft-core region per side.**

.. toctree::
   :maxdepth: 2
   :caption: Guide

   quickstart
   concepts/softcore_repair
   concepts/scoring
   concepts/network_selection
   concepts/replanning
   cli
   plugins

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`

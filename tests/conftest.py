"""Shared fixtures and Dummy plugin implementations.

The Dummy plugins exist so that everything downstream of the plugin boundary can be
tested with no real backend involved. ``DummyMapper`` in particular accepts a
hand-authored contract, which is the only practical way to drive the soft-core repair
into a specific fragmentation: asking a real mapper for a mapping whose soft-core falls
into exactly three pieces means finding a molecule pair that happens to produce one.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

from rbfenetmap.core.meta.exporters import AbstractExporter
from rbfenetmap.core.meta.mappers import AbstractMapper
from rbfenetmap.core.meta.planners import AbstractNetworkPlanner
from rbfenetmap.core.meta.scorers import AbstractScorer
from rbfenetmap.core.models import AtomMapping, EdgeKind, EdgeScore, Ligand, Network, RejectionReason, Transformation
from rbfenetmap.core.options import MappingOptions, NetworkOptions

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "data"


# --- Dummy plugins ---------------------------------------------------------------


class DummyMapper(AbstractMapper):
    """Map atom *i* to atom *i*, or return a caller-supplied contract verbatim."""

    name: ClassVar[str] = "dummy"

    def __init__(self, contract: Mapping[str, Sequence[int]] | None = None) -> None:
        """Store an optional explicit contract to return instead of the identity map."""
        self._contract = contract

    def map_pair(self, source: Ligand, target: Ligand, options: MappingOptions) -> AtomMapping:
        """Return the stored contract, or the index-identity mapping."""
        del options
        if self._contract is not None:
            return AtomMapping.from_contract(
                self._contract, n_atoms_1=source.n_atoms, n_atoms_2=target.n_atoms, method=self.name
            )
        shared = min(source.n_atoms, target.n_atoms)
        return AtomMapping.from_core_pairs(
            {i: i for i in range(shared)}, n_atoms_1=source.n_atoms, n_atoms_2=target.n_atoms, method=self.name
        )


class DummyScorer(AbstractScorer):
    """Return costs from a caller-supplied table, so planner tests are deterministic."""

    name: ClassVar[str] = "dummy"

    def __init__(self, costs: Mapping[tuple[str, str], float] | None = None, default: float = 1.0) -> None:
        """Store the cost table and the fallback cost."""
        self._costs = {tuple(sorted(k)): v for k, v in (costs or {}).items()}
        self._default = default

    def cost_for(self, source: str, target: str) -> float:
        """Look up the cost of an unordered pair."""
        return self._costs.get(tuple(sorted((source, target))), self._default)  # type: ignore[arg-type]

    def score_edge(self, descriptors, *, rejections):
        """Return the tabulated cost, keyed by the descriptors' embedded edge name."""
        if rejections:
            return EdgeScore.rejected(*rejections, scorer=self.name)
        return EdgeScore(
            total=float(descriptors.get("cost", self._default)),
            feasible=True,
            descriptors=dict(descriptors),
            contributions={"cost": float(descriptors.get("cost", self._default))},
            scorer=self.name,
        )


class DummyPlanner(AbstractNetworkPlanner):
    """Select every feasible candidate, with no optimisation."""

    name: ClassVar[str] = "dummy"

    def plan(self, ligands, candidates, options):
        """Return a network containing all feasible candidates."""
        return Network(
            ligands=ligands,
            edges=tuple(c for c in candidates if c.feasible),
            candidates=tuple(candidates),
            planner=self.name,
            options=options,
        )


class DummyExporter(AbstractExporter):
    """Record its calls instead of writing anything."""

    name: ClassVar[str] = "dummy"
    default_suffix: ClassVar[str] = ".dummy"

    def __init__(self) -> None:
        """Start with an empty call log."""
        self.calls: list[tuple[Network, Path, dict[str, Any]]] = []

    def export(self, network, destination, **options):
        """Log the call and report a plausible path."""
        self.calls.append((network, Path(destination), dict(options)))
        return (Path(destination) / f"network{self.default_suffix}",)


# --- Molecule factories ----------------------------------------------------------


def make_ligand(smiles: str, name: str, *, seed: int = 0xF00D) -> Ligand:
    """Build a :class:`Ligand` from SMILES with a deterministic 3D conformer."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0, f"could not embed {smiles!r}"
    AllChem.MMFFOptimizeMolecule(mol)
    return Ligand.from_mol(mol, name)


def make_coposed(smiles_by_name: Mapping[str, str], scaffold_smiles: str) -> dict[str, Ligand]:
    """Build ligands constrained-embedded onto a shared scaffold.

    Ligands for an RBFE network share a binding-site frame. Independently embedded
    conformers do not, and the in-place core RMSD check correctly rejects every edge
    between them -- so any test touching geometry must co-pose its inputs.
    """
    scaffold = Chem.AddHs(Chem.MolFromSmiles(scaffold_smiles))
    assert AllChem.EmbedMolecule(scaffold, randomSeed=0xF00D) == 0
    AllChem.MMFFOptimizeMolecule(scaffold)
    scaffold = Chem.RemoveHs(scaffold)

    ligands: dict[str, Ligand] = {}
    for name, smiles in smiles_by_name.items():
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.ConstrainedEmbed(mol, scaffold, randomseed=0xF00D)
        ligands[name] = Ligand.from_mol(mol, name)
    return ligands


def random_frame(seed: int) -> np.ndarray:
    """Return a deterministic ``(4, 4)`` proper rigid-motion matrix.

    The determinant is forced positive. A scramble that smuggled in a reflection would be
    unrecoverable by any rotation, so an alignment test built on one would fail for a
    perfectly good reason while looking exactly like a bug in the aligner.
    """
    rng = np.random.default_rng(seed)
    rotation, upper = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation = rotation * np.sign(np.diag(upper))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = rng.normal(size=3) * 40.0
    return matrix


def scramble_frame(ligand: Ligand, *, seed: int) -> Ligand:
    """Return *ligand* moved rigidly into a frame of its own."""
    mol = Chem.Mol(ligand.mol)
    rdMolTransforms.TransformConformer(mol.GetConformer(), random_frame(seed))
    return replace(ligand, mol=mol)


# --- Fixtures --------------------------------------------------------------------


@pytest.fixture
def dummy_mapper() -> DummyMapper:
    """An identity mapper."""
    return DummyMapper()


@pytest.fixture
def dummy_scorer() -> DummyScorer:
    """A table-driven scorer."""
    return DummyScorer()


@pytest.fixture
def dummy_planner() -> DummyPlanner:
    """A planner that selects everything feasible."""
    return DummyPlanner()


@pytest.fixture
def dummy_exporter() -> DummyExporter:
    """An exporter that records its calls."""
    return DummyExporter()


@pytest.fixture(scope="session")
def benzamides() -> dict[str, Ligand]:
    """A small co-posed congeneric series, built in memory."""
    return make_coposed(
        {
            "bza_H": "c1ccccc1C(=O)N",
            "bza_F": "Fc1ccccc1C(=O)N",
            "bza_Cl": "Clc1ccccc1C(=O)N",
            "bza_Me": "Cc1ccccc1C(=O)N",
            "bza_CF3": "FC(F)(F)c1ccccc1C(=O)N",
        },
        "c1ccccc1C(=O)N",
    )


@pytest.fixture(scope="session")
def scrambled_benzamides(benzamides: dict[str, Ligand]) -> dict[str, Ligand]:
    """The co-posed benzamides, each pushed into a different frame.

    Stands in for ligands prepared separately -- set up individually for ABFE runs, then
    written to mol2 from their own Amber topologies, each wherever its own box put it. The
    conformations are identical here and only the frames differ, which makes it the
    sharpest available test: perfect alignment is achievable, so any residual RMSD is the
    aligner's doing rather than the chemistry's.
    """
    return {
        name: scramble_frame(ligand, seed=index + 1) for index, (name, ligand) in enumerate(sorted(benzamides.items()))
    }


@pytest.fixture
def benzene() -> Ligand:
    """Benzene, with explicit hydrogens and a 3D conformer."""
    return make_ligand("c1ccccc1", "benzene")


@pytest.fixture
def default_network_options() -> NetworkOptions:
    """Default network options."""
    return NetworkOptions()


def make_transformation(
    source: str,
    target: str,
    *,
    cost: float = 1.0,
    n_atoms: int = 4,
    feasible: bool = True,
    kind: EdgeKind = EdgeKind.RBFE,
) -> Transformation:
    """Build a minimal :class:`Transformation` for planner tests.

    Carries no chemistry: the planner only reads endpoints, feasibility, and cost. A CBFE
    *kind* forces the empty-core mapping that kind requires.
    """
    if kind is EdgeKind.CBFE:
        mapping = AtomMapping(
            cc1=(),
            cc2=(),
            sc1=tuple(range(n_atoms)),
            sc2=tuple(range(n_atoms)),
            n_atoms_1=n_atoms,
            n_atoms_2=n_atoms,
            method="cbfe",
        )
    else:
        mapping = AtomMapping.from_core_pairs(
            {i: i for i in range(n_atoms)}, n_atoms_1=n_atoms, n_atoms_2=n_atoms, method="dummy"
        )
    score = (
        EdgeScore(total=cost, feasible=True, scorer="dummy")
        if feasible
        else EdgeScore.rejected(RejectionReason.SOFTCORE_TOO_LARGE, scorer="dummy")
    )
    return Transformation(source=source, target=target, mapping=mapping, score=score, kind=kind)

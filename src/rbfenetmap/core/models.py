"""Core data types: ligands, atom mappings, transformations, and networks.

Every type here is a frozen dataclass that validates its own invariants on construction.
That choice is deliberate and load-bearing: the soft-core repair, the scorers, and the
planner all assume a mapping is well formed, and the cheapest place to guarantee that is
at the boundary. An :class:`AtomMapping` that exists is a valid one.

The central type is :class:`AtomMapping`, which promotes the ``{"sc1", "sc2", "cc1",
"cc2"}`` dictionary contract used by ``amberstudio``'s ``BuildEdges`` into a real type
with enforced invariants. :meth:`AtomMapping.from_contract` and
:meth:`AtomMapping.to_contract` round-trip that dictionary exactly, which is the seam
that lets ``BuildEdges(mapping_method=...)`` call this package through a small shim.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx as nx
    from rdkit import Chem

    from rbfenetmap.core.options import NetworkOptions

__all__ = (
    "EDGE_SEPARATOR",
    "AtomMapping",
    "EdgeScore",
    "Ligand",
    "Network",
    "RejectionReason",
    "SoftcoreRepair",
    "Transformation",
    "edge_key",
    "parse_edge_key",
)

#: Separator between the two endpoints of an edge in file names and CLI arguments.
#: Ligand names are forbidden from containing it so ``parse_edge_key`` is unambiguous.
EDGE_SEPARATOR = "~"

_LIGAND_NAME_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")


class RejectionReason(str, Enum):
    """Why a candidate transformation is infeasible.

    A ``str`` enum so the values serialize to JSON as themselves and compare equal to
    plain strings, which keeps the exported network readable without a custom decoder.
    """

    SOFTCORE_FRAGMENTED = "softcore_fragmented"
    SOFTCORE_MULTIPLE_ATTACHMENTS = "softcore_multiple_attachments"
    SOFTCORE_TOO_LARGE = "softcore_too_large"
    SOFTCORE_FRACTION = "softcore_fraction_exceeded"
    CORE_TOO_SMALL = "core_too_small"
    NO_COMMON_CORE = "no_common_core"
    MCS_FRACTION_TOO_LOW = "mcs_fraction_too_low"
    NET_CHARGE_CHANGE = "net_charge_change"
    CORE_GEOMETRY_MISMATCH = "core_geometry_mismatch"
    REPAIR_DID_NOT_CONVERGE = "repair_did_not_converge"
    MAPPER_FAILED = "mapper_failed"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def edge_key(source: str, target: str) -> str:
    """Return the canonical ``"source~target"`` key for a directed edge."""
    return f"{source}{EDGE_SEPARATOR}{target}"


def parse_edge_key(key: str) -> tuple[str, str]:
    """Split a ``"source~target"`` key back into its endpoints.

    Parameters
    ----------
    key : str
        An edge key such as ``"lig_a~lig_b"``.

    Returns
    -------
    tuple[str, str]
        The ``(source, target)`` ligand names.

    Raises
    ------
    ValueError
        If *key* does not contain exactly one separator, or either side is empty.
        Ligand names cannot contain ``~`` (see :class:`Ligand`), so a key with two
        separators is user error rather than an ambiguous name.
    """
    parts = key.split(EDGE_SEPARATOR)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Malformed edge key {key!r}. Expected exactly one {EDGE_SEPARATOR!r} "
            f"separating two non-empty ligand names, e.g. 'lig_a{EDGE_SEPARATOR}lig_b'."
        )
    return parts[0], parts[1]


@dataclass(frozen=True)
class Ligand:
    """A network vertex: one molecule with a single 3D conformer.

    Parameters
    ----------
    name : str
        The vertex identifier, also the file-name token used by exporters. Restricted
        to ``[A-Za-z0-9_.+-]+`` so it is filesystem-safe and cannot contain the edge
        separator ``~``.
    mol : rdkit.Chem.Mol
        The molecule. Must carry explicit hydrogens and exactly one 3D conformer.
    charge : int
        Net formal charge, cached at construction so scorers never need to import
        RDKit. Use :meth:`from_mol` to compute it.
    source : pathlib.Path, optional
        Where the molecule was read from, for diagnostics.
    metadata : Mapping[str, Any], optional
        Free-form annotations carried through to exported networks.

    Raises
    ------
    ValueError
        If the name is malformed, the molecule is empty, it does not have exactly one
        3D conformer, or any atom carries implicit hydrogens.

    Notes
    -----
    The implicit-hydrogen check is not pedantry. Mappers address atoms positionally by
    index, and every invariant in :class:`AtomMapping` is stated over
    ``range(mol.GetNumAtoms())``. An implicit hydrogen is an atom that participates in
    the chemistry but has no index, so a mapping over a molecule with implicit Hs is
    silently incomplete: the soft-core region it describes omits atoms that a downstream
    engine will nonetheless have to transform. Requiring ``Chem.AddHs`` up front makes
    that impossible rather than merely unlikely.
    """

    name: str
    mol: "Chem.Mol"
    charge: int
    source: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Validate the name, conformer count, and hydrogen treatment."""
        if not _LIGAND_NAME_RE.match(self.name):
            raise ValueError(
                f"Invalid ligand name {self.name!r}. Names must match "
                f"{_LIGAND_NAME_RE.pattern} so they are filesystem-safe and cannot "
                f"contain the edge separator {EDGE_SEPARATOR!r}."
            )
        if self.mol is None or self.mol.GetNumAtoms() == 0:
            raise ValueError(f"Ligand {self.name!r} has no atoms.")
        if self.mol.GetNumConformers() != 1:
            raise ValueError(
                f"Ligand {self.name!r} has {self.mol.GetNumConformers()} conformers; exactly one is required. "
                "Geometry mappers align conformers pairwise and would otherwise silently use conformer 0."
            )
        if not self.mol.GetConformer().Is3D():
            raise ValueError(
                f"Ligand {self.name!r} has a 2D conformer. Embed it in 3D "
                "(e.g. rdkit.Chem.AllChem.EmbedMolecule) before planning a network."
            )
        implicit = [a.GetIdx() for a in self.mol.GetAtoms() if a.GetNumImplicitHs()]
        if implicit:
            raise ValueError(
                f"Ligand {self.name!r} has implicit hydrogens on atoms {implicit[:8]}"
                f"{'...' if len(implicit) > 8 else ''}. Call rdkit.Chem.AddHs(mol, addCoords=True) first: "
                "mappings index atoms positionally, so implicit hydrogens would be silently "
                "absent from both the common core and the soft-core region."
            )

    @classmethod
    def from_mol(cls, mol: "Chem.Mol", name: str, *, source: Path | None = None, **metadata: Any) -> "Ligand":
        """Build a :class:`Ligand`, computing the net charge from *mol*.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            Molecule with explicit hydrogens and one 3D conformer.
        name : str
            Vertex identifier.
        source : pathlib.Path, optional
            Origin of the molecule.
        **metadata
            Stored on the ligand and carried into exports.

        Returns
        -------
        Ligand
        """
        from rdkit import Chem

        return cls(
            name=name,
            mol=mol,
            charge=Chem.GetFormalCharge(mol),
            source=source,
            metadata=MappingProxyType(dict(metadata)),
        )

    @property
    def n_atoms(self) -> int:
        """Total atom count, including hydrogens."""
        return self.mol.GetNumAtoms()

    @property
    def heavy_indices(self) -> tuple[int, ...]:
        """Indices of the non-hydrogen atoms."""
        return tuple(a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() != 1)

    @property
    def n_heavy(self) -> int:
        """Number of non-hydrogen atoms."""
        return len(self.heavy_indices)

    @property
    def atom_names(self) -> tuple[str, ...]:
        """Per-atom names, used by the Amber exporter to build masks.

        Prefers an existing name from the Tripos mol2 property or PDB residue info;
        falls back to ``element + 1-based index`` (``C1``, ``N2``, ...), which is unique
        by construction.
        """
        names: list[str] = []
        for atom in self.mol.GetAtoms():
            name = ""
            if atom.HasProp("_TriposAtomName"):
                name = atom.GetProp("_TriposAtomName").strip()
            if not name:
                info = atom.GetPDBResidueInfo()
                if info is not None:
                    name = info.GetName().strip()
            if not name:
                name = f"{atom.GetSymbol().upper()}{atom.GetIdx() + 1}"
            names.append(name)
        return tuple(names)


@dataclass(frozen=True)
class AtomMapping:
    """A common-core / soft-core partition of two molecules.

    ``cc1[i]`` and ``cc2[i]`` are corresponding atoms -- the pairing is *positional*,
    not by sorted order of ``cc2``. ``sc1`` and ``sc2`` are the unmapped (soft-core)
    atoms on each side.

    Parameters
    ----------
    cc1, cc2 : tuple[int, ...]
        The common core on each side, paired positionally. ``cc1`` is sorted ascending.
    sc1, sc2 : tuple[int, ...]
        The soft-core atoms on each side, sorted ascending.
    n_atoms_1, n_atoms_2 : int
        Atom counts of the two molecules, so the partition can be checked for
        completeness without holding a reference to the molecules.
    method : str
        Name of the plugin that produced the mapping.

    Raises
    ------
    ValueError
        If any invariant below is violated.

    Notes
    -----
    The enforced invariants are:

    - no duplicate indices within any of the four tuples;
    - every index lies in ``range(n_atoms_k)``;
    - ``sc_k`` and ``cc_k`` are disjoint and together cover ``range(n_atoms_k)``
      (every atom is either transformed or held in common -- there is no third state);
    - ``len(cc1) == len(cc2)``;
    - ``cc2`` contains no duplicates, i.e. the correspondence is injective.

    The last two are worth stating explicitly because together they are exactly Amber's
    linear-scaling constraint, ``len(TI1) - len(SC1) == len(TI2) - len(SC2)``. Since
    ``len(TI_k) - len(SC_k)`` is just ``len(cc_k)``, an :class:`AtomMapping` that exists
    already satisfies it. The Amber exporter re-checks anyway, but only hand-authored
    maps can ever trip it.
    """

    cc1: tuple[int, ...]
    cc2: tuple[int, ...]
    sc1: tuple[int, ...]
    sc2: tuple[int, ...]
    n_atoms_1: int
    n_atoms_2: int
    method: str = "unknown"

    def __post_init__(self) -> None:
        """Enforce every invariant documented on the class."""
        from rbfenetmap.core.validate import validate_mapping

        validate_mapping(self)

    # -- constructors -------------------------------------------------------------

    @classmethod
    def from_contract(
        cls, contract: Mapping[str, Sequence[int]], *, n_atoms_1: int, n_atoms_2: int, method: str = "unknown"
    ) -> "AtomMapping":
        """Build from the ``{"sc1", "sc2", "cc1", "cc2"}`` dictionary contract.

        This is the amberstudio ``BuildEdges`` interchange format. ``cc1``/``cc2`` are
        taken as already paired positionally and are *not* re-sorted independently --
        doing so would silently scramble the correspondence.

        Parameters
        ----------
        contract : Mapping[str, Sequence[int]]
            Must contain the keys ``sc1``, ``sc2``, ``cc1``, ``cc2``.
        n_atoms_1, n_atoms_2 : int
            Atom counts of the two molecules.
        method : str, optional
            Name recorded as the producing method.

        Returns
        -------
        AtomMapping
        """
        missing = {"sc1", "sc2", "cc1", "cc2"} - set(contract)
        if missing:
            raise ValueError(f"Mapping contract is missing key(s) {sorted(missing)}.")
        pairs = sorted(zip((int(i) for i in contract["cc1"]), (int(i) for i in contract["cc2"])))
        return cls(
            cc1=tuple(a for a, _ in pairs),
            cc2=tuple(b for _, b in pairs),
            sc1=tuple(sorted(int(i) for i in contract["sc1"])),
            sc2=tuple(sorted(int(i) for i in contract["sc2"])),
            n_atoms_1=n_atoms_1,
            n_atoms_2=n_atoms_2,
            method=method,
        )

    @classmethod
    def from_core_pairs(
        cls,
        core: Mapping[int, int] | Sequence[tuple[int, int]],
        *,
        n_atoms_1: int,
        n_atoms_2: int,
        method: str = "unknown",
    ) -> "AtomMapping":
        """Build from a ``{idx1: idx2}`` correspondence, inferring the soft-core.

        Every atom not appearing in *core* becomes soft-core on its side. This is the
        usual entry point for a mapper, which naturally produces a correspondence rather
        than a four-way partition.

        Parameters
        ----------
        core : Mapping[int, int] or Sequence[tuple[int, int]]
            The common-core correspondence from molecule 1 to molecule 2.
        n_atoms_1, n_atoms_2 : int
            Atom counts of the two molecules.
        method : str, optional
            Name recorded as the producing method.

        Returns
        -------
        AtomMapping
        """
        items = sorted(core.items() if isinstance(core, Mapping) else core)
        cc1 = tuple(int(a) for a, _ in items)
        cc2 = tuple(int(b) for _, b in items)
        mapped1, mapped2 = set(cc1), set(cc2)
        return cls(
            cc1=cc1,
            cc2=cc2,
            sc1=tuple(i for i in range(n_atoms_1) if i not in mapped1),
            sc2=tuple(i for i in range(n_atoms_2) if i not in mapped2),
            n_atoms_1=n_atoms_1,
            n_atoms_2=n_atoms_2,
            method=method,
        )

    # -- accessors ----------------------------------------------------------------

    def to_contract(self) -> dict[str, tuple[int, ...]]:
        """Return the ``{"sc1", "sc2", "cc1", "cc2"}`` dictionary contract.

        Byte-for-byte what amberstudio's ``cartograph_mapping_method`` returns, so a
        shim on the amberstudio side can hand this straight to ``BuildEdges``. That shim
        also absorbs ``BuildEdges``' two unused ``parmed.Structure`` positionals -- which
        is why ParmEd is not a dependency of this package.
        """
        return {"sc1": self.sc1, "sc2": self.sc2, "cc1": self.cc1, "cc2": self.cc2}

    @property
    def forward(self) -> dict[int, int]:
        """The common-core correspondence as ``{idx1: idx2}``."""
        return dict(zip(self.cc1, self.cc2))

    @property
    def reverse(self) -> dict[int, int]:
        """The common-core correspondence as ``{idx2: idx1}``."""
        return dict(zip(self.cc2, self.cc1))

    @property
    def n_common_core(self) -> int:
        """Number of mapped atom pairs."""
        return len(self.cc1)

    @property
    def n_softcore_1(self) -> int:
        """Soft-core atom count on side 1, hydrogens included."""
        return len(self.sc1)

    @property
    def n_softcore_2(self) -> int:
        """Soft-core atom count on side 2, hydrogens included."""
        return len(self.sc2)

    def swapped(self) -> "AtomMapping":
        """Return the mapping with the two sides exchanged.

        Used when a transformation is reoriented. The positional pairing is preserved by
        re-sorting on the new side 1.
        """
        pairs = sorted(zip(self.cc2, self.cc1))
        return AtomMapping(
            cc1=tuple(a for a, _ in pairs),
            cc2=tuple(b for _, b in pairs),
            sc1=self.sc2,
            sc2=self.sc1,
            n_atoms_1=self.n_atoms_2,
            n_atoms_2=self.n_atoms_1,
            method=self.method,
        )


@dataclass(frozen=True)
class SoftcoreRepair:
    """Outcome of the soft-core connectivity repair for one transformation.

    Parameters
    ----------
    applied : bool
        Whether any atom was demoted from the common core.
    n_fragments_before, n_fragments_after : tuple[int, int]
        Soft-core connected-component counts per side, before and after repair. After a
        successful repair both entries are ``0`` or ``1`` -- the constraint is *at most*
        one region, and an empty soft-core (zero regions) is perfectly legal.
    demoted_1, demoted_2 : tuple[int, ...]
        Atoms moved from the common core into the soft-core, per side.
    iterations : int
        Repair loop iterations consumed.
    rejection : RejectionReason, optional
        Set when the repair concluded the edge is infeasible.
    trace : tuple[str, ...]
        Human-readable log of each repair step, surfaced by ``rbfenet inspect``. This is
        the only window into why a given edge grew the soft-core it did, so it is
        retained even on success.
    """

    applied: bool = False
    n_fragments_before: tuple[int, int] = (0, 0)
    n_fragments_after: tuple[int, int] = (0, 0)
    demoted_1: tuple[int, ...] = ()
    demoted_2: tuple[int, ...] = ()
    iterations: int = 0
    rejection: RejectionReason | None = None
    trace: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        """True when the repair produced at most one soft-core region per side."""
        return self.rejection is None

    @property
    def n_demoted(self) -> int:
        """Total atoms demoted across both sides."""
        return len(self.demoted_1) + len(self.demoted_2)


@dataclass(frozen=True)
class EdgeScore:
    """The cost of a candidate transformation. Lower is better.

    Parameters
    ----------
    total : float
        The scalar cost, ``math.inf`` when the edge is infeasible.
    feasible : bool
        Whether the edge may be selected at all.
    descriptors : Mapping[str, float]
        Raw descriptor values, as produced by
        :func:`rbfenetmap.core.descriptors.compute_descriptors`.
    contributions : Mapping[str, float]
        Weighted terms, summing to *total* for feasible edges.
    rejections : tuple[RejectionReason, ...]
        Why the edge is infeasible; empty when it is not.
    scorer : str
        Name of the scoring plugin.

    Notes
    -----
    Feasibility and cost are kept strictly separate. Rejection is *structural* and
    originates only in the mapper, the repair, or validation -- never from a weighted sum
    crossing a threshold. A merely bad edge has a large finite ``total`` and stays in the
    candidate pool where the planner can still use it if the alternative is a
    disconnected network. This is why infeasible candidates are retained on
    :attr:`Network.candidates` rather than dropped: they are the audit trail that
    explains a disconnection.
    """

    total: float = math.inf
    feasible: bool = False
    descriptors: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    contributions: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    rejections: tuple[RejectionReason, ...] = ()
    scorer: str = "unknown"

    def __post_init__(self) -> None:
        """Check that feasibility, cost, and rejections agree with one another."""
        if self.feasible and self.rejections:
            raise ValueError(f"EdgeScore is marked feasible but carries rejections {list(self.rejections)}.")
        if self.feasible and not math.isfinite(self.total):
            raise ValueError("EdgeScore is marked feasible but its total is not finite.")
        if not self.feasible and not self.rejections:
            raise ValueError("EdgeScore is marked infeasible but records no RejectionReason.")

    @classmethod
    def rejected(cls, *reasons: RejectionReason, scorer: str = "unknown", **kwargs: Any) -> "EdgeScore":
        """Build an infeasible score carrying *reasons*."""
        if not reasons:
            raise ValueError("EdgeScore.rejected requires at least one RejectionReason.")
        return cls(total=math.inf, feasible=False, rejections=tuple(reasons), scorer=scorer, **kwargs)


@dataclass(frozen=True)
class Transformation:
    """A candidate or selected network edge: an alchemical transformation.

    Parameters
    ----------
    source, target : str
        Ligand names. Must differ.
    mapping : AtomMapping
        The common-core / soft-core partition, after repair.
    repair : SoftcoreRepair
        What the repair did to get there.
    score : EdgeScore
        The cost and feasibility verdict.
    """

    source: str
    target: str
    mapping: AtomMapping
    repair: SoftcoreRepair = field(default_factory=SoftcoreRepair)
    score: EdgeScore = field(default_factory=lambda: EdgeScore.rejected(RejectionReason.MAPPER_FAILED))

    def __post_init__(self) -> None:
        """Reject self-loops, which are meaningless as alchemical transformations."""
        if self.source == self.target:
            raise ValueError(f"Transformation source and target are both {self.source!r}; self-loops are not edges.")

    @property
    def key(self) -> str:
        """The directed ``"source~target"`` key."""
        return edge_key(self.source, self.target)

    @property
    def unordered_key(self) -> tuple[str, str]:
        """The endpoints as a sorted pair.

        Selection is undirected -- the free energy of a transformation is antisymmetric,
        so ``a -> b`` and ``b -> a`` are the same experiment. Direction only matters when
        writing files and assigning ``timask``/``scmask``. Keying the candidate pool by
        this rather than by :attr:`key` is what keeps the two from being double-counted.
        """
        return tuple(sorted((self.source, self.target)))  # type: ignore[return-value]

    @property
    def feasible(self) -> bool:
        """Whether this edge may be selected."""
        return self.score.feasible

    def reversed(self) -> "Transformation":
        """Return the transformation with its direction flipped.

        The mapping, fragment counts, and demoted-atom lists are all swapped along with
        the endpoints, so the result stays internally consistent rather than describing
        the old direction under new labels.

        The repair *trace* is free-form text written during the repair and cannot be
        rewritten, so a note is prepended recording that its "side 1" and "side 2" refer
        to the original orientation. Without it a reader comparing the trace against the
        edge's reported soft-core sizes sees them transposed and reasonably concludes one
        of the two is wrong.
        """
        trace = self.repair.trace
        if trace:
            trace = (f"(orientation flipped: sides below refer to {self.source}~{self.target})", *trace)
        return Transformation(
            source=self.target,
            target=self.source,
            mapping=self.mapping.swapped(),
            repair=SoftcoreRepair(
                applied=self.repair.applied,
                n_fragments_before=self.repair.n_fragments_before[::-1],
                n_fragments_after=self.repair.n_fragments_after[::-1],
                demoted_1=self.repair.demoted_2,
                demoted_2=self.repair.demoted_1,
                iterations=self.repair.iterations,
                rejection=self.repair.rejection,
                trace=trace,
            ),
            score=self.score,
        )


@dataclass(frozen=True)
class Network:
    """A planned perturbation network.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
        The vertices, in insertion order.
    edges : tuple[Transformation, ...]
        The selected transformations.
    candidates : tuple[Transformation, ...]
        Every transformation that was scored, feasible or not. Retained as an audit
        trail: when the planner reports a disconnection, this is what explains it.
    planner : str
        Name of the planning plugin.
    options : NetworkOptions, optional
        The options the network was planned under.
    unmet_constraints : tuple[str, ...]
        Best-effort constraints that could not be satisfied (for example a requested
        ``edges_per_ligand`` the candidate pool could not support). Hard conflicts raise
        :class:`~rbfenetmap.core.exceptions.NetworkPlanError` instead of landing here.
    """

    ligands: Mapping[str, Ligand]
    edges: tuple[Transformation, ...] = ()
    candidates: tuple[Transformation, ...] = ()
    planner: str = "unknown"
    options: "NetworkOptions | None" = None
    unmet_constraints: tuple[str, ...] = ()

    def to_networkx(self) -> "nx.Graph":
        """Return the selected edges as an undirected :class:`networkx.Graph`.

        Nodes are ligand names; each edge carries ``transformation`` and ``weight``
        (the score total) attributes.
        """
        import networkx as nx

        graph: nx.Graph = nx.Graph()
        graph.add_nodes_from(self.ligands)
        for edge in self.edges:
            graph.add_edge(edge.source, edge.target, transformation=edge, weight=edge.score.total)
        return graph

    def validate(self, *, require_connected: bool = True) -> None:
        """Check the network is structurally sound.

        Parameters
        ----------
        require_connected : bool, optional
            Whether to require that the selected edges span every ligand.

        Raises
        ------
        ValueError
            If an endpoint is unknown, a self-loop is present, an unordered pair appears
            twice, or (when required) the network is disconnected.
        """
        import networkx as nx

        seen: set[tuple[str, str]] = set()
        for edge in self.edges:
            for endpoint in (edge.source, edge.target):
                if endpoint not in self.ligands:
                    raise ValueError(f"Edge {edge.key!r} references unknown ligand {endpoint!r}.")
            if edge.source == edge.target:
                raise ValueError(f"Edge {edge.key!r} is a self-loop.")
            if edge.unordered_key in seen:
                raise ValueError(f"Edge {edge.key!r} duplicates an already-selected pair {edge.unordered_key}.")
            seen.add(edge.unordered_key)

        if require_connected and len(self.ligands) > 1:
            graph = self.to_networkx()
            if not nx.is_connected(graph):
                components = [sorted(c) for c in nx.connected_components(graph)]
                raise ValueError(
                    f"Network is disconnected: {len(components)} components {components}. "
                    "Plan with require_connected=False if that is intended."
                )

    @property
    def rejected(self) -> tuple[Transformation, ...]:
        """Candidates that were found infeasible."""
        return tuple(c for c in self.candidates if not c.feasible)

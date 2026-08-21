"""What an intermediate generator proposes, and how a proposal becomes a ligand.

A generator invents a molecule that sits *between* two ligands the mapper cannot relate
cheaply, turning one hard edge into two easier ones. This module holds the vocabulary it
speaks in -- :class:`ProposedMolecule`, :class:`ProposedLink`,
:class:`IntermediateProposal` -- plus the naming rule and the one function that turns a
proposal into a real :class:`~rbfenetmap.core.models.Ligand`.

These types live here rather than in :mod:`rbfenetmap.core.models` on purpose: they carry
RDKit molecules and an atom-level correspondence, and ``models`` is deliberately free of
chemistry so the data model stays importable and testable without one.

Three constraints the types enforce, and why
--------------------------------------------

**A proposed molecule carries no conformer.** Posing is centralised in
:mod:`rbfenetmap.core.posing`, for the same reason :mod:`rbfenetmap.core.descriptors`
centralises scoring inputs: a generator that poses its own output makes the quality of
every intermediate depend on which generator produced it, and makes the pose
unauditable. A generator that *knows* where its atoms belong says so through a complete
:attr:`ProposedMolecule.parent_atom_map`, which is strictly more useful than coordinates
-- the poser can act on a correspondence, and can tell you which parent every atom came
from afterwards.

**A link's hint is advisory and can never become an**
:attr:`~rbfenetmap.core.models.EdgeScore.total`. A generator may know that one of its
proposals is more promising than another, and that is worth recording; it is not a cost.
Letting a hint reach a score would put a second scoring system in the package on a
different scale, which is the dual of the rule that a scorer must not invent rejections.

**A proposal's rejection is a plain string.**
:class:`~rbfenetmap.core.models.RejectionReason` is the vocabulary of *edge* feasibility.
Reusing it here would make ``core_geometry_mismatch`` mean two different things depending
on which object it was read from.

Naming
------

:func:`intermediate_name` is content-addressed -- ``int_{a}_{b}_{sha1(smiles)[:6]}`` with
the parents sorted -- rather than counter-based. The same intermediate is often reachable
from the same gap by two routes, and a hash makes that a natural dedupe instead of a pair
of near-identical ligands; it also keeps a run reproducible under ``jobs > 1``, where a
counter's value depends on which worker finished first.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence

from rbfenetmap.core.models import IntermediateRecord, Ligand, LigandProvenance
from rbfenetmap.core.posing import POSE_RMSD_FACTOR, PoseDonor, PoseResult, pose_intermediate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rdkit import Chem

    from rbfenetmap.core.options import MappingOptions, SoftcorePolicy

__all__ = (
    "INTERMEDIATE_KIND",
    "INTERMEDIATE_NAME_PREFIX",
    "INTERMEDIATE_MODES",
    "IntermediateMode",
    "IntermediateOptions",
    "IntermediateProposal",
    "ProposedLink",
    "ProposedMolecule",
    "describe_intermediate_attempts",
    "intermediate_name",
    "reserve_intermediate_names",
    "synthesize_ligand",
)

#: The :attr:`~rbfenetmap.core.models.LigandProvenance.kind` this package writes.
INTERMEDIATE_KIND = "intermediate"

#: Prefix every generated ligand name carries. Matches ``Ligand``'s name pattern and
#: contains no :data:`~rbfenetmap.core.models.EDGE_SEPARATOR`, so a generated name
#: survives :func:`~rbfenetmap.core.models.edge_key` and
#: :func:`~rbfenetmap.core.models.parse_edge_key` unchanged.
INTERMEDIATE_NAME_PREFIX = "int_"

_HASH_LENGTH = 6

#: How freely the pipeline may invent ligands.
#:
#: There is deliberately no ``"all"``. For :attr:`NetworkOptions.cbfe_mode` that member
#: means "every edge is counterpoised", which is a coherent request; "every pair gets an
#: intermediate" is not one, because most pairs already have a perfectly good direct edge
#: and inventing a molecule for them adds two calculations to avoid nothing.
INTERMEDIATE_MODES: tuple[str, ...] = ("off", "bridge", "gaps")

IntermediateMode = Literal["off", "bridge", "gaps"]


@dataclass(frozen=True)
class IntermediateOptions:
    """Whether to invent ligands, how many, and how hard to try posing them.

    Parameters
    ----------
    mode : {"off", "bridge", "gaps"}, optional
        Which gaps are offered to the generator.

        - ``"off"`` (default) -- none. No generator is even constructed, so a run that
          does not ask for intermediates never imports one.
        - ``"bridge"`` -- only pairs whose endpoints fall in different components of the
          feasible candidate graph. This is the mode that turns a hard connectivity
          failure into a planned network, and it is the analogue of
          ``cbfe_mode="bridge"``.
        - ``"gaps"`` -- everything ``"bridge"`` does, and additionally infeasible pairs
          *inside* a component. Those are already reachable by some path, so an
          intermediate there buys accuracy rather than connectivity.

        Forced pairs with no feasible mapping are offered under both non-off modes: the
        user demanded that comparison, and an intermediate is the only way to keep it
        relative.
    generator : str, optional
        Registered name of the generator plugin to construct. Resolved lazily, and only
        when *mode* is not ``"off"``.
    max_intermediates : int, optional
        Cap on how many ligands one run may invent in total. ``None`` means only the
        per-gap cap and the edge budget constrain it.
    max_gaps : int, optional
        Cap on how many gaps are offered to the generator, taken in decreasing
        fingerprint similarity. ``None`` offers every gap. This is the knob that keeps
        generation off the tail of an O(n^2) rejection list, where the pairs are least
        similar and least likely to be bridgeable anyway.
    max_molecules : int, optional
        Cap on how many molecules a generator may propose for one gap. A generator that
        enumerates every single-substituent swap on a heavily decorated pair can produce
        dozens; each one costs an embedding and a minimisation, so the cap is the knob
        that keeps generation from dominating a run.
    seed : int, optional
        Base RDKit random seed for posing, so a run is reproducible.
    max_pose_attempts : int, optional
        Embedding attempts spent on one molecule before giving up on it.
    pose_rmsd_factor : float, optional
        Fraction of ``SoftcorePolicy.core_rmsd_threshold`` an accepted pose must stay
        under. See :data:`~rbfenetmap.core.posing.POSE_RMSD_FACTOR`.
    min_link_score : float, optional
        Lowest link score a generator may consider worth proposing, on a ``(0, 1]``
        similarity scale where 1 is "no atoms change at all".
    max_dist : int, optional
        Longest source-to-target path, in links, the generator may propose. At least 2:
        a one-link path *is* the direct transformation that was already rejected.
    max_cycle : int, optional
        Largest cycle the generator may build to give a proposed link a second,
        independent route. Cycles are what turn a chain of intermediates into a network
        with a closure error to check.
    max_subgraph_dist : int, optional
        How far from either parent, in links, a molecule may sit and still be considered
        for the subnetwork. Bounds the search, and must be at least *max_dist*.
    beta : float, optional
        Decay rate of the exponential link score, in inverse heavy atoms. The published
        default of 0.1 is the same constant LOMAP's similarity uses, and it is what makes
        :attr:`min_link_score` ``0.2`` mean "at most about sixteen heavy atoms change".

    Raises
    ------
    ValueError
        If *mode* is unknown, or any budget is out of range.

    Notes
    -----
    Reachable as :attr:`~rbfenetmap.core.options.NetworkOptions.intermediates`, nested the
    way :class:`~rbfenetmap.core.options.SoftcorePolicy` is. It was deliberately left off
    ``NetworkOptions`` while nothing read it, because a field on the serialized options
    block that no stage consumes is a knob that lies. The pipeline now consumes every one
    of these, so the nesting is what makes a planned network state the settings that
    invented its vertices.

    The last five fields keep the names and the published defaults of the PairMap
    constants (``MIN_SCORE``, ``MAX_DIST``, ``MAX_CYCLE``, ``MAX_SUBGRAPH_DIST``,
    ``beta``) from Furui *et al.*, *J. Chem. Inf. Model.* **2025**, 65, 705-721
    (`doi:10.1021/acs.jcim.4c01634 <https://doi.org/10.1021/acs.jcim.4c01634>`_), so a
    reader can grep them against the paper. They live on the shared options object rather
    than on the generator because a plugin's parameters have to survive serialization to
    make an invented ligand reproducible, and
    :meth:`~rbfenetmap.core.meta.intermediates.AbstractIntermediateGenerator.describe_parameters`
    is a report, not a record. A generator that does not search a subnetwork simply
    ignores them.
    """

    mode: IntermediateMode = "off"
    generator: str = "pairmap"
    max_intermediates: int | None = None
    max_gaps: int | None = None
    max_molecules: int = 4
    seed: int = 0xF00D
    max_pose_attempts: int = 10
    pose_rmsd_factor: float = POSE_RMSD_FACTOR
    min_link_score: float = 0.2
    max_dist: int = 3
    max_cycle: int = 4
    max_subgraph_dist: int = 4
    beta: float = 0.1

    def __post_init__(self) -> None:
        """Reject nonsensical budgets up front."""
        if self.mode not in INTERMEDIATE_MODES:
            raise ValueError(f"mode must be one of {list(INTERMEDIATE_MODES)}; got {self.mode!r}.")
        if not self.generator:
            raise ValueError("generator must name a registered intermediate generator plugin.")
        if self.max_intermediates is not None and self.max_intermediates < 1:
            raise ValueError("max_intermediates must be at least 1 when set.")
        if self.max_gaps is not None and self.max_gaps < 1:
            raise ValueError("max_gaps must be at least 1 when set.")
        if self.max_molecules < 1:
            raise ValueError("max_molecules must be at least 1.")
        if self.max_pose_attempts < 1:
            raise ValueError("max_pose_attempts must be at least 1.")
        if not 0.0 < self.pose_rmsd_factor <= 1.0:
            raise ValueError("pose_rmsd_factor must lie in (0, 1].")
        if not 0.0 < self.min_link_score <= 1.0:
            raise ValueError("min_link_score must lie in (0, 1]; it is a similarity, not a cost.")
        if self.max_dist < 2:
            raise ValueError(
                "max_dist must be at least 2. A path of one link from source to target *is* the direct "
                "transformation that was already rejected, so a shorter bound admits no intermediate at all."
            )
        if self.max_cycle < 3:
            raise ValueError("max_cycle must be at least 3; a shorter cycle is a repeated edge.")
        if self.max_subgraph_dist < self.max_dist:
            raise ValueError(
                f"max_subgraph_dist ({self.max_subgraph_dist}) must be at least max_dist ({self.max_dist}); "
                "the subnetwork has to be able to hold the optimal path it is built around."
            )
        if self.beta <= 0.0:
            raise ValueError("beta must be positive; it is the decay rate of an exponential link score.")

    @property
    def enabled(self) -> bool:
        """Whether generation runs at all."""
        return self.mode != "off"

    @property
    def bridges_components(self) -> bool:
        """Whether cross-component gaps are offered to the generator."""
        return self.mode in ("bridge", "gaps")

    @property
    def fills_internal_gaps(self) -> bool:
        """Whether infeasible pairs inside one component are offered too."""
        return self.mode == "gaps"


@dataclass(frozen=True)
class ProposedMolecule:
    """One molecule a generator suggests inserting into a gap.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The molecule, **without a conformer**. Any conformers present are stripped at
        construction rather than rejected, so a generator that happened to build its
        molecule from a posed parent does not have to remember to clear them -- but the
        coordinates are discarded either way, because posing is centralised.
    parents : tuple[str, ...]
        Names of the ligands it was derived from, sorted at construction.
    parent_atom_map : Mapping[str, Mapping[int, int]]
        ``{parent name: {proposed atom index: parent atom index}}``. May be empty, or
        cover only some parents; the poser falls back to an MCS search for whatever is
        missing and records that it had to.
    hint : float, optional
        The generator's own ordering preference among its proposals, lower being more
        promising. Advisory only -- see the module docstring.
    detail : Mapping[str, Any], optional
        Free-form annotations carried into the ligand's provenance.

    Raises
    ------
    ValueError
        If *parents* is empty, or *parent_atom_map* names a parent that is not in
        *parents*.
    """

    mol: "Chem.Mol"
    parents: tuple[str, ...]
    parent_atom_map: Mapping[str, Mapping[int, int]] = field(default_factory=lambda: MappingProxyType({}))
    hint: float | None = None
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Sort the parents, strip conformers, and check the map's keys."""
        object.__setattr__(self, "parents", tuple(sorted(self.parents)))
        if not self.parents:
            raise ValueError("A ProposedMolecule must name at least one parent ligand.")
        unknown = sorted(set(self.parent_atom_map) - set(self.parents))
        if unknown:
            raise ValueError(f"parent_atom_map names non-parent ligand(s) {unknown}; parents are {list(self.parents)}.")
        if self.mol is not None and self.mol.GetNumConformers():
            stripped = type(self.mol)(self.mol)
            stripped.RemoveAllConformers()
            object.__setattr__(self, "mol", stripped)


@dataclass(frozen=True)
class ProposedLink:
    """A sub-edge the generator expects its molecule to make possible.

    Parameters
    ----------
    source, target : str
        Endpoints. One is normally the proposed molecule, the other a parent.
    hint : float, optional
        How promising the generator believes the link to be, lower being better.
    detail : Mapping[str, Any], optional
        Free-form annotations.

    Notes
    -----
    **The hint can never become an**
    :attr:`~rbfenetmap.core.models.EdgeScore.total`. Nothing in this package reads it as a
    cost, and nothing should: a generator's opinion about its own output is on a scale
    only that generator knows, so promoting it to a score would mean two incomparable
    numbers competing inside the planner's edge ordering. The link exists to say *which
    edges are worth evaluating*; the scorer says what they cost.
    """

    source: str
    target: str
    hint: float | None = None
    detail: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class IntermediateProposal:
    """A generator's complete answer for one gap.

    Parameters
    ----------
    source, target : str
        The gap the generator was asked about.
    generator : str
        Registered plugin name.
    molecules : tuple[ProposedMolecule, ...], optional
        What it suggests inserting. Empty is a legitimate answer.
    links : tuple[ProposedLink, ...], optional
        Sub-edges it expects to become feasible.
    rejection : str, optional
        Why it proposed nothing. A plain ``str``, never a
        :class:`~rbfenetmap.core.models.RejectionReason`.
    trace : tuple[str, ...], optional
        Human-readable log of what it tried.

    Raises
    ------
    ValueError
        If the two endpoints are the same ligand.
    """

    source: str
    target: str
    generator: str
    molecules: tuple[ProposedMolecule, ...] = ()
    links: tuple[ProposedLink, ...] = ()
    rejection: str | None = None
    trace: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a proposal for a gap between a ligand and itself."""
        if self.source == self.target:
            raise ValueError(f"An intermediate proposal needs two distinct ligands; got {self.source!r} twice.")

    @property
    def proposed(self) -> bool:
        """Whether the generator suggested anything at all."""
        return bool(self.molecules)


def intermediate_name(parents: Sequence[str], mol: "Chem.Mol") -> str:
    """Return the content-addressed name for an invented molecule.

    Parameters
    ----------
    parents : Sequence[str]
        Names of the ligands it bridges. Sorted here, so the caller need not.
    mol : rdkit.Chem.Mol
        The molecule. Only its canonical SMILES is used, so a name is stable across a
        re-embedding and across atom reordering.

    Returns
    -------
    str
        ``int_{a}_{b}_{sha1(canonical_smiles)[:6]}``.

    Raises
    ------
    ValueError
        If *parents* is empty, or the resulting name would not be a legal ligand name.

    Notes
    -----
    Content-addressed rather than counter-based, and the parents are part of the address
    rather than only the structure. The hash makes the same molecule proposed twice for
    the same gap collapse to one ligand -- which is the common case, since a gap is
    usually reachable from either end. The parent tokens stay in because two gaps that
    happen to want the same molecule want it for different reasons, and a name that hid
    that would leave two different provenances competing for one vertex.

    Hydrogens are suppressed before canonicalisation. A generator that hands over a
    molecule with explicit hydrogens and one that leaves them implicit are proposing the
    same thing, and they must not get different names for it.
    """
    from rdkit import Chem

    from rbfenetmap.core.models import _LIGAND_NAME_RE

    if not parents:
        raise ValueError("intermediate_name needs at least one parent name.")
    try:
        bare = Chem.RemoveHs(Chem.Mol(mol))
    except Exception:  # noqa: BLE001 - a molecule too broken to strip is named from itself
        bare = mol
    digest = hashlib.sha1(Chem.MolToSmiles(bare).encode("utf-8")).hexdigest()[:_HASH_LENGTH]
    name = INTERMEDIATE_NAME_PREFIX + "_".join((*sorted(parents), digest))
    if not _LIGAND_NAME_RE.match(name):
        raise ValueError(
            f"Generated intermediate name {name!r} is not a legal ligand name. Parent names are "
            "restricted to the same character class, so this means one of them is not."
        )
    return name


def reserve_intermediate_names(names: Iterable[str], *, enabled: bool) -> None:
    """Refuse user ligand names that would collide with generated ones.

    Parameters
    ----------
    names : Iterable[str]
        The user's ligand names.
    enabled : bool
        Whether intermediate generation is switched on for this run.

    Raises
    ------
    ValueError
        If *enabled* and any name starts with :data:`INTERMEDIATE_NAME_PREFIX`.

    Notes
    -----
    The prefix is reserved **only when the feature is on**, which is the whole point of
    taking *enabled* rather than reading it from a global. A user with a ligand honestly
    called ``int_3`` is running a plain network plan that cannot possibly generate a
    conflicting name, and refusing it would be a compatibility break bought for nothing.
    """
    if not enabled:
        return
    clashes = sorted(name for name in names if name.startswith(INTERMEDIATE_NAME_PREFIX))
    if clashes:
        raise ValueError(
            f"Ligand name(s) {clashes} start with the reserved prefix {INTERMEDIATE_NAME_PREFIX!r}, which "
            "intermediate generation uses for the molecules it invents. Rename them, or plan without "
            "intermediate generation, where the prefix is not reserved."
        )


def synthesize_ligand(
    proposed: ProposedMolecule,
    parents: Mapping[str, Ligand],
    *,
    generator: str,
    softcore: "SoftcorePolicy | None" = None,
    options: IntermediateOptions | None = None,
    mapping_options: "MappingOptions | None" = None,
) -> tuple[Ligand | None, PoseResult]:
    """Pose *proposed* against its parents and wrap the result in a ligand.

    Parameters
    ----------
    proposed : ProposedMolecule
        The conformer-free molecule a generator suggested.
    parents : Mapping[str, Ligand]
        The real ligands, keyed by name. Must contain every name in
        :attr:`ProposedMolecule.parents`.
    generator : str
        Registered name of the generator, recorded in the provenance.
    softcore : SoftcorePolicy, optional
        Supplies ``core_rmsd_threshold``. Defaults are used when omitted.
    options : IntermediateOptions, optional
        Supplies the seed, the attempt budget, and the RMSD factor.
    mapping_options : MappingOptions, optional
        Only consulted by the poser's MCS fallback.

    Returns
    -------
    ligand : Ligand or None
        ``None`` when the pose was rejected; the :class:`PoseResult` says why.
    result : PoseResult
        Always returned, successful or not, because the trace is what a user needs when
        an intermediate does not appear.

    Raises
    ------
    KeyError
        If *parents* does not contain a name the proposal claims. That is a caller bug
        rather than a chemistry outcome, so it raises where posing failures do not.

    Notes
    -----
    The parents are handed to the poser in sorted-name order, which is also the order
    :func:`intermediate_name` uses. That matters because donors are consumed in priority
    order: an unordered iteration would let two runs of the same input take scaffold
    coordinates from different parents and produce two slightly different poses under the
    same name.
    """
    from rbfenetmap.core.options import SoftcorePolicy

    options = options or IntermediateOptions()
    softcore = softcore or SoftcorePolicy()

    donors = [
        PoseDonor(name=name, mol=parents[name].mol, atom_map=proposed.parent_atom_map.get(name))
        for name in proposed.parents
    ]
    result = pose_intermediate(
        proposed.mol,
        donors,
        core_rmsd_threshold=softcore.core_rmsd_threshold,
        rmsd_factor=options.pose_rmsd_factor,
        seed=options.seed,
        max_attempts=options.max_pose_attempts,
        mapping_options=mapping_options,
    )
    if result.mol is None:
        return None, result

    provenance = LigandProvenance(
        kind=INTERMEDIATE_KIND,
        generator=generator,
        parents=proposed.parents,
        pose_method=result.method,
        pose_rmsd=result.rmsd,
        detail=MappingProxyType({**dict(proposed.detail), **dict(result.detail)}),
    )
    ligand = Ligand.synthesized(result.mol, intermediate_name(proposed.parents, result.mol), provenance)
    return ligand, result


def describe_intermediate_attempts(records: Sequence["IntermediateRecord"]) -> str:
    """Summarise what generation was asked to do and what came of it.

    Parameters
    ----------
    records : Sequence[IntermediateRecord]
        One per gap *attempted*, in the order they were attempted.

    Returns
    -------
    str
        A short paragraph naming each gap offered to the generator and why it was
        refused, or the empty string when *records* is empty.

    Notes
    -----
    The paragraph exists to discharge the same obligation the CBFE branch of
    :func:`~rbfenetmap.plugins.planners.mst_planner._describe_disconnection` discharges:
    "disconnected" alone tells a user nothing they can act on, and a user who switched
    generation *on* and still got a disconnection needs to know whether their gaps were
    never offered, offered and declined, or bridged by molecules that failed the geometry
    gate. Those three call for entirely different responses -- raise ``max_gaps``, change
    generator, loosen ``core_rmsd_threshold`` -- and only this record distinguishes them.
    """
    if not records:
        return ""
    accepted = [record for record in records if record.accepted]
    lines = [
        f"  Intermediate generation was offered {len(records)} gap(s) and bridged {len(accepted)}:",
        *(
            f"    {record.source}~{record.target}: "
            + (f"accepted {list(record.names)}" if record.accepted else f"refused ({record.rejection or 'unknown'})")
            for record in records
        ),
    ]
    return "\n".join(lines)

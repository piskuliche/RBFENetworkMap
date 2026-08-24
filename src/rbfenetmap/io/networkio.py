"""JSON serialization of a planned network.

The network file is the package's own interchange format, and it is deliberately
self-contained: molecules travel as MDL molblocks embedded in the JSON rather than as
paths to the original inputs. A network that references files by path stops being
reproducible the moment someone reorganises a directory, and the mapping indices in it
are meaningless against a molecule that has been re-read with different atom ordering.

Rejected candidates are written too. They cost little and they are what explains a
disconnected or sparse network after the fact.

Two fields are written only when they carry something -- a ligand's ``provenance`` and a
network's ``intermediates``. Absence already means "not set", so emitting a null or an
empty list would add a difference to every file ever written in order to convey nothing,
and would break the property that regenerating an all-real network reproduces it
byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from rdkit import Chem

from rbfenetmap.core.intermediates import IntermediateOptions
from rbfenetmap.core.models import (
    AtomMapping,
    EdgeKind,
    EdgeScore,
    IntermediateRecord,
    Ligand,
    LigandProvenance,
    Network,
    RejectionReason,
    SoftcoreRepair,
    Transformation,
)
from rbfenetmap.core.options import NetworkOptions, SoftcorePolicy

__all__ = ("SCHEMA_VERSION", "dump_network", "load_network", "network_to_dict")

#: Bumped whenever the on-disk shape changes incompatibly.
SCHEMA_VERSION = 1


def _provenance_to_dict(provenance: LigandProvenance) -> dict[str, Any]:
    """Serialize a ligand provenance."""
    return {
        "kind": provenance.kind,
        "generator": provenance.generator,
        "parents": list(provenance.parents),
        "pose_method": provenance.pose_method,
        "pose_rmsd": provenance.pose_rmsd,
        "detail": dict(provenance.detail),
    }


def _provenance_from_dict(data: dict[str, Any]) -> LigandProvenance:
    """Rebuild a ligand provenance from its serialized form."""
    return LigandProvenance(
        kind=data["kind"],
        generator=data["generator"],
        parents=tuple(data.get("parents", ())),
        pose_method=data.get("pose_method", "unknown"),
        pose_rmsd=float(data.get("pose_rmsd", 0.0)),
        detail=MappingProxyType(dict(data.get("detail") or {})),
    )


def _record_to_dict(record: IntermediateRecord) -> dict[str, Any]:
    """Serialize one intermediate-generation attempt."""
    return {
        "source": record.source,
        "target": record.target,
        "generator": record.generator,
        "accepted": record.accepted,
        "names": list(record.names),
        "rejection": record.rejection,
        "trace": list(record.trace),
    }


def _record_from_dict(data: dict[str, Any]) -> IntermediateRecord:
    """Rebuild an intermediate-generation attempt from its serialized form."""
    return IntermediateRecord(
        source=data["source"],
        target=data["target"],
        generator=data.get("generator", "unknown"),
        accepted=bool(data.get("accepted", False)),
        names=tuple(data.get("names", ())),
        rejection=data.get("rejection"),
        trace=tuple(data.get("trace", ())),
    )


def _ligand_to_dict(ligand: Ligand) -> dict[str, Any]:
    """Serialize a ligand, embedding its molecule as a molblock.

    ``provenance`` is omitted entirely on a ligand that has none, which is every ligand
    read from an input file. See the module docstring.
    """
    return {
        "name": ligand.name,
        "charge": ligand.charge,
        "source": str(ligand.source) if ligand.source else None,
        "metadata": dict(ligand.metadata),
        "molblock": Chem.MolToMolBlock(ligand.mol, kekulize=False),
        **({"provenance": _provenance_to_dict(ligand.provenance)} if ligand.provenance is not None else {}),
    }


def _ligand_from_dict(data: dict[str, Any]) -> Ligand:
    """Rebuild a ligand from its serialized form."""
    mol = Chem.MolFromMolBlock(data["molblock"], removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError(f"Could not parse the stored molblock for ligand {data['name']!r}.")
    return Ligand(
        name=data["name"],
        mol=mol,
        charge=int(data["charge"]),
        source=Path(data["source"]) if data.get("source") else None,
        metadata=MappingProxyType(dict(data.get("metadata") or {})),
        provenance=_provenance_from_dict(data["provenance"]) if data.get("provenance") else None,
    )


def _edge_to_dict(edge: Transformation) -> dict[str, Any]:
    """Serialize one transformation, including its repair trace."""
    return {
        "source": edge.source,
        "target": edge.target,
        "kind": edge.kind.value,
        "mapping": {
            "cc1": list(edge.mapping.cc1),
            "cc2": list(edge.mapping.cc2),
            "sc1": list(edge.mapping.sc1),
            "sc2": list(edge.mapping.sc2),
            "n_atoms_1": edge.mapping.n_atoms_1,
            "n_atoms_2": edge.mapping.n_atoms_2,
            "method": edge.mapping.method,
        },
        "repair": {
            "applied": edge.repair.applied,
            "n_fragments_before": list(edge.repair.n_fragments_before),
            "n_fragments_after": list(edge.repair.n_fragments_after),
            "demoted_1": list(edge.repair.demoted_1),
            "demoted_2": list(edge.repair.demoted_2),
            "iterations": edge.repair.iterations,
            "rejection": edge.repair.rejection.value if edge.repair.rejection else None,
            "trace": list(edge.repair.trace),
        },
        "score": {
            "total": edge.score.total if edge.score.feasible else None,
            "feasible": edge.score.feasible,
            "descriptors": dict(edge.score.descriptors),
            "contributions": dict(edge.score.contributions),
            "rejections": [r.value for r in edge.score.rejections],
            "scorer": edge.score.scorer,
        },
    }


def _edge_from_dict(data: dict[str, Any]) -> Transformation:
    """Rebuild a transformation from its serialized form.

    ``kind`` defaults to RBFE when absent rather than being required, so files written
    before counterpoised edges existed still load. That is why adding the field did not
    bump :data:`SCHEMA_VERSION`: a bump would have made every one of those files
    unreadable to buy a compatibility guarantee the default already provides.

    The same reasoning governs a ligand's ``provenance`` and a network's
    ``intermediates``: absent-means-default is a complete compatibility story, so neither
    bumped the version either. A bump is for *removing* a field or changing what an
    existing one means -- the two cases a default cannot cover.
    """
    mapping_data = data["mapping"]
    mapping = AtomMapping(
        cc1=tuple(mapping_data["cc1"]),
        cc2=tuple(mapping_data["cc2"]),
        sc1=tuple(mapping_data["sc1"]),
        sc2=tuple(mapping_data["sc2"]),
        n_atoms_1=mapping_data["n_atoms_1"],
        n_atoms_2=mapping_data["n_atoms_2"],
        method=mapping_data.get("method", "unknown"),
    )
    repair_data = data["repair"]
    repair = SoftcoreRepair(
        applied=repair_data["applied"],
        n_fragments_before=tuple(repair_data["n_fragments_before"]),  # type: ignore[arg-type]
        n_fragments_after=tuple(repair_data["n_fragments_after"]),  # type: ignore[arg-type]
        demoted_1=tuple(repair_data["demoted_1"]),
        demoted_2=tuple(repair_data["demoted_2"]),
        iterations=repair_data["iterations"],
        rejection=RejectionReason(repair_data["rejection"]) if repair_data.get("rejection") else None,
        trace=tuple(repair_data["trace"]),
    )
    score_data = data["score"]
    rejections = tuple(RejectionReason(r) for r in score_data.get("rejections", ()))
    if score_data["feasible"]:
        score = EdgeScore(
            total=float(score_data["total"]),
            feasible=True,
            descriptors=MappingProxyType(dict(score_data.get("descriptors") or {})),
            contributions=MappingProxyType(dict(score_data.get("contributions") or {})),
            rejections=(),
            scorer=score_data.get("scorer", "unknown"),
        )
    else:
        score = EdgeScore.rejected(
            *rejections,
            scorer=score_data.get("scorer", "unknown"),
            descriptors=MappingProxyType(dict(score_data.get("descriptors") or {})),
        )
    return Transformation(
        source=data["source"],
        target=data["target"],
        mapping=mapping,
        repair=repair,
        score=score,
        kind=EdgeKind(data.get("kind", EdgeKind.RBFE.value)),
    )


def network_to_dict(network: Network) -> dict[str, Any]:
    """Return the JSON-ready representation of *network*."""
    options = network.options
    return {
        "schema_version": SCHEMA_VERSION,
        "planner": network.planner,
        "unmet_constraints": list(network.unmet_constraints),
        "options": (
            {
                "pair_strategy": options.pair_strategy,
                "hub": options.hub,
                "n_edges": options.n_edges,
                "edges_per_ligand": options.edges_per_ligand,
                "min_cycle_coverage": options.min_cycle_coverage,
                "require_connected": options.require_connected,
                "edge_direction": options.edge_direction,
                "selection_objective": options.selection_objective,
                "cycle_coverage_mode": options.cycle_coverage_mode,
                "max_cycle_size": options.max_cycle_size,
                "max_diameter": options.max_diameter,
                "n_redundancy": options.n_redundancy,
                "hub_selection": options.hub_selection,
                "pair_evaluation": options.pair_evaluation,
                "adaptive_initial_neighbors": options.adaptive_initial_neighbors,
                "adaptive_batch_size": options.adaptive_batch_size,
                "consistency": options.consistency,
                "cbfe_mode": options.cbfe_mode,
                "cbfe_base_cost": options.cbfe_base_cost,
                "cbfe_atom_weight": options.cbfe_atom_weight,
                "cluster_by": options.cluster_by,
                "cluster_bridges": options.cluster_bridges,
                "design": options.design,
                "design_candidate_factor": options.design_candidate_factor,
                "design_refine": options.design_refine,
                "design_total_ns": options.design_total_ns,
                "design_lambda_min": options.design_lambda_min,
                "design_lambda_max": options.design_lambda_max,
                "softcore": {
                    "ring_policy": options.softcore.ring_policy,
                    "max_softcore_atoms": options.softcore.max_softcore_atoms,
                    "max_softcore_fraction": options.softcore.max_softcore_fraction,
                    "min_core_atoms": options.softcore.min_core_atoms,
                    "min_mcs_fraction": options.softcore.min_mcs_fraction,
                    "core_rmsd_threshold": options.softcore.core_rmsd_threshold,
                    "charge_change_policy": options.softcore.charge_change_policy,
                },
                # Always written, like "softcore" and unlike "compat". An absent block
                # means "generation off", which is also what the defaults say -- but a
                # network whose vertex set was *invented* has to state the settings that
                # invented it, and a block that appeared only sometimes would make the
                # reader check two things to learn one.
                "intermediates": {
                    "mode": options.intermediates.mode,
                    "generator": options.intermediates.generator,
                    "max_intermediates": options.intermediates.max_intermediates,
                    "max_gaps": options.intermediates.max_gaps,
                    "max_molecules": options.intermediates.max_molecules,
                    "seed": options.intermediates.seed,
                    "max_pose_attempts": options.intermediates.max_pose_attempts,
                    "pose_rmsd_factor": options.intermediates.pose_rmsd_factor,
                    "min_link_score": options.intermediates.min_link_score,
                    "max_dist": options.intermediates.max_dist,
                    "max_cycle": options.intermediates.max_cycle,
                    "max_subgraph_dist": options.intermediates.max_subgraph_dist,
                    "beta": options.intermediates.beta,
                },
                # Omitted entirely when unset, so a network planned without --compat
                # serializes byte-for-byte as it did before the flag existed. An
                # absent key already means "not pinned", so writing a null would add a
                # difference to every existing file to convey nothing.
                **({"compat": options.compat} if options.compat is not None else {}),
            }
            if options is not None
            else None
        ),
        "ligands": [_ligand_to_dict(ligand) for ligand in network.ligands.values()],
        "edges": [_edge_to_dict(edge) for edge in network.edges],
        "candidates": [_edge_to_dict(edge) for edge in network.candidates],
        # Omitted when empty, for the same reason `compat` is: a network planned without
        # intermediate generation must serialize exactly as it did before generation
        # existed, or the golden baseline stops meaning anything.
        **(
            {"intermediates": [_record_to_dict(record) for record in network.intermediates]}
            if network.intermediates
            else {}
        ),
    }


def dump_network(network: Network, path: Path, *, indent: int = 2) -> Path:
    """Write *network* to *path* as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(network_to_dict(network), indent=indent))
    return path


def load_network(path: Path) -> Network:
    """Read a network written by :func:`dump_network`.

    Raises
    ------
    ValueError
        If the file was written by an incompatible schema version.
    """
    data = json.loads(Path(path).read_text())
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path} declares schema version {version!r}, but this build reads version {SCHEMA_VERSION}. "
            "Re-run `rbfenet plan` to regenerate it."
        )

    ligands = {item["name"]: _ligand_from_dict(item) for item in data["ligands"]}
    options_data = data.get("options")
    options = None
    if options_data:
        softcore_data = options_data.get("softcore") or {}
        intermediate_data = options_data.get("intermediates") or {}
        options = NetworkOptions(
            pair_strategy=options_data.get("pair_strategy", "all_unordered_pairs"),
            hub=options_data.get("hub"),
            n_edges=options_data.get("n_edges"),
            edges_per_ligand=options_data.get("edges_per_ligand", 2),
            min_cycle_coverage=options_data.get("min_cycle_coverage", 1.0),
            require_connected=options_data.get("require_connected", True),
            edge_direction=options_data.get("edge_direction", "fewer_softcore_first"),
            selection_objective=options_data.get("selection_objective", "uniform_redundancy"),
            cycle_coverage_mode=options_data.get("cycle_coverage_mode", "node"),
            max_cycle_size=options_data.get("max_cycle_size"),
            max_diameter=options_data.get("max_diameter"),
            n_redundancy=options_data.get("n_redundancy", 2),
            hub_selection=options_data.get("hub_selection", "most_partners"),
            pair_evaluation=options_data.get("pair_evaluation", "eager"),
            adaptive_initial_neighbors=options_data.get("adaptive_initial_neighbors", 3),
            adaptive_batch_size=options_data.get("adaptive_batch_size", 32),
            consistency=options_data.get("consistency", "pairwise"),
            cbfe_mode=options_data.get("cbfe_mode", "off"),
            cbfe_base_cost=options_data.get("cbfe_base_cost", 8.0),
            cbfe_atom_weight=options_data.get("cbfe_atom_weight", 0.05),
            cluster_by=options_data.get("cluster_by", "none"),
            cluster_bridges=options_data.get("cluster_bridges", 2),
            design=options_data.get("design", "none"),
            design_candidate_factor=options_data.get("design_candidate_factor", 3.0),
            design_refine=options_data.get("design_refine", False),
            design_total_ns=options_data.get("design_total_ns"),
            design_lambda_min=options_data.get("design_lambda_min", 12),
            design_lambda_max=options_data.get("design_lambda_max", 24),
            compat=options_data.get("compat"),
            softcore=SoftcorePolicy(**softcore_data) if softcore_data else SoftcorePolicy(),
            intermediates=IntermediateOptions(**intermediate_data) if intermediate_data else IntermediateOptions(),
        )

    return Network(
        ligands=ligands,
        edges=tuple(_edge_from_dict(item) for item in data.get("edges", ())),
        candidates=tuple(_edge_from_dict(item) for item in data.get("candidates", ())),
        planner=data.get("planner", "unknown"),
        options=options,
        unmet_constraints=tuple(data.get("unmet_constraints", ())),
        intermediates=tuple(_record_from_dict(item) for item in data.get("intermediates", ())),
    )

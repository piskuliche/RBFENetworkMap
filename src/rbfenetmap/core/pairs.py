"""Candidate pair generation and prefiltering.

Decides *which* transformations are worth mapping and scoring at all. For anything past
a couple of dozen ligands the all-pairs set is dominated by pairs no one would consider,
and mapping is the expensive stage, so a cheap similarity prefilter pays for itself many
times over.

The prefilter carries one obligation, discharged by :func:`reconnect_pairs`: it must not
be allowed to disconnect the candidate pool behind the user's back.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import TYPE_CHECKING, Mapping, Sequence

import networkx as nx

from rbfenetmap.core.models import Ligand, parse_edge_key
from rbfenetmap.core.options import NetworkOptions, PairStrategy

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = (
    "expand_pairs",
    "fingerprint_pair_similarities",
    "fingerprint_prefilter",
    "generate_candidate_pairs",
    "reconnect_pairs",
)


def fingerprint_pair_similarities(
    ligands: Mapping[str, Ligand], pairs: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], float]:
    """Return Morgan/Tanimoto similarity for each requested pair.

    This is deliberately mapping-free and therefore cheap enough to rank an all-pairs
    pool before any MCS searches are launched.
    """
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import TanimotoSimilarity

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {name: generator.GetFingerprint(ligand.mol) for name, ligand in ligands.items()}
    return {
        pair: float(TanimotoSimilarity(fingerprints[pair[0]], fingerprints[pair[1]])) for pair in pairs
    }


def expand_pairs(
    names: Sequence[str],
    strategy: PairStrategy = "all_unordered_pairs",
    *,
    hub: str | None = None,
    explicit: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Enumerate candidate pairs under *strategy*.

    A port of ``BuildEdges._expand_edges``.

    Parameters
    ----------
    names : Sequence[str]
        Ligand names, in input order.
    strategy : PairStrategy
        ``"all_unordered_pairs"``, ``"all_pairs"``, ``"star"``, ``"linear"``, or
        ``"explicit"``.
    hub : str, optional
        Required by ``"star"``.
    explicit : Sequence[str]
        ``"a~b"`` specifications, required by ``"explicit"``.

    Returns
    -------
    list[tuple[str, str]]
        Ordered pairs, deduplicated.

    Raises
    ------
    ValueError
        For an unknown strategy, a missing or unknown hub, an unknown ligand in
        *explicit*, or a strategy that yields no pairs at all.
    """
    known = set(names)
    if strategy == "all_unordered_pairs":
        pairs = list(combinations(names, 2))
    elif strategy == "all_pairs":
        pairs = list(permutations(names, 2))
    elif strategy == "star":
        if not hub:
            raise ValueError("strategy 'star' requires a hub ligand.")
        if hub not in known:
            raise ValueError(f"Hub {hub!r} is not among the ligands {sorted(known)}.")
        pairs = [(hub, name) for name in names if name != hub]
    elif strategy == "linear":
        pairs = list(zip(names, names[1:]))
    elif strategy == "explicit":
        pairs = []
        for spec in explicit:
            source, target = parse_edge_key(spec)
            unknown = {source, target} - known
            if unknown:
                raise ValueError(f"Explicit edge {spec!r} names unknown ligand(s) {sorted(unknown)}.")
            pairs.append((source, target))
    else:
        raise ValueError(f"Unknown pair strategy {strategy!r}.")

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for pair in pairs:
        if pair[0] == pair[1] or pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)

    if not unique:
        raise ValueError(
            f"Pair strategy {strategy!r} produced no candidate pairs from {len(names)} ligand(s). "
            "At least two distinct ligands are required."
        )
    return unique


def fingerprint_prefilter(
    ligands: Mapping[str, Ligand], pairs: Sequence[tuple[str, str]], *, top_k: int = 8, min_similarity: float = 0.4
) -> list[tuple[str, str]]:
    """Keep only the pairs most likely to yield a usable transformation.

    For each ligand, retains its *top_k* most similar partners, plus every pair above
    *min_similarity*. Similarity is Morgan/Tanimoto.

    Parameters
    ----------
    ligands : Mapping[str, Ligand]
    pairs : Sequence[tuple[str, str]]
        Candidate pairs to filter.
    top_k : int, optional
        Neighbours retained per ligand.
    min_similarity : float, optional
        Tanimoto floor for unconditional retention.

    Returns
    -------
    list[tuple[str, str]]
        The surviving pairs, in the input order.

    Notes
    -----
    Callers must follow this with :func:`reconnect_pairs`. Retaining each ligand's
    nearest neighbours says nothing about whether the resulting graph is connected: a
    series containing two distinct chemical families will happily split into two
    components, each internally well connected, and the planner would then report a
    disconnection the user never asked for.
    """
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.DataStructs import BulkTanimotoSimilarity

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    names = list(ligands)
    fingerprints = {name: generator.GetFingerprint(ligands[name].mol) for name in names}

    similarity: dict[tuple[str, str], float] = {}
    for name in names:
        others = [n for n in names if n != name]
        if not others:
            continue
        scores = BulkTanimotoSimilarity(fingerprints[name], [fingerprints[n] for n in others])
        for other, score in zip(others, scores):
            similarity[tuple(sorted((name, other)))] = float(score)  # type: ignore[index]

    keep: set[tuple[str, str]] = set()
    for name in names:
        ranked = sorted(
            (n for n in names if n != name),
            key=lambda other: (-similarity.get(tuple(sorted((name, other))), 0.0), other),  # type: ignore[arg-type]
        )
        for other in ranked[:top_k]:
            keep.add(tuple(sorted((name, other))))  # type: ignore[arg-type]
    for pair, score in similarity.items():
        if score >= min_similarity:
            keep.add(pair)

    return [pair for pair in pairs if tuple(sorted(pair)) in keep]


def reconnect_pairs(
    names: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    all_pairs: Sequence[tuple[str, str]],
    ligands: Mapping[str, Ligand] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Add pairs back until the candidate graph spans every ligand.

    Parameters
    ----------
    names : Sequence[str]
        Every ligand name.
    pairs : Sequence[tuple[str, str]]
        The prefiltered pairs.
    all_pairs : Sequence[tuple[str, str]]
        The unfiltered pairs, from which bridges may be restored.
    ligands : Mapping[str, Ligand], optional
        Used to rank restoration candidates by similarity. Without it, restoration is
        arbitrary but deterministic.

    Returns
    -------
    tuple[list[tuple[str, str]], list[tuple[str, str]]]
        The reconnected pair list, and the pairs that had to be restored -- reported so
        the user can see the prefilter was overridden rather than silently corrected.

    Notes
    -----
    Mandatory after :func:`fingerprint_prefilter`. Prefiltering is an optimisation, and
    an optimisation that changes the answer -- here, by making a connected network
    impossible -- is a bug. Restoring the best available bridge keeps the prefilter
    honest: it may reorder the work, but it cannot remove an outcome.
    """
    restored: list[tuple[str, str]] = []
    current = list(pairs)

    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(names)
    graph.add_edges_from(current)

    remaining = [p for p in all_pairs if tuple(sorted(p)) not in {tuple(sorted(c)) for c in current}]
    if ligands is not None and remaining:
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit.DataStructs import TanimotoSimilarity

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprints = {name: generator.GetFingerprint(ligands[name].mol) for name in names}
        remaining.sort(key=lambda p: (-TanimotoSimilarity(fingerprints[p[0]], fingerprints[p[1]]), p))

    while not nx.is_connected(graph) if graph.number_of_nodes() > 1 else False:
        bridge = next((p for p in remaining if not nx.has_path(graph, p[0], p[1])), None)
        if bridge is None:
            break  # the unfiltered pool is itself disconnected; the planner will say so
        graph.add_edge(*bridge)
        current.append(bridge)
        restored.append(bridge)
        remaining.remove(bridge)

    return current, restored


def generate_candidate_pairs(
    ligands: Mapping[str, Ligand], options: NetworkOptions
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Produce the pairs to map and score, applying strategy, prefilter, and forcing.

    Returns
    -------
    tuple[list[tuple[str, str]], list[tuple[str, str]]]
        The candidate pairs and any pairs restored by the reconnection pass.
    """
    names = list(ligands)
    pairs = expand_pairs(names, options.pair_strategy, hub=options.hub, explicit=options.explicit_pairs)

    restored: list[tuple[str, str]] = []
    if options.prefilter == "fingerprint":
        filtered = fingerprint_prefilter(
            ligands, pairs, top_k=options.prefilter_k, min_similarity=options.prefilter_min_tanimoto
        )
        pairs, restored = reconnect_pairs(names, filtered, pairs, ligands)

    # A forced edge must be scored even if the strategy or the prefilter excluded it;
    # otherwise the planner would fail it as "infeasible" when it was merely never tried.
    existing = {tuple(sorted(p)) for p in pairs}
    for source, target in sorted(options.forced_pairs):
        if (source, target) not in existing:
            pairs.append((source, target))

    banned = options.banned_pairs
    pairs = [p for p in pairs if tuple(sorted(p)) not in banned]
    return pairs, restored

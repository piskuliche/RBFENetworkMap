"""PairMap-style intermediate generator: a searched subnetwork, not a chain.

Re-derived from the method described in

    K. Furui, S. Shimizu, Y. Akiyama, S. Kimura, T. Terada and M. Ohue,
    "PairMap: An Intermediate Scaffold-Based Approach to Improve Alchemical
    Free Energy Calculations for Complex Perturbations",
    *J. Chem. Inf. Model.* **2025**, 65, 705-721,
    `doi:10.1021/acs.jcim.4c01634 <https://doi.org/10.1021/acs.jcim.4c01634>`_;
    reference implementation at https://github.com/ohuelab/PairMap (CC-BY 4.0).

**No code was copied or adapted from the reference implementation.** The algorithm below
was written from the description in the paper and the epic's specification, against this
package's own R-group decomposition, options object, and plugin contract. The citation is
here because the *method* is theirs and belongs attributed wherever it is used, not because
the file carries a CC-BY obligation.

What this generator does
------------------------

Two ligands that share a core differ at a handful of substituent positions. Every position
offers three groups: the source parent's, the target parent's, and -- when both parents put
something there -- *nothing*, which is the shared core itself. An assignment of one group
per position is a molecule; the source parent is "source everywhere" and the target parent
is "target everywhere". That state space is the recursive enumeration of operations from
both parents toward their MCS, expressed as a product rather than as a recursion, because
whole-substituent operations commute and enumerating a commutative recursion is a product.

Two states are *linked* when a transformation between them is worth running. Its score is

.. math:: s = \\exp(-\\beta \\, \\Delta)

with :math:`\\Delta` the heavy atoms that disappear plus the heavy atoms that appear -- the
LOMAP similarity, which is why the paper's :math:`\\beta` and this package's
``beta = 0.1`` are the same constant. Links below ``min_link_score`` are not links.

A **path** from source to target is scored by the harmonic mean of its squared link scores
divided by its length, which reduces exactly to

.. math:: \\frac{1}{\\sum_i s_i^{-2}}

so maximising it is a shortest-path problem with edge weight :math:`s^{-2}`. That identity
is the whole reason the paper's score is shaped that way: "shortest path" and "highest link
scores", the first two of its four subnetwork requirements, are one Dijkstra rather than
two competing objectives. Paths of one link are excluded -- a one-link path *is* the direct
transformation the pipeline already rejected.

The remaining two requirements shape what is emitted around that path. Every link on it
should sit in a cycle of at most ``max_cycle`` edges, because a cycle is what turns a chain
of intermediates into a network with a closure error you can check; and the subnetwork
should carry no more redundancy than that, because every extra vertex is another molecule
somebody has to parameterise. So cycles are closed one uncovered link at a time, cheapest
first, and the search stops the moment every link is covered.

What is covered, and what is not
--------------------------------

Covered: the state enumeration, the link score, the path score and its shortest-path
identity, the optimal-path search under ``max_dist``, cycle closure under ``max_cycle``,
the subnetwork extent bound ``max_subgraph_dist``, and the ``min_link_score`` cut.

**Not covered: the paper's fine-grained operation set.** PairMap enumerates atom-level and
ring-level operations -- change an element, add or delete one atom, open or close a ring --
so it can walk through intermediates that are *inside* a substituent. This implementation's
operations are whole substituents: put the source's group here, the target's group here, or
nothing here. That is a strict subset. On a pair differing by several R-groups, which is
the common hard case and the one the epic is aimed at, the two enumerations agree on the
useful states; on a pair whose difference is a scaffold hop or a change buried in the middle
of one substituent, this generator will find the truncation to the shared core and nothing
finer, and it will often refuse outright. It says so in its rejection rather than quietly
proposing something worse.

Also not covered: the paper's own scoring of a *proposed* molecule's synthetic accessibility
or its similarity to known chemistry. Nothing in this package would read it -- the
:class:`~rbfenetmap.core.intermediates.ProposedLink` hint is advisory by contract -- and a
number the pipeline cannot act on is a number that lies about its own importance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Iterable, Mapping, Sequence

from rdkit import Chem

from rbfenetmap.core.intermediates import (
    IntermediateOptions,
    IntermediateProposal,
    ProposedLink,
    ProposedMolecule,
    intermediate_name,
)
from rbfenetmap.core.meta.intermediates import AbstractIntermediateGenerator
from rbfenetmap.core.models import Ligand
from rbfenetmap.core.options import MappingOptions
from rbfenetmap.plugins.intermediates._rgroups import Position, assemble, differing_positions, shared_core

__all__ = ("PairMapGenerator",)

#: Choice labels, in the order they are offered. ``"hydrogen"`` is the move *toward* the
#: MCS; the other two are the moves toward a parent.
SOURCE, TARGET, HYDROGEN = "source", "target", "hydrogen"

#: Ceiling on how many states may be enumerated for one gap.
#:
#: The state space is a product, so it grows as ``3**n`` in the number of differing
#: positions and would reach five figures on a pair differing at eight of them -- which is
#: a pair nobody should be bridging with one intermediate anyway. Rather than refuse such a
#: pair outright, :func:`_group_positions` bundles the least consequential positions
#: together so they move as one; the bound then holds and the fine-grained search is spent
#: where the atoms are.
MAX_STATES = 243

#: Ceiling on how many simple paths the cycle-closure search will enumerate per link.
MAX_CLOSURE_PATHS = 4000

#: A state: one choice label per position *group*, in group order.
State = tuple[str, ...]


@dataclass(frozen=True)
class _Group:
    """One or more positions that move together, and the labels they may take.

    Parameters
    ----------
    positions : tuple[int, ...]
        Indices into the position list. More than one means the positions were bundled to
        keep the state space inside :data:`MAX_STATES`.
    labels : tuple[str, ...]
        The choices this group offers.
    """

    positions: tuple[int, ...]
    labels: tuple[str, ...]


def _labels_for(position: Position) -> tuple[str, ...]:
    """Return the distinct groups a single position can carry.

    ``"hydrogen"`` is offered only when both parents put something there. Where one of them
    already carries nothing, truncating reproduces that parent's group rather than
    inventing a third option, and a state space with two names for one molecule makes every
    later dedupe a special case.
    """
    return (SOURCE, TARGET, HYDROGEN) if position.truncatable else (SOURCE, TARGET)


def _group_positions(positions: Sequence[Position]) -> list[_Group]:
    """Bundle the least consequential positions until the state space fits.

    Positions are ranked by how many heavy atoms change at them, most first, and kept
    independent from the top down. Whatever does not fit is bundled into a single group
    that only ever takes ``"source"`` or ``"target"`` -- the bundle moves as one, so the
    search still reaches both parents while spending its resolution where the chemistry is.
    """
    order = sorted(range(len(positions)), key=lambda i: (-(positions[i].source_heavy + positions[i].target_heavy), i))
    groups = [_Group(positions=(index,), labels=_labels_for(positions[index])) for index in order]
    if math.prod(len(group.labels) for group in groups) <= MAX_STATES:
        return sorted(groups, key=lambda group: group.positions)

    kept: list[_Group] = []
    for group in groups:
        trial = math.prod(len(item.labels) for item in (*kept, group)) * 2
        if trial > MAX_STATES:
            break
        kept.append(group)
    bundled = tuple(sorted(index for group in groups[len(kept) :] for index in group.positions))
    kept.append(_Group(positions=bundled, labels=(SOURCE, TARGET)))
    return sorted(kept, key=lambda group: group.positions)


def _enumerate(groups: Sequence[_Group]) -> list[State]:
    """Return every state, in a deterministic order."""
    states: list[State] = [()]
    for group in groups:
        states = [(*state, label) for state in states for label in group.labels]
    return states


def _expand(groups: Sequence[_Group], state: State, n_positions: int) -> list[str]:
    """Expand a per-group state into one label per position."""
    choices = [SOURCE] * n_positions
    for group, label in zip(groups, state):
        for index in group.positions:
            choices[index] = label
    return choices


def _heavy(position: Position, label: str) -> int:
    """Heavy atoms the *label* group puts at *position*."""
    if label == SOURCE:
        return position.source_heavy
    if label == TARGET:
        return position.target_heavy
    return 0


def _link_score(
    groups: Sequence[_Group], positions: Sequence[Position], left: State, right: State, beta: float
) -> float:
    """Return the LOMAP-style similarity of the transformation between two states.

    ``exp(-beta * delta)`` with *delta* the heavy atoms that disappear plus the heavy atoms
    that appear. The two states share every atom of the core and every group they agree on,
    so *delta* is exactly the atom count of the groups they disagree on -- which is why this
    needs no MCS of its own and costs nothing to evaluate over the whole state graph.
    """
    delta = 0
    for group, here, there in zip(groups, left, right):
        if here == there:
            continue
        for index in group.positions:
            delta += _heavy(positions[index], here) + _heavy(positions[index], there)
    return math.exp(-beta * delta)


def _bounded_shortest_path(
    weights: Mapping[tuple[State, State], float], nodes: Sequence[State], start: State, end: State, max_hops: int
) -> list[State] | None:
    """Return the cheapest *start*-to-*end* walk of two to *max_hops* links.

    Dynamic programming over hop count rather than plain Dijkstra, because the bound is on
    the number of links -- the number of simulations somebody has to run -- and not on the
    accumulated weight. Walks of one link are excluded by construction: a one-link path
    from source to target is the direct transformation that was already rejected.

    Returns ``None`` when no such walk exists.
    """
    infinity = float("inf")
    best: list[dict[State, float]] = [{node: infinity for node in nodes} for _ in range(max_hops + 1)]
    came: list[dict[State, State | None]] = [{node: None for node in nodes} for _ in range(max_hops + 1)]
    best[0][start] = 0.0

    adjacency: dict[State, list[tuple[State, float]]] = {node: [] for node in nodes}
    for (left, right), weight in weights.items():
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))

    for hops in range(1, max_hops + 1):
        for node in nodes:
            if best[hops - 1][node] == infinity:
                continue
            for neighbour, weight in adjacency[node]:
                candidate = best[hops - 1][node] + weight
                if candidate < best[hops][neighbour]:
                    best[hops][neighbour] = candidate
                    came[hops][neighbour] = node

    # The dynamic program optimises over *walks*, and a walk may revisit a node -- notably
    # by hopping out to the target and back when a direct link exists. Take the hop counts
    # in increasing cost and return the first one that reconstructs to a simple path, so a
    # cheap walk that is not a path never masks the real answer.
    for hops in sorted(range(2, max_hops + 1), key=lambda h: best[h][end]):
        if best[hops][end] == infinity:
            break
        path: list[State] = [end]
        node, remaining = end, hops
        while remaining > 0:
            previous = came[remaining][node]
            if previous is None:  # pragma: no cover - a reachable node always has a predecessor
                break
            path.append(previous)
            node, remaining = previous, remaining - 1
        path.reverse()
        if len(path) == hops + 1 and len(set(path)) == len(path):
            return path
    return None


def _hop_distances(adjacency: Mapping[State, Iterable[State]], start: State) -> dict[State, int]:
    """Breadth-first link-count distances from *start*."""
    distances = {start: 0}
    frontier = [start]
    while frontier:
        following: list[State] = []
        for node in frontier:
            for neighbour in adjacency[node]:
                if neighbour not in distances:
                    distances[neighbour] = distances[node] + 1
                    following.append(neighbour)
        frontier = following
    return distances


def _simple_paths(
    adjacency: Mapping[State, list[State]], start: State, end: State, max_hops: int, forbidden: tuple[State, State]
) -> list[list[State]]:
    """Enumerate simple *start*-to-*end* paths of at most *max_hops* links.

    *forbidden* is the one link the path may not use -- the link whose cycle is being
    closed, which would otherwise close a two-edge "cycle" with itself.
    """
    found: list[list[State]] = []
    stack: list[list[State]] = [[start]]
    while stack and len(found) < MAX_CLOSURE_PATHS:
        path = stack.pop()
        node = path[-1]
        if len(path) - 1 >= max_hops:
            continue
        for neighbour in adjacency[node]:
            if neighbour in path:
                continue
            if tuple(sorted((node, neighbour))) == forbidden:
                continue
            if neighbour == end:
                found.append([*path, neighbour])
            else:
                stack.append([*path, neighbour])
    return found


class PairMapGenerator(AbstractIntermediateGenerator):
    """Search a subnetwork of R-group recombinations between two parents.

    Notes
    -----
    Emits a *subnetwork*, not a chain: the optimal path plus whatever closes its links into
    cycles of at most ``max_cycle``. ``A-M1-B-M2-A`` is the smallest such shape, and it is a
    genuine consistency check -- two independent routes across a gap whose closure error is
    measurable -- rather than redundancy for its own sake.

    Everything it emits still goes through
    :func:`~rbfenetmap.core.pipeline.build_candidate` like any other edge. The generator's
    link score orders *what to try*; it never becomes a cost, and a molecule whose pose does
    not survive the geometry gate is dropped by the pipeline regardless of how promising the
    generator thought it was.
    """

    name: ClassVar[str] = "pairmap"

    def supports_pair(self, source: Ligand, target: Ligand) -> bool:
        """Refuse a pair whose parents are not both posed.

        The decomposition resolves ring symmetry by in-place RMSD, so a parent without a
        conformer would have its positions assigned by whichever MCS embedding came back
        first -- a coin flip that produces a plausible molecule with a substituent on the
        wrong side of the ring.
        """
        return bool(source.mol.GetNumConformers()) and bool(target.mol.GetNumConformers())

    def describe_parameters(self) -> Mapping[str, object]:
        """Return what this generator's search does, for the run record.

        The numeric knobs are not repeated here: they live on
        :class:`~rbfenetmap.core.intermediates.IntermediateOptions` and are serialized with
        the network, and a second copy that could disagree would be worse than none.
        """
        return {
            "operations": "whole-substituent recombination toward the shared core",
            "link_score": "exp(-beta * heavy atoms changed)",
            "path_score": "harmonic mean of squared link scores / path length",
            "emits": "subnetwork",
            "reference": "Furui et al., J. Chem. Inf. Model. 2025, 65, 705-721, doi:10.1021/acs.jcim.4c01634",
        }

    def propose(
        self, source: Ligand, target: Ligand, options: IntermediateOptions, mapping_options: MappingOptions
    ) -> IntermediateProposal:
        """Return the subnetwork bridging *source* to *target*.

        Parameters
        ----------
        source, target : Ligand
            The gap endpoints, co-posed.
        options : IntermediateOptions
            ``min_link_score``, ``max_dist``, ``max_cycle``, ``max_subgraph_dist`` and
            ``beta`` steer the search; ``max_molecules`` caps what it may emit.
        mapping_options : MappingOptions
            Settings for the MCS that finds the shared core.

        Returns
        -------
        IntermediateProposal
            With no molecules and a ``rejection`` string when no subnetwork was found. The
            rejections are ``no_common_core``, ``no_substituent_difference``,
            ``core_decomposition_incomplete``, ``no_path_within_max_dist``,
            ``no_molecule_built`` and ``max_molecules_leaves_no_room``.
        """
        trace: list[str] = []

        def refuse(reason: str) -> IntermediateProposal:
            """Return an empty proposal carrying the trace built so far."""
            return IntermediateProposal(
                source=source.name, target=target.name, generator=self.name, rejection=reason, trace=tuple(trace)
            )

        core = shared_core(source, target, mapping_options, trace)
        if core is None:
            return refuse("no_common_core")

        positions = differing_positions(source, target, core)
        trace.append(f"{len(positions)} differing substituent position(s) on a {len(core)}-atom core")
        if not positions:
            return refuse("no_substituent_difference")

        groups = _group_positions(positions)
        if any(len(group.positions) > 1 for group in groups):
            trace.append(f"bundled the least consequential positions into {len(groups)} group(s) to bound the search")

        start: State = tuple(SOURCE for _ in groups)
        end: State = tuple(TARGET for _ in groups)
        states = _enumerate(groups)
        trace.append(f"enumerated {len(states)} state(s) between the parents")

        # The decomposition is only trustworthy if its two extreme states really are the
        # two parents. They are not when a difference lives somewhere `differing_positions`
        # refuses to describe -- a fused ring, a position carrying two heavy branches -- and
        # a generator that proceeded anyway would emit molecules related to neither parent.
        incomplete = self._decomposition_gap(source, target, positions, groups, start, end)
        if incomplete is not None:
            trace.append(incomplete)
            return refuse("core_decomposition_incomplete")

        weights, adjacency = self._link_graph(groups, positions, states, options, start, end)
        trace.append(f"{len(weights)} link(s) at or above min_link_score={options.min_link_score}")

        reachable = self._within_reach(adjacency, start, end, options.max_subgraph_dist)
        weights = {pair: weight for pair, weight in weights.items() if pair[0] in reachable and pair[1] in reachable}
        adjacency = {node: [n for n in adjacency[node] if n in reachable] for node in reachable}
        nodes = [state for state in states if state in reachable]
        trace.append(f"{len(nodes)} state(s) within max_subgraph_dist={options.max_subgraph_dist} of both parents")
        if start not in reachable or end not in reachable:
            # The commonest way here is a single difference at a position one parent leaves
            # bare: there is no third group to put there, so the only route between the two
            # states is the direct link -- which is the transformation that was already
            # rejected and is deliberately not in the graph.
            trace.append("no route between the parents once the direct link is excluded")
            return refuse("no_path_within_max_dist")

        path = _bounded_shortest_path(weights, nodes, start, end, options.max_dist)
        if path is None:
            return refuse("no_path_within_max_dist")
        score = 1.0 / sum(weights[_key(a, b)] for a, b in zip(path, path[1:]))
        trace.append(f"optimal path of {len(path) - 1} link(s), path score {score:.4f}")

        chosen_nodes = list(path)
        chosen_links = [_key(a, b) for a, b in zip(path, path[1:])]
        spent = len(chosen_nodes) - 2
        if spent > options.max_molecules:
            trace.append(f"the optimal path needs {spent} molecule(s); max_molecules={options.max_molecules}")
            return refuse("max_molecules_leaves_no_room")

        # Only what the path did not already spend is available for closing cycles. The
        # path is the bridge and the cycles are the check on it, so the bridge is paid for
        # first.
        remaining = options.max_molecules - spent
        self._close_cycles(chosen_nodes, chosen_links, adjacency, weights, options, start, end, remaining, trace)

        return self._emit(source, target, positions, groups, chosen_nodes, chosen_links, weights, start, end, trace)

    # -- internals ----------------------------------------------------------------

    @staticmethod
    def _decomposition_gap(
        source: Ligand,
        target: Ligand,
        positions: Sequence[Position],
        groups: Sequence[_Group],
        start: State,
        end: State,
    ) -> str | None:
        """Return a note when an extreme state is not the parent it should be.

        The check is on canonical SMILES with hydrogens suppressed, which is the same
        identity :func:`~rbfenetmap.core.intermediates.intermediate_name` and the pipeline's
        dedupe use, so a state that passes here cannot come back later as a surprise
        duplicate of a parent.
        """
        for state, parent in ((start, source), (end, target)):
            built = assemble(source, target, positions, _expand(groups, state, len(positions)))
            if built is None:
                return f"the {parent.name} extreme of the decomposition does not build"
            if _identity(built[0]) != _identity(parent.mol):
                return (
                    f"the {parent.name} extreme of the decomposition is {_identity(built[0])}, not "
                    f"{_identity(parent.mol)}; the two parents differ somewhere the R-group "
                    "decomposition cannot describe"
                )
        return None

    @staticmethod
    def _link_graph(
        groups: Sequence[_Group],
        positions: Sequence[Position],
        states: Sequence[State],
        options: IntermediateOptions,
        start: State,
        end: State,
    ) -> tuple[dict[tuple[State, State], float], dict[State, list[State]]]:
        """Return ``{link: 1/score**2}`` and its adjacency, excluding the direct link.

        The weight is the reciprocal square of the link score because that is what makes
        the paper's path score -- harmonic mean of squared link scores divided by path
        length -- collapse to ``1 / sum(weights)``. Maximising the path score and minimising
        the summed weight are then the same problem, which is why this is a shortest-path
        search rather than an enumeration of paths.

        The ``start``-to-``end`` link is left out entirely. It is the direct transformation
        the pipeline already found infeasible, and a "path" that used it would be proposing
        the rejected edge back to the caller.
        """
        weights: dict[tuple[State, State], float] = {}
        adjacency: dict[State, list[State]] = {state: [] for state in states}
        direct = _key(start, end)
        for index, left in enumerate(states):
            for right in states[index + 1 :]:
                pair = _key(left, right)
                if pair == direct:
                    continue
                score = _link_score(groups, positions, left, right, options.beta)
                if score < options.min_link_score:
                    continue
                weights[pair] = 1.0 / (score * score)
                adjacency[left].append(right)
                adjacency[right].append(left)
        return weights, adjacency

    @staticmethod
    def _within_reach(
        adjacency: Mapping[State, list[State]], start: State, end: State, max_subgraph_dist: int
    ) -> set[State]:
        """Return the states close enough to *both* parents to join the subnetwork.

        Bounding by link-count distance from each parent, rather than by anything about the
        molecules themselves, is what makes ``max_subgraph_dist`` mean the same thing as
        ``max_dist``: both are counts of simulations, which is the resource the user is
        actually budgeting.
        """
        from_start = _hop_distances(adjacency, start)
        from_end = _hop_distances(adjacency, end)
        return {
            state
            for state in from_start
            if from_start[state] <= max_subgraph_dist
            and from_end.get(state, max_subgraph_dist + 1) <= max_subgraph_dist
        }

    @staticmethod
    def _close_cycles(
        chosen_nodes: list[State],
        chosen_links: list[tuple[State, State]],
        adjacency: Mapping[State, list[State]],
        weights: Mapping[tuple[State, State], float],
        options: IntermediateOptions,
        start: State,
        end: State,
        budget: int,
        trace: list[str],
    ) -> None:
        """Grow the subnetwork until every optimal link sits in a small cycle.

        One uncovered link at a time, cheapest closure first, and nothing is added once
        every link is covered -- the paper's fourth requirement, minimal redundancy. A
        closure is preferred by how many *new molecules* it costs before how much it weighs,
        because a vertex is a parameterisation and a simulation while an extra link between
        vertices that already exist is only a simulation.

        Mutates *chosen_nodes* and *chosen_links* in place. Runs out of budget quietly: a
        chain with no cycle in it is a worse network than one with cycles, but it is still a
        network, and refusing the gap over it would trade a bridge for nothing.
        """
        for link in list(chosen_links):
            if budget <= 0:
                break
            if _in_small_cycle(link, chosen_links, options.max_cycle):
                continue
            candidates = _simple_paths(adjacency, link[0], link[1], options.max_cycle - 1, link)
            best: tuple[int, float, list[State]] | None = None
            for candidate in candidates:
                new_nodes = [node for node in candidate if node not in chosen_nodes]
                if len(new_nodes) > budget or start in new_nodes or end in new_nodes:
                    continue
                weight = sum(weights[_key(a, b)] for a, b in zip(candidate, candidate[1:]))
                ranked = (len(new_nodes), weight, candidate)
                if best is None or ranked[:2] < best[:2]:
                    best = ranked
            if best is None:
                trace.append(f"no cycle of {options.max_cycle} or fewer edges closes {_label(link)}")
                continue
            _, _, candidate = best
            added = 0
            for node in candidate:
                if node not in chosen_nodes:
                    chosen_nodes.append(node)
                    added += 1
            for pair in (_key(a, b) for a, b in zip(candidate, candidate[1:])):
                if pair not in chosen_links:
                    chosen_links.append(pair)
            budget -= added
            trace.append(f"closed a cycle over {_label(link)} with {added} further molecule(s)")

    def _emit(
        self,
        source: Ligand,
        target: Ligand,
        positions: Sequence[Position],
        groups: Sequence[_Group],
        chosen_nodes: Sequence[State],
        chosen_links: Sequence[tuple[State, State]],
        weights: Mapping[tuple[State, State], float],
        start: State,
        end: State,
        trace: list[str],
    ) -> IntermediateProposal:
        """Build the chosen states as molecules and package them as a proposal.

        A state that will not build, or that turns out to *be* one of the parents, is not a
        failure: it is dropped, its links are re-pointed or discarded, and the subnetwork is
        re-checked for whether it still bridges the gap. Two states that build the same
        molecule collapse onto one name, because the name is content-addressed and a
        proposal naming one molecule twice would ask the pipeline to invent it twice.
        """
        names: dict[State, str] = {start: source.name, end: target.name}
        molecules: list[ProposedMolecule] = []
        seen: dict[str, State] = {}
        for state in chosen_nodes:
            if state in names:
                continue
            built = assemble(source, target, positions, _expand(groups, state, len(positions)))
            if built is None:
                trace.append(f"state {_state_label(state)} does not build; dropped")
                continue
            mol, atom_map = built
            identity = _identity(mol)
            if identity == _identity(source.mol):
                names[state] = source.name
                continue
            if identity == _identity(target.mol):
                names[state] = target.name
                continue
            if identity in seen:
                names[state] = names[seen[identity]]
                trace.append(f"state {_state_label(state)} is the same molecule as {_state_label(seen[identity])}")
                continue
            proposed = ProposedMolecule(
                mol=mol,
                parents=(source.name, target.name),
                parent_atom_map=atom_map,
                detail={"state": _state_label(state)},
            )
            names[state] = intermediate_name(proposed.parents, proposed.mol)
            seen[identity] = state
            molecules.append(proposed)
            trace.append(f"state {_state_label(state)} -> {names[state]}")

        links: list[ProposedLink] = []
        emitted: set[tuple[str, str]] = set()
        for left, right in chosen_links:
            if left not in names or right not in names:
                continue
            a, b = names[left], names[right]
            if a == b or tuple(sorted((a, b))) in emitted:
                continue
            emitted.add(tuple(sorted((a, b))))
            score = 1.0 / math.sqrt(weights[_key(left, right)])
            links.append(ProposedLink(source=a, target=b, hint=1.0 - score, detail={"link_score": round(score, 4)}))

        if not molecules:
            trace.append("no molecule survived construction")
            return IntermediateProposal(
                source=source.name,
                target=target.name,
                generator=self.name,
                rejection="no_molecule_built",
                trace=tuple(trace),
            )
        trace.append(f"proposing {len(molecules)} molecule(s) and {len(links)} link(s)")
        return IntermediateProposal(
            source=source.name,
            target=target.name,
            generator=self.name,
            molecules=tuple(molecules),
            links=tuple(links),
            trace=tuple(trace),
        )


def _key(left: State, right: State) -> tuple[State, State]:
    """Return the unordered key for a link between two states."""
    return (left, right) if left <= right else (right, left)


def _label(link: tuple[State, State]) -> str:
    """Human-readable name for a link, for the trace."""
    return f"{_state_label(link[0])}~{_state_label(link[1])}"


def _state_label(state: State) -> str:
    """Compact, stable name for a state: one letter per position group."""
    return "".join(label[0].upper() for label in state)


def _identity(mol: Chem.Mol) -> str:
    """Canonical SMILES with hydrogens suppressed.

    The same identity the pipeline dedupes synthetic ligands on, so a molecule this
    generator considers new cannot come back from :func:`build_candidate` as a duplicate.
    """
    try:
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
    except Exception:  # noqa: BLE001 - a molecule too broken to strip is named from itself
        return Chem.MolToSmiles(mol)


def _in_small_cycle(link: tuple[State, State], links: Sequence[tuple[State, State]], max_cycle: int) -> bool:
    """Whether *link* already lies in a cycle of at most *max_cycle* edges."""
    adjacency: dict[State, list[State]] = {}
    for left, right in links:
        if _key(left, right) == link:
            continue
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    if link[0] not in adjacency or link[1] not in adjacency:
        return False
    distances = _hop_distances(adjacency, link[0])
    return distances.get(link[1], max_cycle + 1) <= max_cycle - 1

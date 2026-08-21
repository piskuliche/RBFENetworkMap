"""Intermediate generation as a pipeline stage: gaps, budgets, and knob precedence.

Everything here is driven with ``DummyIntermediateGenerator`` and a mapper that declines
named pairs, for the reason ``tests/conftest.py`` gives for every other Dummy: asking a
real generator for a molecule that happens to exercise a particular seam means finding a
ligand pair that happens to produce one. The chemistry is Phase 4c's problem; the wiring
is this file's.

The gap is manufactured, not found. ``GatedMapper`` refuses exactly the pairs it is told
to, which is the only way to get a *specific* pair rejected while every other pair stays
feasible -- and a specific rejected pair is what a gap is.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import pytest
from rdkit import Chem

from tests.conftest import DummyIntermediateGenerator, DummyMapper, DummyScorer, make_transformation
from rbfenetmap.core.exceptions import NetworkPlanError
from rbfenetmap.core.intermediates import (
    IntermediateOptions,
    IntermediateProposal,
    ProposedLink,
    ProposedMolecule,
    describe_intermediate_attempts,
    intermediate_name,
)
from rbfenetmap.core.models import EdgeKind, IntermediateRecord, Ligand, LigandProvenance
from rbfenetmap.core.options import MappingOptions, NetworkOptions
from rbfenetmap.core.pipeline import _intermediate_gaps, augment_with_intermediates, build_candidate, build_network
from rbfenetmap.io.networkio import dump_network, load_network, network_to_dict
from rbfenetmap.plugins.planners import create_planner

#: The molecule every proposal in this file inserts. Bromine is deliberately absent from
#: the ligand set, so an accepted proposal is unambiguously a *new* vertex rather than a
#: rediscovery of one that was already there.
BRIDGE_SMILES = "Brc1ccccc1C(=O)N"


class GatedMapper(DummyMapper):
    """The identity mapper, refusing the unordered pairs it was told to refuse."""

    name: ClassVar[str] = "gated"

    def __init__(self, blocked=()) -> None:
        """Store the pairs to decline, normalized to unordered tuples."""
        super().__init__()
        self.blocked = {tuple(sorted(pair)) for pair in blocked}

    def supports_pair(self, source: Ligand, target: Ligand) -> bool:
        """Decline a blocked pair, so ``build_candidate`` records a mapper rejection."""
        return tuple(sorted((source.name, target.name))) not in self.blocked


class CountingGenerator(DummyIntermediateGenerator):
    """A dummy generator that records every gap it was offered."""

    def __init__(self, proposal=None, **kwargs) -> None:
        """Start with an empty call log."""
        super().__init__(proposal, **kwargs)
        self.offered: list[tuple[str, str]] = []

    def propose(self, source, target, options, mapping_options):
        """Log the gap, then behave like the dummy."""
        self.offered.append(tuple(sorted((source.name, target.name))))
        return super().propose(source, target, options, mapping_options)


@pytest.fixture(scope="module")
def pair(benzamides):
    """Two co-posed ligands whose direct edge the tests block."""
    return {name: benzamides[name] for name in ("bza_F", "bza_Cl")}


def bridge_proposal(source: str, target: str, *, links: bool = True, smiles: str = BRIDGE_SMILES):
    """Return a proposal inserting one molecule between *source* and *target*."""
    mol = Chem.MolFromSmiles(smiles)
    parents = tuple(sorted((source, target)))
    molecule = ProposedMolecule(mol=mol, parents=parents)
    invented = intermediate_name(parents, mol)
    return IntermediateProposal(
        source=source,
        target=target,
        generator="dummy",
        molecules=(molecule,),
        links=(ProposedLink(source, invented), ProposedLink(invented, target)) if links else (),
    )


def augment(ligands, generator, options, *, mapper=None, candidates=None):
    """Run the whole stage over *ligands* with the direct pair already rejected."""
    names = sorted(ligands)
    mapper = mapper or GatedMapper([tuple(names[:2])])
    scorer = DummyScorer()
    mapping_options = MappingOptions()
    if candidates is None:
        candidates = [
            build_candidate(ligands[a], ligands[b], mapper, scorer, mapping_options, options)
            for index, a in enumerate(names)
            for b in names[index + 1 :]
        ]
    return augment_with_intermediates(ligands, candidates, generator, mapper, scorer, mapping_options, options)


def synthetic(ligand: Ligand, parents: tuple[str, str]) -> Ligand:
    """Wrap a real molecule as though the package had invented it."""
    provenance = LigandProvenance(
        kind="intermediate", generator="dummy", parents=parents, pose_method="parent_atom_map", pose_rmsd=0.1
    )
    return Ligand.synthesized(ligand.mol, intermediate_name(parents, ligand.mol), provenance)


class TestOptions:
    def test_generation_is_off_by_default(self):
        options = NetworkOptions()
        assert options.intermediates.mode == "off"
        assert not options.generates_intermediates

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "all"},
            {"generator": ""},
            {"max_intermediates": 0},
            {"max_gaps": 0},
            {"max_molecules": 0},
            {"max_pose_attempts": 0},
            {"pose_rmsd_factor": 0.0},
        ],
    )
    def test_nonsense_is_refused_at_construction(self, kwargs):
        with pytest.raises(ValueError):
            IntermediateOptions(**kwargs)

    def test_network_options_rechecks_the_mode(self):
        """The nested block is validated again from the outside, as ``softcore`` is."""
        options = IntermediateOptions()
        object.__setattr__(options, "mode", "everything")
        with pytest.raises(ValueError, match="intermediates.mode"):
            NetworkOptions(intermediates=options)

    @pytest.mark.parametrize("n_edges,expected", [(None, None), (4, 0), (6, 2), (3, -1)])
    def test_headroom_is_the_budget_minus_the_spanning_tree(self, n_edges, expected):
        options = NetworkOptions(n_edges=n_edges, require_connected=False)
        assert options.intermediate_headroom(5) == expected

    def test_compat_pins_generation_off(self):
        assert NetworkOptions.preset("v0.4").intermediates == IntermediateOptions()

    def test_the_block_round_trips(self, benzamides, tmp_path):
        options = NetworkOptions(intermediates=IntermediateOptions(mode="gaps", max_gaps=3, seed=7))
        network = build_network(dict(list(benzamides.items())[:3]), network_options=options)
        loaded = load_network(dump_network(network, tmp_path / "n.json"))
        assert loaded.options.intermediates == options.intermediates

    def test_a_file_without_the_block_loads_as_off(self, benzamides, tmp_path):
        network = build_network(dict(list(benzamides.items())[:3]))
        path = dump_network(network, tmp_path / "n.json")
        payload = network_to_dict(network)
        del payload["options"]["intermediates"]
        path.write_text(__import__("json").dumps(payload, indent=2))
        assert load_network(path).options.intermediates == IntermediateOptions()


class TestGapSelection:
    def _pool(self, ligands, blocked, options):
        mapper = GatedMapper(blocked)
        scorer = DummyScorer()
        names = sorted(ligands)
        return [
            build_candidate(ligands[a], ligands[b], mapper, scorer, MappingOptions(), options)
            for index, a in enumerate(names)
            for b in names[index + 1 :]
        ]

    def test_bridge_mode_offers_only_cross_component_pairs(self, benzamides):
        """A rejected pair whose ends are still joined by a path is not a bridge gap."""
        ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl")}
        options = NetworkOptions(intermediates=IntermediateOptions(mode="bridge"))
        candidates = self._pool(ligands, [("bza_F", "bza_Cl")], options)
        assert _intermediate_gaps(ligands, candidates, options) == []

    def test_gaps_mode_offers_them(self, benzamides):
        ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl")}
        options = NetworkOptions(intermediates=IntermediateOptions(mode="gaps"))
        candidates = self._pool(ligands, [("bza_F", "bza_Cl")], options)
        assert _intermediate_gaps(ligands, candidates, options) == [("bza_Cl", "bza_F")]

    def test_a_banned_gap_is_never_offered(self, pair):
        """``A~M~B`` runs the pair the ban forbade, so a banned gap is not a gap."""
        options = NetworkOptions(
            banned_edges=("bza_Cl~bza_F",), intermediates=IntermediateOptions(mode="bridge"), require_connected=False
        )
        generator = CountingGenerator(bridge_proposal("bza_F", "bza_Cl"))
        result = augment(pair, generator, options)
        assert generator.offered == []
        assert result.records == ()
        assert set(result.ligands) == set(pair)

    def test_a_forced_pair_with_no_mapping_is_offered(self, pair):
        options = NetworkOptions(
            forced_edges=("bza_F~bza_Cl",), intermediates=IntermediateOptions(mode="bridge"), require_connected=False
        )
        generator = CountingGenerator()
        augment(pair, generator, options)
        assert generator.offered == [("bza_Cl", "bza_F")]

    def test_max_gaps_caps_what_is_offered(self, benzamides):
        ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl", "bza_Me")}
        options = NetworkOptions(intermediates=IntermediateOptions(mode="gaps", max_gaps=1), require_connected=False)
        generator = CountingGenerator()
        augment(ligands, generator, options, mapper=GatedMapper([("bza_F", "bza_Cl"), ("bza_H", "bza_Me")]))
        assert len(generator.offered) == 1


class TestAugmentation:
    def test_an_accepted_proposal_adds_the_vertex_and_its_sub_edges(self, pair):
        options = NetworkOptions(intermediates=IntermediateOptions(mode="bridge"), require_connected=False)
        result = augment(pair, DummyIntermediateGenerator(bridge_proposal("bza_F", "bza_Cl")), options)

        invented = result.synthetic_names
        assert len(invented) == 1
        assert result.ligands[invented[0]].synthetic
        assert result.ligands[invented[0]].provenance.parents == ("bza_Cl", "bza_F")
        sub_edges = {edge.unordered_key for edge in result.candidates if edge.feasible}
        assert sub_edges == {("bza_Cl", invented[0]), ("bza_F", invented[0])}
        assert result.records[0].accepted and result.records[0].names == invented

    def test_links_are_optional(self, pair):
        """No links means the obvious thing: each new molecule against each parent."""
        options = NetworkOptions(intermediates=IntermediateOptions(mode="bridge"), require_connected=False)
        result = augment(pair, DummyIntermediateGenerator(bridge_proposal("bza_F", "bza_Cl", links=False)), options)
        assert len(result.synthetic_names) == 1

    def test_a_proposal_that_does_not_bridge_leaves_the_ligands_exactly_unchanged(self, benzamides):
        """The analogue of "an edge that cannot be repaired is rejected, not mutated".

        ``bza_CF3`` against a bromobenzamide leaves the identity mapper too small a core,
        so one sub-edge survives and the other does not -- and half a bridge is no bridge.
        """
        ligands = {name: benzamides[name] for name in ("bza_H", "bza_CF3")}
        options = NetworkOptions(intermediates=IntermediateOptions(mode="bridge"), require_connected=False)
        before = set(ligands)
        result = augment(ligands, DummyIntermediateGenerator(bridge_proposal("bza_H", "bza_CF3")), options)

        assert set(result.ligands) == before
        assert result.synthetic_names == ()
        assert all(not edge.source.startswith("int_") for edge in result.candidates)
        assert all(not edge.target.startswith("int_") for edge in result.candidates)
        assert result.records[0].rejection == "sub_edges_do_not_bridge"

    def test_a_declining_generator_records_its_reason(self, pair):
        options = NetworkOptions(intermediates=IntermediateOptions(mode="bridge"), require_connected=False)
        result = augment(pair, DummyIntermediateGenerator(rejection="nothing_to_swap"), options)
        assert set(result.ligands) == set(pair)
        assert [record.rejection for record in result.records] == ["nothing_to_swap"]

    def test_a_molecule_that_is_already_a_ligand_is_not_added_twice(self, benzamides):
        """A duplicate is not a failure -- the bridge was already in the set."""
        ligands = {name: benzamides[name] for name in ("bza_F", "bza_Cl", "bza_Me")}
        options = NetworkOptions(intermediates=IntermediateOptions(mode="bridge"), require_connected=False)
        proposal = bridge_proposal("bza_F", "bza_Cl", smiles="Cc1ccccc1C(=O)N")
        result = augment(
            ligands, DummyIntermediateGenerator(proposal), options, mapper=GatedMapper([("bza_F", "bza_Cl")])
        )
        assert set(result.ligands) == set(ligands)
        assert "duplicates bza_Me" in " ".join(result.records[0].trace)

    def test_max_intermediates_stops_generation_and_says_so(self, benzamides):
        ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl", "bza_Me")}
        options = NetworkOptions(
            intermediates=IntermediateOptions(mode="gaps", max_intermediates=1), require_connected=False
        )
        generator = CountingGenerator(bridge_proposal("bza_F", "bza_Cl"))
        result = augment(ligands, generator, options, mapper=GatedMapper([("bza_F", "bza_Cl"), ("bza_H", "bza_Me")]))
        assert len(result.synthetic_names) == 1
        assert any("max_intermediates=1 reached" in message for message in result.unmet_constraints)

    def test_mode_off_returns_its_inputs_untouched(self, pair):
        options = NetworkOptions(require_connected=False)
        result = augment(pair, DummyIntermediateGenerator(bridge_proposal("bza_F", "bza_Cl")), options)
        assert result.ligands is pair
        assert result.records == ()


class TestEdgeBudget:
    def test_a_budget_with_no_headroom_invents_nothing_and_does_not_raise(self, benzamides):
        """``n_edges == n_real - 1`` is a spanning tree exactly; there is nothing to spend."""
        ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl")}
        options = NetworkOptions(n_edges=2, intermediates=IntermediateOptions(mode="gaps"), require_connected=False)
        generator = CountingGenerator(bridge_proposal("bza_F", "bza_Cl"))
        result = augment(ligands, generator, options, mapper=GatedMapper([("bza_F", "bza_Cl")]))

        assert result.synthetic_names == ()
        assert generator.offered == []
        assert any("left no room for intermediates" in message for message in result.unmet_constraints)

    def test_headroom_of_one_buys_exactly_one_vertex(self, pair):
        options = NetworkOptions(n_edges=2, intermediates=IntermediateOptions(mode="bridge"), require_connected=False)
        result = augment(pair, DummyIntermediateGenerator(bridge_proposal("bza_F", "bza_Cl")), options)
        assert len(result.synthetic_names) == 1
        assert result.unmet_constraints == ()

    def test_the_message_reaches_the_planned_network(self, benzamides, monkeypatch):
        ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl")}
        network = plan_with(
            monkeypatch,
            ligands,
            CountingGenerator(bridge_proposal("bza_F", "bza_Cl")),
            NetworkOptions(
                n_edges=2,
                intermediates=IntermediateOptions(mode="gaps"),
                require_connected=False,
                edges_per_ligand=1,
                min_cycle_coverage=0.0,
            ),
            blocked=[("bza_F", "bza_Cl")],
        )
        assert any("left no room for intermediates" in message for message in network.unmet_constraints)


def plan_with(monkeypatch, ligands, generator, options, *, blocked=(), planner="mst"):
    """Run ``build_network`` with *generator* standing in for the registered plugin."""
    import rbfenetmap.plugins.intermediates as registry

    monkeypatch.setattr(registry, "create_intermediate", lambda name, *a, **k: generator)
    return build_network(
        ligands, mapper=GatedMapper(blocked), scorer=DummyScorer(), planner=planner, network_options=options
    )


class TestPipelineWiring:
    def test_mode_off_never_constructs_a_generator(self, pair, monkeypatch):
        """Which is also what keeps the lazy-import rule: no generator, no import."""
        import rbfenetmap.plugins.intermediates as registry

        def explode(*args, **kwargs):
            raise AssertionError("a generator was constructed with generation off")

        monkeypatch.setattr(registry, "create_intermediate", explode)
        network = build_network(
            pair, mapper=DummyMapper(), scorer=DummyScorer(), network_options=NetworkOptions(min_cycle_coverage=0.0)
        )
        assert network.intermediates == ()
        assert network.synthetic_ligands == ()

    def test_the_generator_is_constructed_when_the_mode_asks_for_it(self, pair, monkeypatch):
        """Guards the previous test against passing for the wrong reason."""
        generator = CountingGenerator(bridge_proposal("bza_F", "bza_Cl"))
        network = plan_with(
            monkeypatch,
            pair,
            generator,
            NetworkOptions(intermediates=IntermediateOptions(mode="bridge"), min_cycle_coverage=0.0),
            blocked=[("bza_F", "bza_Cl")],
        )
        assert generator.offered == [("bza_Cl", "bza_F")]
        assert len(network.synthetic_ligands) == 1
        assert len(network.edges) == 2

    def test_the_attempt_record_lands_on_the_network_and_survives_a_round_trip(self, pair, monkeypatch, tmp_path):
        network = plan_with(
            monkeypatch,
            pair,
            CountingGenerator(bridge_proposal("bza_F", "bza_Cl")),
            NetworkOptions(intermediates=IntermediateOptions(mode="bridge"), min_cycle_coverage=0.0),
            blocked=[("bza_F", "bza_Cl")],
        )
        assert [record.accepted for record in network.intermediates] == [True]
        loaded = load_network(dump_network(network, tmp_path / "n.json"))
        assert loaded.intermediates == network.intermediates
        assert loaded.ligands[network.synthetic_ligands[0].name].synthetic

    def test_a_reserved_user_name_is_refused_only_when_generation_is_on(self, benzamides):
        ligands = {"int_3": benzamides["bza_H"], "bza_F": benzamides["bza_F"]}
        ligands = {name: dataclasses.replace(ligand, name=name) for name, ligand in ligands.items()}
        build_network(ligands, mapper=DummyMapper(), scorer=DummyScorer())
        with pytest.raises(ValueError, match="reserved prefix"):
            build_network(
                ligands,
                mapper=DummyMapper(),
                scorer=DummyScorer(),
                network_options=NetworkOptions(intermediates=IntermediateOptions(mode="bridge")),
            )

    def test_the_result_is_identical_at_any_job_count(self, benzamides, monkeypatch):
        """Generation is serial over a pool that was not, so ``jobs`` cannot move it."""
        ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl", "bza_Me")}

        def plan(jobs):
            return plan_with(
                monkeypatch,
                ligands,
                CountingGenerator(bridge_proposal("bza_F", "bza_Cl")),
                NetworkOptions(
                    intermediates=IntermediateOptions(mode="gaps"),
                    jobs=jobs,
                    min_cycle_coverage=0.0,
                    require_connected=False,
                ),
                blocked=[("bza_F", "bza_Cl")],
            )

        serial, parallel = plan(1), plan(4)
        assert list(serial.ligands) == list(parallel.ligands)
        assert [edge.key for edge in serial.edges] == [edge.key for edge in parallel.edges]
        assert len(serial.synthetic_ligands) == 1

    def test_the_adaptive_path_augments_after_the_loop(self, benzamides, monkeypatch):
        ligands = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl")}
        generator = CountingGenerator(bridge_proposal("bza_F", "bza_Cl"))
        network = plan_with(
            monkeypatch,
            ligands,
            generator,
            NetworkOptions(
                pair_evaluation="adaptive", intermediates=IntermediateOptions(mode="gaps"), min_cycle_coverage=0.0
            ),
            blocked=[("bza_F", "bza_Cl")],
        )
        # Once, over the settled pool -- not once per adaptive batch.
        assert generator.offered == [("bza_Cl", "bza_F")]
        assert len(network.synthetic_ligands) == 1


class TestVersusCBFE:
    """Intermediates win over CBFE, and it takes no precedence logic to make them.

    Generation runs before the planner and changes the pool it is handed, so a gap an
    intermediate closed is not a gap when CBFE eligibility is evaluated. Both tests use
    identical options but for the generator's answer.
    """

    def _plan(self, monkeypatch, pair, generator):
        return plan_with(
            monkeypatch,
            pair,
            generator,
            NetworkOptions(
                cbfe_mode="bridge", intermediates=IntermediateOptions(mode="bridge"), min_cycle_coverage=0.0
            ),
            blocked=[("bza_F", "bza_Cl")],
        )

    def test_a_gap_an_intermediate_closed_buys_no_cbfe_edge(self, pair, monkeypatch):
        network = self._plan(monkeypatch, pair, CountingGenerator(bridge_proposal("bza_F", "bza_Cl")))
        assert network.cbfe_edges == ()
        assert len(network.synthetic_ligands) == 1

    def test_a_gap_generation_failed_on_still_does(self, pair, monkeypatch):
        network = self._plan(monkeypatch, pair, CountingGenerator())
        assert [edge.kind for edge in network.cbfe_edges] == [EdgeKind.CBFE]
        assert network.synthetic_ligands == ()


class TestRedundancyTargets:
    """Synthetic vertices are excluded from the targets, never from the pool."""

    def _network(self, ligands, options):
        candidates = [
            make_transformation("bza_H", "bza_F", cost=1.0),
            make_transformation("bza_F", "bza_Cl", cost=1.0),
            make_transformation("bza_Cl", "bza_H", cost=2.0),
            make_transformation("bza_H", sorted(ligands)[-1], cost=1.0),
        ]
        return create_planner("mst").plan(ligands, candidates, options)

    def _ligands(self, benzamides, *, invented: bool):
        real = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl")}
        extra = synthetic(benzamides["bza_Me"], ("bza_H", "bza_F"))
        if not invented:
            extra = dataclasses.replace(extra, provenance=None)
        return {**real, extra.name: extra}

    def test_a_synthetic_pendant_is_neither_deficient_nor_uncovered(self, benzamides):
        ligands = self._ligands(benzamides, invented=True)
        network = self._network(ligands, NetworkOptions())
        assert network.unmet_constraints == ()
        assert len(network.ligands) == 4

    def test_the_same_shape_with_a_real_pendant_reports_both_shortfalls(self, benzamides):
        """The control: without the guard the two cases would be indistinguishable."""
        ligands = self._ligands(benzamides, invented=False)
        with pytest.warns(UserWarning):
            network = self._network(ligands, NetworkOptions())
        assert any("edges_per_ligand" in message for message in network.unmet_constraints)
        assert any("min_cycle_coverage" in message for message in network.unmet_constraints)

    def test_a_synthetic_vertex_may_still_carry_a_cycle(self, benzamides):
        """``A-M-B-...-A`` is a genuine consistency check, so nothing forbids it."""
        real = {name: benzamides[name] for name in ("bza_H", "bza_F", "bza_Cl")}
        extra = synthetic(benzamides["bza_Me"], ("bza_H", "bza_F"))
        ligands = {**real, extra.name: extra}
        candidates = [
            make_transformation("bza_H", extra.name, cost=1.0),
            make_transformation(extra.name, "bza_F", cost=1.0),
            make_transformation("bza_F", "bza_Cl", cost=1.0),
            make_transformation("bza_Cl", "bza_H", cost=1.0),
        ]
        network = create_planner("mst").plan(ligands, candidates, NetworkOptions())
        assert len(network.edges) == 4
        assert network.unmet_constraints == ()


class TestDiagnostics:
    def test_the_disconnection_message_names_generation_and_the_gaps(self, benzamides, monkeypatch):
        ligands = {name: benzamides[name] for name in ("bza_F", "bza_Cl")}
        with pytest.raises(NetworkPlanError) as excinfo:
            plan_with(
                monkeypatch,
                ligands,
                CountingGenerator(rejection="no_common_core"),
                NetworkOptions(intermediates=IntermediateOptions(mode="bridge")),
                blocked=[("bza_F", "bza_Cl")],
            )
        message = str(excinfo.value)
        assert "intermediates.mode='bridge'" in message
        assert "bza_Cl~bza_F: refused (no_common_core)" in message

    def test_an_empty_record_adds_nothing(self):
        assert describe_intermediate_attempts(()) == ""

    def test_the_summary_names_accepted_ligands(self):
        record = IntermediateRecord("a", "b", "dummy", accepted=True, names=("int_a_b_abc123",))
        assert "accepted ['int_a_b_abc123']" in describe_intermediate_attempts((record,))


class TestCLI:
    """The flags reach the options object, and ``--compat`` refuses them."""

    def test_the_flags_build_the_options(self):
        from rbfenetmap.cli._args import build_network_options
        from rbfenetmap.cli.main import build_parser

        args = build_parser().parse_args(
            [
                "plan",
                "--ligands",
                "x.sdf",
                "--out",
                "n.json",
                "--intermediates",
                "gaps",
                "--max-intermediates",
                "3",
                "--max-intermediate-gaps",
                "2",
                "--intermediates-per-gap",
                "1",
            ]
        )
        options = build_network_options(args).intermediates
        assert (options.mode, options.max_intermediates, options.max_gaps, options.max_molecules) == ("gaps", 3, 2, 1)

    def test_the_default_is_off(self):
        from rbfenetmap.cli._args import build_network_options
        from rbfenetmap.cli.main import build_parser

        args = build_parser().parse_args(["plan", "--ligands", "x.sdf", "--out", "n.json"])
        assert build_network_options(args).intermediates == IntermediateOptions()

    def test_compat_refuses_the_flag(self, tmp_path, capsys):
        from rbfenetmap.cli.main import main
        from tests.test_compat import GOLDEN_SDF

        code = main(
            [
                "plan",
                "--ligands",
                str(GOLDEN_SDF),
                "--out",
                str(tmp_path / "n.json"),
                "--compat",
                "v0.4",
                "--intermediates",
                "bridge",
            ]
        )
        assert code == 1
        assert "--intermediates" in capsys.readouterr().err

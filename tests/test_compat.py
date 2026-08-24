"""``--compat``: reproducing a released behaviour exactly.

The golden-baseline tests here are the ones the rest of the network-knobs epic leans on.
Two assertions run, and the difference between them is the whole point:

- the **default** path still reproduces the v0.4.0 baseline. True today, and it is expected
  to stop being true the first time a phase deliberately moves a default;
- **``--compat v0.4``** reproduces it too, and that one must never stop being true.

When the first assertion legitimately fails, the fix is to update it and record why in the
changelog -- never to regenerate ``tests/data/golden_benzamides.json``. Regenerating the
baseline whenever it disagrees with the code turns it into a transcript of the current
behaviour, which checks nothing.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tests.golden import load_golden, network_fingerprint
from rbfenetmap.cli._args import COMPAT_CLI_PINS
from rbfenetmap.cli.main import main
from rbfenetmap.core.options import COMPAT_LEVELS, NetworkOptions
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.io.loaders import load_ligands
from rbfenetmap.io.networkio import dump_network, load_network, network_to_dict

#: The structures the baseline was captured from, tracked in the repository.
#:
#: Deliberately *not* ``examples/data/benzamides.sdf``. That file is gitignored and
#: regenerated on demand by ``examples/data/make_conformers.py``, so it does not exist in a
#: fresh clone and these tests would not run at all in CI. It is also the wrong kind of
#: input for a golden test even where it does exist: its coordinates come from a
#: constrained embedding, so an RDKit upgrade could shift them, move ``core_rmsd``, and
#: change every edge cost -- a baseline failure caused by the input rather than by the
#: planner. A golden test's input has to be pinned exactly as firmly as its expected output.
GOLDEN_SDF = Path(__file__).resolve().parent / "data" / "golden_benzamides.sdf"


@pytest.fixture(scope="module")
def example_ligands():
    """The nine-ligand series the golden baseline was captured from."""
    return load_ligands([str(GOLDEN_SDF)])


class TestPreset:
    def test_sets_the_compat_label(self):
        assert NetworkOptions.preset("v0.4").compat == "v0.4"

    def test_unknown_level_is_refused(self):
        with pytest.raises(ValueError, match="Unknown compat level"):
            NetworkOptions.preset("v0.3")

    def test_unknown_level_is_refused_on_the_field_too(self):
        with pytest.raises(ValueError, match="Unknown compat level"):
            NetworkOptions(compat="v9.9")

    def test_overrides_apply_on_top(self):
        options = NetworkOptions.preset("v0.4", hub="bza_H", jobs=4, banned_edges=("a~b",))
        assert (options.hub, options.jobs, options.banned_edges) == ("bza_H", 4, ("a~b",))
        assert options.compat == "v0.4"
        assert options.edges_per_ligand == 2  # untouched by the overrides

    def test_transcription_matches_the_v0_4_defaults(self):
        """The preset is a literal transcription; this checks it was transcribed right.

        v0.4.0's values *are* the current defaults, so today the two agree on everything
        but the label. When a later phase deliberately moves a default this test starts
        failing, and the correct response is to assert the specific difference here --
        not to rebuild the preset from the defaults, which would silently un-pin it.
        """
        pinned = dataclasses.asdict(NetworkOptions.preset("v0.4"))
        current = dataclasses.asdict(NetworkOptions())
        differing = {k for k in pinned if pinned[k] != current[k]}
        assert differing == {"compat"}

    def test_every_declared_level_is_constructible(self):
        for level in COMPAT_LEVELS:
            assert NetworkOptions.preset(level).compat == level


class TestGoldenBaseline:
    def test_default_path_reproduces_the_baseline(self, example_ligands):
        assert network_fingerprint(build_network(example_ligands)) == load_golden("golden_benzamides")

    def test_compat_reproduces_the_baseline(self, example_ligands):
        network = build_network(example_ligands, network_options=NetworkOptions.preset("v0.4"))
        fingerprint = network_fingerprint(network)
        golden = load_golden("golden_benzamides")
        assert fingerprint == golden

    def test_the_two_paths_agree_with_each_other(self, example_ligands):
        """Guards the case where both drift together and the baseline is regenerated."""
        default = network_fingerprint(build_network(example_ligands))
        pinned = network_fingerprint(build_network(example_ligands, network_options=NetworkOptions.preset("v0.4")))
        assert default == pinned


class TestSerialization:
    def test_compat_round_trips(self, example_ligands, tmp_path):
        network = build_network(example_ligands, network_options=NetworkOptions.preset("v0.4"))
        path = dump_network(network, tmp_path / "n.json")
        assert load_network(path).options.compat == "v0.4"

    def test_an_unpinned_network_writes_no_compat_key(self, example_ligands):
        """Absent means "not pinned", so a null would differ from every existing file."""
        options = network_to_dict(build_network(example_ligands))["options"]
        assert "compat" not in options

    def test_every_network_option_survives_the_round_trip(self, example_ligands, tmp_path):
        """A field missing from networkio silently reverts to its default on reload.

        That is the quietest failure in the package: the file loads, the network is intact,
        and the options block claims a setting the run never used.
        """
        options = NetworkOptions(
            cycle_coverage_mode="edge", max_diameter=4, n_redundancy=3, hub_selection="min_total_cost", max_cycle_size=5
        )
        network = build_network(example_ligands, network_options=options)
        reloaded = load_network(dump_network(network, tmp_path / "n.json")).options
        for field in dataclasses.fields(NetworkOptions):
            if field.name in _NETWORK_DESTS:
                assert getattr(reloaded, field.name) == getattr(options, field.name), field.name

    def test_a_file_without_compat_loads_as_unpinned(self, example_ligands, tmp_path):
        path = dump_network(build_network(example_ligands), tmp_path / "n.json")
        payload = json.loads(path.read_text())
        assert "compat" not in payload["options"]
        assert load_network(path).options.compat is None


class TestCLIConflicts:
    """A pinned knob named alongside ``--compat`` is refused, naming the flag."""

    def _plan(self, tmp_path, *extra):
        return main(
            ["plan", "--ligands", str(GOLDEN_SDF), "--out", str(tmp_path / "n.json"), "--compat", "v0.4", *extra]
        )

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--edges-per-ligand", "3"),
            ("--max-softcore-atoms", "20"),
            ("--cbfe", "bridge"),
            ("--planner", "star"),
            ("--scorer", "lomaplike"),
            ("--min-cycle-coverage", "0.5"),
            ("--pair-evaluation", "adaptive"),
            # Phase 1. A knob absent from the pin tables silently stops --compat from
            # reproducing v0.4, and nothing else in this suite would catch the omission.
            ("--cycle-coverage-mode", "edge"),
            ("--max-diameter", "5"),
            ("--n-redundancy", "3"),
            ("--hub-selection", "min_total_cost"),
            ("--cluster-by", "charge"),
            ("--cluster-bridges", "3"),
        ],
    )
    def test_pinned_knob_is_refused(self, tmp_path, capsys, flag, value):
        assert self._plan(tmp_path, flag, value) == 1
        err = capsys.readouterr().err
        assert flag in err
        assert "--compat v0.4" in err

    def test_a_store_true_knob_is_refused(self, tmp_path, capsys):
        assert self._plan(tmp_path, "--allow-disconnected") == 1
        assert "--allow-disconnected" in capsys.readouterr().err

    def test_several_conflicts_are_reported_together(self, tmp_path, capsys):
        assert self._plan(tmp_path, "--edges-per-ligand", "3", "--cbfe", "bridge") == 1
        err = capsys.readouterr().err
        assert "--cbfe" in err and "--edges-per-ligand" in err

    def test_passing_the_pinned_value_is_still_a_conflict(self, tmp_path, capsys):
        """Naming a knob is the contradiction, not disagreeing with it.

        Accepting ``--edges-per-ligand 2`` because it happens to match would make the
        rule depend on the current default: the same command would start failing the day
        that default moved, which is exactly the surprise ``--compat`` exists to prevent.
        """
        assert self._plan(tmp_path, "--edges-per-ligand", "2") == 1
        assert "--edges-per-ligand" in capsys.readouterr().err


class TestCLIPermitted:
    """Run-specific and operational flags stay usable alongside ``--compat``."""

    @pytest.mark.integration
    @pytest.mark.parametrize("extra", [[], ["--jobs", "2"], ["--no-progress"], ["--banned-edge", "bza_H~bza_F"]])
    def test_unpinned_flags_are_accepted(self, tmp_path, extra):
        out = tmp_path / "n.json"
        assert main(["plan", "--ligands", str(GOLDEN_SDF), "--out", str(out), "--compat", "v0.4", *extra]) == 0
        assert load_network(out).options.compat == "v0.4"

    @pytest.mark.integration
    def test_compat_run_reproduces_the_baseline_end_to_end(self, tmp_path):
        out = tmp_path / "n.json"
        assert main(["plan", "--ligands", str(GOLDEN_SDF), "--out", str(out), "--compat", "v0.4"]) == 0
        assert network_fingerprint(load_network(out)) == load_golden("golden_benzamides")


#: ``NetworkOptions`` field to the CLI destination that sets it, for the two tests below
#: that walk the dataclass rather than the parser. Most are the same string; the handful
#: that are not is exactly why this is written out rather than derived.
_NETWORK_DESTS = {
    "pair_strategy": "pair_strategy",
    "n_edges": "n_edges",
    "edges_per_ligand": "edges_per_ligand",
    "min_cycle_coverage": "min_cycle_coverage",
    "require_connected": "allow_disconnected",
    "edge_direction": "edge_direction",
    "prefilter": "prefilter",
    "prefilter_k": "prefilter_k",
    "prefilter_min_tanimoto": "prefilter_min_tanimoto",
    "selection_objective": "selection_objective",
    "cycle_coverage_mode": "cycle_coverage_mode",
    "max_cycle_size": "max_cycle_size",
    "max_diameter": "max_diameter",
    "n_redundancy": "n_redundancy",
    "hub_selection": "hub_selection",
    "pair_evaluation": "pair_evaluation",
    "adaptive_initial_neighbors": "adaptive_initial_neighbors",
    "adaptive_batch_size": "adaptive_batch_size",
    "consistency": "consistency",
    "cbfe_mode": "cbfe",
    "cbfe_base_cost": "cbfe_base_cost",
    "cbfe_atom_weight": "cbfe_atom_weight",
}


class TestPinnedSurface:
    def test_every_pinned_dest_exists_on_the_plan_parser(self):
        """A typo in the pin table would silently pin nothing."""
        from rbfenetmap.cli.main import build_parser

        plan = build_parser()._subparsers._group_actions[0].choices["plan"]
        dests = {action.dest for action in plan._actions}
        missing = sorted(set(COMPAT_CLI_PINS["v0.4"]) - dests)
        assert not missing, f"pinned destination(s) not on the parser: {missing}"

    def test_every_algorithmic_network_option_is_pinned(self):
        """The omission the other tests cannot catch.

        ``test_every_pinned_dest_exists_on_the_parser`` catches a *typo* in the pin table.
        Nothing catches a knob that was simply never added to it -- the run would succeed,
        report ``compat="v0.4"``, and quietly plan under the new default. So this walks the
        options dataclass instead and requires every algorithmic field to be accounted for,
        which fails the moment a later phase adds a field and forgets the table.
        """
        import dataclasses

        # Not pinned by design: ligand intent, operational knobs, and the compat label
        # itself. See COMPAT_CLI_PINS' docstring for why each is excluded.
        unpinned = {
            "hub",
            "explicit_pairs",
            "forced_edges",
            "banned_edges",
            "show_progress",
            "jobs",
            "softcore",
            "compat",
        }
        pins = COMPAT_CLI_PINS["v0.4"]
        unaccounted = sorted(
            field.name
            for field in dataclasses.fields(NetworkOptions)
            if field.name not in unpinned and _NETWORK_DESTS.get(field.name) not in pins
        )
        assert not unaccounted, (
            f"NetworkOptions field(s) neither pinned in COMPAT_CLI_PINS nor listed as deliberately "
            f"unpinned: {unaccounted}"
        )

    def test_the_two_pin_tables_agree_with_each_other(self):
        """A knob pinned in one table and not the other is only half-pinned.

        ``--compat`` writes the CLI table; the library path writes the preset. They are two
        literal transcriptions of the same release, so they have to agree even after a
        default moves and both stop agreeing with the default.
        """
        pinned = NetworkOptions.preset("v0.4")
        pins = COMPAT_CLI_PINS["v0.4"]
        disagreements = {}
        for field_name, dest in _NETWORK_DESTS.items():
            if dest not in pins:
                continue
            expected = pins[dest]
            if dest == "allow_disconnected":  # the CLI states the negation
                expected = not expected
            actual = getattr(pinned, field_name)
            if actual != expected:
                disagreements[field_name] = (actual, expected)
        assert not disagreements

    def test_run_specific_intent_is_not_pinned(self):
        """Pinning these would make --compat useless for planning a real ligand set."""
        pins = COMPAT_CLI_PINS["v0.4"]
        for dest in ("hub", "forced_edge", "banned_edge", "explicit_edge", "ligands", "jobs", "progress", "out"):
            assert dest not in pins

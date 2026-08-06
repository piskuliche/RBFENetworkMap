"""End-to-end tests: pipeline, serialization, exporters, and the CLI."""

from __future__ import annotations

import json

import networkx as nx
import pytest

from rbfenetmap.cli.main import main
from rbfenetmap.core.exceptions import ExporterError
from rbfenetmap.core.options import NetworkOptions, SoftcorePolicy
from rbfenetmap.core.pipeline import build_network
from rbfenetmap.io.amber_masks import build_amber_masks
from rbfenetmap.io.networkio import dump_network, load_network


@pytest.fixture(scope="module")
def planned(request):
    """A planned network over the co-posed benzamide series."""
    from .conftest import make_coposed

    ligands = make_coposed(
        {
            "bza_H": "c1ccccc1C(=O)N",
            "bza_F": "Fc1ccccc1C(=O)N",
            "bza_Cl": "Clc1ccccc1C(=O)N",
            "bza_Me": "Cc1ccccc1C(=O)N",
            "bza_CF3": "FC(F)(F)c1ccccc1C(=O)N",
        },
        "c1ccccc1C(=O)N",
    )
    del request
    return build_network(
        ligands,
        mapper="mcss-e2",
        scorer="linear",
        planner="mst",
        network_options=NetworkOptions(edges_per_ligand=2, softcore=SoftcorePolicy(max_softcore_atoms=12)),
    )


@pytest.mark.integration
class TestPipeline:
    def test_network_is_connected_and_redundant(self, planned):
        graph = planned.to_networkx()
        assert nx.is_connected(graph)
        assert min(dict(graph.degree()).values()) >= 2

    def test_every_selected_edge_has_at_most_one_softcore_region(self, planned):
        # The invariant the whole package exists to enforce.
        for edge in planned.edges:
            assert edge.repair.n_fragments_after[0] <= 1
            assert edge.repair.n_fragments_after[1] <= 1

    def test_candidates_retain_the_rejected_edges(self, planned):
        assert len(planned.candidates) >= len(planned.edges)
        for candidate in planned.rejected:
            assert candidate.score.rejections, "a rejected candidate must say why"

    def test_fewer_than_two_ligands_is_an_error(self, benzamides):
        with pytest.raises(ValueError, match="at least two ligands"):
            build_network([benzamides["bza_H"]])

    def test_duplicate_names_are_rejected(self, benzamides):
        ligand = benzamides["bza_H"]
        with pytest.raises(ValueError, match="Duplicate ligand name"):
            build_network([ligand, ligand])

    def test_adaptive_evaluation_stops_before_all_pairs(
        self, benzamides, dummy_mapper, dummy_scorer, monkeypatch, capsys
    ):
        from .conftest import make_transformation

        def fake_evaluate(
            ligands, pairs, mapper, scorer, mapping_options, network_options, *, progress_callback=None
        ):
            del ligands, mapper, scorer, mapping_options, network_options
            if progress_callback:
                progress_callback(len(pairs))
            return [make_transformation(*pair) for pair in pairs]

        monkeypatch.setattr("rbfenetmap.core.pipeline.evaluate_pairs", fake_evaluate)
        network = build_network(
            benzamides,
            mapper=dummy_mapper,
            scorer=dummy_scorer,
            planner="mst",
            network_options=NetworkOptions(
                pair_evaluation="adaptive",
                adaptive_initial_neighbors=1,
                adaptive_batch_size=1,
                edges_per_ligand=1,
                min_cycle_coverage=0.0,
                show_progress=True,
            ),
        )

        assert nx.is_connected(network.to_networkx())
        assert len(network.candidates) < len(benzamides) * (len(benzamides) - 1) // 2
        progress = capsys.readouterr().err
        assert "Mapping pairs" in progress
        assert "stopped early" in progress

    def test_adaptive_evaluation_keeps_trying_bridges_until_connected(
        self, benzamides, dummy_mapper, dummy_scorer, monkeypatch
    ):
        from .conftest import make_transformation

        names = sorted(benzamides)
        feasible_chain = {tuple(sorted(pair)) for pair in zip(names, names[1:])}

        def fake_evaluate(
            ligands, pairs, mapper, scorer, mapping_options, network_options, *, progress_callback=None
        ):
            del ligands, mapper, scorer, mapping_options, network_options
            if progress_callback:
                progress_callback(len(pairs))
            return [
                make_transformation(*pair, feasible=tuple(sorted(pair)) in feasible_chain) for pair in pairs
            ]

        monkeypatch.setattr("rbfenetmap.core.pipeline.evaluate_pairs", fake_evaluate)
        network = build_network(
            benzamides,
            mapper=dummy_mapper,
            scorer=dummy_scorer,
            planner="mst",
            network_options=NetworkOptions(
                pair_evaluation="adaptive",
                adaptive_initial_neighbors=1,
                adaptive_batch_size=1,
                edges_per_ligand=1,
                min_cycle_coverage=0.0,
            ),
        )

        assert nx.is_connected(network.to_networkx())
        assert feasible_chain <= {candidate.unordered_key for candidate in network.candidates if candidate.feasible}

    def test_parallel_pair_evaluation_handles_immutable_metadata(
        self, benzamides, dummy_mapper, dummy_scorer, capsys
    ):
        network = build_network(
            dict(list(benzamides.items())[:3]),
            mapper=dummy_mapper,
            scorer=dummy_scorer,
            planner="mst",
            network_options=NetworkOptions(
                jobs=2,
                edges_per_ligand=1,
                min_cycle_coverage=0.0,
                show_progress=True,
            ),
        )

        assert len(network.candidates) == 3
        progress = capsys.readouterr().err
        assert "Mapping pairs" in progress
        assert "3/3" in progress
        assert "stopped early" not in progress


class TestNetworkSerialization:
    def test_round_trip_preserves_the_network(self, planned, tmp_path):
        path = dump_network(planned, tmp_path / "network.json")
        reloaded = load_network(path)

        assert set(reloaded.ligands) == set(planned.ligands)
        assert len(reloaded.edges) == len(planned.edges)
        for original, restored in zip(planned.edges, reloaded.edges):
            assert restored.key == original.key
            assert restored.mapping.cc1 == original.mapping.cc1
            assert restored.mapping.cc2 == original.mapping.cc2
            assert restored.mapping.sc1 == original.mapping.sc1
            assert restored.score.total == pytest.approx(original.score.total)

    def test_rejected_candidates_survive_the_round_trip(self, planned, tmp_path):
        path = dump_network(planned, tmp_path / "network.json")
        reloaded = load_network(path)
        assert len(reloaded.rejected) == len(planned.rejected)

    def test_round_trip_preserves_adaptive_options(self, planned, tmp_path):
        from dataclasses import replace

        planned = replace(
            planned,
            options=NetworkOptions(
                pair_evaluation="adaptive",
                adaptive_initial_neighbors=2,
                adaptive_batch_size=7,
                selection_objective="connectivity_then_cycles",
                max_cycle_size=4,
            ),
        )
        reloaded = load_network(dump_network(planned, tmp_path / "adaptive.json"))

        assert reloaded.options is not None
        assert reloaded.options.pair_evaluation == "adaptive"
        assert reloaded.options.adaptive_initial_neighbors == 2
        assert reloaded.options.adaptive_batch_size == 7
        assert reloaded.options.selection_objective == "connectivity_then_cycles"
        assert reloaded.options.max_cycle_size == 4

    def test_incompatible_schema_version_is_refused(self, planned, tmp_path):
        path = tmp_path / "network.json"
        dump_network(planned, path)
        data = json.loads(path.read_text())
        data["schema_version"] = 999
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="schema version"):
            load_network(path)


class TestAmberMasks:
    def test_masks_reference_softcore_atom_names(self, planned):
        edge = planned.edges[0]
        masks = build_amber_masks(planned.ligands[edge.source], planned.ligands[edge.target], edge.mapping)
        assert masks.timask1 == ":SRC"
        assert masks.timask2 == ":DST"
        if edge.mapping.sc1:
            assert masks.scmask1.startswith(":SRC&@")

    def test_empty_softcore_yields_an_empty_mask(self, benzamides):
        from rbfenetmap.core.models import AtomMapping

        ligand = benzamides["bza_H"]
        mapping = AtomMapping.from_core_pairs(
            {i: i for i in range(ligand.n_atoms)}, n_atoms_1=ligand.n_atoms, n_atoms_2=ligand.n_atoms
        )
        masks = build_amber_masks(ligand, ligand, mapping)
        assert masks.scmask1 == "" and masks.scmask2 == ""

    def test_name_collision_is_caught(self, benzamides, monkeypatch):
        # Amber masks select by name, so a soft-core name shared with a core atom would
        # silently pull that core atom in. This must fail loudly instead.
        from rbfenetmap.core.models import AtomMapping, Ligand

        ligand = benzamides["bza_H"]
        collided = ["X"] * ligand.n_atoms
        monkeypatch.setattr(Ligand, "atom_names", property(lambda self: tuple(collided)))
        mapping = AtomMapping.from_core_pairs(
            {i: i for i in range(ligand.n_atoms - 1)}, n_atoms_1=ligand.n_atoms, n_atoms_2=ligand.n_atoms
        )
        with pytest.raises(ExporterError, match="also used by common-core atoms"):
            build_amber_masks(ligand, ligand, mapping)


@pytest.mark.integration
class TestExporters:
    @pytest.mark.parametrize("name", ["json", "edgelist", "graphml", "html"])
    def test_exporter_writes_files(self, planned, tmp_path, name):
        from rbfenetmap.plugins.exporters import create_exporter

        written = create_exporter(name).export(planned, tmp_path)
        assert written
        for path in written:
            assert path.exists() and path.stat().st_size > 0

    def test_amber_exporter_writes_one_runconfig_per_edge(self, planned, tmp_path):
        pytest.importorskip("yaml")
        import yaml

        from rbfenetmap.plugins.exporters import create_exporter

        written = create_exporter("amber").export(planned, tmp_path)
        runconfigs = [p for p in written if p.name.startswith("atommap_") and "~" in p.name]
        assert len(runconfigs) == len(planned.edges)

        payload = yaml.safe_load(runconfigs[0].read_text())
        assert set(payload) >= {"atommapindices", "atommapnames", "timask1", "scmask1", "scmask2"}
        assert len(payload["atommapindices"]) == len(payload["atommapnames"])
        assert (tmp_path / "edges.dat").exists()

    def test_amber_validate_catches_problems_before_writing(self, planned, tmp_path, monkeypatch):
        from rbfenetmap.core.models import Ligand
        from rbfenetmap.plugins.exporters import create_exporter

        monkeypatch.setattr(Ligand, "atom_names", property(lambda self: ("X",) * self.n_atoms))
        with pytest.raises(ExporterError, match="cannot produce valid Amber masks"):
            create_exporter("amber").validate(planned)
        assert not list(tmp_path.iterdir()), "validate must not write anything"

    def test_html_report_is_self_contained(self, planned, tmp_path):
        import re

        from rbfenetmap.plugins.exporters import create_exporter

        (path,) = create_exporter("html").export(planned, tmp_path)
        text = path.read_text()
        assert "<svg" in text

        # The property that matters is that nothing is *fetched*. XML namespace
        # declarations (xmlns, xmlns:rdkit, xmlns:xlink) contain URLs but are identifiers,
        # never dereferenced, so they are stripped before looking for real references.
        assert not re.search(r"\bsrc\s*=", text), "no element may load an external resource"
        assert not re.search(r"<link\b", text), "no external stylesheet"
        assert "<script" not in text, "no scripts"
        assert "@import" not in text, "no CSS imports"

        without_namespaces = re.sub(r"""xmlns(:\w+)?\s*=\s*['"][^'"]*['"]""", "", text)
        leftover = re.findall(r"""['"](?:https?:)?//[^'"]+""", without_namespaces)
        assert not leftover, f"report references external resources: {leftover[:3]}"

    def test_html_report_links_network_and_edge_cards(self, planned, tmp_path):
        from rbfenetmap.plugins.exporters import create_exporter

        (path,) = create_exporter("html").export(planned, tmp_path)
        text = path.read_text()

        first_edge = sorted(planned.edges, key=lambda e: e.score.total)[0]
        fragment = f"edge-{first_edge.key}"
        assert f'id="{fragment}"' in text or f"id='{fragment}'" in text
        assert f'href="#{fragment}"' in text or f"href='#{fragment}'" in text
        assert "Click a network edge or the index below" in text


class TestCLI:
    def test_help_for_every_subcommand(self, capsys):
        for command in ("plan", "score", "map", "export", "report", "plugins", "inspect"):
            with pytest.raises(SystemExit) as excinfo:
                main([command, "--help"])
            assert excinfo.value.code == 0
            assert capsys.readouterr().out

    def test_plugins_lists_unavailable_backends_with_all(self, capsys):
        assert main(["plugins", "--all"]) == 0
        out = capsys.readouterr().out
        assert "kartograf" in out
        assert "mcss-e2" in out

    def test_plugins_hides_unavailable_by_default(self, capsys):
        from rbfenetmap.plugins.mappers import available_mappers

        if "kartograf" in available_mappers():  # pragma: no cover - depends on the env
            pytest.skip("kartograf is installed")
        assert main(["plugins", "--kind", "mapper"]) == 0
        assert "kartograf" not in capsys.readouterr().out

    @pytest.mark.integration
    def test_plan_writes_a_network(self, planned, tmp_path, capsys):
        from rdkit import Chem

        sdf = tmp_path / "ligands.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            for ligand in planned.ligands.values():
                mol = Chem.Mol(ligand.mol)
                mol.SetProp("_Name", ligand.name)
                writer.write(mol)

        out = tmp_path / "network.json"
        code = main(["plan", "--ligands", str(sdf), "--out", str(out), "--edges-per-ligand", "2", "--show-rejected"])
        assert code == 0
        assert out.exists()
        assert "Planned" in capsys.readouterr().out

        network = load_network(out)
        network.validate()

    def test_n_edges_conflict_exits_nonzero(self, planned, tmp_path, capsys):
        from rdkit import Chem

        sdf = tmp_path / "ligands.sdf"
        with Chem.SDWriter(str(sdf)) as writer:
            for ligand in planned.ligands.values():
                mol = Chem.Mol(ligand.mol)
                mol.SetProp("_Name", ligand.name)
                writer.write(mol)

        code = main(["plan", "--ligands", str(sdf), "--n-edges", "2", "--out", str(tmp_path / "n.json")])
        assert code == 1
        assert "cannot connect" in capsys.readouterr().err

    def test_inspect_reports_an_unknown_edge(self, planned, tmp_path, capsys):
        path = dump_network(planned, tmp_path / "network.json")
        assert main(["inspect", "--network", str(path), "--edge", "nope~alsonope"]) == 1
        assert "not in" in capsys.readouterr().err

    @pytest.mark.integration
    def test_inspect_shows_the_repair_trace(self, planned, tmp_path, capsys):
        path = dump_network(planned, tmp_path / "network.json")
        edge = planned.edges[0]
        code = main(
            [
                "inspect",
                "--network",
                str(path),
                "--edge",
                edge.key,
                "--show-repair-trace",
                "--show-descriptors",
                "--show-masks",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "common core" in out
        assert "amber masks" in out
        assert "descriptors" in out

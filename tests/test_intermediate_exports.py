"""What leaves the package once a vertex was invented.

An invented ligand is only useful if the things downstream of the planner can tell it apart
from one the user supplied. Three of them have to:

- the **Amber exporter**, because ``edges.dat`` names residues and ``BuildEdges`` needs a
  parameterised topology for each one. A name in that file with no structure beside it is
  the single worst thing this feature can produce, so it is asserted directly;
- the **HTML report**, because a reader who takes an invented vertex for a measured ligand
  will act on a number nobody has measured;
- **GraphML**, because a graph that dropped the distinction exports a lie.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from tests.conftest import make_transformation
from rbfenetmap.core.intermediates import IntermediateOptions
from rbfenetmap.core.models import Ligand, LigandProvenance, Network
from rbfenetmap.core.options import NetworkOptions
from rbfenetmap.plugins.exporters.amber_exporter import INTERMEDIATE_MANIFEST, LIGAND_DIRECTORY
from rbfenetmap.viz.gallery import render_report


@pytest.fixture(scope="module")
def invented(benzamides):
    """One synthetic ligand, posed onto a real one so its conformer is in their frame."""
    parent = benzamides["bza_F"]
    mol = Chem.AddHs(Chem.MolFromSmiles("Brc1ccccc1C(=O)N"))
    assert AllChem.EmbedMolecule(mol, randomSeed=0xF00D) == 0
    provenance = LigandProvenance(
        kind="intermediate",
        generator="pairmap",
        parents=("bza_Cl", "bza_F"),
        pose_method="parent_atom_map",
        pose_rmsd=0.12,
        detail=MappingProxyType({"state": "ST"}),
    )
    assert parent.synthetic is False
    return Ligand.synthesized(mol, "int_bza_Cl_bza_F_abc123", provenance)


@pytest.fixture(scope="module")
def augmented(benzamides, invented):
    """A small planned network in which one vertex was invented.

    Built by hand rather than by running the generator: this file is about what the
    exporters do with a synthetic vertex, and manufacturing one keeps the assertions
    independent of which molecule a generator happens to propose today.
    """
    ligands = {**{name: benzamides[name] for name in ("bza_F", "bza_Cl", "bza_Me")}, invented.name: invented}
    edges = (
        make_transformation("bza_F", invented.name),
        make_transformation(invented.name, "bza_Cl"),
        make_transformation("bza_Cl", "bza_Me"),
    )
    return Network(
        ligands=ligands,
        edges=edges,
        candidates=edges,
        planner="mst",
        options=NetworkOptions(intermediates=IntermediateOptions(mode="bridge", generator="pairmap")),
    )


@pytest.fixture(scope="module")
def all_real(benzamides):
    """The same shape with nothing invented, for the comparisons that need a control."""
    ligands = {name: benzamides[name] for name in ("bza_F", "bza_Cl", "bza_Me")}
    edges = (make_transformation("bza_F", "bza_Cl"), make_transformation("bza_Cl", "bza_Me"))
    return Network(ligands=ligands, edges=edges, candidates=edges, planner="mst")


class TestAmberExport:
    def test_every_name_in_edges_dat_has_a_structure_on_disk(self, augmented, tmp_path):
        """The worst failure mode this feature has, asserted directly.

        amberstudio needs a parameterised topology per residue. A residue named in
        ``edges.dat`` whose structure exists nowhere fails deep inside somebody else's
        tooling, hours later, with an error about a missing residue.
        """
        pytest.importorskip("yaml")
        from rbfenetmap.plugins.exporters import create_exporter

        create_exporter("amber").export(augmented, tmp_path)
        named = {name for line in (tmp_path / "edges.dat").read_text().splitlines() if line for name in line.split()}
        assert named
        for name in named:
            assert (tmp_path / LIGAND_DIRECTORY / f"{name}.sdf").exists(), name

    def test_it_holds_for_an_all_real_network_too(self, all_real, tmp_path):
        # The invariant is checkable only if it holds for the whole file. "Every name
        # except the ones you already had" is not something a script can assert.
        pytest.importorskip("yaml")
        from rbfenetmap.plugins.exporters import create_exporter

        create_exporter("amber").export(all_real, tmp_path)
        for line in (tmp_path / "edges.dat").read_text().splitlines():
            for name in line.split():
                assert (tmp_path / LIGAND_DIRECTORY / f"{name}.sdf").exists()

    def test_the_written_structure_round_trips(self, augmented, tmp_path, invented):
        pytest.importorskip("yaml")
        from rbfenetmap.plugins.exporters import create_exporter

        create_exporter("amber").export(augmented, tmp_path)
        loaded = Chem.SDMolSupplier(str(tmp_path / LIGAND_DIRECTORY / f"{invented.name}.sdf"), removeHs=False)[0]
        assert loaded is not None
        assert loaded.GetNumConformers() == 1
        assert loaded.GetProp("rbfenet_generator") == "pairmap"
        assert loaded.GetProp("rbfenet_parents") == "bza_Cl bza_F"

    def test_the_manifest_names_parents_and_generator(self, augmented, tmp_path, invented):
        pytest.importorskip("yaml")
        from rbfenetmap.plugins.exporters import create_exporter

        create_exporter("amber").export(augmented, tmp_path)
        rows = [line.split() for line in (tmp_path / INTERMEDIATE_MANIFEST).read_text().splitlines() if line]
        assert rows == [[invented.name, "bza_Cl", "bza_F", "pairmap"]]

    def test_an_all_real_export_writes_no_manifest(self, all_real, tmp_path):
        # A file that says "nothing happened" is a file somebody has to learn to ignore.
        pytest.importorskip("yaml")
        from rbfenetmap.plugins.exporters import create_exporter

        create_exporter("amber").export(all_real, tmp_path)
        assert not (tmp_path / INTERMEDIATE_MANIFEST).exists()

    def test_validate_reports_the_synthetic_count(self, augmented):
        from rbfenetmap.plugins.exporters import create_exporter

        with pytest.warns(UserWarning, match="need parameterising"):
            create_exporter("amber").validate(augmented)

    def test_validate_warns_pre_flight_when_generation_is_merely_enabled(self, benzamides):
        """The pre-flight network has no synthetic ligands yet, but may grow some."""
        from rbfenetmap.plugins.exporters import create_exporter

        options = NetworkOptions(intermediates=IntermediateOptions(mode="bridge", generator="pairmap"))
        preflight = Network(ligands=benzamides, edges=(), planner="preflight", options=options)
        with pytest.warns(UserWarning, match="Intermediate generation is enabled"):
            create_exporter("amber").validate(preflight)

    def test_validate_stays_quiet_on_an_all_real_network(self, all_real, recwarn):
        from rbfenetmap.plugins.exporters import create_exporter

        create_exporter("amber").validate(all_real)
        assert not [w for w in recwarn if "parameterising" in str(w.message)]


class TestNetworkX:
    def test_nodes_carry_the_synthetic_flag(self, augmented, invented):
        graph = augmented.to_networkx()
        assert graph.nodes[invented.name]["synthetic"] is True
        assert graph.nodes["bza_F"]["synthetic"] is False

    def test_graphml_carries_it(self, augmented, invented, tmp_path):
        import networkx as nx

        path = tmp_path / "network.graphml"
        graph = augmented.to_networkx()
        # The edge payload is not GraphML-typeable; the node attribute is, which is the
        # whole reason it is a bool rather than the provenance object.
        for _, _, data in graph.edges(data=True):
            data.pop("transformation", None)
        nx.write_graphml(graph, path)
        reloaded = nx.read_graphml(path)
        assert reloaded.nodes[invented.name]["synthetic"] is True


class TestReport:
    def test_the_badge_and_provenance_section_are_present(self, augmented, invented):
        report = render_report(augmented)
        assert ">SYN<" in report
        assert "Invented ligands" in report
        assert invented.name in report
        assert "0.120" in report, "the pose RMSD is the number a reviewer actually needs"
        assert "parent_atom_map" in report

    def test_the_node_marker_is_not_colour_alone(self, augmented):
        report = render_report(augmented)
        # A dashed outline in the diagram and the three letters on the label; neither of
        # them is a hue somebody has to distinguish.
        assert "node-synthetic" in report
        assert "stroke-dasharray" in report
        assert " SYN<" in report or ">SYN<" in report

    def test_the_ligand_stat_counts_only_real_ligands(self, augmented):
        report = render_report(augmented)
        real = len(augmented.ligands) - len(augmented.synthetic_ligands)
        assert f"<div class='value'>{real}</div><div class='label'>Ligands</div>" in report
        assert "<div class='label'>Invented</div>" in report

    def test_an_all_real_report_gains_nothing(self, all_real):
        report = render_report(all_real)
        assert "SYN" not in report
        assert "Invented" not in report
        assert f"<div class='value'>{len(all_real.ligands)}</div><div class='label'>Ligands</div>" in report

    def test_it_does_not_crash_when_provenance_is_none(self, all_real):
        # Every ligand here has provenance None, which is the default and the overwhelming
        # majority of every report this package will ever render.
        assert all(ligand.provenance is None for ligand in all_real.ligands.values())
        assert render_report(all_real).endswith("</body></html>")

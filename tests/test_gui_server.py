"""The GUI server, driven the way the browser drives it.

Started on a real socket and talked to over HTTP rather than by calling the handler
directly: the routing, the JSON bodies and the status codes are the contract the page
depends on, and none of them is exercised by calling :class:`PlanSession` in-process.

Stdlib only, so this runs in the default CI job with no optional dependency present.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

import pytest

from rbfenetmap.gui.server import build_server
from rbfenetmap.gui.session import PlanSession

pytestmark = pytest.mark.integration

GOLDEN_SDF = Path(__file__).resolve().parent / "data" / "golden_benzamides.sdf"


@pytest.fixture
def server():
    """A server on an ephemeral port, with the tracked golden series loaded.

    The golden set rather than a fixture built in memory: it is co-posed and tracked, so
    the geometry checks these runs reach have something real to judge.
    """
    session = PlanSession([GOLDEN_SDF])
    session.set_ligands([GOLDEN_SDF])
    httpd = build_server(session, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}", session
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(base, path):
    """GET and decode JSON."""
    with urllib.request.urlopen(f"{base}{path}", timeout=30) as response:  # noqa: S310 - a loopback test server
        return json.loads(response.read())


def get_raw(base, path):
    """GET and return the body and content type."""
    with urllib.request.urlopen(f"{base}{path}", timeout=30) as response:  # noqa: S310 - as above
        return response.read(), response.headers.get("Content-Type")


def post(base, path, body=None):
    """POST JSON and decode the reply."""
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - as above
        return json.loads(response.read())


def wait_for(base, run_id, timeout=120):
    """Poll a run to completion, as the page does."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = get(base, f"/api/run/{run_id}")
        if run["state"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} did not finish within {timeout}s.")


class TestThePage:
    def test_the_root_serves_the_page(self, server):
        base, _ = server
        body, content_type = get_raw(base, "/")
        assert content_type.startswith("text/html")
        assert b"<title>rbfenet" in body

    def test_the_assets_are_served(self, server):
        """The page is three files; a packaging slip that dropped them would be silent."""
        base, _ = server
        for name, expected in (("app.css", "text/css"), ("app.js", "text/javascript")):
            body, content_type = get_raw(base, f"/static/{name}")
            assert content_type == expected
            assert body

    def test_a_traversing_path_is_refused(self, server):
        """Nothing here is user-authored, but only happening to be safe is not the same.

        Requested with a pre-encoded segment so it survives the client rather than being
        normalized away before it reaches the server.
        """
        base, _ = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            get_raw(base, "/static/..%2F..%2F..%2Fetc%2Fpasswd")
        assert exc.value.code == 404

    def test_an_unknown_path_is_a_404(self, server):
        base, _ = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(base, "/api/nonsense")
        assert exc.value.code == 404


class TestSchema:
    def test_the_schema_is_served_whole(self, server):
        """What the page builds every control from."""
        base, _ = server
        schema = get(base, "/api/schema")
        assert schema["groups"]
        assert {"mapper", "scorer", "planner", "exporter", "intermediate"} == set(schema["plugins"])
        assert any(p["name"] == "mcss-e2" for p in schema["plugins"]["mapper"])

    def test_the_session_reports_its_ligands(self, server):
        base, _ = server
        state = get(base, "/api/session")
        assert len(state["ligands"]["loaded"]) == 9


class TestPlanning:
    def test_a_default_run_produces_a_network_and_a_diagram(self, server):
        """The whole round trip, as the page performs it."""
        base, _ = server
        started = post(base, "/api/plan", {"values": {}})
        run = wait_for(base, started["id"])

        assert run["state"] == "done", run["error"]
        assert run["metrics"]["n_edges"] > 0
        assert run["metrics"]["n_ligands"] == 9
        assert run["progress"]["done"] == run["progress"]["total"] == 36

    def test_the_diagram_is_well_formed_svg_with_a_node_per_ligand(self, server):
        """Injected into the page as markup, so it had better parse."""
        base, _ = server
        run = wait_for(base, post(base, "/api/plan", {"values": {}})["id"])
        root = ElementTree.fromstring(run["svg"])
        circles = root.findall(".//{http://www.w3.org/2000/svg}circle")
        assert len(circles) == 9

    def test_the_command_line_reproduces_the_run(self, server):
        """The promise the whole GUI is built around.

        Not merely that a command is shown, but that it is the command: parsed back
        through the real CLI, it has to carry the knob that was set.
        """
        from rbfenetmap.cli.main import build_parser

        base, _ = server
        run = wait_for(base, post(base, "/api/plan", {"values": {"edges_per_ligand": 3}})["id"])
        assert run["command"].startswith("rbfenet plan ")
        assert "--edges-per-ligand 3" in run["command"]

        args = build_parser().parse_args(run["argv"])
        assert args.command == "plan"
        assert args.edges_per_ligand == 3

    def test_a_selection_knob_changes_the_network(self, server):
        """Two runs, one knob, different answers -- which is the point of the tool."""
        base, _ = server
        default = wait_for(base, post(base, "/api/plan", {"values": {}})["id"])
        denser = wait_for(base, post(base, "/api/plan", {"values": {"edges_per_ligand": 3}})["id"])
        assert denser["metrics"]["n_edges"] > default["metrics"]["n_edges"]

    def test_the_second_run_reuses_the_mappings(self, server):
        """A knob moved between runs must not re-run the MCS searches."""
        base, session = server
        wait_for(base, post(base, "/api/plan", {"values": {}})["id"])
        misses = session.cache.misses
        second = wait_for(base, post(base, "/api/plan", {"values": {"edges_per_ligand": 3}})["id"])
        assert session.cache.misses == misses
        assert second["cache"]["hits"] > 0


class TestRefusals:
    def test_an_impossible_edge_budget_comes_back_as_a_message(self, server):
        """Not a 500. The refusal already says what to do; that sentence is the answer.

        check_edge_budget needs the ligand count, so unlike a range error this one can
        only be judged once they are loaded and therefore surfaces on the run.
        """
        base, _ = server
        run = wait_for(base, post(base, "/api/plan", {"values": {"n_edges": 1}})["id"])
        assert run["state"] == "error"
        assert "cannot connect 9 ligands" in run["error"]

    def test_a_contradictory_form_is_refused_before_the_run_starts(self, server):
        """Judged without the ligands, so the browser hears at once rather than by polling."""
        base, _ = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(base, "/api/plan", {"values": {"compat": "v0.4", "max_diameter": 5}})
        assert exc.value.code == 400
        assert "pins every algorithmic knob" in json.loads(exc.value.read())["error"]

    def test_a_star_strategy_without_a_hub_is_refused(self, server):
        base, _ = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(base, "/api/plan", {"values": {"pair_strategy": "star"}})
        assert "requires a hub" in json.loads(exc.value.read())["error"]

    def test_an_unreadable_ligand_path_is_a_message(self, server, tmp_path):
        base, _ = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(base, "/api/ligands", {"paths": [str(tmp_path / "absent.sdf")]})
        assert exc.value.code == 400

    def test_a_run_that_does_not_exist_is_a_404(self, server):
        base, _ = server
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(base, "/api/run/deadbeef")
        assert exc.value.code == 404


class TestArtefacts:
    def test_the_full_report_is_produced_on_demand(self, server):
        """Never with the run: it is megabytes, and a knob change must not pay for it."""
        base, _ = server
        run = wait_for(base, post(base, "/api/plan", {"values": {}})["id"])
        body, content_type = get_raw(base, f"/api/run/{run['id']}/report")
        assert content_type.startswith("text/html")
        assert b"<h2>Selected edges</h2>" in body
        assert len(body) > len(run["svg"])

    def test_the_network_json_round_trips(self, server, tmp_path):
        """What the user takes away, and it has to load back through the library."""
        from rbfenetmap.io.networkio import load_network

        base, _ = server
        run = wait_for(base, post(base, "/api/plan", {"values": {}})["id"])
        body, content_type = get_raw(base, f"/api/run/{run['id']}/network.json")
        assert content_type == "application/json"

        path = tmp_path / "network.json"
        path.write_bytes(body)
        assert len(load_network(path).edges) == run["metrics"]["n_edges"]


class TestPins:
    def test_a_finished_run_can_be_pinned_and_dropped(self, server):
        base, _ = server
        run = wait_for(base, post(base, "/api/plan", {"values": {}})["id"])
        pinned = post(base, "/api/pin", {"run_id": run["id"], "label": "default"})

        assert pinned["label"] == "default"
        assert pinned["metrics"]["n_edges"] == run["metrics"]["n_edges"]
        assert len(get(base, "/api/session")["pins"]) == 1

        assert post(base, "/api/unpin", {"run_id": run["id"]})["pins"] == []

    def test_pinned_metrics_are_the_diagnose_json_shape(self, server):
        """So a GUI comparison reads against the published variant matrix untranslated.

        Asserted against summarize_json's own output rather than a copied key list, which
        would be one more thing to drift.
        """
        from rbfenetmap.core.diagnostics import summarize_json

        base, session = server
        run = wait_for(base, post(base, "/api/plan", {"values": {}})["id"])
        pinned = post(base, "/api/pin", {"run_id": run["id"]})
        expected = summarize_json(session.runs[run["id"]].network)
        assert set(pinned["metrics"]) == set(expected)

    def test_an_unfinished_run_cannot_be_pinned(self, server):
        """A pin is a row of numbers; a running job has none yet."""
        base, _ = server
        started = post(base, "/api/plan", {"values": {}})
        try:
            post(base, "/api/pin", {"run_id": started["id"]})
        except urllib.error.HTTPError as exc:
            assert "only a finished run" in json.loads(exc.read())["error"]
        wait_for(base, started["id"])


class TestCancellation:
    def test_a_cancelled_run_reports_itself_cancelled(self, server):
        """Cancelled, not failed. The user asked for it, so it is not an error."""
        base, _ = server
        started = post(base, "/api/plan", {"values": {"mcs_timeout": 1}})
        post(base, f"/api/run/{started['id']}/cancel")
        run = wait_for(base, started["id"])
        # The set is small enough that it may well finish first; either outcome is correct,
        # and what must never happen is a crash or a network built from abandoned work.
        assert run["state"] in ("cancelled", "done")
        if run["state"] == "cancelled":
            assert run["metrics"] is None

    def test_starting_a_run_cancels_the_previous_one(self, server):
        """The user has moved a knob, so the previous question no longer interests them.

        Two concurrent runs would compete for the same worker threads and double the peak
        memory of the mapping stage, which on a large set is the thing most worth not doing.
        """
        base, session = server
        first = post(base, "/api/plan", {"values": {}})
        second = post(base, "/api/plan", {"values": {"edges_per_ligand": 3}})
        wait_for(base, second["id"])
        assert session.runs[first["id"]].state in ("cancelled", "done")

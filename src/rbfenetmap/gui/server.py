"""A small local HTTP server for the knob explorer.

Standard library only. There are eleven endpoints and one page, which a framework would
make marginally pleasanter to write at the cost of a runtime dependency, an extra to
install, an ``autodoc_mock_imports`` entry, and a module that
``tests/test_smoke.py``'s unconditional import walk would trip over in the default CI job.

Bound to the loopback interface unless told otherwise, and it says so loudly when told
otherwise: it reads whatever ligand path the form names, with the privileges of whoever
started it, and it is not written to face a network.
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from rbfenetmap.gui.schema import plan_schema
from rbfenetmap.gui.session import PlanSession

logger = logging.getLogger(__name__)

__all__ = ("build_server", "serve")

_STATIC = Path(__file__).resolve().parent / "static"

_CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript"}


class _Handler(BaseHTTPRequestHandler):
    """Routes requests to the session. One instance per request, as http.server does it."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args, session: PlanSession, **kwargs) -> None:
        self.session = session
        super().__init__(*args, **kwargs)

    # -- plumbing --------------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        """Send request logs through logging rather than straight to stderr.

        The default writes every request to the terminal, which buries the one thing a
        user running this actually wants to see there: the URL to open.
        """
        logger.debug("%s - %s", self.address_string(), format % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # The page is regenerated per request and the reports are per run; a cached copy
        # of either is always the wrong one.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        """Send *payload* as JSON, refusing anything a browser could not parse.

        ``allow_nan=False`` is the point. Python renders ``math.inf`` as the bare token
        ``Infinity``, which ``json.dumps`` emits happily and ``JSON.parse`` rejects
        outright -- and an infinite cost is not exotic here, it is what every infeasible
        candidate carries. Without this the page fails with a console error and the server
        logs nothing at all. With it, the same mistake is a loud 500 naming the value.
        """
        self._send(status, json.dumps(payload, allow_nan=False).encode(), "application/json")

    def _error(self, exc: Exception, status: int = 400) -> None:
        """Return a failure as a readable message.

        A stack trace in the browser console is not an answer to "why can I not have this
        network". Every refusal the package raises already carries a sentence saying what
        to do instead, and that sentence is what the page shows.
        """
        self._json({"error": str(exc)}, status=status)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routes ----------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        """Serve the page, the static assets, and the read-only endpoints."""
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path.removeprefix("/static/"))
            if path == "/api/schema":
                return self._json(plan_schema())
            if path == "/api/session":
                return self._json(self.session.state())
            if path.startswith("/api/run/"):
                return self._run_get(path.removeprefix("/api/run/"))
            self._json({"error": f"No such path {path!r}."}, status=404)
        except Exception as exc:  # noqa: BLE001 - one failure must not kill the server
            logger.exception("GET %s failed", path)
            self._error(exc, status=500)

    def do_POST(self) -> None:  # noqa: N802 - as above
        """Serve the endpoints that change something."""
        path = urlparse(self.path).path
        try:
            if path == "/api/ligands":
                body = self._body()
                return self._json(self.session.set_ligands(body["paths"], name_property=body.get("name_property")))
            if path == "/api/plan":
                run = self.session.start(self._body().get("values", {}))
                return self._json(run.as_dict(include_svg=False))
            if path == "/api/pin":
                body = self._body()
                return self._json(self.session.pin(body["run_id"], body.get("label")))
            if path == "/api/unpin":
                self.session.unpin(self._body()["run_id"])
                return self._json(self.session.state())
            if path.startswith("/api/run/") and path.endswith("/cancel"):
                self.session.cancel(path.removeprefix("/api/run/").removesuffix("/cancel"))
                return self._json({"ok": True})
            self._json({"error": f"No such path {path!r}."}, status=404)
        except (ValueError, KeyError, OSError) as exc:
            # The expected failures: an unsatisfiable request, a malformed body, an
            # unreadable ligand file. All of them are messages for the user.
            self._error(exc)
        except Exception as exc:  # noqa: BLE001 - as above
            logger.exception("POST %s failed", path)
            self._error(exc, status=500)

    def _run_get(self, rest: str) -> None:
        """Handle ``/api/run/<id>`` and its per-run artifacts."""
        run_id, _, tail = rest.partition("/")
        run = self.session.runs.get(run_id)
        if run is None:
            return self._json({"error": f"No run {run_id!r}."}, status=404)
        kind, _, key = tail.partition("/")
        if kind in ("edge", "candidate"):
            # Which collection to resolve in is in the path, not inferred: one key can name
            # both a selected counterpoised edge and the relative candidate refused for the
            # same pair, and guessing answers only one of the two questions.
            scope = "edges" if kind == "edge" else "candidates"
            # unquote, never unquote_plus: '+' is a legal ligand-name character and must
            # not become a space. The path is not decoded for us -- see the traversal test.
            return self._edge(run, unquote(key), scope=scope)
        if tail == "rejected":
            try:
                return self._json(run.rejected_summary())
            except ValueError as exc:
                return self._json({"error": str(exc)}, status=404)
        if tail == "report":
            return self._send(200, run.report_html().encode(), "text/html; charset=utf-8")
        if tail == "network.json":
            from rbfenetmap.io.networkio import network_to_dict

            if run.network is None:
                return self._json({"error": f"Run {run_id} produced no network."}, status=404)
            return self._send(200, json.dumps(network_to_dict(run.network), indent=2).encode(), "application/json")
        # Poll responses carry the SVG only once the run is finished, so a run being
        # polled every few hundred milliseconds is not also re-sending a diagram.
        self._json(run.as_dict(include_svg=run.state == "done"))

    def _edge(self, run, key: str, *, scope: str) -> None:
        """Serve one edge's facts, masks and depictions.

        Everything a caller can get wrong here is a message rather than a traceback: a run
        with no network yet, a key naming nothing, a ligand missing from the network. The
        page shows all of them in the panel it would otherwise have filled.
        """
        indices = parse_qs(urlparse(self.path).query).get("indices", ["0"])[0] not in ("0", "", "false")
        try:
            return self._json(run.edge_detail(key, scope=scope, show_indices=indices))
        except (ValueError, KeyError) as exc:
            return self._json({"error": str(exc)}, status=404)

    def _static(self, name: str) -> None:
        """Serve one file from the package's static directory.

        The name is resolved and checked to be inside that directory, so a request for
        ``../../etc/passwd`` cannot escape it. Nothing here is user-authored, but a server
        that only happens to be safe is not the same as one that is.
        """
        target = (_STATIC / name).resolve()
        if not target.is_file() or _STATIC not in target.parents:
            return self._json({"error": f"No such asset {name!r}."}, status=404)
        self._send(200, target.read_bytes(), _CONTENT_TYPES.get(target.suffix, "application/octet-stream"))


def build_server(session: PlanSession, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create the server without starting it.

    Parameters
    ----------
    session : PlanSession
    host : str
    port : int
        ``0`` asks the operating system for a free one, which is what the tests use.

    Returns
    -------
    http.server.ThreadingHTTPServer
    """
    return ThreadingHTTPServer((host, port), partial(_Handler, session=session))


def serve(
    ligands: Sequence[Path] | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    name_property: str = "_Name",
    cache_dir: Path | None = None,
    open_browser: bool = True,
) -> None:
    """Serve the knob explorer until interrupted.

    Parameters
    ----------
    ligands : Sequence[Path], optional
        Loaded at startup. Omit to choose a file from the page instead.
    host : str
        Loopback by default. Any other value is a deliberate choice to expose a server
        that reads local files on request, and is warned about.
    port : int
    name_property : str
    cache_dir : Path, optional
        Persists the mapping cache between sessions, which is what makes the *second*
        launch fast rather than only the second run.
    open_browser : bool
    """
    session = PlanSession(ligands, name_property=name_property, cache_dir=cache_dir)
    if ligands:
        summary = session.set_ligands(ligands, name_property=name_property)
        print(f"Loaded {summary['n_ligands']} ligand(s).", flush=True)

    server = build_server(session, host=host, port=port)
    address = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: serving on {host}, not loopback. This server plans networks from any "
            "ligand path it is given and runs with your privileges; it has no authentication "
            "and is not written to face a network. Prefer an SSH tunnel to a remote machine."
        )
    # Flushed: the URL is the one thing a user needs from this terminal, and a piped
    # or redirected stdout would otherwise hold it in a buffer until the server stops.
    print(f"rbfenet gui listening on {address}  (ctrl-c to stop)", flush=True)

    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(address,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.shutdown()
        server.server_close()
        session.cache.save()

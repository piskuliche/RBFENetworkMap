"""One browser session: the loaded ligands, the runs, and the pinned comparisons.

A run is started from a filled form, evaluated on a worker thread, and polled. Threads
rather than a synchronous handler because mapping is quadratic in the ligand count -- past
a thousand pairs a plan takes long enough that a blocking request would look like a hung
browser, and there would be nowhere to put a cancel button.

Everything a run needs beyond :func:`~rbfenetmap.core.pipeline.build_network` is borrowed
from the CLI rather than rebuilt. Ligand loading and alignment come from
:func:`rbfenetmap.cli.commands._load` and the scorer from
:func:`rbfenetmap.cli.commands._make_scorer`, both of which take the same
:class:`argparse.Namespace` this module already has. That matters: alignment deliberately
sits *outside* the pipeline, so a GUI that skipped it would leave ``--align`` doing nothing
while still printing it in the command line it offers to copy.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from rbfenetmap.cli._args import build_mapping_options, build_network_options, explicit_dests, resolve_compat
from rbfenetmap.core.diagnostics import summarize_json
from rbfenetmap.gui.cache import CachingMapper, MappingCache, RunCancelled
from rbfenetmap.gui.schema import to_argv

logger = logging.getLogger(__name__)

__all__ = ("PlanRun", "PlanSession")


class PlanRun:
    """One planning attempt, and whatever it has produced so far.

    Attributes
    ----------
    id : str
    values : dict
        The form as submitted.
    argv : list of str
        The full ``rbfenet`` argument list this run is equivalent to.
    state : {"running", "done", "error", "cancelled"}
    done, total : int
        Candidate pairs mapped, and how many there are to map. Under
        ``pair_evaluation="adaptive"`` *total* is a ceiling the run may stop short of.
    error : str or None
    network : Network or None
    metrics : dict or None
        Exactly what ``rbfenet diagnose --format json`` reports, so a pinned comparison
        reads against the published variant matrix without translation.
    svg : str or None
    """

    def __init__(self, run_id: str, values: dict[str, Any], argv: list[str]) -> None:
        self.id = run_id
        self.values = values
        self.argv = argv
        self.state = "running"
        self.done = 0
        self.total = 0
        self.error: str | None = None
        self.network: Any = None
        self.metrics: dict[str, Any] | None = None
        self.svg: str | None = None
        self.started = time.monotonic()
        self.seconds: float | None = None
        self.cache_hits = 0
        self.cache_misses = 0
        self._cancelled = threading.Event()
        self._report: str | None = None

    @property
    def command(self) -> str:
        """The run as a copy-pasteable shell command."""
        from shlex import quote

        return "rbfenet " + " ".join(quote(part) for part in self.argv)

    def cancel(self) -> None:
        """Ask the run to stop. Takes effect within one ``--mcs-timeout``."""
        self._cancelled.set()

    def as_dict(self, *, include_svg: bool = True) -> dict[str, Any]:
        """Serialize for the browser. The network itself is never sent whole."""
        payload = {
            "id": self.id,
            "state": self.state,
            "values": self.values,
            "argv": self.argv,
            "command": self.command,
            "progress": {"done": self.done, "total": self.total},
            "error": self.error,
            "metrics": self.metrics,
            "seconds": self.seconds,
            "cache": {"hits": self.cache_hits, "misses": self.cache_misses},
            "unmet": list(self.network.unmet_constraints) if self.network is not None else [],
        }
        if include_svg:
            payload["svg"] = self.svg
        return payload

    def report_html(self) -> str:
        """Render the full self-contained report, once, on demand.

        Deliberately not produced with the run. On the sixteen-ligand Tyk2 set the report
        is over two megabytes of inlined SVG, and generating one per knob change is exactly
        what would make the tool feel slow. The network diagram and the metrics are what
        the live panel needs; this is a button.
        """
        if self.network is None:
            raise ValueError(f"Run {self.id} has no network to report on.")
        if self._report is None:
            from rbfenetmap.viz.gallery import render_report

            self._report = render_report(self.network, title=f"RBFE network -- {self.id}")
        return self._report


class PlanSession:
    """The ligands, the mapping cache, the runs and the pins behind one served page.

    Parameters
    ----------
    ligands : Sequence[Path], optional
        Initial ligand files. May also be set later from the browser.
    name_property : str
        Molecule property ligand names are read from.
    cache_dir : Path, optional
        Where the mapping cache is persisted between sessions. ``None`` keeps it in memory.

    Notes
    -----
    One run at a time. A second start cancels the first rather than queueing it: the user
    has moved a knob, which means they are no longer interested in the answer to the
    previous question, and two concurrent runs would compete for the same worker threads
    and double the peak memory of the mapping stage.
    """

    def __init__(
        self, ligands: Sequence[Path] | None = None, *, name_property: str = "_Name", cache_dir: Path | None = None
    ) -> None:
        self.ligand_paths: list[Path] = [Path(p) for p in (ligands or ())]
        self.name_property = name_property
        self.cache = MappingCache(cache_dir / "mappings.json" if cache_dir else None)
        self.runs: dict[str, PlanRun] = {}
        self.pins: list[dict[str, Any]] = []
        self._current: PlanRun | None = None
        self._lock = threading.Lock()
        self._ligands: dict[str, Any] | None = None
        self._ligand_key: tuple | None = None

    # -- ligands ---------------------------------------------------------------------

    def set_ligands(self, paths: Sequence[str | Path], *, name_property: str | None = None) -> dict[str, Any]:
        """Point the session at a new ligand set and load it immediately.

        Returns
        -------
        dict
            ``names``, ``n_ligands`` and the ``paths`` as given, so the browser can show
            what it got rather than only that something happened.
        """
        self.ligand_paths = [Path(p) for p in paths]
        if name_property is not None:
            self.name_property = name_property
        self._ligands = None
        self._ligand_key = None
        ligands = self._load_ligands(self._namespace({}))
        return {"paths": [str(p) for p in self.ligand_paths], "n_ligands": len(ligands), "names": sorted(ligands)}

    def _namespace(self, values: dict[str, Any]):
        """Parse *values* into the namespace ``rbfenet plan`` would have produced."""
        from rbfenetmap.cli.main import build_parser

        argv = self._argv(values)
        args = build_parser().parse_args(argv)
        # A second parser on purpose: explicit_dests suppresses the defaults of whatever it
        # is handed, so it must not be handed the one whose result is used.
        resolve_compat(args, explicit_dests(build_parser(), argv))
        return args

    def _argv(self, values: dict[str, Any]) -> list[str]:
        """Build the full argument list, input flags included."""
        if not self.ligand_paths:
            raise ValueError("No ligands loaded. Choose a file or directory first.")
        argv = ["plan", "--ligands", *[str(p) for p in self.ligand_paths]]
        if self.name_property != "_Name":
            argv += ["--name-property", self.name_property]
        return argv + to_argv(values)

    def _load_ligands(self, args) -> dict[str, Any]:
        """Load and align the ligands, reusing the previous result when nothing changed.

        Alignment is an all-pairs-ish MCS job in its own right, so repeating it on every
        knob change would undo much of what the mapping cache buys. Keyed on the inputs
        that can move an atom -- and on nothing else, because a knob that cannot change the
        coordinates cannot invalidate them.
        """
        from rbfenetmap.cli import commands

        key = (
            tuple(str(p) for p in self.ligand_paths),
            self.name_property,
            args.align,
            args.align_reference,
            args.align_min_atoms,
        )
        if self._ligands is None or self._ligand_key != key:
            self._ligands = commands._load(args)  # noqa: SLF001 - the seam all commands share
            self._ligand_key = key
        return self._ligands

    # -- runs ------------------------------------------------------------------------

    def start(self, values: dict[str, Any]) -> PlanRun:
        """Validate *values*, then plan in the background.

        Raises
        ------
        ValueError
            For anything that can be judged without the ligands: an unparseable flag, a
            ``--compat`` level contradicting a knob it pins, a star strategy with no hub,
            a knob out of range, an edge both forced and banned. Raised here rather than
            on the worker thread so the browser is told at once instead of having to poll
            to discover that the run it just started was never going to work.

            The checks that need the ligand count -- an edge budget too small to span
            them, most of all -- can only run once they are loaded, so those still surface
            on the run as ``state == "error"``.
        """
        args = self._namespace(values)
        # Constructs the options purely for their __post_init__ validation; the worker
        # builds its own from the same namespace.
        build_network_options(args)
        run = PlanRun(uuid.uuid4().hex[:12], dict(values), self._argv(values))

        with self._lock:
            if self._current is not None and self._current.state == "running":
                self._current.cancel()
            self._current = run
            self.runs[run.id] = run

        threading.Thread(target=self._execute, args=(run, args), daemon=True, name=f"plan-{run.id}").start()
        return run

    def _execute(self, run: PlanRun, args) -> None:
        """Run the pipeline, recording whatever comes of it on *run*."""
        from rbfenetmap.core.pipeline import build_network
        from rbfenetmap.plugins.mappers import create_mapper

        try:
            from rbfenetmap.cli import commands

            ligands = self._load_ligands(args)
            network_options = build_network_options(args)
            mapping_options = build_mapping_options(args)
            run.total = self._pair_total(ligands, network_options)

            def advance(count: int) -> None:
                run.done += count

            mapper = CachingMapper(
                create_mapper(args.mapper),
                self.cache,
                should_cancel=run._cancelled.is_set,  # noqa: SLF001 - the run owns its flag
            )
            network = build_network(
                ligands,
                mapper=mapper,
                scorer=commands._make_scorer(args),  # noqa: SLF001 - name-keyed weights dispatch
                planner=args.planner,
                mapping_options=mapping_options,
                network_options=network_options,
                progress_callback=advance,
            )
        except RunCancelled:
            run.state = "cancelled"
        except Exception as exc:  # noqa: BLE001 - every failure is a message for the browser
            # A rejected edge is not an error, so anything reaching here is a genuine
            # refusal -- an unsatisfiable constraint, a missing plugin, an unreadable file.
            # It belongs on the page as text, not in a 500 the user cannot read.
            logger.info("Run %s failed: %s", run.id, exc)
            run.state = "error"
            run.error = str(exc)
        else:
            run.network = network
            run.metrics = summarize_json(network)
            run.svg = self._render(network)
            run.state = "done"
        finally:
            run.seconds = time.monotonic() - run.started
            run.cache_hits, run.cache_misses = self.cache.hits, self.cache.misses
            self.cache.save()

    @staticmethod
    def _pair_total(ligands, network_options) -> int:
        """How many pairs the mapping stage will consider, for the progress bar.

        Enumerated a second time rather than reported by the pipeline. It is only the pair
        list -- the expensive part is what happens to each one -- and the alternative is a
        progress bar whose denominator is wrong whenever a prefilter or an explicit pair
        strategy is in play.
        """
        if network_options.cbfe_mode == "all":
            return 0  # No mapping happens at all; the pool is closed-form.
        from rbfenetmap.core.pairs import generate_candidate_pairs

        pairs, _ = generate_candidate_pairs(ligands, network_options)
        return len(pairs)

    @staticmethod
    def _render(network) -> str:
        """Draw the network at the report's own settings.

        Stock seed, so the picture in the page is the picture in the report a click away.
        A different layout in each would make the two impossible to hold side by side.
        """
        from rbfenetmap.viz.network_svg import render_network_svg

        return render_network_svg(network)

    def cancel(self, run_id: str) -> None:
        """Cancel a run by id, if it is still going."""
        run = self.runs.get(run_id)
        if run is not None and run.state == "running":
            run.cancel()

    # -- pins ------------------------------------------------------------------------

    def pin(self, run_id: str, label: str | None = None) -> dict[str, Any]:
        """Keep a finished run's numbers for comparison.

        Only the metrics, the command and the form are kept; the network is left on the run
        it came from. A pin is something to read a table row from, and holding every pinned
        network would grow the session without bound over an afternoon's exploring.
        """
        run = self.runs.get(run_id)
        if run is None:
            raise ValueError(f"No run {run_id!r}.")
        if run.state != "done":
            raise ValueError(f"Run {run_id} is {run.state}; only a finished run can be pinned.")
        pinned = {
            "run_id": run.id,
            "label": label or f"run {len(self.pins) + 1}",
            "values": run.values,
            "argv": run.argv,
            "command": run.command,
            "metrics": run.metrics,
            "seconds": run.seconds,
        }
        self.pins.append(pinned)
        return pinned

    def unpin(self, run_id: str) -> None:
        """Drop a pin."""
        self.pins = [pin for pin in self.pins if pin["run_id"] != run_id]

    def state(self) -> dict[str, Any]:
        """The whole session, for a browser that has just connected or reconnected."""
        return {
            "ligands": {"paths": [str(p) for p in self.ligand_paths], "loaded": sorted(self._ligands or {})},
            "current": self._current.as_dict() if self._current is not None else None,
            "pins": self.pins,
            "cache": {"entries": len(self.cache), "hits": self.cache.hits, "misses": self.cache.misses},
        }

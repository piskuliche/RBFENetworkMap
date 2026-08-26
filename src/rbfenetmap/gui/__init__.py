"""An optional local GUI for exploring what the network knobs do.

``rbfenet plan`` has some sixty knobs, and choosing among them means running it, opening
the report, changing a flag, and holding the previous answer in your head. This subpackage
serves a small local page that closes that loop: move a knob, see the network and its
metrics, pin the run, move another knob, compare.

The design rule the whole thing hangs on is that **the GUI's state is an argv list**. It
defines no knob and builds no options object; :mod:`rbfenetmap.gui.schema` derives the form
from the CLI's own argument groups, and the filled form is serialized back to flags that
:func:`rbfenetmap.cli._args.build_network_options` turns into options exactly as the command
line does. So the GUI cannot drift from the CLI, and it can always show the user the precise
``rbfenet plan ...`` line that produced what they are looking at.

Standard library only -- there is no extra to install and no optional dependency to probe.

Unlike :mod:`rbfenetmap.viz`, whose output is deliberately script-free because it is an
artifact you email or attach to a ticket, this is an application and does use JavaScript.
"""

from __future__ import annotations

__all__ = ("plan_schema", "to_argv")


def __getattr__(name: str) -> object:
    """Import the schema helpers lazily, keeping this module import free of work."""
    if name in __all__:
        from rbfenetmap.gui import schema

        return getattr(schema, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

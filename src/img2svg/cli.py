"""Typer-based CLI entry point for img2svg.

This module is the user-facing ``img2svg`` command. It exposes three
top-level subcommands (``list-gpus``, ``info``) plus a default
``convert`` command that runs when no subcommand is given.

The "default command" pattern is implemented via a small custom
:class:`typer.core.TyperGroup` subclass. Standard Click/Typer cannot
route a free-standing positional argument (e.g. ``img2svg foo.png``) to
a subcommand because the group callback consumes positionals first.
The subclass overrides ``_click_resolve_command`` to fall back to a
default subcommand (``convert``) when the first non-option arg doesn't
match any registered subcommand. All args are then re-parsed by the
default command.

Exit code mapping (per the plan):

    0  success
    1  partial fail (batch with at least one file error)
    2  invalid args / unsupported format / bad mode / file not found
    3  dependency / model load failure
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from img2svg.api import convert as api_convert
from img2svg.api import convert_batch
from img2svg.device import list_available_devices
from img2svg.enums import DeviceStrategy, GpuVendor, Mode
from img2svg.errors import (
    ConfigError,
    CorruptImageError,
    DeviceUnavailableError,
    Img2SvgError,
    ModelLoadError,
    OutputPathCollisionError,
    UnsupportedFormatError,
)
from img2svg.gpu import list_gpus, recommend_gpu
from img2svg.logging import setup_logging
from img2svg.models import ConversionOptions

__version__ = "0.1.0"

# Glob meta-characters that mark a string as a glob pattern rather than
# a literal path. Used to decide whether to call ``convert()`` (single
# file) or ``convert_batch()`` (glob / directory).
_GLOB_META: frozenset[str] = frozenset("*?[")


# ----------------------------------------------------------------------
# Custom TyperGroup: "default command" routing
# ----------------------------------------------------------------------


class _DefaultCommandGroup(typer.core.TyperGroup):
    """A TyperGroup that routes unknown first args to a default subcommand.

    Standard Click/Typer treats the first non-option token as a
    subcommand name and errors with "No such command" if it doesn't
    match. For img2svg we want ``img2svg foo.png`` to run the
    ``convert`` subcommand with ``foo.png`` as its input argument.

    The fallback works in two parts:

    1. ``get_command`` returns the default subcommand when the name
       is unknown. This satisfies Click's "do you have this command?"
       check.
    2. ``_click_resolve_command`` detects that we hit the fallback
       and passes the *full* remaining arg list (including the
       original first token) to the default subcommand. The default
       subcommand's own parser then handles ``foo.png`` as its
       positional ``input`` argument.
    """

    def __init__(self, *args: object, default_cmd: str = "convert", **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._default_cmd_name: str = default_cmd

    def get_command(  # type: ignore[override]
        self, ctx: typer.Context, cmd_name: str
    ) -> typer.core.TyperCommand | None:
        rv = super().get_command(ctx, cmd_name)
        if rv is not None:
            return rv
        return super().get_command(ctx, self._default_cmd_name)

    def _click_resolve_command(  # type: ignore[override]
        self, ctx: typer.Context, args: list[str]
    ) -> tuple[str | None, typer.core.TyperCommand | None, list[str]]:
        if not args:
            return super()._click_resolve_command(ctx, args)
        cmd_name = args[0]
        if cmd_name in self.commands:
            return super()._click_resolve_command(ctx, args)
        # Fall back to the default command and let IT re-parse the args.
        default_cmd = self.get_command(ctx, self._default_cmd_name)
        if default_cmd is not None:
            return self._default_cmd_name, default_cmd, args
        return super()._click_resolve_command(ctx, args)


# ----------------------------------------------------------------------
# Typer app + shared state
# ----------------------------------------------------------------------


app = typer.Typer(
    name="img2svg",
    cls=_DefaultCommandGroup,
    invoke_without_command=False,
    add_completion=False,
    no_args_is_help=True,
    help="Convert raster images to clean, optimized SVG using YOLO segmentation and vtracer.",
)

_console = Console()


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    """Print version and exit 0 when --version is passed (eager)."""
    if value:
        typer.echo(f"img2svg {__version__}")
        raise typer.Exit(0)


def _mode_callback(value: str) -> str:
    """Validate --mode at the option level. Raises BadParameter for bad values.

    Click translates BadParameter into a clean error message + exit code 2.
    """
    if value is None:
        return value
    try:
        Mode(value)
    except ValueError:
        valid = ", ".join(m.value for m in Mode)
        raise typer.BadParameter(
            f"invalid mode {value!r}. Valid modes: {valid}"
        ) from None
    return value


def _gpu_strategy_callback(value: str) -> str:
    """Validate --gpu-strategy at the option level. Raises BadParameter for bad values."""
    if value is None:
        return value
    try:
        DeviceStrategy(value)
    except ValueError:
        valid = ", ".join(s.value for s in DeviceStrategy)
        raise typer.BadParameter(
            f"invalid gpu-strategy {value!r}. Valid strategies: {valid}"
        ) from None
    return value


# ----------------------------------------------------------------------
# Group-level callback (eager --version)
# ----------------------------------------------------------------------


@app.callback()
def _group_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        hidden=True,
        help="Print version and exit",
    ),
) -> None:
    """Global group-level options. Currently only --version (eager)."""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _is_glob(s: str) -> bool:
    """True if `s` contains any glob meta-character."""
    return any(c in s for c in _GLOB_META)


def _is_batch_input(input_path: Path) -> bool:
    """True if `input_path` should be processed as a batch (glob or directory)."""
    if _is_glob(str(input_path)):
        return True
    return input_path.is_dir()


def _build_options(
    mode: str,
    model: str,
    device: str,
    gpu_strategy: str,
    conf: float,
    no_clobber: bool,
) -> ConversionOptions:
    """Build a ConversionOptions from validated CLI values."""
    # Mode and gpu_strategy are already validated by their callbacks.
    return ConversionOptions(
        mode=Mode(mode),
        model=model,
        device=device,
        conf=conf,
        gpu_strategy=DeviceStrategy(gpu_strategy),
        no_clobber=no_clobber,
    )


def _print_gpu_table(gpus: list, recommended: object | None) -> None:
    """Print a Rich table of detected GPUs. T22 will replace this stub."""
    if not gpus:
        _console.print("[yellow]No GPUs detected.[/yellow]")
        return

    table = Table(title="Available GPUs", show_lines=False)
    table.add_column("Index", justify="right")
    table.add_column("Vendor")
    table.add_column("Name")
    table.add_column("VRAM Total (MB)", justify="right")
    table.add_column("VRAM Free (MB)", justify="right")
    table.add_column("Recommended", justify="center")

    rec_index = recommended.index if recommended is not None else None
    for g in gpus:
        is_rec = "Y" if rec_index == g.index else ""
        table.add_row(
            str(g.index),
            g.vendor.value if isinstance(g.vendor, GpuVendor) else str(g.vendor),
            g.name,
            str(g.vram_total_mb),
            str(g.vram_free_mb),
            is_rec,
        )

    _console.print(table)


# ----------------------------------------------------------------------
# convert command (default)
# ----------------------------------------------------------------------


@app.command(
    "convert",
    help="Convert images (single file, directory, or glob) to SVG.",
)
def _convert_cmd(
    input: Path = typer.Argument(
        ...,
        help="Input file, glob, or directory",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file (or directory for batch)",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="auto, labels, visual, annotated, trace",
        callback=_mode_callback,
    ),
    model: str = typer.Option("yolo11x.pt", "--model", help="YOLO model name"),
    device: str = typer.Option(
        "auto", "--device", help="auto, cpu, cuda, cuda:N, mps, rocm"
    ),
    gpu_strategy: str = typer.Option(
        "power",
        "--gpu-strategy",
        help="GPU recommendation strategy",
        callback=_gpu_strategy_callback,
    ),
    conf: float = typer.Option(0.25, "--conf", help="YOLO confidence threshold"),
    no_clobber: bool = typer.Option(
        False, "--no-clobber", help="Don't overwrite existing output files"
    ),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress non-essential output"),
    verbose: bool = typer.Option(
        False, "-v", "--verbose", help="Enable debug output"
    ),
) -> None:
    """Convert a single image, a glob, or a directory of images to SVG.

    The branch (single file vs. batch) is decided by inspecting the
    input path: glob characters and existing directories go to
    ``convert_batch()``; everything else goes to ``convert()``.
    """
    # Configure logging FIRST so any subsequent log calls use the
    # user's chosen verbosity.
    if quiet:
        setup_logging("quiet")
    elif verbose:
        setup_logging("verbose")
    else:
        setup_logging("default")

    # File-existence check (only for plain files; globs and existing
    # directories are passed through to convert_batch).
    if not _is_batch_input(input):
        if not input.exists():
            _console.print(f"[red]file not found:[/red] {input}")
            raise typer.Exit(2)

    # Build options. The mode / gpu_strategy callbacks have already
    # raised BadParameter (exit 2) on bad values, so Mode(...) and
    # DeviceStrategy(...) below will not raise ValueError.
    try:
        options = _build_options(
            mode=mode,
            model=model,
            device=device,
            gpu_strategy=gpu_strategy,
            conf=conf,
            no_clobber=no_clobber,
        )
    except (ValueError, TypeError) as exc:
        _console.print(f"[red]invalid options:[/red] {exc}")
        raise typer.Exit(2) from exc

    is_batch = _is_batch_input(input)

    # Branch: single file (convert) vs. batch (convert_batch).
    try:
        if is_batch:
            if output is None:
                # Allow default output resolution inside convert_batch.
                out_dir_arg: str | None = None
            else:
                out_dir_arg = str(output)
            results = convert_batch(
                str(input),
                output_dir=out_dir_arg,
                options=options,
                show_progress=not quiet,
            )
            successes = sum(1 for r in results if not r.errors)
            failures = len(results) - successes
            if failures > 0:
                _console.print(
                    f"[yellow]Batch complete: {successes} succeeded, "
                    f"{failures} failed[/yellow]"
                )
                for r in results:
                    if r.errors:
                        _console.print(f"  [red]{r.svg_path.name}:[/red] {r.errors[0]}")
                raise typer.Exit(1)
            _console.print(
                f"[green]Batch complete: {successes} succeeded, 0 failed[/green]"
            )
        else:
            if output is None:
                _console.print(
                    "[red]--output is required when converting a single file[/red]"
                )
                raise typer.Exit(2)
            result = api_convert(str(input), str(output), options=options)
            _console.print(
                f"[green]Converted {result.svg_path}[/green]"
            )
    except typer.Exit:
        # Pass through Typer/Click-managed exits (e.g. from --help).
        raise
    except (FileNotFoundError, UnsupportedFormatError, CorruptImageError) as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except OutputPathCollisionError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except DeviceUnavailableError as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except (ModelLoadError, ConfigError) as exc:
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from exc
    except Img2SvgError as exc:
        # Other img2svg errors → exit 1 (general runtime failure)
        _console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


# ----------------------------------------------------------------------
# list-gpus subcommand
# ----------------------------------------------------------------------


@app.command(
    "list-gpus",
    help="List available GPUs and the recommended one (per --gpu-strategy).",
)
def _list_gpus_cmd(
    strategy: str = typer.Option(
        "power",
        "--strategy",
        help="Recommendation strategy: auto, power, availability",
    ),
) -> None:
    """List detected GPUs. T22 will replace this stub with a richer view."""
    try:
        strategy_enum = DeviceStrategy(strategy)
    except ValueError:
        valid = ", ".join(s.value for s in DeviceStrategy)
        _console.print(
            f"[red]invalid strategy {strategy!r}. Valid strategies: {valid}[/red]"
        )
        raise typer.Exit(2)

    gpus = list_gpus()
    recommended = recommend_gpu(gpus, strategy_enum) if gpus else None
    _print_gpu_table(gpus, recommended)


# ----------------------------------------------------------------------
# info subcommand
# ----------------------------------------------------------------------


@app.command(
    "info",
    help="Print version, Python, OS, and detected compute devices.",
)
def _info_cmd() -> None:
    """Print runtime environment information."""
    try:
        os_name = os.uname().sysname
    except AttributeError:
        # Windows (no os.uname). Fall back to platform.
        os_name = sys.platform

    _console.print(f"[bold]img2svg[/bold] [cyan]{__version__}[/cyan]")
    _console.print(f"Python:  {sys.version.split()[0]}")
    _console.print(f"OS:      {os_name}")
    devs = list_available_devices()
    _console.print(
        f"Devices: {', '.join(devs) if devs else '(none detected)'}"
    )

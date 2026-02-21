"""Shared CLI helpers for simulation experiments."""

from __future__ import annotations

import argparse


def _has_opt(parser: argparse.ArgumentParser, opt: str) -> bool:
    return opt in parser._option_string_actions  # noqa: SLF001


def _add_bool_arg(
    parser: argparse.ArgumentParser,
    *,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    flag_us = f"--{name}"
    flag_dash = f"--{name.replace('_', '-')}"
    opts = [flag_us] if flag_us == flag_dash else [flag_us, flag_dash]
    if any(_has_opt(parser, opt) for opt in opts):
        return
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            *opts,
            action=argparse.BooleanOptionalAction,
            default=bool(default),
            help=help_text,
        )
    else:
        # Fallback for older Python versions.
        parser.add_argument(
            *opts, action="store_true", default=bool(default), help=help_text
        )
        parser.add_argument(
            f"--no-{name}",
            dest=name,
            action="store_false",
            help=f"Disable: {help_text}",
        )
        if flag_us != flag_dash:
            parser.add_argument(
                f"--no-{name.replace('_', '-')}",
                dest=name,
                action="store_false",
                help=f"Disable: {help_text}",
            )


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common runner flags while avoiding duplicate option conflicts."""
    if not _has_opt(parser, "--outdir"):
        parser.add_argument(
            "--outdir", type=str, required=True, help="Output directory."
        )
    if not _has_opt(parser, "--master_seed"):
        parser.add_argument("--master_seed", type=int, default=123, help="Master seed.")
    if not _has_opt(parser, "--test_mode"):
        parser.add_argument(
            "--test_mode", action="store_true", help="Run tiny deterministic settings."
        )

    if not _has_opt(parser, "--n_jobs"):
        parser.add_argument(
            "--n_jobs",
            "--workers",
            dest="n_jobs",
            type=int,
            default=1,
            help="Execution parallelism.",
        )
    if not _has_opt(parser, "--backend"):
        parser.add_argument(
            "--backend",
            type=str,
            choices=["loky", "multiprocessing", "threading"],
            default="loky",
            help="Parallel backend.",
        )
    if not _has_opt(parser, "--chunk_size"):
        parser.add_argument(
            "--chunk_size", type=int, default=25, help="Parallel map chunk size."
        )
    if not _has_opt(parser, "--cache_dir"):
        parser.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Optional cache directory (defaults to run_dir/cache).",
        )
    _add_bool_arg(
        parser,
        name="progress",
        default=True,
        help_text="Enable progress output for long loops.",
    )
    _add_bool_arg(
        parser,
        name="fast_mode",
        default=False,
        help_text="Enable runtime-saving fast path defaults while preserving correctness.",
    )
    if not _has_opt(parser, "--overwrite"):
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing run dir if needed.",
        )
    if not _has_opt(parser, "--run_tag"):
        parser.add_argument(
            "--run_tag", type=str, default=None, help="Optional run tag suffix."
        )
    if not _has_opt(parser, "--dry_run"):
        parser.add_argument(
            "--dry_run",
            action="store_true",
            help="Print cell count/planned run and exit.",
        )
    if not _has_opt(parser, "--artifact_snapshot"):
        parser.add_argument(
            "--artifact_snapshot",
            action="store_true",
            help="Copy minimal code/config snapshot into run folder.",
        )
    if not _has_opt(parser, "--log_level"):
        parser.add_argument(
            "--log_level", type=str, default="INFO", help="Log level (INFO/DEBUG/WARN)."
        )
    if not _has_opt(parser, "--no_plots"):
        parser.add_argument(
            "--no_plots",
            action="store_true",
            help="Skip plotting stage if supported by experiment.",
        )
    if not _has_opt(parser, "--no_reports"):
        parser.add_argument(
            "--no_reports",
            action="store_true",
            help="Skip report generation if supported by experiment.",
        )
    if not _has_opt(parser, "--plots"):
        parser.add_argument(
            "--plots",
            type=str,
            choices=["all", "minimal", "none"],
            default="all",
            help="Plot generation level.",
        )
    if not _has_opt(parser, "--n_example_genes"):
        parser.add_argument(
            "--n_example_genes",
            type=int,
            default=12,
            help="Number of representative genes/cells for example plots.",
        )
    if not _has_opt(parser, "--example_gene_strategy"):
        parser.add_argument(
            "--example_gene_strategy",
            type=str,
            default="top_and_controls",
            help="Strategy for selecting representative example genes.",
        )
    if not _has_opt(parser, "--n_perm_pool"):
        parser.add_argument(
            "--n_perm_pool",
            type=int,
            default=None,
            help="Optional pooled permutation count for fast paths.",
        )
    _add_bool_arg(
        parser,
        name="save_full_profiles",
        default=False,
        help_text="Persist full per-gene profile arrays.",
    )
    _add_bool_arg(
        parser,
        name="skip_heavy_examples",
        default=False,
        help_text="Skip heavy example generation for very large runs.",
    )
    return parser

"""
Author: Jiheng Li
Email: jiheng.li.1@vanderbilt.edu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def _cmd_pipeline(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline_main

    return run_pipeline_main(args.config)


def _cmd_review(args: argparse.Namespace) -> int:
    from .review.cli import run_review_main

    return run_review_main(args.config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hapid")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_pipeline = subparsers.add_parser(
        "pipeline",
        help="Run the auditing pipeline",
    )
    p_pipeline.add_argument(
        "config",
        type=Path,
        help="Path to pipeline TOML config file",
    )
    p_pipeline.set_defaults(func=_cmd_pipeline)

    p_review = subparsers.add_parser(
        "review",
        help="Run the review app",
    )
    p_review.add_argument(
        "config",
        type=Path,
        help="Path to review TOML config file",
    )
    p_review.set_defaults(func=_cmd_review)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
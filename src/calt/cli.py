"""Command line entry point for CALT."""

from __future__ import annotations

import argparse
import sys

from .kaggle import KaggleJobError, KaggleKernelConfig, run_kaggle_job


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calt")
    subparsers = parser.add_subparsers(dest="command")

    kaggle_parser = subparsers.add_parser("kaggle", help="Run jobs on Kaggle kernels")
    kaggle_subparsers = kaggle_parser.add_subparsers(dest="kaggle_command")
    run_parser = kaggle_subparsers.add_parser(
        "run",
        help="Submit a local training job to Kaggle and optionally wait/download outputs.",
    )

    run_parser.add_argument("--source-dir", required=True, help="Job source directory")
    run_parser.add_argument(
        "--script",
        required=True,
        help="Training script path relative to --source-dir (e.g., train.py)",
    )
    run_parser.add_argument(
        "--kernel-id",
        required=True,
        help="Kaggle kernel id in format username/slug",
    )
    run_parser.add_argument(
        "--output-dir",
        required=True,
        help="Local directory to download Kaggle outputs into",
    )
    run_parser.add_argument(
        "--include-path",
        action="append",
        default=[],
        help="Additional local file/dir to copy into the job package (repeatable)",
    )
    run_parser.add_argument(
        "--accelerator",
        default=None,
        help="Kaggle accelerator (e.g., NvidiaTeslaT4)",
    )
    run_parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Timeout in seconds for waiting/submission",
    )
    run_parser.add_argument(
        "--poll-interval-sec",
        type=int,
        default=20,
        help="Polling interval in seconds while waiting for completion",
    )
    run_parser.add_argument(
        "--job-dir",
        default=None,
        help="Optional directory to build the Kaggle upload package into",
    )
    run_parser.add_argument(
        "--title",
        default=None,
        help="Optional kernel title override",
    )
    run_parser.add_argument(
        "--keep-job-dir",
        action="store_true",
        help="Keep local job package directory after submission",
    )
    run_parser.add_argument(
        "--wait",
        dest="wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for job completion before exiting (default: true)",
    )
    run_parser.add_argument(
        "--internet",
        dest="enable_internet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable internet access inside Kaggle runtime (default: false)",
    )
    run_parser.add_argument(
        "--gpu",
        dest="enable_gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GPU runtime for Kaggle job (default: true)",
    )
    run_parser.add_argument(
        "--private",
        dest="is_private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create/update kernel as private (default: true)",
    )
    run_parser.set_defaults(handler=_handle_kaggle_run)

    return parser


def _handle_kaggle_run(args: argparse.Namespace) -> int:
    config = KaggleKernelConfig(
        kernel_id=args.kernel_id,
        title=args.title,
        enable_gpu=args.enable_gpu,
        enable_internet=args.enable_internet,
        is_private=args.is_private,
    )
    result = run_kaggle_job(
        source_dir=args.source_dir,
        script=args.script,
        config=config,
        output_dir=args.output_dir,
        include_paths=args.include_path,
        accelerator=args.accelerator,
        wait=args.wait,
        timeout_sec=args.timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
        keep_job_dir=args.keep_job_dir,
        job_dir=args.job_dir,
    )
    print(f"Kernel: {result.kernel_id}")
    print(f"Submit output: {result.submit_output}")
    if args.wait:
        print("Job completed.")
        if result.output_dir is not None:
            print(f"Outputs downloaded to: {result.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return handler(args)
    except KaggleJobError as exc:
        print(f"[calt kaggle] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

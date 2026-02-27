"""Command line entry point for CALT."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from .kaggle import (
    MANIFEST_FILE_NAME,
    KaggleJobError,
    KaggleKernelConfig,
    run_kaggle_job,
)


def _kaggle_dir() -> Path:
    return Path.home() / ".kaggle"


def _chmod_user_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _write_access_token(token: str) -> Path:
    kaggle_dir = _kaggle_dir()
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_dir.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    token_path = kaggle_dir / "access_token"
    token_path.write_text(token.strip() + "\n", encoding="utf-8")
    _chmod_user_only(token_path)
    return token_path


def _write_kaggle_json(username: str, token: str) -> Path:
    kaggle_dir = _kaggle_dir()
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_dir.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    json_path = kaggle_dir / "kaggle.json"
    payload = {"username": username.strip(), "key": token.strip()}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _chmod_user_only(json_path)
    return json_path


def _run_kaggle_healthcheck() -> tuple[bool, str]:
    if shutil.which("kaggle") is None:
        return False, "kaggle CLI not found"
    proc = subprocess.run(
        ["kaggle", "kernels", "list", "--page-size", "1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, "kaggle API check passed"
    detail = (proc.stderr or proc.stdout or "unknown error").strip()
    return False, f"kaggle API check failed: {detail}"


def _add_run_arguments(run_parser: argparse.ArgumentParser) -> None:
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
        "--debug-package",
        action="store_true",
        help="Keep and print packaged job directory/manifest for debugging",
    )
    run_parser.add_argument(
        "--bundle-dataset-id",
        default=None,
        help="Optional Kaggle dataset id (<owner>/<slug>) used to upload include paths",
    )
    run_parser.add_argument(
        "--bundle-dataset-title",
        default=None,
        help="Optional title for auto-created/updated include bundle dataset",
    )
    run_parser.add_argument(
        "--bundle-dataset-public",
        action="store_true",
        help="Create include bundle dataset as public (default: private)",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="calt")
    subparsers = parser.add_subparsers(dest="command")

    remote_parser = subparsers.add_parser(
        "remote", help="Run jobs on remote backends (Kaggle)."
    )
    remote_subparsers = remote_parser.add_subparsers(dest="remote_command")
    remote_run_parser = remote_subparsers.add_parser(
        "run",
        help="Submit a local training job to Kaggle and optionally wait/download outputs.",
    )
    _add_run_arguments(remote_run_parser)
    remote_run_parser.set_defaults(handler=_handle_kaggle_run, legacy_alias=False)
    remote_init_parser = remote_subparsers.add_parser(
        "init",
        help="Initialize Kaggle credentials for remote runs.",
    )
    remote_init_parser.add_argument(
        "--store",
        choices=["access-token", "kaggle-json", "env"],
        default="access-token",
        help="Credential storage mode (default: access-token).",
    )
    remote_init_parser.add_argument(
        "--token",
        default=None,
        help="Kaggle API token value. If omitted, prompt securely.",
    )
    remote_init_parser.add_argument(
        "--username",
        default=None,
        help="Kaggle username (required for --store kaggle-json).",
    )
    remote_init_parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip API connectivity test after writing credentials.",
    )
    remote_init_parser.set_defaults(handler=_handle_remote_init, legacy_alias=False)
    remote_doctor_parser = remote_subparsers.add_parser(
        "doctor",
        help="Check remote(Kaggle) CLI/auth setup.",
    )
    remote_doctor_parser.add_argument(
        "--skip-api-test",
        action="store_true",
        help="Skip calling Kaggle API and only check local config.",
    )
    remote_doctor_parser.set_defaults(handler=_handle_remote_doctor, legacy_alias=False)

    # Backward-compatible alias: `calt kaggle run`
    kaggle_parser = subparsers.add_parser(
        "kaggle",
        help=argparse.SUPPRESS,
    )
    kaggle_subparsers = kaggle_parser.add_subparsers(dest="kaggle_command")
    kaggle_run_parser = kaggle_subparsers.add_parser("run", help=argparse.SUPPRESS)
    _add_run_arguments(kaggle_run_parser)
    kaggle_run_parser.set_defaults(handler=_handle_kaggle_run, legacy_alias=True)
    kaggle_init_parser = kaggle_subparsers.add_parser("init", help=argparse.SUPPRESS)
    kaggle_init_parser.add_argument(
        "--store",
        choices=["access-token", "kaggle-json", "env"],
        default="access-token",
    )
    kaggle_init_parser.add_argument("--token", default=None)
    kaggle_init_parser.add_argument("--username", default=None)
    kaggle_init_parser.add_argument("--no-test", action="store_true")
    kaggle_init_parser.set_defaults(handler=_handle_remote_init, legacy_alias=True)
    kaggle_doctor_parser = kaggle_subparsers.add_parser("doctor", help=argparse.SUPPRESS)
    kaggle_doctor_parser.add_argument("--skip-api-test", action="store_true")
    kaggle_doctor_parser.set_defaults(handler=_handle_remote_doctor, legacy_alias=True)

    return parser


def _handle_kaggle_run(args: argparse.Namespace) -> int:
    if getattr(args, "legacy_alias", False):
        print(
            "Deprecated: use `calt remote run ...` instead of `calt kaggle run ...`.",
            file=sys.stderr,
        )
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
        keep_job_dir=(args.keep_job_dir or args.debug_package),
        job_dir=args.job_dir,
        use_bootstrap_entrypoint=True,
        write_manifest=True,
        bundle_include_paths_as_dataset=True,
        bundle_dataset_id=args.bundle_dataset_id,
        bundle_dataset_title=args.bundle_dataset_title,
        bundle_dataset_public=args.bundle_dataset_public,
    )
    print(f"Kernel: {result.kernel_id}")
    print(f"Submit output: {result.submit_output}")
    if args.wait:
        print("Job completed.")
        if result.output_dir is not None:
            print(f"Outputs downloaded to: {result.output_dir}")
    if args.debug_package:
        print(f"Packaged job dir: {result.job_dir}")
        print(f"Manifest: {result.job_dir / MANIFEST_FILE_NAME}")
    return 0


def _handle_remote_init(args: argparse.Namespace) -> int:
    if getattr(args, "legacy_alias", False):
        print(
            "Deprecated: use `calt remote init ...` instead of `calt kaggle init ...`.",
            file=sys.stderr,
        )
    token = args.token or getpass.getpass("Kaggle API token: ").strip()
    if not token:
        print("[calt remote] token is empty.", file=sys.stderr)
        return 2

    if args.store == "access-token":
        saved_path = _write_access_token(token)
        print(f"Saved Kaggle token to: {saved_path}")
    elif args.store == "kaggle-json":
        username = args.username or input("Kaggle username: ").strip()
        if not username:
            print("[calt remote] username is required for kaggle-json store.", file=sys.stderr)
            return 2
        saved_path = _write_kaggle_json(username, token)
        print(f"Saved Kaggle credentials to: {saved_path}")
    else:
        os.environ["KAGGLE_API_TOKEN"] = token
        print("Set KAGGLE_API_TOKEN for current process.")
        print("Add this to your shell profile if needed:")
        print("export KAGGLE_API_TOKEN=<your-token>")

    if args.no_test:
        return 0
    ok, message = _run_kaggle_healthcheck()
    if ok:
        print(message)
        return 0
    print(f"[calt remote] {message}", file=sys.stderr)
    return 2


def _handle_remote_doctor(args: argparse.Namespace) -> int:
    if getattr(args, "legacy_alias", False):
        print(
            "Deprecated: use `calt remote doctor ...` instead of `calt kaggle doctor ...`.",
            file=sys.stderr,
        )

    ok = True
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        print(f"CLI: ok ({kaggle_bin})")
    else:
        print("CLI: missing (`pip install \"calt-x[kaggle]\"`)", file=sys.stderr)
        ok = False

    token_env = bool(os.environ.get("KAGGLE_API_TOKEN"))
    access_token_file = _kaggle_dir() / "access_token"
    kaggle_json_file = _kaggle_dir() / "kaggle.json"
    token_file = access_token_file.exists()
    json_file = kaggle_json_file.exists()
    if token_env or token_file or json_file:
        source = []
        if token_env:
            source.append("env")
        if token_file:
            source.append(str(access_token_file))
        if json_file:
            source.append(str(kaggle_json_file))
        print(f"Auth: found ({', '.join(source)})")
    else:
        print("Auth: missing (run `calt remote init`)", file=sys.stderr)
        ok = False

    if not args.skip_api_test:
        health_ok, message = _run_kaggle_healthcheck()
        if health_ok:
            print(f"API: ok ({message})")
        else:
            print(f"API: fail ({message})", file=sys.stderr)
            ok = False

    return 0 if ok else 2


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
        print(f"[calt remote] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

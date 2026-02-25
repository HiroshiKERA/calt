"""Utilities for submitting CALT training jobs to Kaggle kernels."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_POLL_INTERVAL_SEC = 20
ENTRYPOINT_SCRIPT_NAME = "__calt_kaggle_entrypoint__.py"
MANIFEST_FILE_NAME = "__calt_job_manifest__.txt"


class KaggleJobError(RuntimeError):
    """Raised when a Kaggle job operation fails."""


@dataclass
class KaggleKernelConfig:
    """Configuration for a Kaggle kernel run."""

    kernel_id: str
    title: str | None = None
    enable_gpu: bool = True
    enable_internet: bool = False
    is_private: bool = True
    language: str = "python"
    kernel_type: str = "script"
    dataset_sources: list[str] = field(default_factory=list)
    competition_sources: list[str] = field(default_factory=list)
    kernel_sources: list[str] = field(default_factory=list)
    model_sources: list[str] = field(default_factory=list)


@dataclass
class PreparedKaggleJob:
    """Represents a prepared local job directory for Kaggle push."""

    kernel_id: str
    job_dir: Path
    metadata_path: Path
    code_file: str
    target_script: str
    manifest_path: Path | None = None
    managed_temp_dir: bool = False


def _ensure_kaggle_installed() -> None:
    if shutil.which("kaggle") is None:
        raise KaggleJobError(
            "kaggle CLI not found. Install with `pip install kaggle` "
            "or `pip install \"calt-x[kaggle]\"`."
        )


def _run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout_sec: int | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or str(exc)
        raise KaggleJobError(
            f"Command failed: {' '.join(args)}\n{details}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise KaggleJobError(
            f"Command timed out after {timeout_sec}s: {' '.join(args)}"
        ) from exc


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
        ),
    )


def _copy_path(path: Path, destination_root: Path, source_root: Path) -> None:
    if not path.exists():
        raise KaggleJobError(f"Include path not found: {path}")

    try:
        relative = path.resolve().relative_to(source_root.resolve())
        target = destination_root / relative
    except ValueError:
        target = destination_root / path.name

    if path.is_dir():
        _copy_tree(path, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _kernel_title_from_id(kernel_id: str) -> str:
    slug = kernel_id.split("/")[-1]
    return slug.replace("-", " ")


def _to_kaggle_bool(value: bool) -> str:
    return "true" if value else "false"


def _write_bootstrap_entrypoint(job_root: Path, target_script: str) -> Path:
    entrypoint_path = job_root / ENTRYPOINT_SCRIPT_NAME
    entrypoint_path.write_text(
        (
            "from __future__ import annotations\n"
            "\n"
            "import runpy\n"
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "JOB_ROOT = Path(__file__).resolve().parent\n"
            f"TARGET_SCRIPT = JOB_ROOT / {target_script!r}\n"
            "\n"
            "# Ensure bundled project sources are importable (e.g., calt package).\n"
            "candidate_paths = [\n"
            "    JOB_ROOT,\n"
            "    JOB_ROOT / 'src',\n"
            "    JOB_ROOT / 'calt',\n"
            "]\n"
            "for path in candidate_paths:\n"
            "    if path.exists():\n"
            "        path_str = str(path)\n"
            "        if path_str not in sys.path:\n"
            "            sys.path.insert(0, path_str)\n"
            "\n"
            "if not TARGET_SCRIPT.exists():\n"
            "    raise FileNotFoundError(f'Target script not found: {TARGET_SCRIPT}')\n"
            "\n"
            "runpy.run_path(str(TARGET_SCRIPT), run_name='__main__')\n"
        ),
        encoding="utf-8",
    )
    return entrypoint_path


def _write_manifest(job_root: Path) -> Path:
    manifest_path = job_root / MANIFEST_FILE_NAME
    lines = []
    for path in sorted(job_root.rglob("*")):
        if path.is_file():
            lines.append(path.relative_to(job_root).as_posix())
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def build_kernel_metadata(config: KaggleKernelConfig, code_file: str) -> dict[str, Any]:
    """Build a kernel-metadata.json payload for Kaggle kernels push."""
    return {
        "id": config.kernel_id,
        "title": config.title or _kernel_title_from_id(config.kernel_id),
        "code_file": code_file,
        "language": config.language,
        "kernel_type": config.kernel_type,
        "is_private": _to_kaggle_bool(config.is_private),
        "enable_gpu": _to_kaggle_bool(config.enable_gpu),
        "enable_internet": _to_kaggle_bool(config.enable_internet),
        "dataset_sources": config.dataset_sources,
        "competition_sources": config.competition_sources,
        "kernel_sources": config.kernel_sources,
        "model_sources": config.model_sources,
    }


def prepare_job(
    *,
    source_dir: str | Path,
    script: str,
    config: KaggleKernelConfig,
    include_paths: list[str] | None = None,
    job_dir: str | Path | None = None,
    use_bootstrap_entrypoint: bool = True,
    write_manifest: bool = True,
) -> PreparedKaggleJob:
    """Create a Kaggle-ready job directory with metadata and local inputs."""
    source_root = Path(source_dir).resolve()
    if not source_root.is_dir():
        raise KaggleJobError(f"source_dir does not exist or is not a directory: {source_root}")

    script_path = source_root / script
    if not script_path.exists():
        raise KaggleJobError(f"script not found: {script_path}")

    managed_temp_dir = False
    if job_dir is None:
        job_root = Path(tempfile.mkdtemp(prefix="calt-kaggle-job-"))
        managed_temp_dir = True
    else:
        job_root = Path(job_dir).resolve()
        job_root.mkdir(parents=True, exist_ok=True)

    _copy_tree(source_root, job_root)
    for include in include_paths or []:
        include_path = Path(include).expanduser()
        if not include_path.is_absolute():
            include_path = (source_root / include_path).resolve()
        _copy_path(include_path, job_root, source_root)

    target_script = script_path.resolve().relative_to(source_root).as_posix()
    code_file = target_script
    if use_bootstrap_entrypoint:
        _write_bootstrap_entrypoint(job_root, target_script)
        code_file = ENTRYPOINT_SCRIPT_NAME

    metadata = build_kernel_metadata(config, code_file=code_file)
    metadata_path = job_root / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest_path = _write_manifest(job_root) if write_manifest else None

    return PreparedKaggleJob(
        kernel_id=config.kernel_id,
        job_dir=job_root,
        metadata_path=metadata_path,
        code_file=code_file,
        target_script=target_script,
        manifest_path=manifest_path,
        managed_temp_dir=managed_temp_dir,
    )


def submit_job(
    prepared: PreparedKaggleJob,
    *,
    accelerator: str | None = None,
    timeout_sec: int | None = None,
) -> str:
    """Submit a prepared job to Kaggle and return stdout text."""
    _ensure_kaggle_installed()
    command = ["kaggle", "kernels", "push", "-p", str(prepared.job_dir)]
    if accelerator:
        command.extend(["--accelerator", accelerator])
    _result = _run_command(command, timeout_sec=timeout_sec)
    return _result.stdout.strip()


def get_job_status(kernel_id: str) -> str:
    """Return raw `kaggle kernels status` output."""
    _ensure_kaggle_installed()
    result = _run_command(["kaggle", "kernels", "status", kernel_id])
    return result.stdout.strip()


def parse_status_text(status_text: str) -> str:
    """Parse Kaggle status output and normalize to known states."""
    text = status_text.lower()
    if "complete" in text:
        return "complete"
    if "error" in text or "failed" in text:
        return "failed"
    if "cancel" in text:
        return "cancelled"
    if "running" in text:
        return "running"
    if "queued" in text:
        return "queued"
    return "unknown"


def wait_for_job(
    kernel_id: str,
    *,
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC,
    timeout_sec: int | None = None,
) -> str:
    """Poll kernel status until completion or failure."""
    deadline = None if timeout_sec is None else time.time() + timeout_sec
    while True:
        raw_status = get_job_status(kernel_id)
        state = parse_status_text(raw_status)
        if state == "complete":
            return raw_status
        if state in {"failed", "cancelled"}:
            raise KaggleJobError(f"Kaggle kernel ended with state '{state}'.\n{raw_status}")
        if deadline is not None and time.time() > deadline:
            raise KaggleJobError(
                f"Timed out waiting for kernel '{kernel_id}' after {timeout_sec} seconds."
            )
        time.sleep(poll_interval_sec)


def download_output(
    kernel_id: str,
    output_dir: str | Path,
    *,
    force: bool = True,
    quiet: bool = False,
    file_pattern: str | None = None,
) -> str:
    """Download output artifacts for a kernel run to a local directory."""
    _ensure_kaggle_installed()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    command = ["kaggle", "kernels", "output", kernel_id, "-p", str(out)]
    if force:
        command.append("-o")
    if quiet:
        command.append("-q")
    if file_pattern:
        command.extend(["--file-pattern", file_pattern])

    result = _run_command(command)
    return result.stdout.strip()


def cleanup_job_dir(prepared: PreparedKaggleJob) -> None:
    """Delete a temporary job directory created by prepare_job()."""
    if prepared.managed_temp_dir and prepared.job_dir.exists():
        shutil.rmtree(prepared.job_dir, ignore_errors=True)


@dataclass
class KaggleRunResult:
    """Result object for an end-to-end Kaggle run."""

    kernel_id: str
    job_dir: Path
    submit_output: str
    final_status: str | None = None
    output_dir: Path | None = None


def run_kaggle_job(
    *,
    source_dir: str | Path,
    script: str,
    config: KaggleKernelConfig,
    output_dir: str | Path,
    include_paths: list[str] | None = None,
    accelerator: str | None = None,
    wait: bool = True,
    timeout_sec: int | None = None,
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC,
    keep_job_dir: bool = False,
    job_dir: str | Path | None = None,
    use_bootstrap_entrypoint: bool = True,
    write_manifest: bool = True,
) -> KaggleRunResult:
    """Prepare, submit, optionally wait, and download output for a Kaggle run."""
    prepared = prepare_job(
        source_dir=source_dir,
        script=script,
        config=config,
        include_paths=include_paths,
        job_dir=job_dir,
        use_bootstrap_entrypoint=use_bootstrap_entrypoint,
        write_manifest=write_manifest,
    )
    try:
        submit_output = submit_job(
            prepared,
            accelerator=accelerator,
            timeout_sec=timeout_sec,
        )
        result = KaggleRunResult(
            kernel_id=prepared.kernel_id,
            job_dir=prepared.job_dir,
            submit_output=submit_output,
        )
        if wait:
            result.final_status = wait_for_job(
                prepared.kernel_id,
                poll_interval_sec=poll_interval_sec,
                timeout_sec=timeout_sec,
            )
            download_output(prepared.kernel_id, output_dir)
            result.output_dir = Path(output_dir).expanduser().resolve()
        return result
    finally:
        if not keep_job_dir:
            cleanup_job_dir(prepared)

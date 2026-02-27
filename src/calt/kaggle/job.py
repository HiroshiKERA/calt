"""Utilities for submitting CALT training jobs to Kaggle kernels."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

DEFAULT_POLL_INTERVAL_SEC = 20
ENTRYPOINT_SCRIPT_NAME = "__calt_kaggle_entrypoint__.py"
MANIFEST_FILE_NAME = "__calt_job_manifest__.txt"
DEFAULT_DATASET_READY_TIMEOUT_SEC = 300


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
            'or `pip install "calt-x[kaggle]"`.'
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
        raise KaggleJobError(f"Command failed: {' '.join(args)}\n{details}") from exc
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
            "wandb",
            "results",
            ".pytest_cache",
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


def _slugify(value: str) -> str:
    lowered = value.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-")


def _dataset_exists(dataset_id: str) -> bool:
    _ensure_kaggle_installed()
    result = subprocess.run(
        ["kaggle", "datasets", "status", dataset_id],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _wait_for_dataset_ready(
    dataset_id: str,
    *,
    timeout_sec: int = DEFAULT_DATASET_READY_TIMEOUT_SEC,
    poll_interval_sec: int = 5,
) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        result = subprocess.run(
            ["kaggle", "datasets", "status", dataset_id],
            capture_output=True,
            text=True,
            check=False,
        )
        status_text = (result.stdout or result.stderr or "").lower()
        if result.returncode == 0 and (
            "ready" in status_text or "complete" in status_text
        ):
            return
        if "error" in status_text or "failed" in status_text:
            raise KaggleJobError(
                f"Dataset '{dataset_id}' is not ready: {status_text.strip()}"
            )
        time.sleep(poll_interval_sec)
    raise KaggleJobError(
        f"Timed out waiting for dataset '{dataset_id}' to become ready."
    )


def _write_dataset_metadata(dataset_dir: Path, dataset_id: str, title: str) -> Path:
    metadata_path = dataset_dir / "dataset-metadata.json"
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def _derive_bundle_dataset_id(kernel_id: str) -> str:
    owner, slug = kernel_id.split("/", 1)
    return f"{owner}/{_slugify(slug)}-bundle"


def _derive_bundle_dataset_title(kernel_id: str) -> str:
    _, slug = kernel_id.split("/", 1)
    return f"CALT bundle {_slugify(slug)}"


def create_or_update_bundle_dataset(
    *,
    source_dir: str | Path,
    include_paths: list[str],
    dataset_id: str,
    title: str,
    public: bool = False,
    version_message: str = "Update bundle from calt remote run",
) -> str:
    """Create or update a Kaggle dataset containing source/include files."""
    source_root = Path(source_dir).resolve()
    if not source_root.is_dir():
        raise KaggleJobError(
            f"source_dir does not exist or is not a directory: {source_root}"
        )
    if "/" not in dataset_id:
        raise KaggleJobError("dataset_id must be in format <owner>/<slug>")

    bundle_dir = Path(tempfile.mkdtemp(prefix="calt-kaggle-bundle-"))
    try:
        _copy_tree(source_root, bundle_dir)
        for include in include_paths:
            include_path = Path(include).expanduser()
            if not include_path.is_absolute():
                include_path = (source_root / include_path).resolve()
            _copy_path(include_path, bundle_dir, source_root)

        _write_dataset_metadata(bundle_dir, dataset_id, title)
        if _dataset_exists(dataset_id):
            _run_command(
                [
                    "kaggle",
                    "datasets",
                    "version",
                    "-p",
                    str(bundle_dir),
                    "-m",
                    version_message,
                    "-r",
                    "zip",
                    "-q",
                ]
            )
        else:
            command = [
                "kaggle",
                "datasets",
                "create",
                "-p",
                str(bundle_dir),
                "-r",
                "zip",
                "-q",
            ]
            if public:
                command.append("-u")
            _run_command(command)
    finally:
        shutil.rmtree(bundle_dir, ignore_errors=True)

    _wait_for_dataset_ready(dataset_id)
    return dataset_id


def _write_bootstrap_entrypoint(
    job_root: Path,
    target_script: str,
    target_script_text: str,
) -> Path:
    entrypoint_path = job_root / ENTRYPOINT_SCRIPT_NAME
    entrypoint_path.write_text(
        (
            "from __future__ import annotations\n"
            "\n"
            "import runpy\n"
            "import shutil\n"
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "JOB_ROOT = Path(__file__).resolve().parent\n"
            f"TARGET_SCRIPT_REL = Path({target_script!r})\n"
            f"TARGET_SCRIPT_TEXT = {target_script_text!r}\n"
            "\n"
            "def _add_to_syspath(path: Path) -> None:\n"
            "    if path.exists():\n"
            "        path_str = str(path)\n"
            "        if path_str not in sys.path:\n"
            "            sys.path.insert(0, path_str)\n"
            "\n"
            "def _resolve_target_script() -> Path:\n"
            "    direct_candidates = [\n"
            "        JOB_ROOT / TARGET_SCRIPT_REL,\n"
            "        JOB_ROOT / TARGET_SCRIPT_REL.name,\n"
            "        Path('/kaggle/input') / TARGET_SCRIPT_REL,\n"
            "        Path('/kaggle/working') / TARGET_SCRIPT_REL,\n"
            "    ]\n"
            "    for candidate in direct_candidates:\n"
            "        if candidate.exists():\n"
            "            return candidate\n"
            "\n"
            "    kaggle_input = Path('/kaggle/input')\n"
            "    if kaggle_input.exists():\n"
            "        matches = sorted(kaggle_input.rglob(TARGET_SCRIPT_REL.name))\n"
            "        for match in matches:\n"
            "            if match.is_file():\n"
            "                return match\n"
            "\n"
            "    src_listing = sorted(p.name for p in JOB_ROOT.iterdir()) if JOB_ROOT.exists() else []\n"
            "    input_listing = []\n"
            "    if kaggle_input.exists():\n"
            "        input_listing = sorted(p.name for p in kaggle_input.iterdir())\n"
            "    raise FileNotFoundError(\n"
            "        f'Target script not found on filesystem: {JOB_ROOT / TARGET_SCRIPT_REL}. '\n"
            "        f'JOB_ROOT files={src_listing}, /kaggle/input entries={input_listing}. '\n"
            "        'Falling back to embedded script text.'\n"
            "    )\n"
            "\n"
            "def _add_candidate_package_roots() -> None:\n"
            "    base_candidates = [JOB_ROOT, JOB_ROOT / 'src', Path('/kaggle/working')]\n"
            "    for base in base_candidates:\n"
            "        _add_to_syspath(base)\n"
            "\n"
            "    for base in (Path('/kaggle/input'), Path('/kaggle/working')):\n"
            "        if not base.exists():\n"
            "            continue\n"
            "        for calt_dir in base.rglob('calt'):\n"
            "            if calt_dir.is_dir() and (calt_dir / '__init__.py').exists():\n"
            "                _add_to_syspath(calt_dir.parent)\n"
            "                parent_src = calt_dir.parent / 'src'\n"
            "                if (parent_src / 'calt').exists():\n"
            "                    _add_to_syspath(parent_src)\n"
            "\n"
            "_add_candidate_package_roots()\n"
            "import os\n"
            "try:\n"
            "    target_script = _resolve_target_script()\n"
            "except FileNotFoundError as exc:\n"
            "    print(str(exc))\n"
            "    target_script = None\n"
            "\n"
            "if target_script is not None:\n"
            "    runtime_script = target_script\n"
            "    if str(target_script).startswith('/kaggle/input/'):\n"
            "        runtime_root = Path('/kaggle/working/calt_runtime')\n"
            "        if runtime_root.exists():\n"
            "            shutil.rmtree(runtime_root, ignore_errors=True)\n"
            "        shutil.copytree(target_script.parent, runtime_root, dirs_exist_ok=True)\n"
            "        _add_to_syspath(runtime_root)\n"
            "        _add_to_syspath(runtime_root / 'src')\n"
            "        if (runtime_root / 'calt').exists():\n"
            "            _add_to_syspath(runtime_root)\n"
            "        runtime_script = runtime_root / target_script.name\n"
            "        os.chdir(str(runtime_root))\n"
            "    else:\n"
            "        os.chdir(str(target_script.parent))\n"
            "    runpy.run_path(str(runtime_script), run_name='__main__')\n"
            "else:\n"
            "    os.chdir('/kaggle/working')\n"
            "    embedded_globals = {\n"
            "        '__name__': '__main__',\n"
            "        '__file__': str(JOB_ROOT / TARGET_SCRIPT_REL),\n"
            "        '__package__': None,\n"
            "    }\n"
            "    exec(compile(TARGET_SCRIPT_TEXT, embedded_globals['__file__'], 'exec'), embedded_globals)\n"
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
        raise KaggleJobError(
            f"source_dir does not exist or is not a directory: {source_root}"
        )

    script_path = source_root / script
    if not script_path.exists():
        raise KaggleJobError(f"script not found: {script_path}")
    target_script_text = script_path.read_text(encoding="utf-8")

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
        _write_bootstrap_entrypoint(job_root, target_script, target_script_text)
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
            raise KaggleJobError(
                f"Kaggle kernel ended with state '{state}'.\n{raw_status}"
            )
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
    bundle_dataset_id: str | None = None


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
    bundle_include_paths_as_dataset: bool = True,
    bundle_dataset_id: str | None = None,
    bundle_dataset_title: str | None = None,
    bundle_dataset_public: bool = False,
    bundle_dataset_version_message: str = "Update bundle from calt remote run",
) -> KaggleRunResult:
    """Prepare, submit, optionally wait, and download output for a Kaggle run."""
    attached_dataset: str | None = None
    effective_config = config
    if include_paths and bundle_include_paths_as_dataset:
        dataset_id = bundle_dataset_id or _derive_bundle_dataset_id(config.kernel_id)
        dataset_title = bundle_dataset_title or _derive_bundle_dataset_title(
            config.kernel_id
        )
        attached_dataset = create_or_update_bundle_dataset(
            source_dir=source_dir,
            include_paths=include_paths,
            dataset_id=dataset_id,
            title=dataset_title,
            public=bundle_dataset_public,
            version_message=bundle_dataset_version_message,
        )
        dataset_sources = list(config.dataset_sources)
        if attached_dataset not in dataset_sources:
            dataset_sources.append(attached_dataset)
        effective_config = replace(config, dataset_sources=dataset_sources)

    prepared = prepare_job(
        source_dir=source_dir,
        script=script,
        config=effective_config,
        include_paths=None,
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
            bundle_dataset_id=attached_dataset,
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


def delete_kernel(kernel_id: str, *, yes: bool = True) -> str:
    """Delete a Kaggle kernel."""
    _ensure_kaggle_installed()
    command = ["kaggle", "kernels", "delete", kernel_id]
    if yes:
        command.append("--yes")
    result = _run_command(command)
    return result.stdout.strip()


def delete_dataset(dataset_id: str, *, yes: bool = True) -> str:
    """Delete a Kaggle dataset."""
    _ensure_kaggle_installed()
    command = ["kaggle", "datasets", "delete", dataset_id]
    if yes:
        command.append("--yes")
    result = _run_command(command)
    return result.stdout.strip()

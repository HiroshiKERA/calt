"""Tests for calt.kaggle.job."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from calt.kaggle.job import (
    ENTRYPOINT_SCRIPT_NAME,
    MANIFEST_FILE_NAME,
    KaggleKernelConfig,
    build_kernel_metadata,
    create_or_update_bundle_dataset,
    download_output,
    parse_status_text,
    prepare_job,
    run_kaggle_job,
    submit_job,
)


def test_build_kernel_metadata_defaults() -> None:
    config = KaggleKernelConfig(kernel_id="alice/my-kernel")
    metadata = build_kernel_metadata(config, code_file="train.py")

    assert metadata["id"] == "alice/my-kernel"
    assert metadata["title"] == "my kernel"
    assert metadata["code_file"] == "train.py"
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "false"
    assert metadata["is_private"] == "true"


def test_prepare_job_copies_source_and_include(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
    (source_dir / "configs").mkdir()
    (source_dir / "configs" / "train.yaml").write_text("train:\n  seed: 42\n", encoding="utf-8")

    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    (extra_dir / "notes.txt").write_text("hello\n", encoding="utf-8")

    config = KaggleKernelConfig(kernel_id="alice/test-kernel")
    prepared = prepare_job(
        source_dir=source_dir,
        script="train.py",
        config=config,
        include_paths=[str(extra_dir)],
    )

    try:
        assert (prepared.job_dir / "train.py").exists()
        assert (prepared.job_dir / "configs" / "train.yaml").exists()
        assert (prepared.job_dir / "extra" / "notes.txt").exists()
        assert (prepared.job_dir / ENTRYPOINT_SCRIPT_NAME).exists()
        assert (prepared.job_dir / MANIFEST_FILE_NAME).exists()
        assert prepared.code_file == ENTRYPOINT_SCRIPT_NAME
        assert prepared.target_script == "train.py"
        metadata_text = prepared.metadata_path.read_text(encoding="utf-8")
        assert '"id": "alice/test-kernel"' in metadata_text
        assert f'"code_file": "{ENTRYPOINT_SCRIPT_NAME}"' in metadata_text
    finally:
        # Cleanup is covered in run flow; explicit removal keeps test independent.
        if prepared.job_dir.exists():
            shutil.rmtree(prepared.job_dir, ignore_errors=True)


def test_submit_and_download_build_commands(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run_command(args, **kwargs):  # noqa: ANN001, ANN003
        calls.append(args)
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr("calt.kaggle.job._ensure_kaggle_installed", lambda: None)
    monkeypatch.setattr("calt.kaggle.job._run_command", fake_run_command)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
    config = KaggleKernelConfig(kernel_id="alice/test-kernel")
    prepared = prepare_job(source_dir=source_dir, script="train.py", config=config)

    submit_job(prepared, accelerator="NvidiaTeslaT4")
    download_output("alice/test-kernel", tmp_path / "outputs")

    assert calls[0][:4] == ["kaggle", "kernels", "push", "-p"]
    assert "--accelerator" in calls[0]
    assert calls[1][:4] == ["kaggle", "kernels", "output", "alice/test-kernel"]


def test_parse_status_text() -> None:
    assert parse_status_text("status: running") == "running"
    assert parse_status_text("status: complete") == "complete"
    assert parse_status_text("status: error") == "failed"


def test_entrypoint_contains_fallback_resolution(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
    config = KaggleKernelConfig(kernel_id="alice/test-kernel")
    prepared = prepare_job(source_dir=source_dir, script="train.py", config=config)
    try:
        entrypoint = prepared.job_dir / ENTRYPOINT_SCRIPT_NAME
        content = entrypoint.read_text(encoding="utf-8")
        assert "TARGET_SCRIPT_TEXT" in content
        assert "Path('/kaggle/input')" in content
        assert "rglob(TARGET_SCRIPT_REL.name)" in content
        assert "Path('/kaggle/working/calt_runtime')" in content
        assert "_add_candidate_package_roots()" in content
    finally:
        if prepared.job_dir.exists():
            shutil.rmtree(prepared.job_dir, ignore_errors=True)


def test_create_or_update_bundle_dataset_create(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):  # noqa: ANN001, ANN003
        calls.append(args)
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr("calt.kaggle.job._dataset_exists", lambda dataset_id: False)
    monkeypatch.setattr("calt.kaggle.job._run_command", fake_run)
    monkeypatch.setattr("calt.kaggle.job._wait_for_dataset_ready", lambda *a, **k: None)

    result = create_or_update_bundle_dataset(
        source_dir=source_dir,
        include_paths=[],
        dataset_id="alice/test-bundle",
        title="CALT bundle test",
    )
    assert result == "alice/test-bundle"
    assert calls
    assert calls[0][:3] == ["kaggle", "datasets", "create"]


def test_run_kaggle_job_adds_bundle_dataset_source(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")

    captured_dataset_sources: list[str] = []

    def fake_bundle(**kwargs):  # noqa: ANN003
        return "alice/test-bundle"

    def fake_prepare_job(**kwargs):  # noqa: ANN003
        captured_dataset_sources.extend(kwargs["config"].dataset_sources)
        return SimpleNamespace(
            kernel_id="alice/test-kernel",
            job_dir=tmp_path / "job",
            metadata_path=tmp_path / "job" / "kernel-metadata.json",
            code_file=ENTRYPOINT_SCRIPT_NAME,
            target_script="train.py",
            manifest_path=None,
            managed_temp_dir=False,
        )

    monkeypatch.setattr("calt.kaggle.job.create_or_update_bundle_dataset", fake_bundle)
    monkeypatch.setattr("calt.kaggle.job.prepare_job", fake_prepare_job)
    monkeypatch.setattr("calt.kaggle.job.submit_job", lambda *a, **k: "submitted")
    monkeypatch.setattr("calt.kaggle.job.download_output", lambda *a, **k: "downloaded")
    monkeypatch.setattr(
        "calt.kaggle.job.wait_for_job", lambda *a, **k: "alice/test-kernel has status complete"
    )

    run_kaggle_job(
        source_dir=source_dir,
        script="train.py",
        config=KaggleKernelConfig(kernel_id="alice/test-kernel"),
        output_dir=tmp_path / "out",
        include_paths=["extra"],
    )
    assert "alice/test-bundle" in captured_dataset_sources

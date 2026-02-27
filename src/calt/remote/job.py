"""Remote job backend (Kaggle-powered) for CALT."""

from __future__ import annotations

from ..kaggle.job import (
    DEFAULT_DATASET_READY_TIMEOUT_SEC,
    DEFAULT_POLL_INTERVAL_SEC,
    ENTRYPOINT_SCRIPT_NAME,
    MANIFEST_FILE_NAME,
    KaggleJobError,
    KaggleKernelConfig,
    KaggleRunResult,
    PreparedKaggleJob,
    build_kernel_metadata,
    cleanup_job_dir,
    create_or_update_bundle_dataset,
    delete_dataset,
    delete_kernel,
    download_output,
    get_job_status,
    parse_status_text,
    prepare_job,
    run_kaggle_job,
    submit_job,
    wait_for_job,
)

RemoteJobError = KaggleJobError
RemoteRunConfig = KaggleKernelConfig
PreparedRemoteJob = PreparedKaggleJob
RemoteRunResult = KaggleRunResult
run_remote_job = run_kaggle_job

__all__ = [
    "DEFAULT_DATASET_READY_TIMEOUT_SEC",
    "DEFAULT_POLL_INTERVAL_SEC",
    "ENTRYPOINT_SCRIPT_NAME",
    "MANIFEST_FILE_NAME",
    "PreparedRemoteJob",
    "RemoteJobError",
    "RemoteRunConfig",
    "RemoteRunResult",
    "build_kernel_metadata",
    "cleanup_job_dir",
    "create_or_update_bundle_dataset",
    "delete_dataset",
    "delete_kernel",
    "download_output",
    "get_job_status",
    "parse_status_text",
    "prepare_job",
    "run_remote_job",
    "submit_job",
    "wait_for_job",
]

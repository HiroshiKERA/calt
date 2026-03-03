"""Local registry utilities for remote job records."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_jobs_registry_path() -> Path:
    """Return local jobs registry path."""
    return Path.home() / ".calt" / "remote" / "jobs.jsonl"


def generate_job_id() -> str:
    """Generate a user-facing job id."""
    return f"job-{uuid.uuid4().hex[:12]}"


def append_job_record(record: dict[str, Any]) -> Path:
    """Append a job record to local registry."""
    path = get_jobs_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


def load_job_records() -> list[dict[str, Any]]:
    """Load all local job records."""
    path = get_jobs_registry_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def find_job_record(job_id: str) -> dict[str, Any] | None:
    """Find a job record by id."""
    for record in reversed(load_job_records()):
        if record.get("job_id") == job_id:
            return record
    return None


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601."""
    return datetime.now(UTC).isoformat()

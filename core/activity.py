"""Lightweight activity logger for Streamlit workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import os
import uuid

LOG_FILENAME = "activity_log.csv"
HEADER = ["op", "fecha", "accion", "detalle", "trace_id"]


def _log_path() -> Path:
    base = Path(os.getenv("DATA_DIR", "data"))
    return base / LOG_FILENAME


def _ensure_header(path: Path | None = None) -> Path:
    target = path or _log_path()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="|")
            writer.writerow(HEADER)
    return target


def get_log_path(ensure: bool = False) -> Path:
    """Return the CSV path where events are stored."""

    path = _log_path()
    if ensure:
        _ensure_header(path)
    return path


def log_event(op: str, accion: str, detalle: str = "", trace_id: str | None = None) -> str:
    """Append a new activity row and return the trace identifier used."""

    path = _ensure_header()
    now = datetime.now().isoformat(timespec="seconds")
    trace = trace_id or uuid.uuid4().hex[:8]
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow([op.strip() or "operador", now, accion, detalle, trace])
    return trace


def new_trace(prefix: str = "") -> str:
    base = uuid.uuid4().hex[:8]
    return f"{prefix}{base}" if prefix else base


def read_events(limit: int | None = None) -> list[list[str]]:
    """Return the most recent events as a list of rows."""

    path = get_log_path()
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="|")
        rows = list(reader)

    if not rows:
        return []

    header, *entries = rows
    if header != HEADER:
        entries = rows

    if limit is not None and limit >= 0:
        entries = entries[-limit:]
    return entries


__all__ = ["log_event", "new_trace", "get_log_path", "read_events"]

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import tempfile
import uuid
from typing import Iterable

LOG_FILENAME = "activity_log.csv"
PIPE = "|"
COLUMNS = ("timestamp", "level", "scope", "op", "accion", "detalle", "trace")
HEADER = PIPE.join(COLUMNS) + "\n"


def _candidate_log_dirs() -> Iterable[Path]:
    """Yield candidate directories where the activity log can be stored."""

    env_log = os.getenv("GEFEED_LOG_DIR")
    if env_log:
        yield Path(env_log)

    env_data = os.getenv("GEFEED_DATA_DIR")
    if env_data:
        yield Path(env_data) / "logs"

    yield Path("data") / "logs"

    yield Path(tempfile.gettempdir()) / "gefeed-logs"


def get_log_path(ensure: bool = True) -> Path:
    """Return the path to ``activity_log.csv`` in a writable directory."""

    last_err: OSError | None = None

    for base in _candidate_log_dirs():
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            last_err = exc
            continue

        log_path = base / LOG_FILENAME

        if ensure:
            try:
                exists = log_path.exists()
            except OSError:
                exists = False

            if not exists:
                try:
                    with open(log_path, "w", encoding="utf-8") as handle:
                        handle.write(HEADER)
                except OSError as exc:
                    last_err = exc
                    continue

        return log_path

    # As a last resort, try to create a brand-new temporary directory so the
    # application can keep running even if the usual locations are not
    # available (for instance when an existing ``gefeed-logs`` directory has
    # restrictive permissions).
    try:
        fallback_dir = Path(tempfile.mkdtemp(prefix="gefeed-logs-"))
        fallback_path = fallback_dir / LOG_FILENAME

        if ensure and not fallback_path.exists():
            with open(fallback_path, "w", encoding="utf-8") as handle:
                handle.write(HEADER)

        return fallback_path
    except OSError as exc:  # pragma: no cover - extremely unlikely
        last_err = exc

    raise RuntimeError(
        "No se pudo crear / usar activity_log.csv en ningún directorio candidato. "
        f"Último error: {last_err}"
    )


def _sanitize(value: str) -> str:
    return value.replace(PIPE, "/").replace("\n", " ").strip()


def append_log(message: str, level: str = "INFO", scope: str = "app") -> None:
    """Append a line to the activity log without raising if it fails."""

    try:
        path = get_log_path(ensure=True)
    except Exception:
        return

    timestamp = f"{datetime.utcnow().isoformat()}Z"

    if scope == "activity":
        payload = {column: "" for column in COLUMNS}
        payload["timestamp"] = timestamp
        payload["level"] = level
        payload["scope"] = scope

        for raw_part in message.split("|"):
            part = raw_part.strip()
            if not part:
                continue

            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = _sanitize(value)
                if key in payload:
                    payload[key] = value
                elif key == "message" and not payload["detalle"]:
                    payload["detalle"] = value
            elif not payload["detalle"]:
                payload["detalle"] = _sanitize(part)

        line = PIPE.join(payload[column] for column in COLUMNS) + "\n"
    else:
        safe_message = _sanitize(message)
        line = PIPE.join([
            timestamp,
            level,
            scope,
            "",
            "",
            safe_message,
            "",
        ]) + "\n"

    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        return


def new_trace(prefix: str = "") -> str:
    base = uuid.uuid4().hex[:8]
    return f"{prefix}{base}" if prefix else base


__all__ = ["_candidate_log_dirs", "get_log_path", "append_log", "new_trace"]

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import tempfile
import uuid
from typing import Iterable

LOG_FILENAME = "activity.log"
HEADER = "timestamp,level,scope,message\n"


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
    """Return the path to ``activity.log`` in a writable directory."""

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

    raise RuntimeError(
        "No se pudo crear / usar activity.log en ningún directorio candidato. "
        f"Último error: {last_err}"
    )


def append_log(message: str, level: str = "INFO", scope: str = "app") -> None:
    """Append a line to the activity log without raising if it fails."""

    try:
        path = get_log_path(ensure=True)
    except Exception:
        return

    safe_message = message.replace(",", ";")
    line = f"{datetime.utcnow().isoformat()}Z,{level},{scope},{safe_message}\n"

    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        return


def new_trace(prefix: str = "") -> str:
    base = uuid.uuid4().hex[:8]
    return f"{prefix}{base}" if prefix else base


__all__ = ["_candidate_log_dirs", "get_log_path", "append_log", "new_trace"]

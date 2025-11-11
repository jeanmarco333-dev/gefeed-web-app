"""Daily ZIP backups with optional GitHub upload."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import base64
import hashlib
import json
import os
from typing import Any

import requests

FILES = [
    "alimentos.csv",
    "raciones_base.csv",
    "mixer_sim_log.csv",
    "activity_log.csv",
    "raciones_dadas.csv",
    "raciones_log.csv",
    "racion_vigente.json",
]


def _data_dir() -> Path:
    return Path(os.getenv("DATA_DIR", "data"))


def _backup_dir() -> Path:
    raw = os.getenv("BACKUP_DIR")
    base = Path(raw) if raw else Path("backups")
    base.mkdir(parents=True, exist_ok=True)
    return base


def create_daily_zip() -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    zip_path = _backup_dir() / f"backup_{today}.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zipf:
        for name in FILES:
            path = _data_dir() / name
            if path.exists():
                zipf.write(path, arcname=name)
    return zip_path


def upload_github(zip_path: Path) -> dict[str, Any]:
    repo = os.getenv("GITHUB_REPO", "")
    token = os.getenv("GH_TOKEN", "")
    branch = os.getenv("GITHUB_BRANCH", "main")
    if not (repo and token):
        return {"status": "skip"}

    with open(zip_path, "rb") as handle:
        content = handle.read()

    short_hash = hashlib.sha1(content).hexdigest()[:8]
    name = f"{zip_path.stem}_{short_hash}.zip"
    api = f"https://api.github.com/repos/{repo}/contents/backups/{name}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    payload = {
        "message": f"backup {name}",
        "branch": branch,
        "content": base64.b64encode(content).decode("utf-8"),
    }

    response = requests.put(api, headers=headers, data=json.dumps(payload), timeout=15)
    if response.status_code not in (200, 201):
        raise RuntimeError(response.text)
    return response.json()


def backup_flow() -> dict[str, Any]:
    zip_path = create_daily_zip()
    try:
        upload_result = upload_github(zip_path)
        return {"zip": str(zip_path), "upload": upload_result}
    except Exception as exc:
        return {"zip": str(zip_path), "upload_error": str(exc)}


__all__ = ["backup_flow", "create_daily_zip", "upload_github"]

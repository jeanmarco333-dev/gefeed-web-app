"""Utilities for the multi-step mixer loading wizard."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any
import shutil

import streamlit as st


def _data_dir() -> Path:
    """Return the base directory where wizard data should be stored."""
    env_path = os.environ.get("DATA_DIR")
    base = Path(env_path) if env_path else Path("data")
    base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_draft_id() -> str:
    """Ensure the current session has an associated draft identifier."""
    if "draft_id" not in st.session_state:
        st.session_state["draft_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    return st.session_state["draft_id"]


def _draft_dir(draft_id: str) -> Path:
    return _data_dir() / "drafts" / draft_id


def save_step_data(draft_id: str, step_name: str, payload: dict[str, Any]) -> None:
    """Persist step data for the current draft as prettified JSON."""
    target_dir = _draft_dir(draft_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{step_name}.json"

    tmp_path = file_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(file_path)


def load_step_data(draft_id: str, step_name: str) -> dict[str, Any] | None:
    """Load persisted data for the requested step, if available."""
    file_path = _draft_dir(draft_id) / f"{step_name}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def guardar_registro_definitivo(registro: dict[str, Any]) -> Path:
    """Append a finalized mixer record to the JSONL history file."""
    file_path = _data_dir() / "registros_mixer.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return file_path


def delete_draft(draft_id: str) -> None:
    """Remove all persisted data for the provided draft identifier."""
    draft_directory = _draft_dir(draft_id)
    if draft_directory.exists():
        shutil.rmtree(draft_directory)

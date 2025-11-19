# app.py — GE-Feedlot v0.26-beta (free) — UI mejorada (transiciones + menús + validaciones)
# Pestañas: 📊 Stock & Corrales | 🧾 Ajustes de raciones | 📦 Alimentos | 🧮 Mixer | ⚙️ Parámetros | ⬇️ Exportar
# Estructura:
#   app.py, calc_engine.py, requirements.txt
#   data/: alimentos.csv, raciones_base.csv, mixers.csv, pesos.csv, raciones_catalog.csv, raciones_recipes.csv
#         requerimientos_energeticos.csv, requerimiento_proteico.csv
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import zipfile
import hashlib
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qrcode
import requests
import streamlit as st

from core.activity import append_log, get_log_path, new_trace
from core.backup import backup_flow
from core.numbers import normalize_animal_counts

from calc_engine import (
    Food,
    Ingredient,
    calculate_gmd,
    mixer_kg_by_ingredient,
    optimize_ration,
    ration_split_from_pv_cv,
    sugerencia_balance,
)
# --- AUTH (login por usuario/clave) ---
import bcrypt
import yaml
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# ------------------------------------------------------------------------------
# Paths (multiusuario)
# ------------------------------------------------------------------------------
DATA_DIR_ENV = os.getenv("DATA_DIR")


def _ensure_writable_dir(preferred: Path) -> Path:
    """Return a directory that we can write to, falling back to /tmp if needed."""

    candidates: list[Path] = []

    if preferred is not None:
        candidates.append(preferred)

    candidates.append(Path(tempfile.gettempdir()) / "gefeed-data")

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            test_file = base / ".write_test"
            with open(test_file, "w") as handle:
                handle.write("ok")
            test_file.unlink(missing_ok=True)
            return base
        except OSError:
            continue

    raise RuntimeError("No writable data directory found")


GLOBAL_DATA_DIR = _ensure_writable_dir(Path(DATA_DIR_ENV) if DATA_DIR_ENV else Path("data"))

os.environ.setdefault("DATA_DIR", str(GLOBAL_DATA_DIR))
os.environ.setdefault("BACKUP_DIR", str(GLOBAL_DATA_DIR / "backups"))

from core.wizard import (  # noqa: E402  # Imported after DATA_DIR is configured
    delete_draft,
    ensure_draft_id,
    guardar_registro_definitivo,
    load_step_data,
    save_step_data,
)

# --- Config de ADMIN y almacenamiento de usuarios editables ---
DEFAULT_ADMIN_USERS = {"admin"}  # usuarios que verán la pestaña de administración por defecto

AUTH_DIR = GLOBAL_DATA_DIR / "auth"
AUTH_DIR.mkdir(parents=True, exist_ok=True)
AUTH_STORE = AUTH_DIR / "users.yaml"  # acá persistimos los usuarios editados por UI

SESSION_AUTH_STATUS = "auth_status"
SESSION_AUTH_USER = "auth_username"
SESSION_AUTH_NAME = "auth_name"
SESSION_AUTH_TOKEN = "auth_token_seed"


def _clear_auth_state() -> None:
    for key in (SESSION_AUTH_STATUS, SESSION_AUTH_USER, SESSION_AUTH_NAME, SESSION_AUTH_TOKEN):
        st.session_state.pop(key, None)


def _trigger_rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun is None:
        raise RuntimeError("Streamlit rerun function is not available")
    rerun()


def _logout_user() -> None:
    _clear_auth_state()
    _trigger_rerun()


def _verify_password(plain_password: str, stored_hash: str | None) -> bool:
    if not plain_password or not stored_hash:
        return False

    try:
        if isinstance(stored_hash, bytes):
            hashed_bytes = stored_hash
        else:
            hashed_bytes = str(stored_hash).encode("utf-8")
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_bytes)
    except (TypeError, ValueError):
        return False

BASE_STYLE = dedent(
    """
    <style>
    :root {
        --shadow-soft: 0 16px 48px -24px rgba(15, 23, 42, 0.35);
        --shadow-card: 0 18px 42px -24px rgba(15, 23, 42, 0.28);
        --radius-card: 18px;
        --radius-pill: 999px;
        --transition-base: all 0.22s ease;
    }

    html, body, .block-container {
        font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
        letter-spacing: 0.01em;
    }

    section.main > div {
        padding-top: 0.6rem;
    }

    .section-enter {
        opacity: 0;
        transform: translateY(4px);
        animation: fadeSlideIn .3s ease-out forwards;
    }

    @keyframes fadeSlideIn {
        to { opacity: 1; transform: none; }
    }

    .card {
        padding: 1.35rem;
        border-radius: var(--radius-card);
        border: 1px solid transparent;
        background: var(--card-bg);
        box-shadow: var(--shadow-card);
        transition: var(--transition-base);
    }

    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 28px 60px -32px rgba(15, 23, 42, 0.35);
    }

    .card > .stMarkdown:first-child p {
        margin-bottom: 0.75rem;
    }

    .stButton>button,
    .stDownloadButton>button {
        border-radius: var(--radius-pill);
        font-weight: 600;
        padding: 0.55rem 1.35rem;
        border: none;
        box-shadow: 0 14px 32px -18px rgba(37, 99, 235, 0.65);
        transition: transform 0.08s ease, background 0.22s ease, box-shadow 0.22s ease;
    }

    .stButton>button:active,
    .stDownloadButton>button:active {
        transform: scale(0.98);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.65rem 1.35rem;
        border-radius: var(--radius-pill);
        font-weight: 600;
        transition: var(--transition-base);
        margin-right: 0.35rem;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        border-radius: var(--radius-pill);
    }

    .stMarkdown h3,
    .stMarkdown h4 {
        margin-top: 0.35rem;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }

    .stMarkdown h3 {
        font-size: 1.25rem;
    }

    .stMarkdown h4 {
        font-size: 1.05rem;
    }

    details > summary {
        cursor: pointer;
        list-style: none;
        transition: color 0.2s ease;
    }

    details > summary::-webkit-details-marker {
        display: none;
    }

    details > summary::after {
        content: "▸";
        display: inline-block;
        margin-left: 0.5rem;
        transition: transform 0.2s ease;
    }

    details[open] > summary::after {
        transform: rotate(90deg);
    }

    details[open] .expander-body {
        animation: fadeIn 0.2s ease both;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .chip-ok {
        background: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: var(--radius-pill);
        font-size: 0.85rem;
        font-weight: 600;
    }

    .chip-bad {
        background: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: var(--radius-pill);
        font-size: 0.85rem;
        font-weight: 600;
    }

    .erp-link-button {
        display: block;
        padding: 0.75rem 1rem;
        border-radius: var(--radius-pill);
        font-weight: 600;
        text-decoration: none;
        text-align: center;
        transition: filter 0.18s ease, box-shadow 0.25s ease;
        box-shadow: 0 20px 52px -24px rgba(56, 189, 248, 0.55);
    }

    .erp-link-button:hover {
        text-decoration: none;
        filter: brightness(0.96);
    }

    .stDataFrame div[data-testid="stTable"] {
        border-radius: var(--radius-card);
        overflow: hidden;
    }

    .stDataFrame [role="table"] th,
    .stDataFrame [role="table"] td {
        font-size: 0.95rem;
        padding: 0.65rem;
    }

    .stSelectbox div[data-baseweb="select"],
    .stMultiselect div[data-baseweb="select"],
    .stNumberInput input,
    .stTextInput input,
    textarea {
        border-radius: 12px !important;
        min-height: 48px;
        font-size: 0.95rem;
    }

    .metric-small .stMetric {
        background: rgba(37, 99, 235, 0.08);
        padding: 0.8rem 1.1rem;
        border-radius: var(--radius-card);
    }
    </style>
    """
)


def inject_theme_styles(dark_mode: bool) -> None:
    palette_light = {
        "app_bg": "#F8FAFC",
        "sidebar_bg": "#F1F5F9",
        "app_fg": "#0F172A",
        "card_bg": "#FFFFFF",
        "card_border": "rgba(15,23,42,0.08)",
        "button_bg": "#2563EB",
        "button_fg": "#FFFFFF",
        "button_hover_bg": "#DC2626",
        "accent": "#2563EB",
    }
    palette_dark = {
        "app_bg": "#0F172A",
        "sidebar_bg": "#111C32",
        "app_fg": "#E2E8F0",
        "card_bg": "#172554",
        "card_border": "rgba(148,163,184,0.25)",
        "button_bg": "#1D4ED8",
        "button_fg": "#F8FAFC",
        "button_hover_bg": "#B91C1C",
        "accent": "#38BDF8",
    }
    pal = palette_dark if dark_mode else palette_light
    theme_css = dedent(
        f"""
        <style>
        :root {{
            --app-bg: {pal['app_bg']};
            --sidebar-bg: {pal['sidebar_bg']};
            --app-fg: {pal['app_fg']};
            --card-bg: {pal['card_bg']};
            --card-border: {pal['card_border']};
            --button-bg: {pal['button_bg']};
            --button-fg: {pal['button_fg']};
            --button-hover-bg: {pal['button_hover_bg']};
            --accent-color: {pal['accent']};
            --border-subtle: rgba(148, 163, 184, 0.32);
        }}

        div[data-testid="stAppViewContainer"] {{
            background: linear-gradient(145deg, var(--app-bg) 0%, rgba(56, 189, 248, 0.12) 100%);
            color: var(--app-fg);
        }}

        section[data-testid="stSidebar"] {{
            padding-top: 1.5rem;
        }}

        div[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {pal['sidebar_bg']} 0%, rgba(37, 99, 235, 0.08) 100%);
            color: var(--app-fg);
            border-right: 1px solid var(--card-border);
        }}

        div[data-testid="stSidebar"] .stButton>button,
        div[data-testid="stSidebar"] .stDownloadButton>button {{
            width: 100%;
            margin-top: 0.35rem;
        }}

        .stMarkdown, .stText, .stDataFrame {{
            color: var(--app-fg);
        }}

        .card {{
            border: 1px solid var(--card-border);
            color: var(--app-fg);
        }}

        .stDataFrame div[data-testid="stTable"] {{
            color: var(--app-fg);
        }}

        .stButton>button,
        .stDownloadButton>button,
        .erp-link-button {{
            background: var(--button-bg);
            color: var(--button-fg);
        }}

        .stButton>button:hover,
        .stDownloadButton>button:hover,
        .erp-link-button:hover {{
            background: var(--button-hover-bg);
            box-shadow: 0 18px 44px -22px rgba(220, 38, 38, 0.55);
            color: var(--button-fg);
        }}

        .stButton>button:focus,
        .stDownloadButton>button:focus,
        .erp-link-button:focus,
        .stButton>button:active,
        .stDownloadButton>button:active,
        .erp-link-button:active {{
            color: var(--button-fg);
        }}

        .stSelectbox div[data-baseweb="select"],
        .stMultiselect div[data-baseweb="select"],
        .stNumberInput input,
        .stTextInput input,
        textarea {{
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid var(--border-subtle) !important;
            color: var(--app-fg);
        }}

        .stNumberInput input:focus,
        .stTextInput input:focus,
        textarea:focus,
        .stSelectbox div[data-baseweb="select"]:focus-within,
        .stMultiselect div[data-baseweb="select"]:focus-within {{
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18);
        }}

        .stTabs [data-baseweb="tab"] {{
            background: rgba(148, 163, 184, 0.18);
            color: var(--app-fg);
        }}

        .stTabs [aria-selected="true"] {{
            background: var(--accent-color) !important;
            color: var(--button-fg) !important;
        }}

        .stMetric label {{
            color: rgba(148, 163, 184, 0.9);
        }}

        .stMetric .metric-value {{
            color: var(--app-fg);
        }}

        .metric-small .stMetric {{
            background: rgba(37, 99, 235, 0.12);
            border: 1px solid rgba(37, 99, 235, 0.22);
        }}

        .stMarkdown a {{
            color: var(--accent-color);
            font-weight: 600;
        }}

        .stMarkdown a:hover {{
            text-decoration: none;
        }}
        </style>
        """
    )
    st.markdown(BASE_STYLE + theme_css, unsafe_allow_html=True)

# --- Columnas base (constantes) ---
EXPECTED_ALIM_COLS = [
    "ORIGEN",
    "PRESENTACION",
    "TIPO",
    "MS",
    "TND (%)",
    "PB",
    "EE",
    "COEF ATC",
    "$/KG",
    "EM",
    "ENP2",
]

ALIM_COLS = list(EXPECTED_ALIM_COLS)

ALIM_TEMPLATE_ROWS = [
    {
        "ORIGEN": "MAIZ GRANO SECO",
        "PRESENTACION": "Maíz - grano seco",
        "TIPO": "Ingrediente - Energético",
        "MS": 88,
        "TND (%)": 86,
        "PB": 9,
        "EE": 4,
        "COEF ATC": 1.15,
        "$/KG": 150,
        "EM": 3.2,
        "ENP2": 1.6,
    },
    {
        "ORIGEN": "SILO MAIZ",
        "PRESENTACION": "Silaje de maíz",
        "TIPO": "Ingrediente - Fibra",
        "MS": 35,
        "TND (%)": 62,
        "PB": 7,
        "EE": 2,
        "COEF ATC": 2.5,
        "$/KG": 55,
        "EM": 1.8,
        "ENP2": 0.65,
    },
]

REQENER_COLS = ["peso", "cat", "requerimiento_energetico", "ap"]
REQPROT_COLS = ["peso", "cat", "ap", "req_proteico"]

SYNTHETIC_EM_COEFS = {
    "novillos": {
        "intercept": 0.0376962101,
        "pv075": 0.0765280134,
        "ap_pv": 1.49631094e-05,
    },
    "vaquillonas": {
        "intercept": 0.0885280667,
        "pv075": 0.0759192350,
        "ap_pv": 8.32212548e-06,
    },
}


def _num(x, default=0.0):
    try:
        val = float(pd.to_numeric(x, errors="coerce"))
        return default if pd.isna(val) else val
    except Exception:
        return default


def _num_or_none(x):
    try:
        val = float(pd.to_numeric(x, errors="coerce"))
    except Exception:
        return None
    if pd.isna(val):
        return None
    return val


def _interpolate_requirement(
    df: pd.DataFrame,
    peso_kg: float,
    ap_kg_dia: float | None,
    categoria: str | None,
    value_col: str,
) -> float | None:
    if df is None or df.empty or value_col not in df.columns:
        return None

    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work["peso"] = pd.to_numeric(work.get("peso"), errors="coerce")
    if "ap" in work.columns:
        work["ap"] = pd.to_numeric(work.get("ap"), errors="coerce")
    else:
        work["ap"] = np.nan
    work["cat"] = work.get("cat", pd.Series(dtype=str)).fillna("").astype(str)
    work = work.dropna(subset=["peso", value_col])
    if work.empty:
        return None

    cat_key = (categoria or "").strip().lower()
    if cat_key:
        subset = work[work["cat"].str.lower() == cat_key]
        if subset.empty:
            subset = work
    else:
        subset = work

    peso_val = _num_or_none(peso_kg)
    if peso_val is None or peso_val <= 0:
        return None
    ap_val = _num_or_none(ap_kg_dia)

    def _idw(rows: pd.DataFrame, use_ap: bool) -> float | None:
        weights: list[float] = []
        values: list[float] = []
        for _, r in rows.iterrows():
            val = _num_or_none(r.get(value_col))
            peso_r = _num_or_none(r.get("peso"))
            if val is None or peso_r is None:
                continue
            if use_ap:
                ap_r = _num_or_none(r.get("ap"))
                if ap_r is None or ap_val is None:
                    continue
                dp = peso_val - peso_r
                da = ap_val - ap_r
                dist = float((dp ** 2 + da ** 2) ** 0.5)
            else:
                dist = abs(peso_val - peso_r)
            if dist < 1e-6:
                return float(val)
            weights.append(1.0 / (dist + 1e-6))
            values.append(float(val))
        if weights:
            return float(np.dot(weights, values) / np.sum(weights))
        return None

    if ap_val is not None and subset["ap"].notna().any():
        with_ap = subset.dropna(subset=["ap"])
        val = _idw(with_ap, True)
        if val is not None:
            return val

    val = _idw(subset, False)
    return val


def compute_requirement_em(pv_kg: float, ap_kg_dia: float | None, categoria: str | None) -> float | None:
    req_df = load_reqener()
    val = _interpolate_requirement(req_df, pv_kg, ap_kg_dia, categoria, "requerimiento_energetico")
    if val is None:
        return synthetic_em_requirement(pv_kg, ap_kg_dia or 0.0, categoria)
    return float(val)


def compute_requirement_pb(pv_kg: float, ap_kg_dia: float | None, categoria: str | None) -> float | None:
    req_df = load_reqprot()
    val = _interpolate_requirement(req_df, pv_kg, ap_kg_dia, categoria, "req_proteico")
    if val is None:
        return None
    return float(val)


def synthetic_em_requirement(pv_kg: float, ap_kg_dia: float, categoria: str | None) -> float | None:
    """Calcula EM (Mcal/día) con una fórmula sintética ajustada por categoría."""

    pv = _num(pv_kg, 0.0)
    ap = _num(ap_kg_dia, 0.0)
    if pv <= 0:
        return None

    cat_key = (categoria or "").strip().lower()
    if "vaq" in cat_key:
        coeffs = SYNTHETIC_EM_COEFS["vaquillonas"]
    elif "nov" in cat_key:
        coeffs = SYNTHETIC_EM_COEFS["novillos"]
    else:
        coeffs = SYNTHETIC_EM_COEFS["novillos"]

    pv075 = pv ** 0.75
    value = (
        coeffs["intercept"]
        + coeffs["pv075"] * pv075
        + coeffs["ap_pv"] * (pv * ap)
    )
    return round(max(value, 0.0), 3)

# ------------------------------------------------------------------------------
# Configuración global (debe ir antes de cualquier llamada a Streamlit)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="JM P-Feedlot v0.26 — Web", layout="wide")

CFG_PATH = Path("config_users.yaml")  # opcional (dev local o repo privado)

def load_base_cfg():
    """Carga credenciales base: YAML en repo o st.secrets (solo lectura)."""
    if CFG_PATH.exists():
        try:
            return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
        except Exception as e:
            st.error(f"config_users.yaml inválido: {e}"); st.stop()
    if "auth" in st.secrets:
        return dict(st.secrets["auth"])
    # Si no hay nada, creamos estructura mínima vacía (permitirá que el admin cree users desde cero)
    return {
        "credentials": {"usernames": {}},
        "cookie": {"name": "gefeed_cookie", "key": "feedlot_key", "expiry_days": 7},
        "preauthorized": {"emails": []},
    }


def load_user_store():
    """Carga/crea el YAML editable de usuarios."""
    empty_store = {"credentials": {"usernames": {}}, "preauthorized": {"emails": []}}

    try:
        raw_store = AUTH_STORE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return empty_store
    except OSError as exc:
        st.warning(
            "No se pudo acceder al archivo editable de usuarios"
            f" ({AUTH_STORE}). Se usará una configuración vacía. Detalle: {exc}"
        )
        return empty_store
    except Exception as exc:
        st.warning(
            "Error inesperado al leer el archivo editable de usuarios."
            f" Se usará una configuración vacía. Detalle: {exc}"
        )
        return empty_store

    try:
        store = yaml.safe_load(raw_store) or {}
    except Exception as exc:
        st.warning(
            "El archivo editable de usuarios es inválido y se ignorará."
            f" Se usará una configuración vacía. Detalle: {exc}"
        )
        return empty_store

    if not isinstance(store, dict):
        st.warning(
            "El archivo editable de usuarios no contiene una estructura válida."
            " Se usará una configuración vacía."
        )
        return empty_store

    return store


def merge_credentials(base_cfg, store_cfg):
    """El store (editable) pisa/añade usuarios sobre la config base (solo lectura)."""
    out = dict(base_cfg)
    base_users = (out.get("credentials") or {}).get("usernames", {}) or {}
    store_users = (store_cfg.get("credentials") or {}).get("usernames", {}) or {}
    merged_users = dict(base_users)
    merged_users.update(store_users)  # lo editable tiene prioridad
    out.setdefault("credentials", {})["usernames"] = merged_users
    if store_cfg.get("cookie"):
        cookie_base = dict(out.get("cookie") or {})
        cookie_base.update(store_cfg["cookie"])
        out["cookie"] = cookie_base
    if store_cfg.get("preauthorized"):
        preauth_base = dict(out.get("preauthorized") or {})
        preauth_base.update(store_cfg["preauthorized"])
        out["preauthorized"] = preauth_base
    return out


BASE_CFG = load_base_cfg()
STORE_CFG = load_user_store()
CFG = merge_credentials(BASE_CFG, STORE_CFG)

# --- Detección de rol administrador --------------------------------------------------

def is_admin(username: str) -> bool:
    """Determina si un usuario tiene permisos de administrador."""

    user = str(username)
    admin_candidates: set[str] = set()
    try:
        admin_candidates = {str(u) for u in st.secrets.get("admins", [])}
    except Exception:
        admin_candidates = set()

    if admin_candidates:
        return user in admin_candidates

    roles_cfg = CFG.get("roles")
    if isinstance(roles_cfg, dict):
        role = roles_cfg.get(user)
        if isinstance(role, str) and role.strip().lower() == "admin":
            return True

    return user in DEFAULT_ADMIN_USERS

# --- Autenticación manual (sin dependencias externas) -------------------------
credentials_cfg_root = CFG.get("credentials") or {}
if not isinstance(credentials_cfg_root, dict):
    st.error("Config inválida: falta 'credentials' (dict).")
    st.stop()

credentials_cfg = credentials_cfg_root.get("usernames")
if not isinstance(credentials_cfg, dict):
    st.error("Config inválida: falta 'credentials.usernames' (dict).")
    st.stop()

cookie_cfg = CFG.get("cookie") or {}
cookie_name = str(cookie_cfg.get("name", "gefeed_cookie"))
cookie_key = str(cookie_cfg.get("key", "feedlot_key"))
try:
    cookie_expiry_days = int(cookie_cfg.get("expiry_days", 7))
except Exception:
    cookie_expiry_days = 7

cookie_token_seed = f"{cookie_name}|{cookie_key}|{cookie_expiry_days}"
stored_seed = st.session_state.get(SESSION_AUTH_TOKEN)
if stored_seed and stored_seed != cookie_token_seed:
    _clear_auth_state()

auth_status = st.session_state.get(SESSION_AUTH_STATUS)
username = st.session_state.get(SESSION_AUTH_USER)
name = st.session_state.get(SESSION_AUTH_NAME)

if auth_status is True and username:
    st.session_state[SESSION_AUTH_TOKEN] = cookie_token_seed
else:
    if auth_status is False:
        st.error("Usuario o contraseña inválidos")

    with st.form("login_form"):
        input_username = st.text_input("Usuario", placeholder="ej: admin")
        input_password = st.text_input("Contraseña", type="password")
        submit_login = st.form_submit_button("Ingresar", type="primary")

    if submit_login:
        user_entry = credentials_cfg.get(input_username)
        stored_hash = (user_entry or {}).get("password")
        if _verify_password(input_password, stored_hash):
            resolved_name = (user_entry or {}).get("name") or input_username
            st.session_state[SESSION_AUTH_STATUS] = True
            st.session_state[SESSION_AUTH_USER] = input_username
            st.session_state[SESSION_AUTH_NAME] = resolved_name
            st.session_state[SESSION_AUTH_TOKEN] = cookie_token_seed
            _trigger_rerun()
        else:
            st.session_state[SESSION_AUTH_STATUS] = False
            st.error("Usuario o contraseña inválidos")
            st.stop()
    else:
        if auth_status is not False:
            st.info("Ingresá tus credenciales")
        st.stop()

# Si llegamos aquí, hay una sesión válida
auth_status = True
username = st.session_state.get(SESSION_AUTH_USER)
name = st.session_state.get(SESSION_AUTH_NAME) or username

if not username:
    st.error("Sesión inválida. Iniciá sesión nuevamente.")
    _clear_auth_state()
    st.stop()

user_profile = credentials_cfg.get(username, {}) or {}

USER_IS_ADMIN = is_admin(username)

user_email = str(user_profile.get("email", "") or "").strip()
if user_email:
    st.session_state["email"] = user_email

# Carpeta sandbox del usuario autenticado
USER_DIR = GLOBAL_DATA_DIR / "users" / username
USER_DIR.mkdir(parents=True, exist_ok=True)

if os.getenv("DATA_DIR") in (None, "", str(GLOBAL_DATA_DIR)):
    os.environ["DATA_DIR"] = str(USER_DIR)
if os.getenv("BACKUP_DIR") in (None, "", str(GLOBAL_DATA_DIR / "backups")):
    os.environ["BACKUP_DIR"] = str(USER_DIR / "backups")


def user_path(fname: str) -> Path:
    p = USER_DIR / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


META_DIR = USER_DIR / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)
LAST_CHANGED = META_DIR / "last_changed.json"
PREFS_PATH = META_DIR / "preferences.json"


def _load_user_prefs() -> dict:
    if PREFS_PATH.exists():
        try:
            data = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_user_prefs(prefs: dict) -> None:
    try:
        PREFS_PATH.write_text(
            json.dumps(prefs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[PREFS] No se pudo guardar preferencias: {exc}", flush=True)


def _prefs_get(section: str, key: str, default=None):
    prefs = st.session_state.setdefault("user_prefs", _load_user_prefs())
    section_val = prefs.get(section)
    if isinstance(section_val, dict):
        return section_val.get(key, default)
    return default


def _prefs_set(section: str, key: str, value) -> None:
    prefs = st.session_state.setdefault("user_prefs", _load_user_prefs())
    section_dict = prefs.get(section)
    if not isinstance(section_dict, dict):
        section_dict = {}
        prefs[section] = section_dict
    section_dict[key] = value
    st.session_state["user_prefs"] = prefs
    _save_user_prefs(prefs)


def _format_whatsapp_link(number: str, message: str) -> str:
    digits = "".join(ch for ch in str(number) if str(ch).isdigit())
    if digits.startswith("00"):
        digits = digits[2:]
    if not digits:
        digits = "5493874073236"
    return f"https://wa.me/{digits}?text={quote_plus(message)}"


with st.sidebar:
    st.title("⚙️ Opciones")
    st.write(f"👤 {name} (@{username})")
    st.markdown("### Apariencia")
    if "ui_theme_toggle" not in st.session_state:
        st.session_state["ui_theme_toggle"] = bool(
            _prefs_get("ui", "theme_dark", False)
        )
    dark_mode_active = st.toggle(
        "🌙 Tema oscuro / claro",
        key="ui_theme_toggle",
    )
    stored_theme = bool(_prefs_get("ui", "theme_dark", False))
    if bool(dark_mode_active) != stored_theme:
        _prefs_set("ui", "theme_dark", bool(dark_mode_active))

    st.markdown("### Seguimiento")
    default_operator = (
        st.session_state.get("operador")
        or str(name or username or "operador")
    )
    operador_input = st.text_input(
        "Operador", value=default_operator, help="Figura en activity_log.csv"
    )
    st.session_state["operador"] = operador_input.strip() or default_operator
    st.caption("Los eventos clave quedan en data/activity_log.csv (pipe).")

    st.markdown("### Soporte")
    whatsapp_pref = _prefs_get("integrations", "whatsapp_number", "+54 9 387 407 3236")
    wa_message = "Hola equipo Physis, necesito asistencia técnica para GE-Feedlot."
    wa_link = _format_whatsapp_link(whatsapp_pref, wa_message)
    st.markdown(
        f"<a class='erp-link-button' href='{wa_link}' target='_blank'>💬 Solicitar asistencia técnica</a>",
        unsafe_allow_html=True,
    )

    if st.button("Salir", type="secondary", use_container_width=True):
        _logout_user()
APP_VERSION = "JM P-Feedlot v0.26-beta (free)"

dark_mode_active = bool(st.session_state.get("ui_theme_toggle", False))
st.session_state["theme_dark"] = dark_mode_active
inject_theme_styles(dark_mode_active)
def _logo_block() -> None:
    logo_path = Path("assets/logo.png")
    if logo_path.exists():
        try:
            b64 = base64.b64encode(logo_path.read_bytes()).decode()
        except Exception:
            b64 = None
        if b64:
            st.markdown(
                f"""
        <div style="display:flex;align-items:center;gap:.75rem;margin:.25rem 0 1rem 0;">
          <img src="data:image/png;base64,{b64}" height="36" />
          <div style="font-weight:700; letter-spacing:.4px;">JM P-Feedlot v0.26 — Web</div>
        </div>
        """,
                unsafe_allow_html=True,
            )
            return
    st.markdown("### JM P-Feedlot v0.26 — Web")


_logo_block()

st.info("🚧 Versión beta sin costo: validando con clientes iniciales. Guardá y exportá seguido por seguridad.")

# Helpers de UI
@contextmanager
def card(title: str, subtitle: str | None = None, icon: str = ""):
    st.markdown(f'<div class="card section-enter">', unsafe_allow_html=True)
    st.markdown(f"**{icon} {title}**" + (f"<br><span style='color:#6B7280'>{subtitle}</span>" if subtitle else ""), unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)

@contextmanager
def dropdown(title: str, open: bool=False):
    open_attr = " open" if open else ""
    st.markdown(f'<details class="section-enter"{open_attr}><summary><b>{title}</b></summary><div class="expander-body">', unsafe_allow_html=True)
    yield
    st.markdown("</div></details>", unsafe_allow_html=True)

def chip(text: str, ok: bool=True):
    klass = "chip-ok" if ok else "chip-bad"
    st.markdown(f'<span class="{klass}">{text}</span>', unsafe_allow_html=True)


def generate_summary_pdf(
    file_path: Path,
    logo_path: str | None,
    titulo: str,
    datos: dict,
    tabla_detalle: pd.DataFrame | None = None,
    recomendaciones: list[str] | None = None,
):
    c = canvas.Canvas(str(file_path), pagesize=A4)
    W, H = A4
    y = H - 2 * cm

    try:
        if logo_path and os.path.exists(logo_path):
            c.drawImage(
                logo_path,
                x=2 * cm,
                y=y - 2 * cm,
                width=3 * cm,
                height=2 * cm,
                preserveAspectRatio=True,
                mask="auto",
            )
    except Exception:
        pass

    c.setFont("Helvetica-Bold", 16)
    c.drawString(6 * cm, y - 0.5 * cm, titulo)

    c.setFont("Helvetica", 9)
    c.drawString(6 * cm, y - 1.2 * cm, datetime.now().strftime("%Y-%m-%d %H:%M"))

    y -= 3 * cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, y, "Resumen")
    c.setFont("Helvetica", 10)
    y -= 0.5 * cm

    for k in [
        "Ración",
        "PV (kg)",
        "CV (%)",
        "Consumo MS (kg/día)",
        "EM calculada (Mcal/d)",
        "EM requerida (Mcal/d)",
        "PB calculada (g/d)",
        "PB requerida (g/d)",
        "As-fed total (kg/día)",
        "Costo total ($/día)",
        "Costo por cabeza ($/día)",
    ]:
        if k in datos:
            c.drawString(2 * cm, y, f"- {k}: {datos[k]}")
            y -= 0.45 * cm

    if recomendaciones:
        y -= 0.35 * cm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, y, "Recomendaciones")
        y -= 0.5 * cm
        c.setFont("Helvetica", 9)
        for rec in recomendaciones:
            if y < 2 * cm:
                c.showPage()
                y = H - 2 * cm
                c.setFont("Helvetica-Bold", 11)
                c.drawString(2 * cm, y, "Recomendaciones (cont.)")
                y -= 0.5 * cm
                c.setFont("Helvetica", 9)
            c.drawString(2.2 * cm, y, f"• {rec}")
            y -= 0.35 * cm

    if tabla_detalle is not None and not tabla_detalle.empty:
        y -= 0.35 * cm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, y, "Detalle por ingrediente")
        y -= 0.5 * cm
        c.setFont("Helvetica", 9)
        cols = ["ingrediente", "pct_ms", "ms_kg_dia", "asfed_kg_dia", "costo"]
        header = [
            "Ingrediente",
            "%MS",
            "MS (kg/d)",
            "Tal cual (kg/d)",
            "Costo ($)",
        ]
        x0 = 2 * cm
        widths = [6 * cm, 2 * cm, 3 * cm, 4 * cm, 3 * cm]
        for i, h in enumerate(header):
            c.drawString(x0 + sum(widths[:i]), y, h)
        y -= 0.4 * cm

        for _, row in tabla_detalle[cols].iterrows():
            if y < 2 * cm:
                c.showPage()
                y = H - 2 * cm
                c.setFont("Helvetica-Bold", 11)
                c.drawString(2 * cm, y, "Detalle por ingrediente (cont.)")
                y -= 0.5 * cm
                c.setFont("Helvetica", 9)
                for i, h in enumerate(header):
                    c.drawString(x0 + sum(widths[:i]), y, h)
                y -= 0.4 * cm
            vals = [
                str(row["ingrediente"]),
                f"{float(row['pct_ms']):.2f}",
                f"{float(row['ms_kg_dia']):.3f}",
                f"{float(row['asfed_kg_dia']):.3f}",
                f"{float(row['costo']):.2f}",
            ]
            for i, v in enumerate(vals):
                c.drawString(x0 + sum(widths[:i]), y, v)
            y -= 0.35 * cm

    c.showPage()
    c.save()

ALIM_PATH    = user_path("alimentos.csv")
BASE_PATH    = user_path("raciones_base.csv")
MIXERS_PATH  = user_path("mixers.csv")
PESOS_PATH   = user_path("pesos.csv")
CATALOG_PATH = user_path("raciones_catalog.csv")
RECIPES_PATH = user_path("raciones_recipes.csv")
REQENER_PATH = user_path("requerimientos_energeticos.csv")
REQPROT_PATH = user_path("requerimiento_proteico.csv")
AUDIT_LOG_PATH = user_path("audit_log.csv")
RACIONES_LOG_PATH = user_path("raciones_log.csv")
RACION_VIGENTE_PATH = user_path("racion_vigente.json")
RACIONES_DADAS_PATH = user_path("raciones_dadas.csv")
MIXER_SIM_LOG = user_path("mixer_sim_log.csv")
METRICS_PATH = user_path("metrics.json")
ACTIVITY_LOG_PATH = get_log_path(ensure=True)


def _load_metrics() -> dict:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "visits_total": 0,
        "visits_by_day": {},
        "visits_by_user": {},
        "simulations_total": 0,
        "simulations_by_day": {},
        "simulations_by_user": {},
        "last_update": None,
    }


def _save_metrics(metrics: dict, *, backup_user: str | None = None) -> None:
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if backup_user:
        try:
            github_upsert(
                METRICS_PATH,
                message=f"backup({backup_user}): {METRICS_PATH.name}",
            )
        except Exception:
            pass


def metrics_increment_visit(user: str) -> None:
    if st.session_state.get("visit_counted", False):
        return
    st.session_state["visit_counted"] = True

    metrics = _load_metrics()
    today = str(date.today())
    metrics["visits_total"] = int(metrics.get("visits_total", 0)) + 1
    metrics.setdefault("visits_by_day", {})[today] = int(
        metrics.get("visits_by_day", {}).get(today, 0)
    ) + 1
    metrics.setdefault("visits_by_user", {})[user] = int(
        metrics.get("visits_by_user", {}).get(user, 0)
    ) + 1
    metrics["last_update"] = datetime.now().isoformat(timespec="seconds")
    _save_metrics(metrics, backup_user=user)


def metrics_increment_simulation(user: str) -> None:
    metrics = _load_metrics()
    today = str(date.today())
    metrics["simulations_total"] = int(metrics.get("simulations_total", 0)) + 1
    metrics.setdefault("simulations_by_day", {})[today] = int(
        metrics.get("simulations_by_day", {}).get(today, 0)
    ) + 1
    metrics.setdefault("simulations_by_user", {})[user] = int(
        metrics.get("simulations_by_user", {}).get(user, 0)
    ) + 1
    metrics["last_update"] = datetime.now().isoformat(timespec="seconds")
    _save_metrics(metrics, backup_user=user)


def metrics_get_snapshot() -> dict:
    metrics = _load_metrics()
    today = str(date.today())
    return {
        "visits_total": int(metrics.get("visits_total", 0) or 0),
        "simulations_total": int(metrics.get("simulations_total", 0) or 0),
        "today_visits": int(metrics.get("visits_by_day", {}).get(today, 0) or 0),
        "today_simulations": int(metrics.get("simulations_by_day", {}).get(today, 0) or 0),
        "last_update": metrics.get("last_update"),
    }


def _current_operator() -> str:
    value = str(st.session_state.get("operador", "")).strip()
    if value:
        return value
    if "username" in globals() and username:
        return str(username)
    return "operador"


def activity_log_event(
    accion: str,
    detalle: str = "",
    *,
    trace_id: str | None = None,
    trace_prefix: str | None = None,
) -> str | None:
    """Helper to log events without breaking the UI if the CSV is unavailable."""

    trace = trace_id or (new_trace(trace_prefix) if trace_prefix else None)
    operator = _current_operator()
    message_parts = [
        f"op={operator}",
        f"accion={accion.strip()}",
        f"detalle={detalle.strip()}" if detalle else None,
        f"trace={trace}" if trace else None,
    ]
    message = " | ".join(part for part in message_parts if part)
    append_log(message, scope="activity")
    return trace


metrics_increment_visit(username)

MAX_CORRALES = 200
MAX_UPLOAD_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

BASE_EXPECTED_COLUMNS = [
    "nro_corral",
    "nombre_racion",
    "categ",
    "PV_kg",
    "CV_pct",
    "AP_preten",
    "nro_cab",
    "mixer_id",
    "capacidad_kg",
    "turnos",
    "meta_salida",
]

BASE_PREVIEW_COLUMNS = [
    "nro_corral",
    "nombre_racion",
    "categ",
    "nro_cab",
    "mixer_id",
    "capacidad_kg",
    "turnos",
    "meta_salida",
]

BASE_OPTIONAL_COLUMNS = ["tipo_racion", "cod_racion", "kg_turno"]


def mark_changed(event: str, username: str):
    """Marca el último cambio persistido por el usuario autenticado."""

    payload = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "event": str(event),
        "username": str(username),
        "app_version": APP_VERSION,
    }
    try:
        LAST_CHANGED.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[META] No se pudo actualizar last_changed: {exc}", flush=True)


def _hash_app_version() -> str:
    """Genera un hash corto basado en archivos críticos de la app."""

    hasher = hashlib.sha256()
    for path_name in ["app.py", "calc_engine.py", "requirements.txt"]:
        try:
            hasher.update(Path(path_name).read_bytes())
        except Exception:
            continue
    return hasher.hexdigest()[:10]


def _read_last_changed_payload():
    if LAST_CHANGED.exists():
        try:
            return json.loads(LAST_CHANGED.read_text(encoding="utf-8"))
        except Exception:
            try:
                return LAST_CHANGED.read_text(encoding="utf-8").strip()
            except Exception:
                return None
    return None


def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _file_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_mixer_simulation_snapshot(
    *,
    username: str,
    plans_dict: dict,
    version_app: str = "JM P-Feedlot v0.26",
    comment: str = "",
):
    """Guarda un snapshot de la simulación del mixer (historial + CSV detallado)."""

    if not plans_dict:
        return False, "No hay descargas/plans para respaldar."

    import pandas as pd

    frames: list[pd.DataFrame] = []
    for name, df in plans_dict.items():
        if df is None or df.empty:
            continue
        aux = df.copy()
        aux["descarga"] = name
        frames.append(aux)

    if not frames:
        return False, "No se encontraron filas en las descargas."

    consolidado = pd.concat(frames, ignore_index=True)

    fname = f"mixer_sim_{_file_tag()}.csv"
    fpath = user_path(fname)
    consolidado.to_csv(fpath, index=False, encoding="utf-8-sig")

    resumen = []
    for name, df in plans_dict.items():
        if df is None or df.empty:
            continue
        df_local = df.copy()
        if "turno" in df_local.columns and not df_local["turno"].empty:
            turno_min = pd.to_numeric(df_local["turno"], errors="coerce").fillna(0).min()
            df_turno = df_local[df_local["turno"] == turno_min]
            turnos = int(pd.to_numeric(df_local["turno"], errors="coerce").fillna(0).max())
        else:
            df_turno = df_local
            turnos = 1
        kg_turno = float(
            pd.to_numeric(df_turno.get("kg_por_turno", 0.0), errors="coerce").fillna(0.0).sum()
        )
        turnos = max(turnos, 1)
        kg_dia = kg_turno * turnos
        mix_id = (
            str(df_local["mixer_id"].iloc[0])
            if "mixer_id" in df_local.columns and not df_local.empty
            else ""
        )
        tipo = (
            str(df_local["tipo_racion"].iloc[0])
            if "tipo_racion" in df_local.columns and not df_local.empty
            else ""
        )
        rac = (
            str(df_local["racion"].iloc[0])
            if "racion" in df_local.columns and not df_local.empty
            else ""
        )
        resumen.append(
            {
                "descarga": name,
                "mixer_id": mix_id,
                "tipo_racion": tipo,
                "racion": rac,
                "turnos": turnos,
                "kg_por_turno": round(kg_turno, 1),
                "kg_totales_dia": round(kg_dia, 1),
            }
        )

    resumen_json = json.dumps(resumen, ensure_ascii=False)

    total_descargas = len(frames)

    row = {
        "ts": _now_tag(),
        "usuario": username,
        "version_app": version_app,
        "num_descargas": total_descargas,
        "size_filas": len(consolidado),
        "archivo_csv": fpath.name,
        "resumen_kg_dia": resumen_json,
        "comentario": comment,
        "em_calc": np.nan,
        "pb_calc": np.nan,
        "cost_total": np.nan,
    }

    if MIXER_SIM_LOG.exists():
        logdf = pd.read_csv(MIXER_SIM_LOG, encoding="utf-8-sig")
        logdf = pd.concat([logdf, pd.DataFrame([row])], ignore_index=True)
    else:
        logdf = pd.DataFrame([row])

    logdf.to_csv(MIXER_SIM_LOG, index=False, encoding="utf-8-sig")

    try:
        github_upsert(
            MIXER_SIM_LOG,
            message=f"backup({username}): {MIXER_SIM_LOG.name}",
        )
        github_upsert(
            fpath,
            message=f"backup({username}): {fpath.name}",
        )
    except Exception:
        pass

    return True, f"Respaldo creado: {fpath.name}"


def build_methodology_doc() -> tuple[str, dict]:
    """Genera el markdown de metodología y los metadatos de auditoría."""

    alim = load_alimentos()
    rec = load_recipes()
    cat = load_catalog()
    base = load_base()
    reqE = load_reqener()
    reqP = load_reqprot()

    resumen_tablas = {
        "alimentos": [str(c) for c in alim.columns],
        "recetas": [str(c) for c in rec.columns],
        "catalogo": [str(c) for c in cat.columns],
        "base": [str(c) for c in base.columns],
        "req_energetico": [str(c) for c in reqE.columns],
        "req_proteico": [str(c) for c in reqP.columns],
    }

    build_hash = _hash_app_version()
    last_changed_payload = _read_last_changed_payload()
    if isinstance(last_changed_payload, dict):
        last_changed_display = json.dumps(
            last_changed_payload, ensure_ascii=False, indent=2
        )
    elif isinstance(last_changed_payload, str) and last_changed_payload:
        last_changed_display = last_changed_payload
    else:
        last_changed_display = "s/d"
    if isinstance(last_changed_display, str):
        last_changed_display = last_changed_display.replace("\n", " ")

    md = dedent(
        f"""
        # 📐 Metodología y Cálculo — {APP_VERSION}

        **Build:** `{build_hash}`  
        **Último cambio:** `{last_changed_display}`

        ---
        ## 1) Normalización de alimentos
        - Columns destino: `ORIGEN, PRESENTACION, TIPO, MS, TND (%), PB, EE, COEF ATC, $/KG, EM, ENP2`
        - Limpieza numérica: remoción de `$`, `%`, espacios; `,`→`.`; a `float`.
        - `MS` y `TND (%)` en fracción (≤1) se multiplican ×100.
        - Unicidad por `ORIGEN` (case-insensitive), se conserva la última fila.
        - Encoding CSV: UTF-8 con BOM.

        ## 2) Receta 100% MS
        - Cada ración define **hasta 6 ingredientes** con `%MS` que suma **100±0.5**.
        - Se usa MS del alimento para calcular **factor as-fed**:  
          `MS_frac_i = MS_i/100`, `factor_asfed = 1 / Σ( pct_ms_i/100 × MS_frac_i )`.

        ## 3) Cálculo diario (PV, CV → MS → reparto)
        - `Consumo_MS_dia = PV_kg × (CV_pct / 100)`
        - Por ingrediente *i*:
          - `MS_kg_i = Consumo_MS_dia × (pct_ms_i/100)`
          - `asfed_kg_i = MS_kg_i / max(MS_frac_i, 1e-6)`
          - `EM_Mcal_i = MS_kg_i × EM_i`
          - `PB_g_i = MS_kg_i × (PB_i/100) × 1000`
          - `Costo_dia_i = asfed_kg_i × ($/KG_i)`
        - Totales:
          - `asfed_total_kg_dia = Σ asfed_kg_i`
          - `EM_Mcal_dia = Σ EM_Mcal_i`
          - `PB_g_dia = Σ PB_g_i`
          - `Costo_dia = Σ Costo_dia_i`

        ## 4) Requerimientos (interpolación)
        - Tablas: `requerimientos_energeticos.csv` (Mcal/día), `requerimiento_proteico.csv` (g PB/día).
        - Filtro por `cat` (case-insensitive).
        - Si hay columna `ap`: **interpolación bilineal** (peso, ap). Si no, **1D por peso**.
        - Coberturas:
          - `cov_EM = EM_Mcal_dia / req_EM`
          - `cov_PB = PB_g_dia / req_PB`
          - Semáforos: verde ≥98%, amarillo 95–98%, rojo <95%.

        ## 5) Mixer — plan por turno y por descarga
        - `kg_por_turno = Σ kg_turno_asfed_calc` de corrales (tipo+ración).
        - `kg_totales_dia = kg_por_turno × turnos`.
        - Expansión por turnos (1..N) y distribución a corrales (`kg/CORRAL`).
        - Verificación contra `capacidad_kg` del mixer (por turno).

        ## 6) Aislamiento por usuario
        - Ruta de trabajo: `data/users/<username>/...`.
        - Todos los `save_*` guardan en esa sandbox.

        ## 7) Export/Import/Backup
        - Export ZIP con todas las tablas del usuario.
        - Import ZIP para restaurar.
        - Backup GitHub opcional mediante `GH_TOKEN` (commit a `data/users/<username>/...`).

        ---
        ### Columnas detectadas (runtime)
        ```json
        {json.dumps(resumen_tablas, ensure_ascii=False, indent=2)}
        ```
        """
    )

    meta = {
        "build_hash": build_hash,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "app_version": APP_VERSION,
        "tables": resumen_tablas,
        "last_changed": last_changed_payload,
    }
    return md, meta


def _get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default


def _secret_lookup(container, key: str, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    try:
        return container[key]
    except Exception:
        return default


_github_section = _get_secret("github", {})
GITHUB_TOKEN = (
    _secret_lookup(_github_section, "token")
    or _get_secret("GH_TOKEN")
    or os.getenv("GH_TOKEN")
)
GITHUB_REPO = (
    _secret_lookup(_github_section, "repo")
    or _get_secret("GH_REPO")
    or os.getenv("GH_REPO")
)
GITHUB_BRANCH = (
    _secret_lookup(_github_section, "branch")
    or _get_secret("GH_BRANCH")
    or os.getenv("GH_BRANCH")
    or "main"
)
GITHUB_DATA_DIR = (
    _secret_lookup(_github_section, "data_dir")
    or _get_secret("GH_DATA_DIR")
    or os.getenv("GH_DATA_DIR")
    or "data"
)
GITHUB_ENABLED = bool(GITHUB_TOKEN and GITHUB_REPO)
_github_warning_cache: set[str] = set()


# Crear archivos mínimos si faltan
if not ALIM_PATH.exists():
    pd.DataFrame(columns=ALIM_COLS).to_csv(ALIM_PATH, index=False, encoding="utf-8")
if not MIXERS_PATH.exists():
    pd.DataFrame({"mixer_id": ["MX-4200", "MX-6000"], "capacidad_kg": [4200, 6000]}).to_csv(MIXERS_PATH, index=False, encoding="utf-8")
if not PESOS_PATH.exists():
    pd.DataFrame({"peso_kg":[150,162.5,175,187.5,200,212.5,225,237.5,250,262.5,275,287.5,300,312.5,325,337.5,350,362.5,375,387.5,400,412.5,425,437.5,450]}).to_csv(PESOS_PATH, index=False, encoding="utf-8")
if not CATALOG_PATH.exists():
    pd.DataFrame({"id":[1,2,3], "nombre":["R-JOSE","term","R-DTTE"], "etapa":["RECRIA","RECRIA","RECRIA"]}).to_csv(CATALOG_PATH, index=False, encoding="utf-8")
if not RECIPES_PATH.exists():
    pd.DataFrame(columns=["id_racion","nombre_racion","ingrediente","pct_ms"]).to_csv(RECIPES_PATH, index=False, encoding="utf-8")
if not BASE_PATH.exists():
    pd.DataFrame(columns=BASE_EXPECTED_COLUMNS).to_csv(
        BASE_PATH, index=False, encoding="utf-8"
    )
if not REQENER_PATH.exists():
    pd.DataFrame(columns=REQENER_COLS).to_csv(REQENER_PATH, index=False, encoding="utf-8")
if not REQPROT_PATH.exists():
    pd.DataFrame(columns=REQPROT_COLS).to_csv(REQPROT_PATH, index=False, encoding="utf-8")
if not AUDIT_LOG_PATH.exists():
    pd.DataFrame(
        columns=["timestamp", "user", "event", "details", "status", "path", "meta"]
    ).to_csv(AUDIT_LOG_PATH, index=False, encoding="utf-8")

def clear_streamlit_cache():
    try:
        st.cache_data.clear()
    except Exception:
        pass

def rerun_with_cache_reset():
    clear_streamlit_cache()
    st.rerun()

# ------------------------------------------------------------------------------
# Normalización de alimentos
# ------------------------------------------------------------------------------
ALIM_NUMERIC_COLS = ["MS", "TND (%)", "PB", "EE", "COEF ATC", "$/KG", "EM", "ENP2"]
ALIM_TEXT_COLS = ["ORIGEN", "PRESENTACION", "TIPO"]

ALIM_COLUMN_ALIASES = {
    "presentacion": "PRESENTACION",
    "origen": "ORIGEN",
    "descripcion": "PRESENTACION",
    "nombre": "ORIGEN",
    "tipo": "TIPO",
    "tipo para despues generar informe": "TIPO",
    "ms": "MS",
    "%ms": "MS",
    "m.s.": "MS",
    "ms (%)": "MS",
    "tnd": "TND (%)",
    "tnd (%)": "TND (%)",
    "tnd%": "TND (%)",
    "tnd(%)": "TND (%)",
    "pb": "PB",
    "%pb": "PB",
    "ee": "EE",
    "%ee": "EE",
    "coef atc": "COEF ATC",
    "coefatc": "COEF ATC",
    "coef. atc": "COEF ATC",
    "precio": "$/KG",
    "precio/kilo": "$/KG",
    "precio/kg": "$/KG",
    "precio x kg": "$/KG",
    "$xkg": "$/KG",
    "$ / kg": "$/KG",
    "$": "$/KG",
    "$kg": "$/KG",
    "$ x kg": "$/KG",
    "$/kg": "$/KG",
    "costo": "$/KG",
    "costo/kg": "$/KG",
    "precio (para calcula consto de la racion por cab)": "$/KG",
    "precio (para calcula costo de la racion por cab)": "$/KG",
    "costo (para calcula consto de la racion por cab)": "$/KG",
    "costo (para calcula costo de la racion por cab)": "$/KG",
    "em": "EM",
    "energía metabolizable": "EM",
    "energia metabolizable": "EM",
    "enp2": "ENP2",
    "enp": "ENP_ALT",
}


def _clean_header(name: str) -> str:
    base = str(name or "").strip().replace("\ufeff", "")
    normalized = (
        base.lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )
    return normalized


def _clean_numeric(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    text = (
        text.replace("\u00A0", " ")
        .replace("$", "")
        .replace("%", "")
        .replace(" ", "")
        .replace(",", ".")
    )
    try:
        return float(text)
    except Exception:
        return default


def normalize_alimentos(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        out = pd.DataFrame(columns=EXPECTED_ALIM_COLS)
        out.attrs["discarded_rows"] = 0
        return out

    working = df.copy()

    rename_map = {}
    for original in list(working.columns):
        key = _clean_header(original)
        target = ALIM_COLUMN_ALIASES.get(key)
        if target:
            rename_map[original] = target

    working = working.rename(columns=rename_map)

    recognized = [col for col in working.columns if col in EXPECTED_ALIM_COLS or col == "ENP_ALT"]
    if not recognized:
        raise ValueError("Archivo sin columnas reconocibles")

    has_origen = "ORIGEN" in working.columns
    has_presentacion = "PRESENTACION" in working.columns
    if not has_origen and has_presentacion:
        working["ORIGEN"] = working["PRESENTACION"]
        has_origen = True
    if not has_presentacion and has_origen:
        working["PRESENTACION"] = working["ORIGEN"]
        has_presentacion = True

    for col in EXPECTED_ALIM_COLS + ["ENP_ALT"]:
        if col not in working.columns:
            if col in ALIM_TEXT_COLS:
                working[col] = ""
            elif col == "ENP_ALT":
                working[col] = 0.0
            else:
                working[col] = 0.0

    for col in ALIM_TEXT_COLS:
        working[col] = working[col].fillna("").astype(str).str.strip()

    for col in ALIM_NUMERIC_COLS + ["ENP_ALT"]:
        working[col] = working[col].map(lambda v: _clean_numeric(v, 0.0))

    enp_alt = working.get("ENP_ALT")
    if enp_alt is not None:
        mask = working["ENP2"].isna() | (working["ENP2"] == 0)
        working.loc[mask, "ENP2"] = enp_alt[mask]
        working = working.drop(columns=["ENP_ALT"])

    ms_mask = (working["MS"] > 0) & (working["MS"] <= 1.0)
    working.loc[ms_mask, "MS"] = working.loc[ms_mask, "MS"] * 100.0

    tnd_mask = (working["TND (%)"] > 0) & (working["TND (%)"] <= 1.0)
    working.loc[tnd_mask, "TND (%)"] = working.loc[tnd_mask, "TND (%)"] * 100.0

    before_drop = len(working)
    working = working[working["ORIGEN"].astype(str).str.strip() != ""]
    discarded = before_drop - len(working)

    if not working.empty:
        working = working.drop_duplicates(subset=["ORIGEN"], keep="last")

    working = working[EXPECTED_ALIM_COLS].reset_index(drop=True)
    working.attrs["discarded_rows"] = max(discarded, 0)
    return working


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_alimentos(df)


def _notify_github_warning(message: str) -> None:
    if message in _github_warning_cache:
        return
    _github_warning_cache.add(message)
    st.warning(message)


def github_upsert(local_path: Path, *, message: str, repo_path: str | None = None) -> bool:
    if not GITHUB_ENABLED:
        return False
    if not local_path.exists():
        return False

    if repo_path is None:
        try:
            relative = local_path.relative_to(GLOBAL_DATA_DIR).as_posix()
        except ValueError:
            relative = local_path.name
        repo_path = f"{GITHUB_DATA_DIR.strip('/')}/{relative}".replace("//", "/")
    else:
        repo_path = repo_path.lstrip("/")

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{repo_path}"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

    try:
        content_bytes = local_path.read_bytes()
    except Exception as exc:
        _notify_github_warning(f"No se pudo leer {local_path.name} para el backup en GitHub: {exc}")
        return False

    encoded = base64.b64encode(content_bytes).decode("utf-8")

    sha = None
    try:
        existing = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH}, timeout=10)
    except Exception as exc:
        _notify_github_warning(f"GitHub no disponible: {exc}")
        return False

    if existing.status_code == 200:
        sha = existing.json().get("sha")
    elif existing.status_code not in (200, 404):
        _notify_github_warning(
            f"No se pudo consultar el estado del backup ({existing.status_code})."
        )
        return False

    payload = {
        "message": message,
        "content": encoded,
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    try:
        response = requests.put(url, headers=headers, json=payload, timeout=10)
    except Exception as exc:
        _notify_github_warning(f"Error subiendo backup a GitHub: {exc}")
        return False

    if response.status_code not in (200, 201):
        _notify_github_warning(
            f"Backup en GitHub falló ({response.status_code}): {response.text[:120]}"
        )
        return False

    return True


def audit_log_append(
    event: str,
    details: str = "",
    *,
    status: str = "ok",
    path: str | None = None,
    meta: dict | None = None,
) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user": username,
        "event": event,
        "details": details,
        "status": status,
        "path": path or "",
        "meta": json.dumps(meta, ensure_ascii=False) if meta else "",
    }
    try:
        df = pd.DataFrame([record])
        df.to_csv(
            AUDIT_LOG_PATH,
            mode="a",
            header=not AUDIT_LOG_PATH.exists() or AUDIT_LOG_PATH.stat().st_size == 0,
            index=False,
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[AUDIT] No se pudo escribir el log: {exc}", flush=True)
    else:
        print(f"[AUDIT] {record}", flush=True)


def backup_user_file(local_path: Path, label: str) -> bool:
    message = f"[{username}] {label}"
    return github_upsert(local_path, message=message)


def validate_upload_size(uploaded_file, *, label: str) -> bool:
    size = getattr(uploaded_file, "size", None)
    if size and size > MAX_UPLOAD_BYTES:
        mb = size / (1024 * 1024)
        st.error(f"{label} excede el límite de {MAX_UPLOAD_MB} MB (actual: {mb:.2f} MB).")
        audit_log_append(
            "upload_blocked",
            f"{label} demasiado grande",
            status="blocked",
            meta={"size_mb": round(mb, 2)},
        )
        return False
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    return True

# ------------------------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------------------------
def load_alimentos() -> pd.DataFrame:
    try:
        df = pd.read_csv(ALIM_PATH, encoding="utf-8-sig")
    except FileNotFoundError:
        return pd.DataFrame(columns=ALIM_COLS)
    except Exception:
        df = pd.DataFrame(columns=ALIM_COLS)
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(ALIM_PATH, sep=";", encoding="utf-8-sig")
        except Exception:
            pass
    return _normalize_columns(df)

def save_alimentos(df: pd.DataFrame):
    _normalize_columns(df.copy()).to_csv(ALIM_PATH, index=False, encoding="utf-8")
    success = backup_user_file(ALIM_PATH, "Actualizar catálogo de alimentos")
    audit_log_append(
        "save_alimentos",
        "Catálogo actualizado",
        path=str(ALIM_PATH),
        meta={"github_backup": success},
    )
    mark_changed("save_alimentos", username)
    activity_log_event(
        "edicion",
        f"alimentos filas={len(df)}",
        trace_prefix="ALIM-",
    )


def append_ration_log(
    *,
    username: str,
    racion_nombre: str,
    tipo_racion: str,
    pv_kg: float,
    cv_pct: float,
    categoria: str,
    sim: dict,
    ingredientes_df: pd.DataFrame,
    tips: list[str] | None = None,
    gmd_kg_dia: float | None = None,
) -> None:
    """Registra la ración calculada y la marca como vigente."""

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detalle = sim.get("detalle", []) if isinstance(sim, dict) else []
    receta_records = (
        ingredientes_df[["ingrediente", "pct_ms"]]
        .copy()
        .to_dict(orient="records")
        if not ingredientes_df.empty
        else []
    )
    balance_tips = tips or []

    row = {
        "ts": ts,
        "usuario": username,
        "tipo_racion": str(tipo_racion),
        "racion": str(racion_nombre),
        "cat": str(categoria),
        "pv_kg": float(pv_kg),
        "cv_pct": float(cv_pct),
        "consumo_ms_dia": float(sim.get("Consumo_MS_dia", 0.0) if isinstance(sim, dict) else 0.0),
        "asfed_total_kg_dia": float(sim.get("asfed_total_kg_dia", 0.0) if isinstance(sim, dict) else 0.0),
        "em_mcal_dia": float(sim.get("EM_Mcal_dia", 0.0) if isinstance(sim, dict) else 0.0),
        "pb_g_dia": float(sim.get("PB_g_dia", 0.0) if isinstance(sim, dict) else 0.0),
        "costo_dia": float(sim.get("costo_dia", 0.0) if isinstance(sim, dict) else 0.0),
        "gmd_kg_dia": float(gmd_kg_dia) if isinstance(gmd_kg_dia, (int, float)) else np.nan,
        "receta_pct_ms": json.dumps(receta_records, ensure_ascii=False),
        "detalle_calc": json.dumps(detalle, ensure_ascii=False),
        "balance_tips": json.dumps(balance_tips, ensure_ascii=False),
    }

    try:
        existing = pd.read_csv(RACIONES_LOG_PATH, encoding="utf-8-sig")
    except Exception:
        existing = pd.DataFrame()

    if existing.empty:
        df = pd.DataFrame([row])
    else:
        df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(RACIONES_LOG_PATH, index=False, encoding="utf-8-sig")
    df.to_csv(RACIONES_DADAS_PATH, index=False, encoding="utf-8-sig")

    try:
        RACION_VIGENTE_PATH.write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        st.warning(f"No se pudo actualizar la ración vigente: {exc}")

    backup_user_file(RACIONES_LOG_PATH, "Registrar ración dada")
    backup_user_file(RACION_VIGENTE_PATH, "Actualizar ración vigente")

    audit_log_append(
        "racion_log_append",
        f"Ración {racion_nombre} registrada",
        meta={
            "tipo_racion": str(tipo_racion),
            "cat": str(categoria),
            "pv_kg": float(pv_kg),
            "cv_pct": float(cv_pct),
        },
    )
    mark_changed("append_ration_log", username)
    activity_log_event(
        "edicion",
        f"racion={racion_nombre} tipo={tipo_racion}",
        trace_prefix="RAC-",
    )


def build_erp_payload(limit: int | None = None) -> dict:
    """Construye el JSON con raciones dadas para integraciones externas."""

    generated_at = datetime.now().isoformat(timespec="seconds")
    records: list[dict[str, Any]] = []
    total_asfed = total_cost = total_ms = 0.0
    gmd_values: list[float] = []

    try:
        df = pd.read_csv(RACIONES_LOG_PATH, encoding="utf-8-sig")
    except Exception:
        df = pd.DataFrame()

    if not df.empty:
        with pd.option_context("mode.chained_assignment", None):
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            df = df.sort_values("ts", ascending=False)

        if limit is not None and limit > 0:
            df = df.head(limit)

        for _, row in df.iterrows():
            tips_raw = row.get("balance_tips", "[]")
            try:
                tips_list = json.loads(tips_raw) if isinstance(tips_raw, str) else list(tips_raw)
            except Exception:
                tips_list = []

            consumo_ms_row = _num(row.get("consumo_ms_dia"), 0.0)
            asfed_row = _num(row.get("asfed_total_kg_dia"), 0.0)
            costo_row = _num(row.get("costo_dia"), 0.0)
            gmd_row = _num(row.get("gmd_kg_dia"), np.nan)

            total_ms += consumo_ms_row
            total_asfed += asfed_row
            total_cost += costo_row
            if not np.isnan(gmd_row):
                gmd_values.append(gmd_row)

            records.append(
                {
                    "ts": str(row.get("ts")),
                    "usuario": row.get("usuario"),
                    "tipo_racion": row.get("tipo_racion"),
                    "racion": row.get("racion"),
                    "cat": row.get("cat"),
                    "pv_kg": _num(row.get("pv_kg"), 0.0),
                    "cv_pct": _num(row.get("cv_pct"), 0.0),
                    "consumo_ms_dia": consumo_ms_row,
                    "asfed_total_kg_dia": asfed_row,
                    "costo_dia": costo_row,
                    "gmd_kg_dia": None if np.isnan(gmd_row) else gmd_row,
                    "tips": tips_list,
                }
            )

    promedio_gmd = None
    if gmd_values:
        promedio_gmd = float(np.mean(gmd_values))

    payload = {
        "generated_at": generated_at,
        "usuario": username,
        "raciones_dadas": records,
        "consumo_ms_total_kg_dia": round(total_ms, 3),
        "consumo_kg_dia": round(total_asfed, 3),
        "costo_total_dia": round(total_cost, 2),
        "gmd_promedio_kg_dia": promedio_gmd,
        "total_registros": len(records),
    }
    return payload


def send_payload_to_erp(api_url: str, payload: dict) -> tuple[bool, str]:
    if not api_url:
        return False, "Definí una URL de API ERP."
    try:
        response = requests.post(api_url, json=payload, timeout=10)
    except Exception as exc:
        return False, str(exc)
    if response.status_code >= 400:
        return False, f"{response.status_code} – {response.text}"
    return True, response.text or "OK"


def sync_food_costs_from_api(api_url: str) -> tuple[bool, str, int]:
    if not api_url:
        return False, "Definí la URL del módulo Compras/Stock.", 0
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
    except Exception as exc:
        return False, str(exc), 0

    try:
        data = response.json()
    except Exception as exc:
        return False, f"Respuesta JSON inválida: {exc}", 0

    if isinstance(data, dict):
        items = data.get("alimentos") or data.get("items") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []

    if not items:
        return False, "La API no devolvió alimentos para actualizar.", 0

    alim_df = load_alimentos().copy()
    if alim_df.empty:
        alim_df = pd.DataFrame(columns=ALIM_COLS)

    if "ORIGEN" not in alim_df.columns:
        alim_df["ORIGEN"] = ""
    if "$/KG" not in alim_df.columns:
        alim_df["$/KG"] = 0.0

    alim_df["_origen_lower"] = alim_df["ORIGEN"].astype(str).str.strip().str.lower()

    updates = 0
    for entry in items:
        if not isinstance(entry, dict):
            continue
        origen = str(entry.get("ORIGEN") or entry.get("origen") or "").strip()
        if not origen:
            continue
        price_raw = entry.get("precio")
        if price_raw is None:
            price_raw = entry.get("$/kg") or entry.get("$/KG")
        price_val = _num_or_none(price_raw)
        if price_val is None:
            continue
        mask = alim_df["_origen_lower"] == origen.lower()
        if mask.any():
            alim_df.loc[mask, "$/KG"] = float(price_val)
            updates += int(mask.sum())

    if updates <= 0:
        return False, "No se encontraron alimentos coincidentes para actualizar.", 0

    alim_df = alim_df.drop(columns=["_origen_lower"])
    save_alimentos(alim_df)
    return True, f"Actualizados {updates} precios desde Compras/Stock.", updates

@st.cache_data
def load_mixers() -> pd.DataFrame:
    try: df = pd.read_csv(MIXERS_PATH, encoding="utf-8-sig")
    except: df = pd.DataFrame({"mixer_id":[], "capacidad_kg":[]})
    df["capacidad_kg"] = pd.to_numeric(df["capacidad_kg"], errors="coerce").fillna(0).astype(float)
    return df

def save_mixers(df: pd.DataFrame):
    df.to_csv(MIXERS_PATH, index=False, encoding="utf-8")
    success = backup_user_file(MIXERS_PATH, "Actualizar mixers")
    audit_log_append(
        "save_mixers",
        "Mixers actualizados",
        path=str(MIXERS_PATH),
        meta={"github_backup": success},
    )
    mark_changed("save_mixers", username)
    activity_log_event(
        "edicion",
        f"mixers filas={len(df)}",
        trace_prefix="MIX-",
    )

@st.cache_data
def load_pesos() -> pd.DataFrame:
    try: df = pd.read_csv(PESOS_PATH, encoding="utf-8-sig")
    except: df = pd.DataFrame({"peso_kg":[]})
    df["peso_kg"] = pd.to_numeric(df["peso_kg"], errors="coerce").fillna(0).astype(float)
    return df[df["peso_kg"]>0].drop_duplicates().sort_values("peso_kg").reset_index(drop=True)

def save_pesos(df: pd.DataFrame):
    out = df.copy()
    out["peso_kg"] = pd.to_numeric(out["peso_kg"], errors="coerce")
    out = out.dropna().drop_duplicates().sort_values("peso_kg").reset_index(drop=True)
    out.to_csv(PESOS_PATH, index=False, encoding="utf-8")
    success = backup_user_file(PESOS_PATH, "Actualizar pesos")
    audit_log_append(
        "save_pesos",
        "Pesos actualizados",
        path=str(PESOS_PATH),
        meta={"github_backup": success},
    )
    mark_changed("save_pesos", username)
    activity_log_event(
        "edicion",
        f"pesos filas={len(out)}",
        trace_prefix="PESO-",
    )

@st.cache_data
def load_reqener() -> pd.DataFrame:
    try:
        df = pd.read_csv(REQENER_PATH, encoding="utf-8-sig")
    except Exception:
        df = pd.DataFrame(columns=REQENER_COLS)
    for col in REQENER_COLS:
        if col not in df.columns: df[col] = None
    df = df[REQENER_COLS]
    df["peso"] = pd.to_numeric(df["peso"], errors="coerce")
    df["requerimiento_energetico"] = pd.to_numeric(df["requerimiento_energetico"], errors="coerce")
    df["ap"] = pd.to_numeric(df["ap"], errors="coerce")
    df["cat"] = df["cat"].fillna("").astype(str)
    return df.sort_values(["cat","peso","ap"], na_position="last").reset_index(drop=True)

def save_reqener(df: pd.DataFrame):
    out = df.copy()
    out["peso"] = pd.to_numeric(out["peso"], errors="coerce")
    out["requerimiento_energetico"] = pd.to_numeric(out["requerimiento_energetico"], errors="coerce")
    out["ap"] = pd.to_numeric(out["ap"], errors="coerce")
    out["cat"] = out["cat"].fillna("").astype(str)
    out = out[REQENER_COLS]
    out.to_csv(REQENER_PATH, index=False, encoding="utf-8")
    success = backup_user_file(REQENER_PATH, "Actualizar requerimientos energéticos")
    audit_log_append(
        "save_reqener",
        "Requerimientos energéticos actualizados",
        path=str(REQENER_PATH),
        meta={"github_backup": success},
    )
    mark_changed("save_reqener", username)
    activity_log_event(
        "edicion",
        f"req_em filas={len(out)}",
        trace_prefix="REQE-",
    )

@st.cache_data
def load_reqprot() -> pd.DataFrame:
    try:
        df = pd.read_csv(REQPROT_PATH, encoding="utf-8-sig")
    except Exception:
        df = pd.DataFrame(columns=REQPROT_COLS)
    rename_map = {}
    for col in df.columns:
        cname = str(col).strip().lower()
        if cname in ("peso","peso_kg","pv","pv_kg"): rename_map[col] = "peso"
        elif cname in ("cat","categoria","categoría"): rename_map[col] = "cat"
        elif cname in ("ap","ap_kg_dia","ap_kg/dia","ap_kg-dia"): rename_map[col] = "ap"
        elif cname in ("req_proteico","requerimiento_proteico","proteina","proteína"): rename_map[col] = "req_proteico"
    df = df.rename(columns=rename_map)
    for col in REQPROT_COLS:
        if col not in df.columns: df[col] = None
    df = df[REQPROT_COLS]
    df["peso"] = pd.to_numeric(df["peso"], errors="coerce")
    df["ap"] = pd.to_numeric(df["ap"], errors="coerce")
    df["req_proteico"] = pd.to_numeric(df["req_proteico"], errors="coerce")
    df["cat"] = df["cat"].fillna("").astype(str)
    return df.sort_values(["cat","peso","ap"], na_position="last").reset_index(drop=True)

def save_reqprot(df: pd.DataFrame):
    out = df.copy()
    out["peso"] = pd.to_numeric(out["peso"], errors="coerce")
    out["ap"] = pd.to_numeric(out["ap"], errors="coerce")
    out["req_proteico"] = pd.to_numeric(out["req_proteico"], errors="coerce")
    out["cat"] = out["cat"].fillna("").astype(str)
    out = out[REQPROT_COLS]
    out.to_csv(REQPROT_PATH, index=False, encoding="utf-8")
    success = backup_user_file(REQPROT_PATH, "Actualizar requerimientos proteicos")
    audit_log_append(
        "save_reqprot",
        "Requerimientos proteicos actualizados",
        path=str(REQPROT_PATH),
        meta={"github_backup": success},
    )
    mark_changed("save_reqprot", username)
    activity_log_event(
        "edicion",
        f"req_pb filas={len(out)}",
        trace_prefix="REQP-",
    )

@st.cache_data
def load_catalog() -> pd.DataFrame:
    try:
        df = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    except:
        return pd.DataFrame({"id": [], "nombre": [], "etapa": []})
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "id" not in df.columns: df["id"] = pd.Series(dtype="int64")
    if "nombre" not in df.columns: df["nombre"] = pd.Series(dtype="object")
    if "etapa" not in df.columns: df["etapa"] = pd.Series(dtype="object")
    if "sexo" not in df.columns: df["sexo"] = pd.Series(dtype="object")
    if "pv_kg" not in df.columns: df["pv_kg"] = pd.Series(dtype="float64")
    if "cv_pct" not in df.columns: df["cv_pct"] = pd.Series(dtype="float64")
    if "corral_comparacion" not in df.columns: df["corral_comparacion"] = pd.Series(dtype="float64")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
    df["nombre"] = df["nombre"].fillna("").astype(str)
    df["etapa"] = df["etapa"].fillna("").astype(str)
    df["sexo"] = df["sexo"].fillna("").astype(str)
    df["pv_kg"] = pd.to_numeric(df["pv_kg"], errors="coerce").fillna(0.0)
    df["cv_pct"] = pd.to_numeric(df["cv_pct"], errors="coerce").fillna(0.0)
    df["corral_comparacion"] = pd.to_numeric(df["corral_comparacion"], errors="coerce").fillna(0.0)
    return df

def save_catalog(df: pd.DataFrame):
    df.to_csv(CATALOG_PATH, index=False, encoding="utf-8")
    success = backup_user_file(CATALOG_PATH, "Actualizar catálogo de raciones")
    audit_log_append(
        "save_catalog",
        "Catálogo de raciones actualizado",
        path=str(CATALOG_PATH),
        meta={"github_backup": success},
    )
    mark_changed("save_catalog", username)
    activity_log_event(
        "edicion",
        f"catalogo filas={len(df)}",
        trace_prefix="CAT-",
    )

@st.cache_data
def load_recipes() -> pd.DataFrame:
    try: df = pd.read_csv(RECIPES_PATH, encoding="utf-8-sig")
    except: df = pd.DataFrame(columns=["id_racion","nombre_racion","ingrediente","pct_ms"])
    df["pct_ms"] = pd.to_numeric(df["pct_ms"], errors="coerce").fillna(0.0)
    df["id_racion"] = pd.to_numeric(df["id_racion"], errors="coerce").fillna(0).astype(int)
    df["nombre_racion"] = df["nombre_racion"].fillna("").astype(str)
    df["ingrediente"] = df["ingrediente"].fillna("").astype(str)
    return df

def save_recipes(df: pd.DataFrame):
    out = df.copy()
    out["pct_ms"] = pd.to_numeric(out["pct_ms"], errors="coerce").fillna(0.0)
    out = out[out["ingrediente"].astype(str).str.strip()!=""]
    out.to_csv(RECIPES_PATH, index=False, encoding="utf-8")
    success = backup_user_file(RECIPES_PATH, "Actualizar recetas de raciones")
    audit_log_append(
        "save_recipes",
        "Recetas actualizadas",
        path=str(RECIPES_PATH),
        meta={"github_backup": success},
    )
    mark_changed("save_recipes", username)
    activity_log_event(
        "edicion",
        f"recetas filas={len(out)}",
        trace_prefix="REC-",
    )

def build_raciones_from_recipes() -> list:
    cat = load_catalog()
    rec = load_recipes()
    if cat.empty or rec.empty:
        return []

    alimentos = load_alimentos()
    if alimentos.empty:
        return []

    # --- lookup: ORIGEN (lower) -> dict (no Series)
    lookup = {}
    for _, row in alimentos.iterrows():
        nombre = str(row.get("ORIGEN", "")).strip()
        if not nombre:
            continue
        lookup[nombre.lower()] = row.to_dict()

    def _num(value, default=0.0):
        try:
            val = float(pd.to_numeric(value, errors="coerce"))
        except Exception:
            return default
        return default if pd.isna(val) else val

    def _text(value):
        if pd.isna(value):
            return ""
        return str(value)

    raciones = []
    for _, row in cat.iterrows():
        rid = int(row.get("id", 0))
        nombre = _text(row.get("nombre", "")).strip() or f"Ración {rid}"
        receta = rec[rec["id_racion"] == rid]
        if receta.empty:
            continue

        ingredientes = []
        for _, ing in receta.iterrows():
            ing_name = _text(ing.get("ingrediente", "")).strip()
            if not ing_name:
                continue

            ref = lookup.get(ing_name.lower())
            if ref is None:
                ref = {}

            ingredientes.append({
                "ORIGEN": ing_name,
                "PRESENTACION": _text(ref.get("PRESENTACION", "")),
                "TIPO": _text(ref.get("TIPO", "")),
                "MS": _num(ref.get("MS", 100.0), 100.0),
                "TND (%)": _num(ref.get("TND (%)", 0.0), 0.0),
                "PB": _num(ref.get("PB", 0.0), 0.0),
                "EE": _num(ref.get("EE", 0.0), 0.0),
                "COEF ATC": _num(ref.get("COEF ATC", 0.0), 0.0),
                "$/KG": _num(ref.get("$/KG", 0.0), 0.0),
                "EM": _num(ref.get("EM", 0.0), 0.0),
                "ENP2": _num(ref.get("ENP2", 0.0), 0.0),
                "inclusion_pct": _num(ing.get("pct_ms", 0.0), 0.0),
            })

        # Filtro de inclusiones > 0 (correctamente indentado)
        ingredientes = [i for i in ingredientes if i["inclusion_pct"] > 0.0]
        if not ingredientes:
            continue

        raciones.append({
            "id": rid,
            "nombre": nombre,
            "ingredientes": ingredientes,
        })

    return raciones
def load_base() -> pd.DataFrame:
    try:
        df = pd.read_csv(BASE_PATH, encoding="utf-8-sig")
    except FileNotFoundError:
        return pd.DataFrame(columns=BASE_EXPECTED_COLUMNS)
    except Exception:
        return pd.DataFrame(columns=BASE_EXPECTED_COLUMNS)
    drop_cols = [col for col in BASE_OPTIONAL_COLUMNS if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    for col in BASE_EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    ordered_cols = [col for col in BASE_EXPECTED_COLUMNS if col in df.columns]
    df = df[ordered_cols]

    if "nro_cab" in df.columns:
        df["nro_cab"] = normalize_animal_counts(df["nro_cab"], index=df.index)
        df["nro_cab"] = (
            pd.to_numeric(df["nro_cab"], errors="coerce").fillna(0).astype(int)
        )
    return df


def save_base(df: pd.DataFrame):
    out = df.copy()
    for col in BASE_EXPECTED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[[col for col in BASE_EXPECTED_COLUMNS if col in out.columns]]
    if "nro_cab" in out.columns:
        out["nro_cab"] = (
            pd.to_numeric(out["nro_cab"], errors="coerce").fillna(0).astype(int)
        )
    out.to_csv(BASE_PATH, index=False, encoding="utf-8")
    success = backup_user_file(BASE_PATH, "Actualizar base de corrales")
    audit_log_append(
        "save_base",
        "Base de corrales actualizada",
        path=str(BASE_PATH),
        meta={"github_backup": success},
    )
    mark_changed("save_base", username)
    activity_log_event(
        "edicion",
        f"corrales filas={len(df)}",
        trace_prefix="BASE-",
    )


def enrich_and_calc_base(df: pd.DataFrame) -> pd.DataFrame:
    cat_df = load_catalog()
    mix_df = load_mixers()
    cat_names = cat_df["nombre"].astype(str) if "nombre" in cat_df.columns else pd.Series(dtype=str)
    nombre_to_id = dict(zip(cat_names, cat_df["id"])) if "id" in cat_df.columns else {}
    etapa_series = cat_df["etapa"] if "etapa" in cat_df.columns else pd.Series([""] * len(cat_df))
    etapa_series = etapa_series.fillna("").astype(str)
    nombre_to_tipo = dict(zip(cat_names, etapa_series))
    cv_series = cat_df.get("cv_pct")
    if cv_series is not None:
        nombre_to_cv = dict(
            zip(
                cat_names,
                pd.to_numeric(cv_series, errors="coerce").fillna(0.0),
            )
        )
    else:
        nombre_to_cv = {}

    sexo_series = cat_df.get("sexo")
    if sexo_series is not None:
        nombre_to_categoria = dict(
            zip(
                cat_names,
                sexo_series.fillna("").astype(str).str.strip(),
            )
        )
    else:
        nombre_to_categoria = {}

    pv_series = cat_df.get("pv_kg")
    if pv_series is not None:
        nombre_to_pv = dict(
            zip(
                cat_names,
                pd.to_numeric(pv_series, errors="coerce").fillna(0.0),
            )
        )
    else:
        nombre_to_pv = {}
    mix_clean = mix_df.dropna(subset=["mixer_id"]).copy()
    mix_clean["mixer_id"] = mix_clean["mixer_id"].astype(str)
    mixer_cap_map = dict(zip(mix_clean["mixer_id"], mix_clean["capacidad_kg"]))

    df = df.copy()
    df = df.drop(columns=["AP_obt", "dias_TERM", "semanas_TERM", "EFC_conv"], errors="ignore")

    for col, default in {
        "nombre_racion": "",
        "tipo_racion": "",
        "categ": "",
        "CV_pct": 0.0,
        "turnos": 1,
        "nro_cab": 0,
        "PV_kg": 0.0,
        "meta_salida": 0.0,
        "AP_preten": 0.0,
        "mixer_id": "",
    }.items():
        if col not in df.columns:
            df[col] = default

    df["nombre_racion"] = df["nombre_racion"].fillna("").astype(str)
    df["mixer_id"] = df["mixer_id"].fillna("").astype(str)

    # Normalizamos las cabezas antes de realizar cualquier cálculo numérico.
    # Esto evita que valores con separadores de miles (por ejemplo "1.200")
    # se interpreten como decimales (1.2) cuando se convierten con ``float``.
    df["nro_cab"] = normalize_animal_counts(df.get("nro_cab"), index=df.index)

    df["cod_racion"] = df["nombre_racion"].map(nombre_to_id).fillna("")
    df["tipo_racion"] = df["nombre_racion"].map(nombre_to_tipo).fillna("")

    def _cv(row):
        current = pd.to_numeric(row.get("CV_pct", 0.0), errors="coerce")
        nombre = str(row.get("nombre_racion", ""))
        if pd.isna(current) or float(current) == 0.0:
            return float(nombre_to_cv.get(nombre, current if not pd.isna(current) else 0.0))
        return float(current)

    df["CV_pct"] = df.apply(_cv, axis=1)

    if nombre_to_categoria:
        df["categ"] = df.apply(
            lambda row: (
                nombre_to_categoria.get(str(row.get("nombre_racion", "")), "")
                if not str(row.get("categ", "")).strip()
                else str(row.get("categ", "")).strip()
            ),
            axis=1,
        )

    if nombre_to_pv:
        def _pv(row: pd.Series) -> float:
            current = pd.to_numeric(row.get("PV_kg"), errors="coerce")
            if pd.isna(current) or float(current) == 0.0:
                nombre = str(row.get("nombre_racion", ""))
                return float(nombre_to_pv.get(nombre, 0.0))
            return float(current)

        df["PV_kg"] = df.apply(_pv, axis=1)

    df["capacidad_kg"] = df["mixer_id"].map(mixer_cap_map).fillna(0)

    def kg_turno_calc(r):
        try:
            return round(
                (
                    float(r["PV_kg"]) * (float(r["CV_pct"]) / 100.0) * float(r["nro_cab"])
                )
                / max(float(r["turnos"]), 1.0),
                1,
            )
        except Exception:
            return 0.0

    df["kg_turno_calc"] = df.apply(kg_turno_calc, axis=1)
    df["kg_turno"] = df["kg_turno_calc"]

    recipes = load_recipes()
    alimentos = load_alimentos()[["ORIGEN", "MS"]]
    ms_map: dict[str, float] = {}
    for nombre in df["nombre_racion"].dropna().unique():
        sub = recipes[recipes["nombre_racion"] == nombre]
        if not sub.empty:
            sub = sub.merge(alimentos, left_on="ingrediente", right_on="ORIGEN", how="left")
            sub["MS_frac"] = pd.to_numeric(sub["MS"], errors="coerce").fillna(100.0) / 100.0
            w = (
                pd.to_numeric(sub["pct_ms"], errors="coerce").fillna(0.0) / 100.0 * sub["MS_frac"]
            ).sum()
            ms_map[nombre] = float(w) if w > 0 else 1.0

    def kg_turno_asfed(r):
        try:
            ms = ms_map.get(str(r["nombre_racion"]), 1.0)
            base = float(r["kg_turno"])
            return round(base / max(ms, 1e-6), 1)
        except Exception:
            return 0.0

    df["kg_turno_asfed_calc"] = df.apply(kg_turno_asfed, axis=1)

    df["nro_corral"] = pd.to_numeric(df.get("nro_corral", 0), errors="coerce").fillna(0).astype(int)
    df = df.sort_values("nro_corral").reset_index(drop=True)

    return df

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------
tab_labels = [
    "🏠 Dashboard",
    "📊 Stock & Corrales",
    "🧾 Ajustes de raciones",
    "📦 Alimentos",
    "🧮 Mixer",
    "⚙️ Parámetros",
    "⬇️ Exportar",
    "🌾 Acerca de",
]

admin_tab_labels: list[str] = []
if USER_IS_ADMIN:
    admin_tab_labels.extend([
        "📐 Metodología y Cálculo (Admin)",
        "👤 Usuarios (Admin)",
    ])

tabs = st.tabs(tab_labels + admin_tab_labels)

tab_home, tab_corrales, tab_raciones, tab_alimentos, tab_mixer, tab_parametros, tab_export, tab_presentacion, *admin_tabs = tabs

# ------------------------------------------------------------------------------
# 🏠 Dashboard
# ------------------------------------------------------------------------------
with tab_home:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Resumen operativo")

    snap = metrics_get_snapshot() if "metrics_get_snapshot" in globals() else {
        "visits_total": 0,
        "simulations_total": 0,
        "today_visits": 0,
        "today_simulations": 0,
        "last_update": None,
    }

    base_df = load_base()
    total_animales = 0
    va = 0
    nov = 0
    if isinstance(base_df, pd.DataFrame) and not base_df.empty:
        base_df = base_df.copy()
        base_df["nro_cab"] = normalize_animal_counts(
            base_df.get("nro_cab"), index=base_df.index
        )
        total_animales = int(base_df["nro_cab"].sum())
        categ_norm = (
            base_df.get("categ", "")
            .astype(str)
            .str.lower()
            .str.strip()
        )
        va_mask = categ_norm.str.contains("vaq", case=False, na=False)
        nov_mask = categ_norm.str.contains("nov", case=False, na=False)
        va = int(base_df.loc[va_mask, "nro_cab"].sum())
        nov = int(base_df.loc[nov_mask, "nro_cab"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Visitas (hoy)", snap.get("today_visits", 0))
    c2.metric("Simulaciones (hoy)", snap.get("today_simulations", 0))
    c3.metric("Animales totales", f"{total_animales:,}")
    c4.metric("Vaquillonas / Nov.", f"{va:,} / {nov:,}")

    st.markdown('</div>', unsafe_allow_html=True)

    mpath = user_path("metrics.json")
    if mpath.exists():
        try:
            data = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        sim_by_day = data.get("simulations_by_day", {}) if isinstance(data, dict) else {}
        if sim_by_day:
            df_plot = pd.DataFrame(
                [{"fecha": k, "sim": v} for k, v in sim_by_day.items()]
            ).sort_values("fecha")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Simulaciones por día")
            fig = plt.figure()
            plt.plot(df_plot["fecha"], df_plot["sim"], marker="o")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Simulaciones")
            plt.xlabel("Fecha")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Aún no hay datos de simulaciones para graficar.")
    else:
        st.info("Aún no hay datos de simulaciones para graficar.")

tab_methodology = tab_admin = None
if USER_IS_ADMIN and admin_tabs:
    if len(admin_tabs) == 2:
        tab_methodology, tab_admin = admin_tabs
    elif len(admin_tabs) == 1:
        # Solo se definió la pestaña de usuarios (compatibilidad defensiva)
        tab_admin = admin_tabs[0]

    else:
        st.info("Aún no hay datos de simulaciones para graficar.")

    log_path = get_log_path()
    if log_path.exists():
        try:
            activity_df = pd.read_csv(log_path, sep="|", encoding="utf-8")
        except Exception as exc:
            st.warning(f"No se pudo leer activity_log.csv: {exc}")
        else:
            if not activity_df.empty:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Actividad reciente")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Eventos", len(activity_df))

                accion_series = activity_df.get("accion")
                if accion_series is None:
                    st.warning(
                        "El archivo activity_log.csv no contiene la columna 'accion'."
                    )
                    simulaciones = ediciones = exportaciones = 0
                else:
                    accion_series = accion_series.astype(str)
                    simulaciones = int((accion_series == "simulacion").sum())
                    ediciones = int((accion_series == "edicion").sum())
                    exportaciones = int((accion_series == "exportacion").sum())

                c2.metric("Simulaciones", simulaciones)
                c3.metric("Ediciones", ediciones)
                c4.metric("Exportaciones", exportaciones)

                last_df = activity_df.tail(20).iloc[::-1]
                st.dataframe(last_df, use_container_width=True, hide_index=True)

                with st.expander("Filtros"):
                    filt_df = activity_df.copy()

                    acciones = []
                    if "accion" in filt_df.columns:
                        acciones = (
                            filt_df["accion"].dropna().astype(str).unique().tolist()
                        )
                        acciones.sort()
                    else:
                        st.info(
                            "No se puede filtrar por acciones porque falta la columna 'accion'."
                        )

                    operadores = []
                    if "op" in filt_df.columns:
                        operadores = (
                            filt_df["op"].dropna().astype(str).unique().tolist()
                        )
                        operadores.sort()
                    else:
                        st.info(
                            "No se puede filtrar por operadores porque falta la columna 'op'."
                        )

                    sel_acc = st.multiselect("Acciones", acciones)
                    sel_ops = st.multiselect("Operadores", operadores)
                    if sel_acc and "accion" in filt_df.columns:
                        filt_df = filt_df[filt_df["accion"].isin(sel_acc)]
                    if sel_ops and "op" in filt_df.columns:
                        filt_df = filt_df[filt_df["op"].isin(sel_ops)]
                    st.dataframe(
                        filt_df.tail(200).iloc[::-1],
                        use_container_width=True,
                        hide_index=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Aún no hay actividad registrada.")
    else:
        st.info("Aún no hay actividad registrada.")

# ------------------------------------------------------------------------------
# 📊 Stock & Corrales (principal)
# ------------------------------------------------------------------------------
with tab_corrales:
    with card("📊 Stock, categorías y corrales", "Actualizá ración, categoría, cabezas y mezcla asignada por corral."):
        cat_df = load_catalog()
        mix_df = load_mixers()
        base = load_base()

        etapas_series = cat_df.get("etapa", pd.Series(dtype=str))
        tipos = (
            sorted(
                {t.strip() for t in etapas_series.dropna().astype(str) if t and t.strip()},
                key=str.lower,
            )
            or ["Terminación", "Recría"]
        )

        categoria_series: list[pd.Series] = []

        if "categ" in base.columns and not base["categ"].dropna().empty:
            categoria_series.append(base["categ"].dropna())
        elif "sexo" in cat_df.columns and not cat_df["sexo"].dropna().empty:
            categoria_series.append(cat_df["sexo"].dropna())

        categorias_raw: set[str] = set()
        for series in categoria_series:
            for value in series.astype(str):
                v = value.strip()
                if v:
                    categorias_raw.add(v)

        categorias = sorted(categorias_raw, key=str.lower)

        if not categorias:
            categorias = ["va", "nov"]

        pesos_series = load_pesos().get("peso_kg", pd.Series(dtype=float))
        pesos_numeric = pd.to_numeric(pesos_series, errors="coerce").dropna()
        pesos_lista = sorted({float(p) for p in pesos_numeric.tolist()})
        mix_clean = mix_df.dropna(subset=["mixer_id"]).copy()
        mix_clean["mixer_id"] = mix_clean["mixer_id"].astype(str)
        mixers = mix_clean["mixer_id"].tolist()
        mixer_cap_map = dict(zip(mix_clean["mixer_id"], mix_clean["capacidad_kg"]))

        if base.empty:
            base = pd.DataFrame({
                "nro_corral": list(range(1, 21)),
                "nombre_racion": ["" for _ in range(20)],
                "categ": ["va"] * 20,
                "PV_kg": [275] * 20,
                "CV_pct": [2.8] * 20,
                "AP_preten": [1.0] * 20,
                "nro_cab": [0] * 20,
                "mixer_id": [mixers[0] if mixers else ""] * 20,
                "capacidad_kg": [mixer_cap_map.get(mixers[0], 0) if mixers else 0] * 20,
                "turnos": [4] * 20,
                "meta_salida": [350] * 20,
            })

        export_df = base.copy()
        for column in BASE_EXPECTED_COLUMNS:
            if column not in export_df.columns:
                export_df[column] = pd.NA
        export_df = export_df[BASE_EXPECTED_COLUMNS]

        st.markdown("#### 📁 Importar / exportar CSV de corrales")
        export_col, import_col = st.columns(2)
        with export_col:
            st.download_button(
                "⬇️ Descargar CSV actual",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="corrales.csv",
                mime="text/csv",
                key="corrales_export_csv",
            )
        with import_col:
            uploaded_corrales = st.file_uploader(
                "Subí un CSV con la misma estructura",
                type=["csv"],
                key="corrales_import_csv",
                help="Reemplaza toda la base guardada.",
            )
            st.caption(
                "Columnas esperadas: " + ", ".join(BASE_EXPECTED_COLUMNS)
            )
            if (
                uploaded_corrales is not None
                and validate_upload_size(uploaded_corrales, label="CSV de corrales")
            ):
                try:
                    uploaded_corrales.seek(0)
                    imported_df = pd.read_csv(uploaded_corrales, encoding="utf-8-sig")
                except Exception as exc:
                    st.error(f"Error leyendo el archivo: {exc}")
                    audit_log_append(
                        "import_corrales_error",
                        f"Lectura fallida de {getattr(uploaded_corrales, 'name', 'archivo')}",
                        status="error",
                        meta={"error": str(exc)},
                    )
                else:
                    imported_df.columns = imported_df.columns.astype(str).str.strip()
                    duplicated = (
                        imported_df.columns[imported_df.columns.duplicated()].tolist()
                    )
                    if duplicated:
                        st.error(
                            "⚠️ El archivo tiene columnas duplicadas: "
                            + ", ".join(duplicated)
                        )
                        audit_log_append(
                            "import_corrales_invalid_columns",
                            "Columnas duplicadas en CSV de corrales",
                            status="blocked",
                            meta={"duplicated": duplicated},
                        )
                    else:
                        allowed_extra = set(BASE_OPTIONAL_COLUMNS)
                        missing = [
                            col for col in BASE_EXPECTED_COLUMNS if col not in imported_df.columns
                        ]
                        extra = [
                            col
                            for col in imported_df.columns
                            if col not in BASE_EXPECTED_COLUMNS and col not in allowed_extra
                        ]
                        if missing or extra:
                            message = "⚠️ El archivo no coincide con la estructura esperada."
                            if missing:
                                message += "\nFaltan: " + ", ".join(sorted(missing))
                            if extra:
                                message += "\nSobran: " + ", ".join(sorted(extra))
                            st.error(message)
                            audit_log_append(
                                "import_corrales_invalid_columns",
                                "Columnas inesperadas en CSV de corrales",
                                status="blocked",
                                meta={"missing": missing, "extra": extra},
                            )
                        else:
                            ordered_df = imported_df[BASE_EXPECTED_COLUMNS].copy()

                            if "nro_cab" in ordered_df.columns:
                                ordered_df["nro_cab"] = normalize_animal_counts(
                                    ordered_df["nro_cab"], index=ordered_df.index
                                )

                            save_base(ordered_df)
                            st.success(
                                "✅ Corrales actualizados correctamente. Se recargará la vista."
                            )
                            st.toast("Corrales actualizados desde CSV.", icon="📥")
                            rerun_with_cache_reset()

        enriched = enrich_and_calc_base(base)
        base_animals = enriched.copy()
        if not base_animals.empty:
            base_animals["nro_cab"] = normalize_animal_counts(
                base_animals.get("nro_cab"), index=base_animals.index
            )
            base_animals["nro_cab"] = (
                pd.to_numeric(base_animals["nro_cab"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            base_animals["categ_norm"] = (
                base_animals.get("categ", "")
                .astype(str)
                .str.lower()
                .str.strip()
            )
            total_animales = int(base_animals["nro_cab"].sum())
            va_total = int(
                base_animals.loc[
                    base_animals["categ_norm"].str.contains(
                        "vaq", case=False, na=False
                    ),
                    "nro_cab",
                ].sum()
            )
            nov_total = int(
                base_animals.loc[
                    base_animals["categ_norm"].str.contains(
                        "nov", case=False, na=False
                    ),
                    "nro_cab",
                ].sum()
            )
        else:
            total_animales = va_total = nov_total = 0

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total animales", f"{total_animales:,}")
        mc2.metric("Vaquillonas", f"{va_total:,}")
        mc3.metric("Novillos", f"{nov_total:,}")

        stock_checks: dict[str, int] = {}
        stock_preview = base_animals.copy()
        if not stock_preview.empty:
            stock_preview["nro_cab"] = normalize_animal_counts(
                stock_preview.get("nro_cab"), index=stock_preview.index
            )
            stock_preview["nro_cab"] = (
                pd.to_numeric(stock_preview["nro_cab"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            stock_preview["categ_display"] = (
                stock_preview.get("categ", "")
                .fillna("")
                .astype(str)
                .str.strip()
            )
            stock_preview["categ_display"] = stock_preview["categ_display"].replace(
                {"": "Sin categoría"}
            )

            stock_cat = (
                stock_preview.groupby("categ_display", dropna=False)["nro_cab"]
                .sum()
                .reset_index()
                .rename(columns={"categ_display": "Categoría", "nro_cab": "Cabezas"})
            )
            stock_cat = stock_cat.sort_values("Cabezas", ascending=False)
            total_row = pd.DataFrame(
                [{"Categoría": "Total", "Cabezas": int(stock_cat["Cabezas"].sum())}]
            )
            stock_cat = pd.concat([stock_cat, total_row], ignore_index=True)

            st.markdown("#### Stock por categoría")
            st.dataframe(
                stock_cat,
                use_container_width=True,
                hide_index=True,
            )

            stock_preview["nombre_racion"] = (
                stock_preview.get("nombre_racion", "")
                .fillna("")
                .astype(str)
                .str.strip()
            )
            stock_preview["nombre_racion"] = stock_preview["nombre_racion"].replace(
                {"": "Sin ración"}
            )

            stock_racion = (
                stock_preview.groupby("nombre_racion", dropna=False)["nro_cab"]
                .sum()
                .reset_index()
                .rename(columns={"nombre_racion": "Ración", "nro_cab": "Cabezas"})
            )

            if not cat_df.empty and {"nombre", "etapa"}.issubset(cat_df.columns):
                catalog_info = (
                    cat_df[["nombre", "etapa"]]
                    .drop_duplicates()
                    .rename(columns={"nombre": "Ración", "etapa": "Etapa"})
                )
                stock_racion = stock_racion.merge(catalog_info, on="Ración", how="left")

            stock_racion = stock_racion.sort_values("Cabezas", ascending=False)

            st.markdown("#### Stock por ración")
            st.dataframe(
                stock_racion,
                use_container_width=True,
                hide_index=True,
            )

            stock_checks = {
                "filas": int(len(stock_preview)),
                "cabezas": int(stock_preview["nro_cab"].sum()),
                "sin_racion": int(
                    stock_racion.loc[stock_racion["Ración"] == "Sin ración", "Cabezas"].sum()
                ),
            }

        with st.expander("🔍 Verificaciones rápidas de corrales", expanded=False):
            st.caption(f"Filas cargadas: {stock_checks.get('filas', 0)}")
            st.caption(f"Cabezas totales: {stock_checks.get('cabezas', 0)}")
            if stock_checks.get("sin_racion", 0) > 0:
                st.warning(
                    f"Hay {stock_checks['sin_racion']:,} cabezas sin ración asignada. Revisá la tabla."
                )
            if not enriched.empty:
                key_cols = ["kg_turno_calc", "kg_turno_asfed_calc"]
                available_cols = [col for col in key_cols if col in enriched.columns]
                if available_cols:
                    na_pct = (
                        enriched[available_cols].isna().mean().mul(100).round(1).to_dict()
                    )
                    resumen = ", ".join(f"{col}: {pct}%" for col, pct in na_pct.items())
                    st.caption(f"NaN en cálculos clave (%): {resumen or '0%'}")

        preview_cols = [col for col in BASE_PREVIEW_COLUMNS if col in base.columns]
        if preview_cols:
            clean_preview = base[preview_cols].copy()
            if "nro_cab" in clean_preview.columns:
                clean_preview["nro_cab"] = normalize_animal_counts(
                    clean_preview["nro_cab"], index=clean_preview.index
                )
            if "nro_corral" in clean_preview.columns:
                clean_preview = clean_preview.sort_values(
                    by="nro_corral",
                    key=lambda s: pd.to_numeric(s, errors="coerce"),
                )
            st.markdown("#### 📋 Vista de corrales (limpia)")
            st.dataframe(
                clean_preview.reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

        racion_options = cat_df["nombre"].astype(str).tolist() if "nombre" in cat_df.columns else []

        colcfg = {
            "nro_corral": st.column_config.NumberColumn("n° de Corral", min_value=1, max_value=9999, step=1),
            "nombre_racion": st.column_config.SelectboxColumn(
                "nombre la ración",
                options=[""] + racion_options,
                help="Autocompleta tipo y puede pisar CV%",
            ),
            "categ": st.column_config.SelectboxColumn("categ", options=categorias),
            "PV_kg": st.column_config.SelectboxColumn("PV (kg)", options=pesos_lista) if pesos_lista else st.column_config.NumberColumn("PV (kg)", min_value=0.0, max_value=1000.0, step=5.0),
            "CV_pct": st.column_config.NumberColumn("CV (%)", min_value=0.0, max_value=20.0, step=0.1),
            "AP_preten": st.column_config.NumberColumn("AP (kg) PRETEN", min_value=0.0, max_value=5.0, step=0.1),
            "nro_cab": st.column_config.NumberColumn("NRO CAB (und)", min_value=0, max_value=100000, step=1),
            "mixer_id": st.column_config.SelectboxColumn("Mixer", options=[""] + mixers, help="Trae capacidad"),
            "capacidad_kg": st.column_config.NumberColumn(
                "capacidad (kg)", min_value=0, max_value=200000, step=10, disabled=True
            ),
            "turnos": st.column_config.NumberColumn("turnos", min_value=1, max_value=24, step=1),
            "meta_salida": st.column_config.NumberColumn("META DE SALIDA (kg)", min_value=0, max_value=2000, step=5),
        }

        editor_cols = [col for col in BASE_EXPECTED_COLUMNS if col in enriched.columns]
        if not editor_cols:
            editor_cols = BASE_EXPECTED_COLUMNS.copy()

        grid_source = enriched[editor_cols].copy()
        if "nro_cab" in grid_source.columns:
            grid_source["nro_cab"] = normalize_animal_counts(
                grid_source.get("nro_cab"), index=grid_source.index
            )
        if "nro_corral" in grid_source.columns:
            grid_source = grid_source.sort_values(
                by="nro_corral",
                key=lambda s: pd.to_numeric(s, errors="coerce"),
            )
        grid_source = grid_source.reset_index(drop=True)

        with st.form("form_base"):
            grid = st.data_editor(
                grid_source,
                column_config=colcfg,
                column_order=editor_cols,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key="grid_corrales",
            )
            c1, c2 = st.columns(2)
            save = c1.form_submit_button("💾 Guardar base", type="primary")
            refresh = c2.form_submit_button("🔄 Recargar")
            if save:
                if grid is None:
                    grid_df = grid_source.copy()
                else:
                    grid_df = pd.DataFrame(grid).reset_index(drop=True)
                out = enriched.copy().reset_index(drop=True)
                for col in editor_cols:
                    if col in grid_df.columns:
                        out[col] = grid_df[col]
                out = enrich_and_calc_base(out)
                for col in ["kg_turno_calc", "kg_turno_asfed_calc"]:
                    if col in out.columns:
                        out = out.drop(columns=[col])
                out_to_save = out.copy()
                mask_nonempty = pd.Series(False, index=out_to_save.index)
                if "nro_corral" in out_to_save.columns:
                    mask_nonempty |= out_to_save["nro_corral"].notna()
                if "nombre_racion" in out_to_save.columns:
                    mask_nonempty |= out_to_save["nombre_racion"].astype(str).str.strip() != ""
                if "nro_cab" in out_to_save.columns:
                    mask_nonempty |= normalize_animal_counts(out_to_save["nro_cab"]) > 0
                used_rows = out_to_save[mask_nonempty]
                if len(used_rows) > MAX_CORRALES:
                    st.error(f"Máximo permitido: {MAX_CORRALES} corrales. Estás intentando guardar {len(used_rows)}.")
                    audit_log_append(
                        "save_base_blocked",
                        "Supera máximo de corrales",
                        status="blocked",
                        path=str(BASE_PATH),
                        meta={"rows": int(len(used_rows))},
                    )
                else:
                    save_base(out_to_save)
                    st.success("Base guardada.")
                    st.toast("Base actualizada.", icon="📦")
                    rerun_with_cache_reset()
            if refresh:
                rerun_with_cache_reset()

# ------------------------------------------------------------------------------
# 📦 Alimentos
# ------------------------------------------------------------------------------
with tab_alimentos:
    with card("📦 Catálogo de alimentos", "MS/COEF/Precio editables — EM/PB fijos — máx. 30 filas"):
        col_fr, _ = st.columns([1, 1])
        if col_fr.button("🔄 Forzar recarga de catálogo"):
            rerun_with_cache_reset()

        alimentos_df = load_alimentos().copy()

        with dropdown("📥 Importar planilla de alimentos"):
            st.markdown("Descargá la plantilla base, completala y luego importala en formato CSV o Excel.")

            template_df = pd.DataFrame(ALIM_TEMPLATE_ROWS, columns=EXPECTED_ALIM_COLS)
            csv_bytes = template_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar plantilla (CSV)",
                data=csv_bytes,
                file_name="plantilla_alimentos.csv",
                mime="text/csv",
                key="alimentos_template_csv",
            )

            excel_bytes = None
            try:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:  # type: ignore[arg-type]
                    template_df.to_excel(writer, index=False, sheet_name="Alimentos")
                buffer.seek(0)
                excel_bytes = buffer.getvalue()
            except Exception:
                excel_bytes = None

            if excel_bytes:
                st.download_button(
                    "⬇️ Descargar plantilla (Excel)",
                    data=excel_bytes,
                    file_name="plantilla_alimentos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="alimentos_template_excel",
                )
            else:
                st.caption("Si preferís Excel, descargá la planilla CSV y abrila con tu editor favorito.")

            uploaded = st.file_uploader(
                "Subí tu planilla (.xlsx, .xls, .csv)",
                type=["xlsx", "xls", "csv"],
                key="alimentos_import_file",
            )
            if uploaded is not None and validate_upload_size(uploaded, label="Planilla de alimentos"):
                try:
                    uploaded.seek(0)
                    if uploaded.name.lower().endswith((".xlsx", ".xls")):
                        raw_df = pd.read_excel(uploaded)
                    else:
                        uploaded.seek(0)
                        raw_df = pd.read_csv(uploaded, encoding="utf-8-sig")
                except Exception as exc:
                    st.error(f"No se pudo leer el archivo: {exc}")
                    audit_log_append(
                        "import_alimentos_error",
                        f"Lectura fallida de {uploaded.name}",
                        status="error",
                        meta={"error": str(exc)},
                    )
                else:
                    try:
                        df_norm = normalize_alimentos(raw_df)
                    except ValueError as exc:
                        st.error(str(exc))
                        audit_log_append(
                            "import_alimentos_error",
                            f"Normalización fallida {uploaded.name}",
                            status="error",
                            meta={"error": str(exc)},
                        )
                    else:
                        st.dataframe(df_norm.head(20), use_container_width=True)
                        st.caption(f"Filas: {len(df_norm)} — Columnas: {len(df_norm.columns)}")
                        dropped = int(df_norm.attrs.get("discarded_rows", 0))
                        if dropped > 0:
                            st.warning(f"Se descartaron {dropped} filas sin ORIGEN válido.")

                        if st.button(
                            "💾 Cargar y reemplazar catálogo actual",
                            type="primary",
                            key="import_replace_alimentos",
                        ):
                            if len(df_norm) > 30:
                                st.error("El catálogo admite máximo 30 alimentos. Ajustá el archivo antes de importar.")
                            else:
                                save_alimentos(df_norm)
                                st.success("Catálogo actualizado y guardado.")
                                st.toast("Alimentos cargados correctamente.", icon="🧾")
                                audit_log_append(
                                    "import_alimentos",
                                    f"Archivo {uploaded.name} importado",
                                    meta={"rows": int(len(df_norm))},
                                )
                                rerun_with_cache_reset()

        show_cols = [
            "ORIGEN",
            "PRESENTACION",
            "TIPO",
            "MS",
            "TND (%)",
            "PB",
            "EE",
            "COEF ATC",
            "$/KG",
            "EM",
            "ENP2",
        ]
        for col in show_cols:
            if col not in alimentos_df.columns:
                alimentos_df[col] = None
        alimentos_df = alimentos_df[show_cols]

        column_cfg = {
            "ORIGEN": st.column_config.TextColumn("Origen", disabled=True),
            "PRESENTACION": st.column_config.TextColumn("Presentación", disabled=True),
            "TIPO": st.column_config.TextColumn("Tipo", disabled=True),
            "MS": st.column_config.NumberColumn("MS (%)", min_value=0.0, max_value=100.0, step=0.1),
            "TND (%)": st.column_config.NumberColumn("TND (%)", disabled=True),
            "PB": st.column_config.NumberColumn("PB (%)", disabled=True),
            "EE": st.column_config.NumberColumn("EE (%)", disabled=True),
            "COEF ATC": st.column_config.NumberColumn("Coef. ATC", step=0.01),
            "$/KG": st.column_config.NumberColumn("$ por kg (as-fed)", format="$ %.2f", step=0.01),
            "EM": st.column_config.NumberColumn("EM (Mcal/kg MS)", disabled=True),
            "ENP2": st.column_config.NumberColumn("ENp (Mcal/kg MS)", disabled=True),
        }

        st.caption(f"Registros actuales: **{len(alimentos_df)} / 30**")
        grid = st.data_editor(
            alimentos_df,
            column_config=column_cfg,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key="grid_alimentos_fixed",
        )

        estado_cols = st.columns(2)
        with estado_cols[0]:
            st.write("**Estado del catálogo:**")
            ms_vals = pd.to_numeric(grid.get("MS", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
            cero_ms = int((ms_vals <= 0).sum())
            chip("MS > 0 en todos", cero_ms == 0)
            chip(f"Alimentos listados: {len(grid)}", len(grid) > 0)
            if cero_ms > 0:
                st.warning("Hay alimentos con MS=0. Revisá antes de guardar.")

        c1, c2 = st.columns(2)
        if c1.button("💾 Guardar cambios (MS, COEF, $)", type="primary"):
            edited = grid.copy()
            if len(edited) > 30:
                st.error("El catálogo admite máximo 30 alimentos. Recortá filas antes de guardar.")
            else:
                base = load_alimentos().copy()
                base = base.merge(
                    edited[["ORIGEN", "MS", "COEF ATC", "$/KG"]],
                    on="ORIGEN",
                    how="left",
                    suffixes=("", "_NEW"),
                )
                for col in ["MS", "COEF ATC", "$/KG"]:
                    new_col = f"{col}_NEW"
                    if new_col in base.columns:
                        base[col] = base[new_col].where(base[new_col].notna(), base[col])
                        base.drop(columns=[new_col], inplace=True)
                base = base.head(30)
                save_alimentos(base)
                st.success("Catálogo actualizado (MS, COEF ATC y $/KG).")
                st.toast("Alimentos guardados y respaldados.", icon="🧾")
                rerun_with_cache_reset()

        if c2.button("🔄 Recargar"):
            rerun_with_cache_reset()

# ------------------------------------------------------------------------------
# Mixer wizard helpers
# ------------------------------------------------------------------------------
WIZARD_STEPS = ["datos_basicos", "raciones", "mixer", "corrales", "resumen"]
WIZARD_STATE_KEYS = ["draft_id", "wizard_step", *WIZARD_STEPS]


def get_step_index() -> int:
    return int(st.session_state.get("wizard_step", 0))


def set_step_index(i: int) -> None:
    st.session_state["wizard_step"] = int(max(0, min(i, len(WIZARD_STEPS) - 1)))


def go_next() -> None:
    set_step_index(get_step_index() + 1)


def go_prev() -> None:
    set_step_index(get_step_index() - 1)


def _parse_saved_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    return datetime.today().date()


def reset_wizard_state() -> None:
    for key in WIZARD_STATE_KEYS:
        st.session_state.pop(key, None)


def _load_existing_payload(step_name: str) -> dict[str, Any]:
    draft_id = ensure_draft_id()
    payload = st.session_state.get(step_name)
    if not payload:
        payload = load_step_data(draft_id, step_name) or {}
        if payload:
            st.session_state[step_name] = payload
    return payload or {}


def step_datos_basicos() -> None:
    draft_id = ensure_draft_id()
    existing = _load_existing_payload("datos_basicos")

    st.subheader("Paso 1 de 5 — Datos básicos de la carga")

    fecha_valor = _parse_saved_date(existing.get("fecha"))

    with st.form("form_datos_basicos"):
        nombre_carga = st.text_input(
            "Nombre o código de carga",
            value=existing.get("nombre_carga", ""),
        )
        fecha = st.date_input(
            "Fecha",
            value=fecha_valor,
        )
        observaciones = st.text_area(
            "Observaciones",
            value=existing.get("observaciones", ""),
        )

        col1, col2 = st.columns([1, 1])
        btn_guardar = col1.form_submit_button("💾 Guardar y continuar", type="primary")
        col2.form_submit_button("⬅️ Volver", disabled=True)

    if btn_guardar:
        if not nombre_carga.strip():
            st.error("Ingresá un nombre o código para la carga.")
            return

        payload = {
            "nombre_carga": nombre_carga.strip(),
            "fecha": str(fecha),
            "observaciones": observaciones.strip(),
        }
        st.session_state["datos_basicos"] = payload
        save_step_data(draft_id, "datos_basicos", payload)
        go_next()
        st.rerun()


def step_raciones() -> None:
    draft_id = ensure_draft_id()
    existing = _load_existing_payload("raciones")

    st.subheader("Paso 2 de 5 — Configuración de raciones")

    with st.form("form_raciones"):
        racion_base = st.text_input(
            "Ración base",
            value=existing.get("racion_base", ""),
            help="Nombre de la ración que se utilizará como base de la carga.",
        )
        porcentaje_ms = st.number_input(
            "% MS estimado",
            min_value=0.0,
            max_value=100.0,
            value=float(existing.get("porcentaje_ms", 0.0)),
            step=0.1,
        )
        consumo_objetivo = st.number_input(
            "Consumo objetivo (kg/cab/día)",
            min_value=0.0,
            value=float(existing.get("consumo_objetivo", 0.0)),
            step=0.1,
        )
        notas = st.text_area(
            "Observaciones de la ración",
            value=existing.get("notas", ""),
        )

        col1, col2 = st.columns([1, 1])
        btn_guardar = col1.form_submit_button("💾 Guardar y continuar", type="primary")
        btn_volver = col2.form_submit_button("⬅️ Volver")

    if btn_volver:
        go_prev()
        st.rerun()

    if btn_guardar:
        if not racion_base.strip():
            st.error("Seleccioná o ingresá una ración base.")
            return

        payload = {
            "racion_base": racion_base.strip(),
            "porcentaje_ms": porcentaje_ms,
            "consumo_objetivo": consumo_objetivo,
            "notas": notas.strip(),
        }
        st.session_state["raciones"] = payload
        save_step_data(draft_id, "raciones", payload)
        go_next()
        st.rerun()


def step_mixer() -> None:
    draft_id = ensure_draft_id()
    existing = _load_existing_payload("mixer")

    st.subheader("Paso 3 de 5 — Mixer y carga total")

    mixers_df = load_mixers()
    mixer_options = [""]
    if not mixers_df.empty:
        mixer_options.extend(sorted(mixers_df["mixer_id"].dropna().astype(str)))
    selected_mixer = str(existing.get("mixer_sel", ""))
    selected_index = mixer_options.index(selected_mixer) if selected_mixer in mixer_options else 0

    with st.form("form_mixer"):
        mixer_sel = st.selectbox(
            "Mixer",
            options=mixer_options,
            index=selected_index,
        )
        capacidad = st.number_input(
            "Capacidad del mixer (kg)",
            min_value=0.0,
            value=float(existing.get("capacidad", 0.0)),
            step=10.0,
        )
        kilos_cargar = st.number_input(
            "Kilos totales a cargar",
            min_value=0.0,
            value=float(existing.get("kilos_cargar", 0.0)),
            step=10.0,
        )
        vueltas = st.number_input(
            "Número de vueltas",
            min_value=1,
            value=int(existing.get("vueltas", 1) or 1),
            step=1,
        )
        notas = st.text_area(
            "Observaciones del mixer",
            value=existing.get("notas", ""),
        )

        col1, col2 = st.columns([1, 1])
        btn_guardar = col1.form_submit_button("💾 Guardar y continuar", type="primary")
        btn_volver = col2.form_submit_button("⬅️ Volver")

    if btn_volver:
        go_prev()
        st.rerun()

    if btn_guardar:
        if not mixer_sel:
            st.error("Elegí un mixer para continuar.")
            return
        if capacidad <= 0:
            st.error("Ingresá la capacidad del mixer en kilogramos.")
            return

        payload = {
            "mixer_sel": mixer_sel,
            "capacidad": capacidad,
            "kilos_cargar": kilos_cargar,
            "vueltas": vueltas,
            "notas": notas.strip(),
        }
        st.session_state["mixer"] = payload
        save_step_data(draft_id, "mixer", payload)
        go_next()
        st.rerun()


def step_corrales() -> None:
    draft_id = ensure_draft_id()
    existing = _load_existing_payload("corrales")

    st.subheader("Paso 4 de 5 — Distribución por corrales")

    with st.form("form_corrales"):
        descripcion = st.text_area(
            "Detalle de distribución",
            value=existing.get("descripcion", ""),
            help="Anotá cómo se reparte la carga entre corrales o categorías.",
        )
        total_corrales = st.number_input(
            "Cantidad de corrales a abastecer",
            min_value=0,
            value=int(existing.get("total_corrales", 0)),
            step=1,
        )
        kilos_totales = st.number_input(
            "Kilos totales distribuidos",
            min_value=0.0,
            value=float(existing.get("kilos_totales", 0.0)),
            step=10.0,
        )

        col1, col2 = st.columns([1, 1])
        btn_guardar = col1.form_submit_button("💾 Guardar y continuar", type="primary")
        btn_volver = col2.form_submit_button("⬅️ Volver")

    if btn_volver:
        go_prev()
        st.rerun()

    if btn_guardar:
        if total_corrales <= 0:
            st.error("Indicá al menos un corral para la distribución.")
            return

        payload = {
            "descripcion": descripcion.strip(),
            "total_corrales": total_corrales,
            "kilos_totales": kilos_totales,
        }
        st.session_state["corrales"] = payload
        save_step_data(draft_id, "corrales", payload)
        go_next()
        st.rerun()


def step_resumen() -> None:
    draft_id = ensure_draft_id()

    datos_basicos = st.session_state.get("datos_basicos") or load_step_data(draft_id, "datos_basicos")
    raciones = st.session_state.get("raciones") or load_step_data(draft_id, "raciones")
    mixer_info = st.session_state.get("mixer") or load_step_data(draft_id, "mixer")
    corrales = st.session_state.get("corrales") or load_step_data(draft_id, "corrales")

    st.subheader("Paso 5 de 5 — Resumen y confirmación")

    st.write("### Datos básicos")
    st.json(datos_basicos)
    st.write("### Raciones")
    st.json(raciones)
    st.write("### Mixer")
    st.json(mixer_info)
    st.write("### Corrales")
    st.json(corrales)

    col1, col2, col3 = st.columns([1, 1, 1])

    if col1.button("⬅️ Volver"):
        go_prev()
        st.rerun()

    if col3.button("✅ Confirmar y guardar definitivo", type="primary"):
        registro_final = {
            "draft_id": draft_id,
            "datos_basicos": datos_basicos,
            "raciones": raciones,
            "mixer": mixer_info,
            "corrales": corrales,
            "confirmado_en": datetime.now().isoformat(),
        }
        guardar_registro_definitivo(registro_final)
        delete_draft(draft_id)
        st.success("Carga registrada correctamente.")
        reset_wizard_state()
        st.rerun()


def run_mixer_wizard() -> None:
    ensure_draft_id()
    step_name = WIZARD_STEPS[get_step_index()]

    if step_name == "datos_basicos":
        step_datos_basicos()
    elif step_name == "raciones":
        step_raciones()
    elif step_name == "mixer":
        step_mixer()
    elif step_name == "corrales":
        step_corrales()
    elif step_name == "resumen":
        step_resumen()


# ------------------------------------------------------------------------------
# 🧮 Mixer
# ------------------------------------------------------------------------------
with tab_mixer:
    with card("JM P-Feedlot v0.26 — Carga de ración (Wizard)", "Carga guiada paso a paso"):
        st.caption("🚧 Versión beta: guardá la carga paso a paso. Podés cerrar y volver, se recuperan los borradores.")
        run_mixer_wizard()

    with card("🧮 Cálculo de descarga de mixer (as-fed)", "Plan diario por tipo de ración"):
        st.markdown("### Planificación integral del mixer")

        fecha_plan = st.date_input("Fecha", datetime.today().date(), key="mixer_plan_fecha")

        base_df = load_base()
        mixers_df = load_mixers()
        raciones_recetas = build_raciones_from_recipes()
        plan_exports: list[pd.DataFrame] = []

        if base_df.empty:
            st.info("No hay corrales configurados en la base.")
        elif mixers_df.empty or mixers_df["mixer_id"].dropna().empty:
            st.warning("Definí mixers en la pestaña ⚙️ Parámetros para poder planificar la descarga.")
        elif not raciones_recetas:
            st.info("Definí recetas en la pestaña 🧾 Ajustes de raciones para calcular la carga base.")
        else:
            base_calc = enrich_and_calc_base(base_df)
            tipos_racion = [t for t in base_calc["tipo_racion"].dropna().astype(str).unique() if t.strip()]
            if not tipos_racion:
                st.info("Configura tipos de ración en la base para generar el plan del mixer.")
            else:
                mixer_options = mixers_df["mixer_id"].dropna().astype(str).tolist()
                recetas_por_nombre = {r["nombre"]: r for r in raciones_recetas}

                combos = base_calc.copy()
                combos["turnos"] = pd.to_numeric(combos.get("turnos", 0), errors="coerce").fillna(0).astype(int)
                combos = combos[combos["nombre_racion"].astype(str).str.strip() != ""]
                combo_turnos = {}
                for (tipo, nombre), sub in combos.groupby(["tipo_racion", "nombre_racion"]):
                    valid_turnos = sub["turnos"][sub["turnos"] > 0]
                    if not valid_turnos.empty:
                        combo_turnos[(str(tipo), str(nombre))] = int(valid_turnos.mode().iloc[0])

                tipo_options = sorted(tipos_racion)
                opciones_por_tipo = {}
                for tipo in tipo_options:
                    subset = base_calc.loc[base_calc["tipo_racion"] == tipo, "nombre_racion"].dropna().astype(str)
                    opciones_por_tipo[str(tipo)] = sorted([n for n in subset.unique() if str(n).strip()])

                st.markdown("#### Planificar descargas individuales (hasta 3 cargas)")
                st.caption("Cada descarga replica Mixer 1: carga base, turnos y descarga por corral.")

                plans_state = st.session_state.setdefault("mixer_plans", {})

                for slot in range(1, 4):
                    plan_key = f"descarga_{slot}"
                    if plan_key not in plans_state:
                        plans_state[plan_key] = None
                    st.markdown(f"##### Descarga {slot}")
                    col_mix, col_tipo, col_rac, col_turnos = st.columns((1.2, 1.1, 1.1, 0.8))
                    mixer_sel = col_mix.selectbox(
                        "Mixer",
                        [""] + mixer_options,
                        key=f"slot_mixer_{slot}",
                        help="Elegí el mixer que realizará esta descarga.",
                    )
                    tipo_sel = col_tipo.selectbox(
                        "Tipo de ración",
                        [""] + tipo_options,
                        key=f"slot_tipo_{slot}",
                    )
                    racion_opts = opciones_por_tipo.get(str(tipo_sel), [])
                    racion_sel = col_rac.selectbox(
                        "Ración",
                        [""] + racion_opts,
                        key=f"slot_racion_{slot}",
                    )
                    default_turnos = combo_turnos.get((str(tipo_sel), str(racion_sel)), 1)
                    turnos_val = col_turnos.number_input(
                        "Turnos",
                        min_value=1,
                        max_value=24,
                        value=default_turnos,
                        step=1,
                        key=f"slot_turnos_{slot}",
                    )

                    if not (mixer_sel and tipo_sel and racion_sel):
                        st.caption("Seleccioná mixer, tipo y ración para generar el plan de esta descarga.")
                        plans_state[plan_key] = None
                        continue

                    subset = base_calc[
                        (base_calc["tipo_racion"] == tipo_sel)
                        & (base_calc["nombre_racion"] == racion_sel)
                    ].copy()
                    if subset.empty:
                        st.info("No hay corrales asignados a esa combinación.")
                        plans_state[plan_key] = None
                        continue

                    receta = recetas_por_nombre.get(str(racion_sel))
                    if not receta:
                        st.warning("No se encontró una receta para esta ración. Definila en 🧾 Ajustes de raciones.")
                        plans_state[plan_key] = None
                        continue

                    subset["kg_turno_asfed_calc"] = pd.to_numeric(
                        subset.get("kg_turno_asfed_calc", 0.0), errors="coerce"
                    ).fillna(0.0)
                    subset["nro_cab"] = normalize_animal_counts(
                        subset.get("nro_cab"), index=subset.index
                    )
                    if "categ" in subset.columns:
                        subset["categ"] = (
                            subset["categ"]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            .str.lower()
                        )
                    else:
                        subset["categ"] = ""

                    cap_turno = float(
                        mixers_df.loc[mixers_df["mixer_id"] == mixer_sel, "capacidad_kg"].fillna(0).max()
                    )

                    kg_turno_base = float(subset["kg_turno_asfed_calc"].sum())
                    if kg_turno_base <= 0 and cap_turno > 0:
                        kg_turno_base = cap_turno

                    ingredientes_objs: list[Ingredient] = []
                    for ing in receta.get("ingredientes", []):
                        dm_pct = float(pd.to_numeric(ing.get("MS", 100.0), errors="coerce") or 100.0)
                        food = Food(
                            name=str(ing.get("ORIGEN", "Ingrediente")),
                            em=float(pd.to_numeric(ing.get("EM", 0.0), errors="coerce") or 0.0),
                            pb=float(pd.to_numeric(ing.get("PB", 0.0), errors="coerce") or 0.0),
                            dm=dm_pct,
                        )
                        ingredientes_objs.append(
                            Ingredient(
                                food=food,
                                inclusion_pct=float(
                                    pd.to_numeric(ing.get("inclusion_pct", 0.0), errors="coerce") or 0.0
                                ),
                            )
                        )

                    carga_turno_inicial = mixer_kg_by_ingredient(ingredientes_objs, kg_turno_base)
                    carga_df = pd.DataFrame(
                        {
                            "Ingrediente": list(carga_turno_inicial.keys()),
                            "kg_totales_dia": [
                                val * float(turnos_val) for val in carga_turno_inicial.values()
                            ],
                        }
                    )

                    if carga_df.empty:
                        st.warning("No hay ingredientes con inclusión > 0 en la receta seleccionada.")
                        plans_state[plan_key] = None
                        continue

                    carga_df["kg_totales_dia"] = carga_df["kg_totales_dia"].round(1)
                    carga_df["kg_por_turno"] = (carga_df["kg_totales_dia"] / float(turnos_val)).round(1)

                    editor_df = st.data_editor(
                        carga_df,
                        hide_index=True,
                        key=f"carga_editor_{slot}_{racion_sel}",
                        num_rows="fixed",
                        column_config={
                            "Ingrediente": st.column_config.TextColumn("Ingrediente", disabled=True),
                            "kg_totales_dia": st.column_config.NumberColumn(
                                "kg totales * día", format="%.1f"
                            ),
                            "kg_por_turno": st.column_config.NumberColumn(
                                "kg * turno", format="%.1f", disabled=True
                            ),
                        },
                        use_container_width=True,
                    )

                    editor_df = editor_df.copy()
                    editor_df["kg_totales_dia"] = pd.to_numeric(
                        editor_df["kg_totales_dia"], errors="coerce"
                    ).fillna(0.0)
                    editor_df["kg_por_turno"] = (
                        editor_df["kg_totales_dia"] / float(turnos_val)
                    ).round(1)
                    editor_df["kg_totales_dia"] = editor_df["kg_totales_dia"].round(1)

                    total_turno = float(editor_df["kg_por_turno"].sum())
                    total_dia = float(editor_df["kg_totales_dia"].sum())

                    st.markdown(
                        f"**Totales** · Día: {total_dia:,.1f} kg · Turno: {total_turno:,.1f} kg"
                    )

                    if cap_turno > 0:
                        diff_pct = abs(total_turno - cap_turno) / cap_turno if cap_turno else 0.0
                        if diff_pct <= 0.005:
                            st.success(
                                f"{tipo_sel} / {racion_sel}: {total_turno:,.1f} kg por turno dentro de {cap_turno:,.0f} kg del mixer {mixer_sel}."
                            )
                        elif total_turno > cap_turno:
                            st.error(
                                f"{tipo_sel} / {racion_sel}: {total_turno:,.1f} kg por turno supera {cap_turno:,.0f} kg del mixer {mixer_sel}."
                            )
                        else:
                            st.warning(
                                f"El total calculado ({total_turno:,.1f} kg) difiere del total del mixer ({cap_turno:,.1f} kg)."
                            )

                    if cap_turno > 0:
                        resumen_capacidad = (
                            f"{total_turno:,.1f} kg por turno dentro de {cap_turno:,.1f} kg del mixer {mixer_sel}."
                        )
                    else:
                        resumen_capacidad = (
                            f"{total_turno:,.1f} kg por turno — definí capacidad para {mixer_sel} en ⚙️ Parámetros."
                        )
                    st.caption(f"{tipo_sel} / {racion_sel}: {resumen_capacidad}")

                    st.caption(
                        f"Turnos programados: {int(turnos_val)} — Total diario: {total_dia:,.1f} kg"
                    )

                    peso_referencia = subset["kg_turno_asfed_calc"]
                    if peso_referencia.sum() <= 0:
                        peso_referencia = subset["nro_cab"].astype(float)
                    if peso_referencia.sum() <= 0:
                        peso_referencia = pd.Series([1.0] * len(subset), index=subset.index)

                    total_peso = float(peso_referencia.sum()) or 1.0
                    subset["kg_corral_turno"] = (total_turno * (peso_referencia / total_peso)).round(1)
                    subset["kg_corral_dia"] = (subset["kg_corral_turno"] * float(turnos_val)).round(1)

                    def _categoria_val(row, prefijos):
                        cat = str(row.get("categ", "")).strip().lower()
                        for pref in prefijos:
                            if cat.startswith(pref):
                                return int(row.get("nro_cab", 0))
                        return 0

                    corrales_df = pd.DataFrame(
                        {
                            "Corral": pd.to_numeric(
                                subset["nro_corral"], errors="coerce"
                            ).fillna(0).astype(int),
                            "kg/CORRAL": subset["kg_corral_turno"],
                            "kg totales * día": subset["kg_corral_dia"],
                            "vaquillonas": subset.apply(
                                lambda r: _categoria_val(r, ("va",)), axis=1
                            ),
                            "novillos": subset.apply(
                                lambda r: _categoria_val(r, ("nov",)), axis=1
                            ),
                        }
                    )

                    resumen = pd.DataFrame(
                        [
                            {
                                "Corral": "Total",
                                "kg/CORRAL": round(corrales_df["kg/CORRAL"].sum(), 1),
                                "kg totales * día": round(
                                    corrales_df["kg totales * día"].sum(), 1
                                ),
                                "vaquillonas": int(corrales_df["vaquillonas"].sum()),
                                "novillos": int(corrales_df["novillos"].sum()),
                            }
                        ]
                    )
                    corrales_show = pd.concat([corrales_df, resumen], ignore_index=True)
                    st.dataframe(corrales_show, use_container_width=True, hide_index=True)

                    subset_animals = subset.copy()
                    subset_animals["nro_cab"] = normalize_animal_counts(
                        subset_animals.get("nro_cab"), index=subset_animals.index
                    )
                    subset_animals["categ"] = (
                        subset_animals.get("categ", "")
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )
                    hd_total = int(subset_animals["nro_cab"].sum())
                    va_slot = int(
                        subset_animals.loc[
                            subset_animals["categ"].str.startswith("va"), "nro_cab"
                        ].sum()
                    )
                    nov_slot = int(
                        subset_animals.loc[
                            subset_animals["categ"].str.startswith("nov"), "nro_cab"
                        ].sum()
                    )
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.metric("Animales en descarga", f"{hd_total:,}")
                    cc2.metric("Vaquillonas", f"{va_slot:,}")
                    cc3.metric("Novillos", f"{nov_slot:,}")

                    st.caption(
                        f"Turnos programados: {int(turnos_val)} — Total mixer: {total_turno * float(turnos_val):,.1f} kg"
                    )

                    fecha_str = (
                        fecha_plan.strftime("%Y-%m-%d")
                        if hasattr(fecha_plan, "strftime")
                        else str(fecha_plan)
                    )

                    plan_turnos_rows: list[dict[str, Any]] = []
                    for turno_idx in range(1, int(turnos_val) + 1):
                        for _, corral_row in subset.iterrows():
                            corral_val = pd.to_numeric(
                                corral_row.get("nro_corral", 0), errors="coerce"
                            )
                            if pd.isna(corral_val):
                                corral_val = 0
                            cab_val = pd.to_numeric(
                                corral_row.get("nro_cab", 0), errors="coerce"
                            )
                            if pd.isna(cab_val):
                                cab_val = 0
                            plan_turnos_rows.append(
                                {
                                    "descarga": plan_key,
                                    "turno": int(turno_idx),
                                    "mixer_id": mixer_sel,
                                    "tipo_racion": tipo_sel,
                                    "racion": racion_sel,
                                    "nro_corral": int(corral_val),
                                    "kg_por_turno": float(
                                        corral_row.get("kg_corral_turno", 0.0)
                                    ),
                                    "kg_totales_dia": float(
                                        corral_row.get("kg_corral_dia", 0.0)
                                    ),
                                    "cabezas": int(cab_val),
                                    "cat": str(corral_row.get("categ", "")),
                                }
                            )

                    plan_turnos = pd.DataFrame(plan_turnos_rows)
                    plans_state[plan_key] = plan_turnos if not plan_turnos.empty else None

                    export_rows = []
                    for _, row in editor_df.iterrows():
                        export_rows.append(
                            {
                                "fecha": fecha_str,
                                "mixer_id": mixer_sel,
                                "tipo_racion": tipo_sel,
                                "racion": racion_sel,
                                "turno": int(turnos_val),
                                "ingrediente": row["Ingrediente"],
                                "kg_turno": float(row["kg_por_turno"]),
                                "kg_total_dia": float(row["kg_totales_dia"]),
                                "corral": "",
                                "kg_corral": 0.0,
                                "categoria": "",
                            }
                        )

                    for _, row in subset.iterrows():
                        export_rows.append(
                            {
                                "fecha": fecha_str,
                                "mixer_id": mixer_sel,
                                "tipo_racion": tipo_sel,
                                "racion": racion_sel,
                                "turno": int(turnos_val),
                                "ingrediente": "",
                                "kg_turno": float(row["kg_corral_turno"]),
                                "kg_total_dia": float(row["kg_corral_dia"]),
                                "corral": int(row.get("nro_corral", 0)),
                                "kg_corral": float(row["kg_corral_turno"]),
                                "categoria": str(row.get("categ", "")),
                            }
                        )

                    export_df = pd.DataFrame(export_rows)
                    plan_exports.append(export_df)

                    file_name = (
                        f"plan_mixer{slot}_{racion_sel}_{fecha_str}.csv".replace(" ", "_")
                    )
                    download_plan = st.download_button(
                        f"⬇️ Exportar plan (Descarga {slot})",
                        data=export_df.to_csv(index=False).encode("utf-8"),
                        file_name=file_name,
                        mime="text/csv",
                        key=f"download_plan_{slot}",
                    )
                    if download_plan:
                        activity_log_event(
                            "exportacion",
                            f"mixer_descarga={slot} archivo={file_name}",
                            trace_prefix="EXP-",
                        )

        st.markdown("---")
        if st.button(
            "🔄 Actualizar Mixer (generar backup de simulación)",
            type="primary",
        ):
            plans = {
                name: df
                for name, df in st.session_state.get("mixer_plans", {}).items()
                if isinstance(df, pd.DataFrame) and not df.empty
            }
            ok, msg = save_mixer_simulation_snapshot(
                username=username,
                plans_dict=plans,
                version_app=APP_VERSION,
                comment="backup manual mixer",
            )
            if ok:
                metrics_increment_simulation(username)
                trace_id = activity_log_event(
                    "simulacion",
                    f"descargas={len(plans)} filas={sum(len(df) for df in plans.values())}",
                    trace_prefix="SIM-",
                )
                backup_result = backup_flow()
                success_msg = f"{msg}"
                if trace_id:
                    success_msg = f"{msg} · ID: {trace_id}"
                st.success(success_msg)
                st.toast(
                    "Backup de simulación registrado en mixer_sim_log.csv",
                    icon="🗂️",
                )
                upload_info = backup_result.get("upload")
                if upload_info == {"status": "skip"}:
                    st.caption("Backup local generado (sin subir a GitHub).")
                elif "upload_error" in backup_result:
                    st.warning(
                        "Backup local OK, error al subir a GitHub (revisar GH_TOKEN/GITHUB_REPO)."
                    )
                else:
                    st.caption("Backup local + subido a GitHub ✔️")
            else:
                st.warning(msg)

        with st.expander("📚 Historial de simulaciones del mixer"):
            if MIXER_SIM_LOG.exists():
                hist = pd.read_csv(MIXER_SIM_LOG, encoding="utf-8-sig")
                st.dataframe(
                    hist.sort_values("ts", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Todavía no hay backups de simulación.")

        if plan_exports:
            consolidado = pd.concat(plan_exports, ignore_index=True)
            fecha_str = fecha_plan.strftime("%Y-%m-%d") if hasattr(fecha_plan, "strftime") else str(fecha_plan)
            st.markdown("---")
            file_consolidado = f"plan_mixers_consolidado_{fecha_str}.csv"
            download_consolidado = st.download_button(
                "⬇️ Exportar plan consolidado (CSV)",
                data=consolidado.to_csv(index=False).encode("utf-8"),
                file_name=file_consolidado,
                mime="text/csv",
                type="primary",
                key="download_plan_consolidado",
            )
            if download_consolidado:
                activity_log_event(
                    "exportacion",
                    f"mixer_consolidado archivo={file_consolidado}",
                    trace_prefix="EXP-",
                )

# ------------------------------------------------------------------------------
# ⬇️ Exportar
# ------------------------------------------------------------------------------
with tab_export:
    with card(
        "⬇️ Exportar datos y simulaciones",
        "Descargá todas las bases en un ZIP (ámbito de tu usuario)",
    ):
        names = [
            "alimentos.csv",
            "raciones_base.csv",
            "mixers.csv",
            "pesos.csv",
            "raciones_catalog.csv",
            "raciones_recipes.csv",
            "requerimientos_energeticos.csv",
            "requerimiento_proteico.csv",
            "audit_log.csv",
            "raciones_log.csv",
            "raciones_dadas.csv",
            "activity_log.csv",
            "mixer_sim_log.csv",
        ]
        files_to_zip = [USER_DIR / fname for fname in names if (USER_DIR / fname).exists()]

        if files_to_zip:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                manifest = {
                    "user": username,
                    "exported_at": datetime.now().isoformat(timespec="seconds"),
                    "files": [f.name for f in files_to_zip],
                }
                zf.writestr("manifest.json", pd.Series(manifest).to_json())

                for f in files_to_zip:
                    zf.write(f, arcname=f.name)
            buffer.seek(0)
            ts = datetime.now().strftime("%Y%m%d-%H%M")
            file_zip = f"simulaciones_{username}_{ts}.zip"
            download_zip = st.download_button(
                "⬇️ Descargar ZIP (todas las bases)",
                data=buffer,
                file_name=file_zip,
                mime="application/zip",
                type="primary",
            )
            if download_zip:
                activity_log_event(
                    "exportacion",
                    f"zip_bases archivos={len(files_to_zip)}",
                    trace_prefix="EXP-",
                )
        else:
            st.info("No se encontraron archivos en tu carpeta de usuario para exportar.")

        if st.button("⬇️ Exportar métricas (JSON)", key="export_metrics_button"):
            metrics_payload = _load_metrics()
            download_metrics = st.download_button(
                "Descargar métricas",
                data=json.dumps(metrics_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="metrics.json",
                mime="application/json",
                key="download_metrics_json",
            )
            if download_metrics:
                activity_log_event(
                    "exportacion",
                    "metrics.json",
                    trace_prefix="EXP-",
                )

        st.markdown("---")
        st.markdown("### 🧱 Integraciones ERP y Compras/Stock")
        erp_api_pref = str(_prefs_get("integrations", "erp_api_url", "")).strip()
        compras_api_pref = str(_prefs_get("integrations", "compras_api_url", "")).strip()
        dashboard_url_pref = str(_prefs_get("integrations", "dashboard_url", "")).strip()
        whatsapp_pref = str(_prefs_get("integrations", "whatsapp_number", "+54 9 387 407 3236")).strip()

        erp_api_input = st.text_input(
            "📡 Endpoint API ERP (POST JSON)",
            value=erp_api_pref,
            key="erp_api_url_input",
            help="URL para enviar el resumen diario de raciones (formato JSON).",
        ).strip()
        if erp_api_input != erp_api_pref:
            _prefs_set("integrations", "erp_api_url", erp_api_input)
            erp_api_pref = erp_api_input

        compras_api_input = st.text_input(
            "🧾 Endpoint Compras/Stock (GET costos)",
            value=compras_api_pref,
            key="compras_api_url_input",
            help="Endpoint que devuelve precios de alimentos en JSON.",
        ).strip()
        if compras_api_input != compras_api_pref:
            _prefs_set("integrations", "compras_api_url", compras_api_input)
            compras_api_pref = compras_api_input

        dashboard_input = st.text_input(
            "📊 URL Dashboard Physis", value=dashboard_url_pref, key="dashboard_url_input"
        ).strip()
        if dashboard_input != dashboard_url_pref:
            _prefs_set("integrations", "dashboard_url", dashboard_input)
            dashboard_url_pref = dashboard_input

        whatsapp_input = st.text_input(
            "💬 WhatsApp soporte", value=whatsapp_pref, key="whatsapp_number_input"
        ).strip()
        if whatsapp_input != whatsapp_pref:
            _prefs_set("integrations", "whatsapp_number", whatsapp_input)
            whatsapp_pref = whatsapp_input

        payload = build_erp_payload()
        payload_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

        with st.expander("Ver payload ERP", expanded=False):
            st.json(payload)

        st.caption(
            f"Registros exportables: {payload.get('total_registros', 0)} · Consumo diario: {payload.get('consumo_kg_dia', 0):,.2f} kg"
        )

        export_filename = f"erp_payload_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        st.download_button(
            "⬇️ Descargar JSON ERP",
            data=payload_bytes,
            file_name=export_filename,
            mime="application/json",
            key="download_erp_payload",
        )

        action_col1, action_col2 = st.columns(2)
        if action_col1.button(
            "📡 Enviar a ERP (REST)",
            key="send_payload_erp",
            disabled=not erp_api_pref,
        ):
            ok, msg = send_payload_to_erp(erp_api_pref, payload)
            if ok:
                st.success("Payload enviado correctamente al ERP.")
                activity_log_event(
                    "integracion",
                    f"erp_post registros={payload.get('total_registros', 0)}",
                    trace_prefix="ERP-",
                )
            else:
                st.error(f"No se pudo enviar al ERP: {msg}")
                activity_log_event(
                    "integracion_error",
                    f"erp_post error={msg}",
                    trace_prefix="ERP-",
                )

        if action_col2.button(
            "🔄 Sincronizar costos (Compras/Stock)",
            key="sync_compras_stock",
            disabled=not compras_api_pref,
        ):
            ok, msg, updated = sync_food_costs_from_api(compras_api_pref)
            if ok:
                st.success(msg)
                activity_log_event(
                    "integracion",
                    f"compras_sync actualizados={updated}",
                    trace_prefix="CMP-",
                )
                st.toast("Catálogo de costos actualizado.", icon="🧾")
                rerun_with_cache_reset()
            else:
                st.warning(msg)
                activity_log_event(
                    "integracion_error",
                    f"compras_sync error={msg}",
                    trace_prefix="CMP-",
                )

        st.markdown("---")
        st.markdown("### 📤 Importar ZIP (restaurar backup)")
        uploaded_zip = st.file_uploader(
            "Seleccioná un ZIP exportado previamente",
            type=["zip"],
            key="import_zip_backup",
        )
        if uploaded_zip is not None and validate_upload_size(uploaded_zip, label="Backup ZIP"):
            try:
                uploaded_zip.seek(0)
                with zipfile.ZipFile(uploaded_zip) as zf:
                    restored: list[str] = []
                    allowed = set(names)
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        safe_name = Path(info.filename).name
                        if safe_name not in allowed:
                            continue
                        target = USER_DIR / safe_name
                        data = zf.read(info)
                        target.write_bytes(data)
                        restored.append(safe_name)
                        backup_user_file(target, f"Importar ZIP {uploaded_zip.name} -> {safe_name}")
            except zipfile.BadZipFile:
                st.error("El archivo no es un ZIP válido.")
                audit_log_append(
                    "import_zip_error",
                    f"ZIP inválido: {uploaded_zip.name}",
                    status="error",
                )
            except Exception as exc:
                st.error(f"No se pudo procesar el ZIP: {exc}")
                audit_log_append(
                    "import_zip_error",
                    f"Error restaurando {uploaded_zip.name}",
                    status="error",
                    meta={"error": str(exc)},
                )
            else:
                if restored:
                    st.success(
                        "Backup restaurado. Se actualizaron: " + ", ".join(sorted(restored))
                    )
                    st.toast("Datos restaurados desde ZIP.", icon="📦")
                    audit_log_append(
                        "import_zip",
                        f"ZIP {uploaded_zip.name} restaurado",
                        meta={"files": restored},
                    )
                    activity_log_event(
                        "importacion",
                        f"zip={uploaded_zip.name} archivos={len(restored)}",
                        trace_prefix="IMP-",
                    )
                    rerun_with_cache_reset()
                else:
                    st.warning("No se encontraron archivos reconocidos en el ZIP.")
                    audit_log_append(
                        "import_zip_empty",
                        f"ZIP {uploaded_zip.name} sin archivos válidos",
                        status="warning",
                    )

# ------------------------------------------------------------------------------
# ⚙️ Parámetros
# ------------------------------------------------------------------------------
with tab_parametros:
    with card("⚙️ Parámetros técnicos", "Alimentos, Mixers, PV, Requerimientos"):
        config_summary = pd.DataFrame([
            {"Archivo": "alimentos.csv", "Descripción": "Catálogo de alimentos"},
            {"Archivo": "raciones_base.csv", "Descripción": "Asignación de corrales y raciones"},
            {"Archivo": "mixers.csv", "Descripción": "Capacidad de mixers"},
            {"Archivo": "pesos.csv", "Descripción": "Lista de PV disponibles"},
            {"Archivo": "raciones_catalog.csv", "Descripción": "Catálogo de raciones"},
            {"Archivo": "raciones_recipes.csv", "Descripción": "Recetas por ración"},
            {"Archivo": "requerimientos_energeticos.csv", "Descripción": "Requerimientos energéticos"},
            {"Archivo": "requerimiento_proteico.csv", "Descripción": "Requerimientos proteicos"},
        ])
        st.markdown("### Tablas de configuración disponibles")
        st.dataframe(config_summary, hide_index=True, use_container_width=True)

        st.markdown("### Catálogo de alimentos")
        alim_df = load_alimentos().copy()
        for col in show_cols:
            if col not in alim_df.columns:
                alim_df[col] = None
        alim_df = alim_df[show_cols]
        grid_alim_p = st.data_editor(
            alim_df,
            column_config=column_cfg,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="param_alimentos",
        )
        c1, c2 = st.columns(2)
        if c1.button("💾 Guardar alimentos (parámetros)", type="primary"):
            edited = grid_alim_p.copy()
            if len(edited) > 30:
                st.error("El catálogo admite máximo 30 alimentos. Recortá filas antes de guardar.")
            else:
                base = load_alimentos().copy()
                base = base.merge(
                    edited[["ORIGEN", "MS", "COEF ATC", "$/KG"]],
                    on="ORIGEN",
                    how="left",
                    suffixes=("", "_NEW"),
                )
                for col in ["MS", "COEF ATC", "$/KG"]:
                    new_col = f"{col}_NEW"
                    if new_col in base.columns:
                        base[col] = base[new_col].where(base[new_col].notna(), base[col])
                        base.drop(columns=[new_col], inplace=True)
                base = base.head(30)
                save_alimentos(base)
                st.success("Alimentos guardados.")
                st.toast("Alimentos actualizados.", icon="🧾")
                rerun_with_cache_reset()

        st.markdown("---")
        st.markdown("### Mixers (capacidad)")
        mix_df = load_mixers()
        grid_mix = st.data_editor(
            mix_df,
            column_config={
                "mixer_id": st.column_config.TextColumn("Mixer ID"),
                "capacidad_kg": st.column_config.NumberColumn("Capacidad (kg)", min_value=0, step=10)
            },
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="param_mixers",
        )
        if c2.button("💾 Guardar mixers", type="primary"):
            save_mixers(grid_mix); st.success("Mixers guardados."); st.toast("Mixers actualizados.", icon="🛠"); rerun_with_cache_reset()

        st.markdown("---")
        st.markdown("### PV (kg) — lista de opciones")
        pesos_df = load_pesos()
        grid_pes = st.data_editor(
            pesos_df,
            column_config={"peso_kg": st.column_config.NumberColumn("PV (kg)", min_value=1.0, max_value=2000.0, step=0.5)},
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="param_pesos",
        )
        p1, p2 = st.columns(2)
        if p1.button("💾 Guardar PV (kg)", type="primary"):
            save_pesos(grid_pes); st.success("Lista de PV guardada."); st.toast("PV actualizado.", icon="⚖️"); rerun_with_cache_reset()

        st.markdown("---")
        st.markdown("### Requerimientos proteicos (g PB/día)")
        reqprot_df = load_reqprot()
        st.dataframe(
            reqprot_df,
            column_config={
                "peso": st.column_config.NumberColumn("PV (kg)", min_value=0.0, max_value=2000.0, step=0.5),
                "cat": st.column_config.TextColumn("Categoría"),
                "ap": st.column_config.NumberColumn("AP (kg/día)", min_value=0.0, max_value=20.0, step=0.1),
                "req_proteico": st.column_config.NumberColumn("Req. proteico (g PB/día)", min_value=0.0, max_value=5000.0, step=1.0),
            },
            use_container_width=True,
            hide_index=True,
        )
        if st.button("🔄 Recargar requerimientos proteicos", key="reload_reqprot"):
            rerun_with_cache_reset()
        st.caption("Tabla fija: los valores se editan fuera de la aplicación.")

# ------------------------------------------------------------------------------
# 🌾 Presentación / Acerca de
# ------------------------------------------------------------------------------
with tab_presentacion:
    st.header("🌾 Physis Feedlot – Sistema Ganadero Integral")
    st.markdown(
        """
        Software modular para la **gestión ganadera y de alimentación**, diseñado para feedlots y empresas agropecuarias.
        """
    )
    st.markdown(f"Versión: **{APP_VERSION}**")

    tab_info_general, tab_info_tecnologias = st.tabs(
        ["ℹ️ Información general", "🧩 Tecnologías y lenguajes usados"]
    )

    with tab_info_general:
        active_display = f"{name} (@{username})"
        st.write(f"👤 Usuario activo: **{active_display}**")

        active_email = str(
            st.session_state.get("email") or user_email or "demo@physis.com.ar"
        )
        if active_email:
            st.write(f"✉️ Email registrado: **{active_email}**")

        snap = metrics_get_snapshot()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Visitas (hoy)", snap.get("today_visits", 0))
        m2.metric("Visitas (total)", snap.get("visits_total", 0))
        m3.metric("Simulaciones (hoy)", snap.get("today_simulations", 0))
        m4.metric("Simulaciones (total)", snap.get("simulations_total", 0))
        if snap.get("last_update"):
            st.caption(f"Última actualización: {snap['last_update']}")

        payload_dashboard = build_erp_payload()
        if payload_dashboard.get("total_registros", 0):
            st.markdown("### 📊 Indicadores de alimentación")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                "Raciones registradas",
                payload_dashboard.get("total_registros", 0),
            )
            col_b.metric(
                "Consumo diario (kg)",
                f"{payload_dashboard.get('consumo_kg_dia', 0):,.2f}",
            )
            col_c.metric(
                "Costo diario ($)",
                f"$ {payload_dashboard.get('costo_total_dia', 0):,.2f}",
            )
            gmd_prom = payload_dashboard.get("gmd_promedio_kg_dia")
            if gmd_prom is not None:
                st.caption(f"GMD promedio estimada: {gmd_prom:.2f} kg/día")
        else:
            st.caption("Aún no hay raciones registradas para el dashboard general.")

        dashboard_url_pref = str(_prefs_get("integrations", "dashboard_url", "")).strip()
        if dashboard_url_pref:
            st.markdown(
                f"🔗 [Abrir dashboard Physis Feedlot]({dashboard_url_pref})"
            )

        mp_slug = "".join(ch for ch in active_email if str(ch).isalnum()) or "physisfeedlot"
        mp_link = f"https://mpago.la/{mp_slug}"

        qr = qrcode.QRCode(box_size=10, border=2)
        qr.add_data(mp_link)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")
        st.image(buf.getvalue(), caption="📱 Escaneá para abonar o contactar")

        st.markdown(f"🔗 [Link directo de pago o contacto]({mp_link})")

        st.markdown("---")
        st.subheader("📧 Contacto y soporte")
        st.markdown(
            """
            - 📍 Salta, Argentina
            - 📞 +54 9 387 407 3236
            - ✉️ [jeanmarco333@outlook.com](mailto:jeanmarco333@outlook.com)
            - 🌐 [www.physis.com.ar](https://www.physis.com.ar)
            """
        )

        st.markdown("---")
        st.caption("© 2025 Sistema Ganadero Integral – Todos los derechos reservados.")

    with tab_info_tecnologias:
        st.subheader("🧩 Tecnologías y Lenguajes Utilizados")

        tech_cfg: dict[str, Any] = {}
        tech_cfg_path = Path("config/about_tech.yaml")

        if tech_cfg_path.exists():
            try:
                loaded_cfg = yaml.safe_load(
                    tech_cfg_path.read_text(encoding="utf-8")
                )
                if isinstance(loaded_cfg, dict):
                    tech_cfg = loaded_cfg
                else:
                    st.warning("El archivo de tecnologías no tiene el formato esperado.")
            except Exception as exc:
                st.error(f"No se pudo leer config/about_tech.yaml: {exc}")
        else:
            st.info("Aún no se cargó el archivo config/about_tech.yaml.")

        version_actual = str(tech_cfg.get("version", "s/d"))
        st.markdown(f"**Versión actual:** `{version_actual}`")

        categorias_raw = tech_cfg.get("categories", {}) if isinstance(tech_cfg, dict) else {}
        categorias: list[tuple[str, list[str]]] = []
        if isinstance(categorias_raw, dict):
            for categoria, items in categorias_raw.items():
                if isinstance(items, (list, tuple, set)):
                    valores = [str(item) for item in items if str(item).strip()]
                elif items not in (None, ""):
                    valores = [str(items)]
                else:
                    valores = []
                if valores:
                    categorias.append((str(categoria), valores))
        elif categorias_raw:
            st.warning("Las categorías de tecnologías no son válidas.")

        if categorias:
            for idx, (categoria, valores) in enumerate(categorias):
                st.markdown(f"### {categoria}")
                st.markdown("\n".join(f"- {valor}" for valor in valores))
                if idx < len(categorias) - 1:
                    st.markdown("---")
        else:
            st.info("No hay tecnologías cargadas para mostrar.")

        md_lines = [
            "# Tecnologías y Lenguajes",
            "",
            f"**Versión actual:** `{version_actual}`",
            "",
        ]
        for categoria, valores in categorias:
            md_lines.append(f"## {categoria}")
            md_lines.extend(f"- {valor}" for valor in valores)
            md_lines.append("")


        if tech_cfg_path.exists():
            try:
                loaded_cfg = yaml.safe_load(
                    tech_cfg_path.read_text(encoding="utf-8")
                )
                if isinstance(loaded_cfg, dict):
                    tech_cfg = loaded_cfg
                else:
                    st.warning("El archivo de tecnologías no tiene el formato esperado.")
            except Exception as exc:
                st.error(f"No se pudo leer config/about_tech.yaml: {exc}")
        else:
            st.info("Aún no se cargó el archivo config/about_tech.yaml.")

        version_actual = str(tech_cfg.get("version", "s/d"))
        st.markdown(f"**Versión actual:** `{version_actual}`")

        categorias_raw = tech_cfg.get("categories", {}) if isinstance(tech_cfg, dict) else {}
        categorias: list[tuple[str, list[str]]] = []
        if isinstance(categorias_raw, dict):
            for categoria, items in categorias_raw.items():
                if isinstance(items, (list, tuple, set)):
                    valores = [str(item) for item in items if str(item).strip()]
                elif items not in (None, ""):
                    valores = [str(items)]
                else:
                    valores = []
                if valores:
                    categorias.append((str(categoria), valores))
        elif categorias_raw:
            st.warning("Las categorías de tecnologías no son válidas.")

        if categorias:
            for idx, (categoria, valores) in enumerate(categorias):
                st.markdown(f"### {categoria}")
                st.markdown("\n".join(f"- {valor}" for valor in valores))
                if idx < len(categorias) - 1:
                    st.markdown("---")
        else:
            st.info("No hay tecnologías cargadas para mostrar.")

        md_lines = [
            "# Tecnologías y Lenguajes",
            "",
            f"**Versión actual:** `{version_actual}`",
            "",
        ]
        for categoria, valores in categorias:
            md_lines.append(f"## {categoria}")
            md_lines.extend(f"- {valor}" for valor in valores)
            md_lines.append("")

        md_content = "\n".join(md_lines).rstrip() + "\n"

        download_tecnologias = st.download_button(
            "⬇️ Exportar documentación técnica (Markdown)",
            data=md_content.encode("utf-8"),
            file_name="tecnologias_y_lenguajes.md",
            mime="text/markdown",
            disabled=not categorias,
        )
        if download_tecnologias:
            activity_log_event(
                "exportacion",
                "tecnologias_y_lenguajes.md",
                trace_prefix="EXP-",
            )

# ------------------------------------------------------------------------------
# 📐 Metodología y Cálculo (solo admin)
# ------------------------------------------------------------------------------
if USER_IS_ADMIN and tab_methodology is not None:
    with tab_methodology:
        st.subheader("📐 Metodología y Cálculo (solo admin)")
        md, meta = build_methodology_doc()
        st.markdown(md)

        download_metodo = st.download_button(
            "⬇️ Exportar metodología (Markdown)",
            data=md.encode("utf-8"),
            file_name="metodologia_y_calculo.md",
            mime="text/markdown",
            type="primary",
        )
        if download_metodo:
            activity_log_event(
                "exportacion",
                "metodologia_y_calculo.md",
                trace_prefix="EXP-",
            )

        download_meta = st.download_button(
            "⬇️ Exportar metadatos (JSON)",
            data=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="metodologia_meta.json",
            mime="application/json",
        )
        if download_meta:
            activity_log_event(
                "exportacion",
                "metodologia_meta.json",
                trace_prefix="EXP-",
            )

# ------------------------------------------------------------------------------
# 👤 Usuarios (Admin)
# ------------------------------------------------------------------------------
if tab_admin is not None:
    with tab_admin:
        if not USER_IS_ADMIN:
            st.warning("No tenés permisos para administrar usuarios.")
        else:
            with card("👤 Administración de usuarios", "Crear, editar y cambiar contraseñas"):
                import re

                def save_user_store(store_dict):
                    try:
                        AUTH_STORE.write_text(
                            yaml.safe_dump(store_dict, allow_unicode=True, sort_keys=False),
                            encoding="utf-8",
                        )
                    except OSError as exc:
                        st.error(
                            "No se pudo guardar el archivo editable de usuarios."
                            f" Verificá permisos de escritura. Detalle: {exc}"
                        )
                        return
                    except Exception as exc:
                        st.error(
                            "Error inesperado al guardar el archivo editable de usuarios."
                            f" Detalle: {exc}"
                        )
                        return

                    st.success("Cambios guardados.")
                    st.toast("Usuarios actualizados.", icon="✅")
                    st.rerun()

                # Cargar estado actual del store
                store = load_user_store()
                users = (store.get("credentials") or {}).get("usernames", {}) or {}

                st.markdown("### Usuarios existentes")
                if users:
                    df_users = pd.DataFrame(
                        [
                            {"usuario": u, "nombre": v.get("name", ""), "email": v.get("email", "")}
                            for u, v in sorted(users.items())
                        ]
                    )
                    st.dataframe(df_users, use_container_width=True, hide_index=True)
                else:
                    st.info("No hay usuarios en el store editable (data/auth/users.yaml).")

                st.markdown("---")
                st.markdown("### Agregar / Editar usuario")
                col_u1, col_u2 = st.columns(2)
                new_user = col_u1.text_input("Usuario", placeholder="ej: juan").strip()
                new_name = col_u2.text_input("Nombre visible", placeholder="ej: Juan Pérez").strip()
                new_email = st.text_input("Email", placeholder="juan@example.com").strip()

                col_p1, col_p2 = st.columns(2)
                pw = col_p1.text_input(
                    "Contraseña (dejar vacío para no cambiar si el usuario ya existe)",
                    type="password",
                )
                pw2 = col_p2.text_input("Repetir contraseña", type="password")

                ccol1, ccol2, ccol3 = st.columns(3)
                create_or_update = ccol1.button("💾 Crear / Actualizar", type="primary")
                delete_btn = ccol2.button("🗑️ Eliminar usuario", type="secondary", disabled=(not new_user))
                reset_cookie = ccol3.checkbox(
                    "Rotar cookie (forzar re-login)",
                    value=False,
                    help="Se incrementará el sufijo del nombre de cookie.",
                )

                if create_or_update:
                    if not new_user:
                        st.error("Ingresá un nombre de usuario."); st.stop()
                    if not re.match(r"^[a-zA-Z0-9_.-]{3,}$", new_user):
                        st.error("Usuario inválido (usa letras/números/._-, mínimo 3)."); st.stop()
                    if not new_name:
                        st.error("Ingresá un nombre visible."); st.stop()
                    if new_email and "@" not in new_email:
                        st.error("Email inválido."); st.stop()

                    entry = users.get(new_user, {}).copy()
                    entry["name"] = new_name
                    if new_email:
                        entry["email"] = new_email

                    if pw or pw2:
                        if pw != pw2:
                            st.error("Las contraseñas no coinciden."); st.stop()
                        hashed = bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
                        entry["password"] = hashed

                    users[new_user] = entry
                    store.setdefault("credentials", {})["usernames"] = users

                    if reset_cookie:
                        base_cookie = CFG.get("cookie", {}).get("name", "gefeed_cookie")
                        import re as _re

                        m = _re.match(r"^(.*)_v(\d+)$", base_cookie)
                        if m:
                            base, n = m.group(1), int(m.group(2))
                            new_cookie = f"{base}_v{n+1}"
                        else:
                            new_cookie = f"{base_cookie}_v2"
                        store.setdefault("cookie", {})["name"] = new_cookie
                        st.info(f"Cookie rotada a: {new_cookie}")

                    save_user_store(store)

                if delete_btn:
                    if new_user in users:
                        if is_admin(new_user):
                            st.error("No podés eliminar un usuario admin definido en la configuración.")
                        else:
                            users.pop(new_user, None)
                            store.setdefault("credentials", {})["usernames"] = users
                            save_user_store(store)
                    else:
                        st.warning("Ese usuario no existe en el store.")

# ------------------------------------------------------------------------------
# 🧾 Ajustes de raciones (catálogo + recetas)
# ------------------------------------------------------------------------------
with tab_raciones:
    with card("🧾 Ajustes de raciones", "Editá ingredientes y porcentajes (máx. 6, suma 100% MS)"):
        cat = load_catalog()
        rec = load_recipes()
        alimentos_df = load_alimentos()
        alimentos_norm = _normalize_columns(alimentos_df.copy())
        opciones_ingred = [""] + sorted(alimentos_norm["ORIGEN"].dropna().astype(str).unique().tolist())

        st.markdown("### Catálogo de raciones")
        cat_display = cat.copy()
        cat_display["cv_ms_kg"] = (
            pd.to_numeric(cat_display.get("pv_kg", 0.0), errors="coerce").fillna(0.0)
            * pd.to_numeric(cat_display.get("cv_pct", 0.0), errors="coerce").fillna(0.0)
            / 100.0
        ).round(2)
        cols_order = ["id","nombre","etapa","sexo","pv_kg","cv_pct","corral_comparacion","cv_ms_kg"]
        existing_cols = [c for c in cols_order if c in cat_display.columns]
        cat_display = cat_display.reindex(columns=existing_cols)

        column_config = {}
        if "id" in cat_display.columns:
            column_config["id"] = st.column_config.NumberColumn("ID", min_value=0, step=1)
        if "nombre" in cat_display.columns:
            column_config["nombre"] = st.column_config.TextColumn("Nombre")
        if "etapa" in cat_display.columns:
            column_config["etapa"] = st.column_config.TextColumn("Etapa")
        if "sexo" in cat_display.columns:
            column_config["sexo"] = st.column_config.TextColumn("Sexo")
        if "pv_kg" in cat_display.columns:
            column_config["pv_kg"] = st.column_config.NumberColumn("PV (kg)", min_value=0.0, max_value=1000.0, step=0.5)
        if "cv_pct" in cat_display.columns:
            column_config["cv_pct"] = st.column_config.NumberColumn("CV (%)", min_value=0.0, max_value=20.0, step=0.1)
        if "corral_comparacion" in cat_display.columns:
            column_config["corral_comparacion"] = st.column_config.NumberColumn(
                "Corral de comparación", min_value=0.0, max_value=1000.0, step=1.0
            )
        if "cv_ms_kg" in cat_display.columns:
            column_config["cv_ms_kg"] = st.column_config.NumberColumn(
                "CV (kg MS)", disabled=True, format="%.2f"
            )

        grid_cat = st.data_editor(
            cat_display,
            column_config=column_config,
            column_order=existing_cols,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="grid_rac_catalog",
        )
        cat_preview = grid_cat.copy()
        cat_preview["cv_ms_kg"] = (
            pd.to_numeric(cat_preview.get("pv_kg", 0.0), errors="coerce").fillna(0.0)
            * pd.to_numeric(cat_preview.get("cv_pct", 0.0), errors="coerce").fillna(0.0)
            / 100.0
        ).round(2)
        st.caption("Vista previa de PV × CV% (solo lectura)")
        st.dataframe(
            cat_preview[[c for c in ["id","nombre","cv_ms_kg"] if c in cat_preview.columns]],
            hide_index=True,
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        if c1.button("💾 Guardar catálogo", type="primary"):
            if cat_preview.get("id", pd.Series(dtype=int)).duplicated().any():
                st.error("IDs duplicados en el catálogo.")
            else:
                out_catalog = cat_preview.drop(columns=["cv_ms_kg"], errors="ignore")
                save_catalog(out_catalog)
                st.success("Catálogo guardado.")
                st.toast("Raciones actualizadas.", icon="🧾")
                rerun_with_cache_reset()

        st.markdown("---")
        st.markdown("### Receta por ración (máx. 6 ingredientes, 100% MS)")
        st.info("Las estimaciones de EM y PB son orientativas y no definen automáticamente la cantidad de cada alimento.")
        cat = load_catalog()
        if cat.empty:
            st.info("Primero cargá/guardá raciones en el catálogo.")
        else:
            pick = st.selectbox("Seleccioná la ración", cat["nombre"].tolist())
            rid  = int(cat.loc[cat["nombre"]==pick, "id"].iloc[0])
            try:
                racion_row = cat.loc[cat["id"] == rid].iloc[0]
            except Exception:
                racion_row = pd.Series(dtype=object)
            tipo_racion_val = str(racion_row.get("etapa", "")) if not racion_row.empty else ""
            categoria_val = str(racion_row.get("sexo", "")) if not racion_row.empty else ""

            rec = load_recipes()
            rec_r = rec[rec["id_racion"]==rid].copy()
            if rec_r.shape[0] < 6:
                faltan = 6 - rec_r.shape[0]
                rec_r = pd.concat([rec_r, pd.DataFrame({
                    "id_racion":[rid]*faltan,
                    "nombre_racion":[pick]*faltan,
                    "ingrediente":[""]*faltan,
                    "pct_ms":[0.0]*faltan
                })], ignore_index=True)

            pv_catalog = 0.0
            cv_catalog = 0.0
            if "pv_kg" in cat.columns:
                vals = pd.to_numeric(cat.loc[cat["id"] == rid, "pv_kg"], errors="coerce").dropna()
                if not vals.empty:
                    pv_catalog = float(vals.iloc[0])
            if "cv_pct" in cat.columns:
                vals = pd.to_numeric(cat.loc[cat["id"] == rid, "cv_pct"], errors="coerce").dropna()
                if not vals.empty:
                    cv_catalog = float(vals.iloc[0])

            corrales_df = load_base()
            corral_choices = ["(sin corral)"]
            corral_lookup: dict[str, pd.Series | None] = {"(sin corral)": None}
            if not corrales_df.empty:
                work = corrales_df.copy()
                work.columns = [str(c).strip().lower() for c in work.columns]
                if "nombre_racion" in work.columns:
                    work["nombre_racion"] = work["nombre_racion"].fillna("").astype(str)
                    subset = work[
                        work["nombre_racion"].str.strip().str.lower()
                        == pick.strip().lower()
                    ].copy()
                    if not subset.empty and "nro_corral" in subset.columns:
                        subset["nro_corral"] = pd.to_numeric(
                            subset["nro_corral"], errors="coerce"
                        ).fillna(0).astype(int)
                        subset = subset[subset["nro_corral"] > 0]
                        subset = subset.drop_duplicates("nro_corral", keep="last")
                        subset = subset.sort_values("nro_corral")
                        for _, crow in subset.iterrows():
                            nro = int(crow["nro_corral"])
                            cabezas = int(_num(crow.get("nro_cab", 0), 0.0))
                            label = f"Corral {nro}"
                            if cabezas > 0:
                                label += f" ({cabezas} cab)"
                            corral_choices.append(label)
                            corral_lookup[label] = crow

            corral_choice_key = f"ration_corral_{rid}"
            if (
                corral_choice_key not in st.session_state
                or st.session_state[corral_choice_key] not in corral_choices
            ):
                st.session_state[corral_choice_key] = corral_choices[0]

            pv_key = f"ration_pv_{rid}"
            cv_key = f"ration_cv_{rid}"
            prev_corral_key = f"{corral_choice_key}_last"

            params_cols = st.columns([2.2, 1.1, 1.1, 1.1])
            selected_corral_label = params_cols[0].selectbox(
                "Corral (opcional)",
                corral_choices,
                key=corral_choice_key,
            )
            selected_corral = corral_lookup.get(selected_corral_label)
            base_pv = pv_catalog
            base_cv = cv_catalog
            if selected_corral is not None:
                base_pv = _num(selected_corral.get("pv_kg"), pv_catalog)
                base_cv = _num(selected_corral.get("cv_pct"), cv_catalog)

            if pv_key not in st.session_state:
                st.session_state[pv_key] = base_pv
            if cv_key not in st.session_state:
                st.session_state[cv_key] = base_cv

            if st.session_state.get(prev_corral_key) != selected_corral_label:
                st.session_state[pv_key] = base_pv
                st.session_state[cv_key] = base_cv
                st.session_state[prev_corral_key] = selected_corral_label

            pv_input = params_cols[1].number_input(
                "PV (kg)",
                min_value=0.0,
                max_value=1500.0,
                value=float(st.session_state[pv_key]),
                step=1.0,
                key=pv_key,
            )
            cv_input = params_cols[2].number_input(
                "CV (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(st.session_state[cv_key]),
                step=0.1,
                key=cv_key,
            )
            pv_value = _num(pv_input, 0.0)
            cv_value = _num(cv_input, 0.0)
            consumo_ms = pv_value * (cv_value / 100.0)
            categoria_req = categoria_val
            if selected_corral is not None:
                categoria_req = str(selected_corral.get("categ", categoria_req) or categoria_req)
            ap_value = (
                _num_or_none(selected_corral.get("ap_preten"))
                if selected_corral is not None
                else None
            )
            em_req_val = compute_requirement_em(pv_value, ap_value, categoria_req) or 0.0
            pb_req_val = compute_requirement_pb(pv_value, ap_value, categoria_req) or 0.0
            params_cols[3].metric("CV MS (kg)", f"{consumo_ms:.2f}")

            rec_edit_cols = ["ingrediente", "pct_ms"]
            auto_recipe_key = f"auto_recipe_{rid}"
            auto_df_key = f"auto_ration_df_{rid}"
            auto_summary_key = f"auto_ration_summary_{rid}"

            st.markdown("### 🧠 Sugerir ración automática")
            col_auto_a, col_auto_b = st.columns(2)
            aplicar_atc = col_auto_a.toggle(
                "Aplicar COEF ATC al tal cual",
                value=False,
                key=f"auto_apply_atc_{rid}",
            )
            max_ing = col_auto_b.number_input(
                "Máx. ingredientes",
                min_value=2,
                max_value=6,
                value=6,
                step=1,
                key=f"auto_max_ing_{rid}",
            )

            w_em = st.slider("Peso EM", 0.0, 3.0, 1.0, 0.1, key=f"auto_w_em_{rid}")
            w_pb = st.slider("Peso PB", 0.0, 3.0, 1.0, 0.1, key=f"auto_w_pb_{rid}")
            w_cst = st.slider("Peso costo", 0.0, 1.0, 0.15, 0.05, key=f"auto_w_cost_{rid}")

            disponibles_opts = sorted([opt for opt in opciones_ingred if opt])
            disp = st.multiselect(
                "Alimentos disponibles",
                options=disponibles_opts,
                default=disponibles_opts,
                key=f"auto_disp_{rid}",
            )

            if st.button("🧠 Generar ración por aproximación", type="primary", key=f"auto_btn_{rid}"):
                if consumo_ms <= 0:
                    st.warning("Definí PV y CV mayores a 0 para calcular la ración automática.")
                elif not disp:
                    st.warning("Seleccioná al menos un alimento disponible para optimizar.")
                elif alimentos_df.empty:
                    st.warning("No hay catálogo de alimentos cargado.")
                else:
                    df_auto, res_auto = optimize_ration(
                        alimentos_df=alimentos_df,
                        disponibles=disp,
                        consumo_ms_dia=consumo_ms,
                        em_req_mcal_dia=float(em_req_val),
                        pb_req_g_dia=float(pb_req_val),
                        max_ingredientes=int(max_ing),
                        pesos_obj={"em": w_em, "pb": w_pb, "cost": w_cst},
                        aplicar_coef_atc=aplicar_atc,
                    )
                    if res_auto.get("status") != "ok" or df_auto.empty:
                        st.warning(res_auto.get("msg", "No se pudo optimizar la ración."))
                        st.session_state.pop(auto_df_key, None)
                        st.session_state.pop(auto_summary_key, None)
                    else:
                        tips = sugerencia_balance(
                            float(res_auto.get("em_calc", 0.0)),
                            float(em_req_val),
                            float(res_auto.get("pb_calc", 0.0)),
                            float(pb_req_val),
                        )
                        df_rec_out = df_auto[["ingrediente", "pct_ms"]].head(int(max_ing)).copy()
                        df_rec_out = df_rec_out.reset_index(drop=True)
                        while len(df_rec_out) < 6:
                            df_rec_out.loc[len(df_rec_out)] = {"ingrediente": "", "pct_ms": 0.0}
                        st.session_state[auto_recipe_key] = df_rec_out[rec_edit_cols].copy()
                        st.session_state[auto_df_key] = df_auto.copy()
                        cabezas = _num(selected_corral.get("nro_cab"), 0.0) if selected_corral is not None else 0.0
                        costo_total_val = float(res_auto.get("cost_total", 0.0))
                        em_calc_val = float(res_auto.get("em_calc", 0.0))
                        pb_calc_val = float(res_auto.get("pb_calc", 0.0))
                        costo_por_cabeza = (costo_total_val / cabezas) if cabezas > 0 else None
                        costo_por_mcal = (costo_total_val / em_calc_val) if em_calc_val > 0 else None
                        pb_kg = pb_calc_val / 1000.0 if pb_calc_val > 0 else 0.0
                        costo_por_kg_pb = (costo_total_val / pb_kg) if pb_kg > 0 else None
                        gmd_val = calculate_gmd(
                            peso_inicial=pv_value,
                            ap_kg_dia=ap_value,
                        )
                        st.session_state[auto_summary_key] = {
                            "resumen": res_auto,
                            "em_req": float(em_req_val),
                            "pb_req": float(pb_req_val),
                            "consumo_ms": consumo_ms,
                            "tips": tips,
                            "costo_por_cabeza": costo_por_cabeza,
                            "costo_por_mcal": costo_por_mcal,
                            "costo_por_kg_pb": costo_por_kg_pb,
                            "cabezas": cabezas,
                            "aplicado_atc": bool(aplicar_atc),
                            "categoria": categoria_req,
                            "pv_kg": pv_value,
                            "cv_pct": cv_value,
                            "ap_kg_dia": ap_value if ap_value is not None else 0.0,
                            "gmd_kg_dia": float(gmd_val) if isinstance(gmd_val, (int, float)) else None,
                        }
                        st.success("Ración sugerida generada.")

            auto_df = st.session_state.get(auto_df_key)
            auto_summary = st.session_state.get(auto_summary_key)
            if isinstance(auto_summary, dict) and isinstance(auto_summary.get("resumen"), dict):
                res_auto = auto_summary.get("resumen", {})
                em_calc_val = float(res_auto.get("em_calc", 0.0))
                pb_calc_val = float(res_auto.get("pb_calc", 0.0))
                costo_total_val = float(res_auto.get("cost_total", 0.0))
                asfed_total_val = float(res_auto.get("asfed_total_kg_dia", 0.0))
                em_req_val = float(auto_summary.get("em_req", 0.0))
                pb_req_val = float(auto_summary.get("pb_req", 0.0))
                consumo_ms_val = float(auto_summary.get("consumo_ms", consumo_ms))
                gmd_auto = calculate_gmd(
                    peso_inicial=auto_summary.get("pv_kg", pv_value),
                    ap_kg_dia=auto_summary.get("ap_kg_dia"),
                )
                if isinstance(auto_df, pd.DataFrame) and not auto_df.empty:
                    st.dataframe(auto_df, use_container_width=True, hide_index=True)
                st.info(
                    f"""**Resumen**
- EM calculada: **{em_calc_val:.2f} Mcal/día** (req {em_req_val:.2f})
- PB calculada: **{pb_calc_val:.0f} g/día** (req {pb_req_val:.0f})
- As-fed total: **{asfed_total_val:.2f} kg/día**
- Consumo MS objetivo: **{consumo_ms_val:.2f} kg/día**
- Costo total: **$ {costo_total_val:.2f}**"""
                )
                metric_cols = st.columns(5)
                metric_cols[0].metric("Costo total (día)", f"$ {costo_total_val:.2f}")
                costo_cabeza_val = auto_summary.get("costo_por_cabeza")
                metric_cols[1].metric(
                    "Costo por cabeza (día)",
                    f"$ {costo_cabeza_val:.2f}" if isinstance(costo_cabeza_val, (float, int)) and costo_cabeza_val is not None else "s/d",
                )
                costo_em_val = auto_summary.get("costo_por_mcal")
                metric_cols[2].metric(
                    "$ por Mcal EM",
                    f"$ {costo_em_val:.2f}" if isinstance(costo_em_val, (float, int)) and costo_em_val is not None else "s/d",
                )
                costo_pb_val = auto_summary.get("costo_por_kg_pb")
                metric_cols[3].metric(
                    "$ por kg PB",
                    f"$ {costo_pb_val:.2f}" if isinstance(costo_pb_val, (float, int)) and costo_pb_val is not None else "s/d",
                )
                metric_cols[4].metric(
                    "GMD estimada (kg/día)",
                    f"{gmd_auto:.2f}" if isinstance(gmd_auto, (int, float)) and gmd_auto is not None else "s/d",
                )
                for tip in auto_summary.get("tips", []):
                    st.write("• " + tip)

                if isinstance(auto_df, pd.DataFrame) and not auto_df.empty:
                    pdf_detail = auto_df.copy()
                    for col in ["ingrediente", "pct_ms", "ms_kg_dia", "asfed_kg_dia", "costo"]:
                        if col not in pdf_detail.columns:
                            pdf_detail[col] = 0.0
                    pdf_name = f"ficha_racion_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
                    pdf_path = user_path(pdf_name)
                    datos_pdf = {
                        "Ración": pick,
                        "PV (kg)": f"{auto_summary.get('pv_kg', pv_value):.0f}",
                        "CV (%)": f"{auto_summary.get('cv_pct', cv_value):.2f}",
                        "Consumo MS (kg/día)": f"{consumo_ms_val:.3f}",
                        "EM calculada (Mcal/d)": f"{em_calc_val:.2f}",
                        "EM requerida (Mcal/d)": f"{em_req_val:.2f}",
                        "PB calculada (g/d)": f"{pb_calc_val:.0f}",
                        "PB requerida (g/d)": f"{pb_req_val:.0f}",
                        "As-fed total (kg/día)": f"{asfed_total_val:.2f}",
                        "Costo total ($/día)": f"{costo_total_val:.2f}",
                    }
                    cabezas_pdf = auto_summary.get("cabezas", 0)
                    if isinstance(cabezas_pdf, (int, float)) and cabezas_pdf and cabezas_pdf > 0:
                        datos_pdf["Costo por cabeza ($/día)"] = f"{(costo_total_val / cabezas_pdf):.2f}"

                    generate_summary_pdf(
                        file_path=pdf_path,
                        logo_path="assets/logo.png",
                        titulo="JM P-Feedlot — Ficha de Ración",
                        datos=datos_pdf,
                        tabla_detalle=pdf_detail,
                        recomendaciones=list(auto_summary.get("tips", [])),
                    )

                    st.download_button(
                        "⬇️ Descargar PDF de la ración",
                        data=pdf_path.read_bytes(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        type="primary",
                        key=f"pdf_auto_{rid}",
                    )

                if MIXER_SIM_LOG.exists():
                    try:
                        logdf = pd.read_csv(MIXER_SIM_LOG, encoding="utf-8-sig")
                    except Exception as exc:
                        st.caption(f"No se pudo leer mixer_sim_log.csv ({exc}).")
                    else:
                        last = logdf.sort_values("ts", ascending=False).head(1)
                        if not last.empty:
                            def _last_val(col: str) -> float:
                                if col not in last.columns:
                                    return np.nan
                                series = pd.to_numeric(last[col], errors="coerce")
                                if series.empty:
                                    return np.nan
                                val = series.iloc[0]
                                return float(val) if not pd.isna(val) else np.nan

                            em_prev = _last_val("em_calc")
                            pb_prev = _last_val("pb_calc")
                            cost_prev = _last_val("cost_total")

                            def _pct(delta: float, base: float) -> float:
                                if base is None or np.isnan(base) or abs(base) < 1e-6:
                                    return np.nan
                                return (delta / base) * 100.0

                            pct_em = _pct(em_calc_val - em_prev, em_prev)
                            pct_pb = _pct(pb_calc_val - pb_prev, pb_prev)
                            pct_cost = _pct(costo_total_val - cost_prev, cost_prev)

                            def _fmt_pct(val: float) -> str:
                                return f"{val:+.1f}%" if not np.isnan(val) else "s/d"

                            st.caption("Comparación con la última simulación guardada:")
                            st.write(
                                f"- Δ EM: {_fmt_pct(pct_em)}  |  Δ PB: {_fmt_pct(pct_pb)}  |  Δ Costo: {_fmt_pct(pct_cost)}"
                            )
                        else:
                            st.caption("Aún no hay simulaciones registradas para comparar.")

            colcfg = {
                "ingrediente": st.column_config.SelectboxColumn("Ingrediente (ORIGEN)", options=opciones_ingred),
                "pct_ms": st.column_config.NumberColumn("% inclusión (MS)", min_value=0.0, max_value=100.0, step=0.1)
            }

            editor_source = st.session_state.get(auto_recipe_key)
            if isinstance(editor_source, pd.DataFrame) and not editor_source.empty:
                base_editor = editor_source.copy()
            else:
                base_editor = rec_r[rec_edit_cols].copy()

            grid_rec = st.data_editor(
                base_editor,
                column_config=colcfg,
                column_order=rec_edit_cols,
                use_container_width=True,
                hide_index=True,
                num_rows=6,
                key="grid_rac_recipe",
            )
            grid_rec = grid_rec.reset_index(drop=True).copy()
            if "ingrediente" not in grid_rec.columns:
                grid_rec["ingrediente"] = ""
            if "pct_ms" not in grid_rec.columns:
                grid_rec["pct_ms"] = 0.0
            st.session_state[auto_recipe_key] = grid_rec[rec_edit_cols].copy()

            total_pct = float(pd.to_numeric(grid_rec["pct_ms"], errors="coerce").fillna(0.0).sum())
            st.progress(min(int(total_pct), 100), text=f"Suma MS: {total_pct:.1f}%")
            ok100 = abs(total_pct-100) <= 0.5
            ingredientes_clean = grid_rec["ingrediente"].astype(str).str.strip()
            ingredientes_nonempty = ingredientes_clean[ingredientes_clean != ""]
            duplicados = ingredientes_nonempty.str.lower().duplicated().any()
            lookup_ms = alimentos_norm.copy()
            lookup_ms["ORIGEN"] = lookup_ms["ORIGEN"].str.strip().str.lower()
            ms_series = lookup_ms.set_index("ORIGEN")["MS"] if not lookup_ms.empty else pd.Series(dtype=float)
            ms_missing = False
            if not ingredientes_nonempty.empty:
                ms_vals = ingredientes_nonempty.str.lower().map(ms_series)
                ms_missing = ms_vals.isna().any()

            st.write("**Estado:**")
            chip("Suma 100% ±0,5", ok100)
            chip("Ingredientes únicos", not duplicados)
            chip("MS disponible", not ms_missing)

            if duplicados:
                st.warning("Hay ingredientes duplicados en la receta.")
            if ms_missing:
                st.warning("Al menos un ingrediente no tiene MS cargada en el catálogo de alimentos.")

            can_save = ok100 and not duplicados and not ms_missing

            if st.button("💾 Guardar receta de esta ración", type="primary", disabled=not can_save):
                out = grid_rec.copy()
                out["id_racion"] = rid; out["nombre_racion"] = pick
                out = out[out["ingrediente"].astype(str).str.strip()!=""]
                rec2 = load_recipes(); rec2 = rec2[rec2["id_racion"]!=rid]
                rec2 = pd.concat([rec2, out], ignore_index=True)
                save_recipes(rec2)
                st.success("Receta guardada.")
                st.toast("Receta 100% MS guardada.", icon="✅")
                rerun_with_cache_reset()

            calc_ready = ok100 and consumo_ms > 0 and not ingredientes_nonempty.empty
            if not ok100:
                st.warning("La ración debe sumar 100% MS (±0,5) para calcular el reparto diario.")
            if consumo_ms <= 0:
                st.info("Ingresá PV y CV mayores a 0 para estimar consumos y nutrientes.")

            if calc_ready:
                resultado = ration_split_from_pv_cv(grid_rec, alimentos_norm, pv_value, cv_value)
                detail_df = pd.DataFrame(resultado.get("detalle", []))
                categoria_req_manual = categoria_val
                if selected_corral is not None:
                    categoria_req_manual = str(
                        selected_corral.get("categ", categoria_req_manual) or categoria_req_manual
                    )
                ap_manual = _num_or_none(selected_corral.get("ap_preten")) if selected_corral is not None else None
                req_em_manual = compute_requirement_em(pv_value, ap_manual, categoria_req_manual) or 0.0
                req_pb_manual = compute_requirement_pb(pv_value, ap_manual, categoria_req_manual) or 0.0
                cabezas_manual = _num(selected_corral.get("nro_cab"), 0.0) if selected_corral is not None else 0.0
                tips_manual = sugerencia_balance(
                    float(resultado.get("EM_Mcal_dia", 0.0)),
                    float(req_em_manual),
                    float(resultado.get("PB_g_dia", 0.0)),
                    float(req_pb_manual),
                )
                gmd_manual = calculate_gmd(
                    peso_inicial=pv_value,
                    ap_kg_dia=ap_manual,
                )

                st.markdown("#### Métricas diarias de la ración")
                metrics_cols = st.columns(6)
                metrics_cols[0].metric("CV MS (kg)", f"{resultado['Consumo_MS_dia']:.2f}")
                metrics_cols[1].metric(
                    "Tal cual total (kg/día)", f"{resultado['asfed_total_kg_dia']:.2f}"
                )
                metrics_cols[2].metric("EM total (Mcal/día)", f"{resultado['EM_Mcal_dia']:.2f}")
                metrics_cols[3].metric("PB total (g/día)", f"{resultado['PB_g_dia']:.0f}")
                metrics_cols[4].metric("Costo/día", f"$ {resultado['costo_dia']:.2f}")
                metrics_cols[5].metric(
                    "GMD estimada (kg/día)",
                    f"{gmd_manual:.2f}" if isinstance(gmd_manual, (int, float)) and gmd_manual is not None else "s/d",
                )

                if tips_manual:
                    st.markdown("**Balance nutricional:**")
                    for tip in tips_manual:
                        st.write("• " + tip)

                if detail_df.empty:
                    st.info("Definí inclusiones (> 0% MS) para ver el detalle por ingrediente.")
                else:
                    def _alertas(row):
                        flags = []
                        if bool(row.get("missing_ms")):
                            flags.append("MS")
                        if bool(row.get("missing_em")):
                            flags.append("EM")
                        if bool(row.get("missing_pb")):
                            flags.append("PB")
                        if bool(row.get("missing_precio")):
                            flags.append("Precio")
                        return "⚠️ " + ", ".join(flags) if flags else ""

                    detail_df["Alertas"] = detail_df.apply(_alertas, axis=1)
                    display_df = detail_df[
                        [
                            "ingrediente",
                            "pct_ms",
                            "MS_kg",
                            "MS_%_alimento",
                            "asfed_kg",
                            "EM_Mcal",
                            "PB_g",
                            "precio",
                            "costo_dia",
                            "Alertas",
                        ]
                    ].rename(
                        columns={
                            "ingrediente": "Ingrediente",
                            "pct_ms": "%MS",
                            "MS_kg": "MS kg/día",
                            "MS_%_alimento": "MS % alimento",
                            "asfed_kg": "Tal cual kg/día",
                            "EM_Mcal": "EM Mcal/día",
                            "PB_g": "PB g/día",
                            "precio": "$ /kg (tal cual)",
                            "costo_dia": "$ /día",
                        }
                    )

                    column_config = {
                        "%MS": st.column_config.NumberColumn("%MS", format="%.2f"),
                        "MS kg/día": st.column_config.NumberColumn("MS kg/día", format="%.3f"),
                        "MS % alimento": st.column_config.NumberColumn("MS % alimento", format="%.1f"),
                        "Tal cual kg/día": st.column_config.NumberColumn("Tal cual kg/día", format="%.3f"),
                        "EM Mcal/día": st.column_config.NumberColumn("EM Mcal/día", format="%.3f"),
                        "PB g/día": st.column_config.NumberColumn("PB g/día", format="%.0f"),
                        "$ /kg (tal cual)": st.column_config.NumberColumn("$/kg (tal cual)", format="$ %.2f"),
                        "$ /día": st.column_config.NumberColumn("$/día", format="$ %.2f"),
                        "Alertas": st.column_config.TextColumn("Alertas"),
                    }

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=column_config,
                    )

                    if (display_df["Alertas"].str.len() > 0).any():
                        st.caption(
                            "⚠️ Revisá los ingredientes con datos incompletos en el catálogo de alimentos."
                        )

                    export_buffer = io.StringIO()
                    display_df.to_csv(export_buffer, index=False)
                    export_buffer.write("\n")
                    resumen = pd.DataFrame(
                        [
                            {"Métrica": "Ración", "Valor": pick},
                            {"Métrica": "Corral", "Valor": selected_corral_label},
                            {"Métrica": "PV (kg)", "Valor": pv_value},
                            {"Métrica": "CV (%)", "Valor": cv_value},
                            {
                                "Métrica": "Consumo MS (kg/día)",
                                "Valor": resultado["Consumo_MS_dia"],
                            },
                            {
                                "Métrica": "Tal cual total (kg/día)",
                                "Valor": resultado["asfed_total_kg_dia"],
                            },
                            {"Métrica": "EM total (Mcal/día)", "Valor": resultado["EM_Mcal_dia"]},
                            {"Métrica": "PB total (g/día)", "Valor": resultado["PB_g_dia"]},
                            {"Métrica": "Costo/día", "Valor": resultado["costo_dia"]},
                        ]
                    )
                    resumen.to_csv(export_buffer, index=False)
                    calc_filename = f"racion_{rid}_calculo.csv"
                    download_calc = st.download_button(
                        "⬇️ Exportar cálculo (CSV)",
                        data=export_buffer.getvalue().encode("utf-8"),
                        file_name=calc_filename,
                        mime="text/csv",
                    )
                    if download_calc:
                        activity_log_event(
                            "exportacion",
                            f"racion_calculo archivo={calc_filename}",
                            trace_prefix="EXP-",
                        )

                    pdf_detail = detail_df.rename(
                        columns={
                            "MS_kg": "ms_kg_dia",
                            "asfed_kg": "asfed_kg_dia",
                            "costo_dia": "costo",
                        }
                    ).copy()
                    for col in ["ingrediente", "pct_ms", "ms_kg_dia", "asfed_kg_dia", "costo"]:
                        if col not in pdf_detail.columns:
                            pdf_detail[col] = 0.0
                    res_manual = {
                        "em_calc": float(resultado.get("EM_Mcal_dia", 0.0)),
                        "pb_calc": float(resultado.get("PB_g_dia", 0.0)),
                        "asfed_total_kg_dia": float(resultado.get("asfed_total_kg_dia", 0.0)),
                        "cost_total": float(resultado.get("costo_dia", 0.0)),
                    }
                    pdf_name = f"ficha_racion_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"
                    pdf_path = user_path(pdf_name)
                    datos_pdf = {
                        "Ración": pick,
                        "PV (kg)": f"{pv_value:.0f}",
                        "CV (%)": f"{cv_value:.2f}",
                        "Consumo MS (kg/día)": f"{resultado['Consumo_MS_dia']:.3f}",
                        "EM calculada (Mcal/d)": f"{res_manual['em_calc']:.2f}",
                        "EM requerida (Mcal/d)": f"{req_em_manual:.2f}",
                        "PB calculada (g/d)": f"{res_manual['pb_calc']:.0f}",
                        "PB requerida (g/d)": f"{req_pb_manual:.0f}",
                        "As-fed total (kg/día)": f"{res_manual['asfed_total_kg_dia']:.2f}",
                        "Costo total ($/día)": f"{res_manual['cost_total']:.2f}",
                    }
                    if cabezas_manual > 0:
                        datos_pdf["Costo por cabeza ($/día)"] = f"{(res_manual['cost_total'] / cabezas_manual):.2f}"

                    generate_summary_pdf(
                        file_path=pdf_path,
                        logo_path="assets/logo.png",
                        titulo="JM P-Feedlot — Ficha de Ración",
                        datos=datos_pdf,
                        tabla_detalle=pdf_detail,
                        recomendaciones=list(tips_manual),
                    )

                    try:
                        download_pdf = st.download_button(
                            "⬇️ Descargar PDF de la ración",
                            data=pdf_path.read_bytes(),
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            type="primary",
                        )
                    except Exception as exc:
                        st.warning(f"No se pudo generar el PDF: {exc}")
                    else:
                        if download_pdf:
                            activity_log_event(
                                "exportacion",
                                f"ficha_racion archivo={pdf_path.name}",
                                trace_prefix="EXP-",
                            )

                if st.button("💾 Guardar como ración dada (registrar)", type="primary"):
                    try:
                        ingredientes_min = grid_rec.copy()
                        if "ingrediente" not in ingredientes_min.columns:
                            ingredientes_min["ingrediente"] = ""
                        if "pct_ms" not in ingredientes_min.columns:
                            ingredientes_min["pct_ms"] = 0.0
                        ingredientes_min = ingredientes_min[
                            ingredientes_min["ingrediente"].astype(str).str.strip() != ""
                        ][["ingrediente", "pct_ms"]]
                        append_ration_log(
                            username=username,
                            racion_nombre=pick,
                            tipo_racion=tipo_racion_val,
                            pv_kg=float(pv_value),
                            cv_pct=float(cv_value),
                            categoria=categoria_val,
                            sim=resultado,
                            ingredientes_df=ingredientes_min,
                            tips=tips_manual,
                            gmd_kg_dia=gmd_manual,
                        )
                        st.success("Ración guardada como 'ración dada' (vigente).")
                        st.toast("Registro creado en raciones_log.csv", icon="📚")
                    except Exception as exc:
                        st.error(f"No se pudo guardar el registro: {exc}")
            else:
                st.info("Completá la ración (100% MS) para ver el reparto diario de MS, EM y PB.")

        with st.expander("📚 Historial de raciones dadas"):
            if RACIONES_LOG_PATH.exists():
                try:
                    log_df = pd.read_csv(RACIONES_LOG_PATH, encoding="utf-8-sig")
                except Exception as exc:
                    st.error(f"No se pudo leer el historial: {exc}")
                else:
                    if not log_df.empty:
                        cols_show = [
                            "ts",
                            "tipo_racion",
                            "racion",
                            "cat",
                            "pv_kg",
                            "cv_pct",
                            "em_mcal_dia",
                            "pb_g_dia",
                            "asfed_total_kg_dia",
                            "costo_dia",
                        ]
                        existing_cols = [c for c in cols_show if c in log_df.columns]
                        display_log = log_df[existing_cols].copy()
                        if "ts" in display_log.columns:
                            with pd.option_context("mode.chained_assignment", None):
                                display_log["ts"] = pd.to_datetime(
                                    display_log["ts"], errors="coerce"
                                )
                            display_log = display_log.sort_values("ts", ascending=False)
                            display_log["ts"] = display_log["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
                        st.dataframe(
                            display_log,
                            use_container_width=True,
                            hide_index=True,
                        )

                        if "racion" in log_df.columns:
                            ration_names = (
                                log_df["racion"].dropna().astype(str).str.strip().sort_values().unique().tolist()
                            )
                            if ration_names:
                                pick_r = st.selectbox("Ración para comparar", ration_names)
                                sub = log_df[log_df["racion"].astype(str).str.strip() == pick_r].copy()
                                if not sub.empty:
                                    sub["ts"] = pd.to_datetime(sub["ts"], errors="coerce")
                                    sub = sub.sort_values("ts", ascending=False).head(2)
                                    if len(sub) >= 2:
                                        a, b = sub.iloc[0], sub.iloc[1]
                                        def _num_metric(series_name: str) -> tuple[float, float]:
                                            val_a = pd.to_numeric(a.get(series_name), errors="coerce")
                                            val_b = pd.to_numeric(b.get(series_name), errors="coerce")
                                            aval = float(val_a) if not pd.isna(val_a) else 0.0
                                            bval = float(val_b) if not pd.isna(val_b) else 0.0
                                            return aval, bval

                                        em_a, em_b = _num_metric("em_mcal_dia")
                                        pb_a, pb_b = _num_metric("pb_g_dia")
                                        asfed_a, asfed_b = _num_metric("asfed_total_kg_dia")
                                        costo_a, costo_b = _num_metric("costo_dia")
                                        st.metric("EM (Mcal/día): Δ", f"{em_a - em_b:.2f}")
                                        st.metric("PB (g/día): Δ", f"{pb_a - pb_b:.0f}")
                                        st.metric("As-fed (kg/día): Δ", f"{asfed_a - asfed_b:.2f}")
                                        st.metric("Costo/día: Δ", f"${costo_a - costo_b:.2f}")
                                    else:
                                        st.info("Se necesitan al menos dos registros para comparar.")
                                else:
                                    st.info("No hay registros para esa ración.")
                    else:
                        st.info("Aún no hay registros de raciones dadas.")
            else:
                st.info("Aún no hay registros de raciones dadas.")

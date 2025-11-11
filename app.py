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
import zipfile
import hashlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any

import pandas as pd
import streamlit as st
import requests
import qrcode

from calc_engine import (
    Food,
    Ingredient,
    mixer_kg_by_ingredient,
    ration_split_from_pv_cv,
)
# --- AUTH (login por usuario/clave) ---
import yaml
import streamlit_authenticator as stauth

# ------------------------------------------------------------------------------
# Paths (multiusuario)
# ------------------------------------------------------------------------------
DATA_DIR_ENV = os.getenv("DATA_DIR")
GLOBAL_DATA_DIR = Path(DATA_DIR_ENV) if DATA_DIR_ENV else Path("data")
GLOBAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Config de ADMIN y almacenamiento de usuarios editables ---
DEFAULT_ADMIN_USERS = {"admin"}  # usuarios que verán la pestaña de administración por defecto

AUTH_DIR = GLOBAL_DATA_DIR / "auth"
AUTH_DIR.mkdir(parents=True, exist_ok=True)
AUTH_STORE = AUTH_DIR / "users.yaml"  # acá persistimos los usuarios editados por UI

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

# ------------------------------------------------------------------------------
# Preferencias de UI (persistidas en sesión)
# ------------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False


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
    if AUTH_STORE.exists():
        try:
            return yaml.safe_load(AUTH_STORE.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    # Si no existe, inicializamos con estructura mínima
    return {"credentials": {"usernames": {}}, "preauthorized": {"emails": []}}


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

# --- Crear Authenticate con ARGUMENTOS POSICIONALES (compat 0.3.2) ---
# Validaciones mínimas:
creds = CFG.get("credentials") or {}
if not isinstance(creds, dict) or "usernames" not in creds or not isinstance(creds["usernames"], dict):
    st.error("Config inválida: falta 'credentials.usernames' (dict).")
    st.stop()

cookie_cfg = CFG.get("cookie") or {}
cookie_name = str(cookie_cfg.get("name", "gefeed_cookie"))
cookie_key = str(cookie_cfg.get("key", "feedlot_key"))
try:
    cookie_expiry_days = int(cookie_cfg.get("expiry_days", 7))
except Exception:
    cookie_expiry_days = 7

# 'preauthorized' en 0.3.2 puede venir como dict {"emails":[...]} o lista
preauth_cfg = CFG.get("preauthorized") or []
preauthorized = (
    preauth_cfg.get("emails", []) if isinstance(preauth_cfg, dict) else preauth_cfg
) or []

# IMPORTANTE: firma posicional para 0.3.2 => (credentials, cookie_name, key, cookie_expiry_days, preauthorized)
authenticator = stauth.Authenticate(
    creds,
    cookie_name,
    cookie_key,
    cookie_expiry_days,
    preauthorized,
)

# Manejo defensivo del retorno (tupla/dict/None)
result = authenticator.login(location="main")
if isinstance(result, tuple) and len(result) == 3:
    name, auth_status, username = result
elif isinstance(result, dict):
    name = result.get("name")
    auth_status = result.get("authentication_status")
    username = result.get("username")
elif result is None:
    st.info("Ingresá tus credenciales"); st.stop()
else:
    st.error("Retorno inesperado de authenticator.login()"); st.stop()

if auth_status is False:
    st.error("Usuario o contraseña inválidos"); st.stop()
elif auth_status is None:
    st.info("Ingresá tus credenciales"); st.stop()

user_profile = {}
credentials_cfg = (CFG.get("credentials") or {}).get("usernames", {})
if isinstance(credentials_cfg, dict):
    user_profile = credentials_cfg.get(username, {}) or {}

USER_IS_ADMIN = is_admin(username)

user_email = str(user_profile.get("email", "") or "").strip()
if user_email:
    st.session_state["email"] = user_email

with st.sidebar:
    st.title("⚙️ Opciones")
    st.write(f"👤 {name} (@{username})")
    toggle_label = (
        "☀️ Desactivar modo oscuro"
        if st.session_state["dark_mode"]
        else "🌙 Activar modo oscuro"
    )
    if st.button(toggle_label, key="toggle_dark_mode"):
        st.session_state["dark_mode"] = not st.session_state["dark_mode"]

authenticator.logout("Salir", "sidebar")
APP_VERSION = "JM P-Feedlot v0.26-beta (free)"
st.title(APP_VERSION)
st.caption("Stock corrales • Ajustes de raciones • Alimentos • Mixer • Parámetros • Export ZIP")
st.info("🚧 Versión beta sin costo: validando con clientes iniciales. Guardá y exportá seguido por seguridad.")

# ------------------------------------------------------------------------------
# CSS (transiciones, tarjetas, dropdowns, micro-interacciones)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
.section-enter { opacity:0; transform: translateY(4px); animation: fadeSlideIn .25s ease-out forwards; }
@keyframes fadeSlideIn { to { opacity:1; transform:none; } }

.card     { padding:16px; border:1px solid #E5E7EB; border-radius:14px;
            box-shadow:0 1px 2px rgba(0,0,0,.03); margin:6px 0;
            transition: box-shadow .2s ease, transform .2s ease; }
.card:hover { transform: translateY(-2px); box-shadow:0 6px 18px rgba(0,0,0,.07); }

.stButton>button { transition: transform .08s ease, filter .2s ease; }
.stButton>button:active { transform: scale(.98); }
.stButton>button:hover  { filter: brightness(1.03); }

details > summary { cursor:pointer; list-style:none; transition: color .2s ease; }
details > summary::-webkit-details-marker { display:none; }
details > summary::after { content:"▸"; display:inline-block; margin-left:.5rem; transition: transform .2s ease; }
details[open] > summary::after { transform: rotate(90deg); }
details[open] .expander-body { animation: fadeIn .2s ease both; }
@keyframes fadeIn { from {opacity:0} to {opacity:1} }

.chip-ok  { background:#DCFCE7; color:#166534; padding:4px 8px; border-radius:999px; font-size:.85rem; }
.chip-bad { background:#FEE2E2; color:#991B1B; padding:4px 8px; border-radius:999px; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)

if st.session_state["dark_mode"]:
    st.markdown(
        """
        <style>
        html, body, .stApp {
            background-color: #121212;
            color: #EAEAEA;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #1B1B1B;
            color: #EAEAEA;
        }
        .main {
            background-color: #1B1B1B;
            color: #EAEAEA;
        }
        div[data-testid="stHeader"] {
            background: linear-gradient(90deg, #3E2723, #1B1B1B);
            color: #F5F0E6;
        }
        .card {
            background: #242424;
            border-color: #3A3A3A;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.45);
        }
        .card:hover {
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.60);
        }
        .stButton>button {
            background-color: #4E342E;
            color: #FFF;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #6D4C41;
        }
        h1, h2, h3 {
            color: #E0C097 !important;
        }
        [data-testid="stDataEditor"] thead tr th {
            background: #2A2623;
            color: #F5F0E6;
            border-bottom: 2px solid #4E3B33;
        }
        [data-testid="stDataEditor"] thead tr th:first-child {
            background: #3B2F29;
        }
        [data-testid="stDataEditor"] thead tr th:nth-child(2) {
            background: #393224;
        }
        [data-testid="stDataEditor"] thead tr th:nth-child(3) {
            background: #2F3A2C;
        }
        [data-testid="stDataEditor"] thead tr th:hover {
            filter: brightness(1.08);
        }
        a {
            color: #E0C097;
        }
        :root {
            color-scheme: dark;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        html, body, .stApp {
            background-color: #F5F0E6;
            color: #3E2723;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #F5F0E6;
            color: #3E2723;
        }
        .main {
            background-color: #F5F0E6;
            color: #3E2723;
        }
        div[data-testid="stHeader"] {
            background: linear-gradient(90deg, #6D4C41, #4E342E);
            color: #FFFFFF;
        }
        .card {
            background: #FFFFFF;
            border-color: #D7CCC8;
        }
        .card:hover {
            box-shadow: 0 6px 18px rgba(109, 78, 65, 0.18);
        }
        .stButton>button {
            background-color: #6D4C41;
            color: #FFF;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #8D6E63;
        }
        h1, h2, h3 {
            color: #4E342E !important;
        }
        [data-testid="stDataEditor"] thead tr th {
            background: #EFE5DA;
            color: #3E2723;
            border-bottom: 2px solid #D7CCC8;
        }
        [data-testid="stDataEditor"] thead tr th:first-child {
            background: #E8D3C1;
        }
        [data-testid="stDataEditor"] thead tr th:nth-child(2) {
            background: #E3E0CF;
        }
        [data-testid="stDataEditor"] thead tr th:nth-child(3) {
            background: #E2EBE1;
        }
        [data-testid="stDataEditor"] thead tr th:hover {
            filter: brightness(0.98);
        }
        a {
            color: #4E342E;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

# Carpeta sandbox del usuario autenticado
USER_DIR = GLOBAL_DATA_DIR / "users" / username
USER_DIR.mkdir(parents=True, exist_ok=True)

def user_path(fname: str) -> Path:
    p = USER_DIR / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

META_DIR = USER_DIR / "meta"
META_DIR.mkdir(parents=True, exist_ok=True)
LAST_CHANGED = META_DIR / "last_changed.json"

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
MIXER_SIM_LOG = user_path("mixer_sim_log.csv")

MAX_CORRALES = 200
MAX_UPLOAD_MB = 5
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


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
    pd.DataFrame(columns=[
        "tipo_racion","nro_corral","cod_racion","nombre_racion","categ",
        "PV_kg","CV_pct","AP_preten","nro_cab","mixer_id","capacidad_kg",
        "kg_turno","turnos","meta_salida"
    ]).to_csv(BASE_PATH, index=False, encoding="utf-8")
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
# IO helpers (cache)
# ------------------------------------------------------------------------------
@st.cache_data
def load_alimentos() -> pd.DataFrame:
    try: df = pd.read_csv(ALIM_PATH, encoding="utf-8-sig")
    except Exception: df = pd.DataFrame(columns=ALIM_COLS)
    if df.shape[1] == 1:
        try: df = pd.read_csv(ALIM_PATH, sep=";", encoding="utf-8-sig")
        except: pass
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
        "receta_pct_ms": json.dumps(receta_records, ensure_ascii=False),
        "detalle_calc": json.dumps(detalle, ensure_ascii=False),
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
@st.cache_data
def load_base() -> pd.DataFrame:
    try: return pd.read_csv(BASE_PATH, encoding="utf-8-sig")
    except: return pd.DataFrame()

def save_base(df: pd.DataFrame):
    df.to_csv(BASE_PATH, index=False, encoding="utf-8")
    success = backup_user_file(BASE_PATH, "Actualizar base de corrales")
    audit_log_append(
        "save_base",
        "Base de corrales actualizada",
        path=str(BASE_PATH),
        meta={"github_backup": success},
    )
    mark_changed("save_base", username)


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
    mix_clean = mix_df.dropna(subset=["mixer_id"]).copy()
    mix_clean["mixer_id"] = mix_clean["mixer_id"].astype(str)
    mixer_cap_map = dict(zip(mix_clean["mixer_id"], mix_clean["capacidad_kg"]))

    df = df.copy()
    df = df.drop(columns=["AP_obt", "dias_TERM", "semanas_TERM", "EFC_conv"], errors="ignore")

    for col, default in {
        "nombre_racion": "",
        "tipo_racion": "",
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

    df["cod_racion"] = df["nombre_racion"].map(nombre_to_id).fillna("")
    df["tipo_racion"] = df["nombre_racion"].map(nombre_to_tipo).fillna("")

    def _cv(row):
        current = pd.to_numeric(row.get("CV_pct", 0.0), errors="coerce")
        nombre = str(row.get("nombre_racion", ""))
        if pd.isna(current) or float(current) == 0.0:
            return float(nombre_to_cv.get(nombre, current if not pd.isna(current) else 0.0))
        return float(current)

    df["CV_pct"] = df.apply(_cv, axis=1)
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

tab_corrales, tab_raciones, tab_alimentos, tab_mixer, tab_parametros, tab_export, tab_presentacion, *admin_tabs = tabs

tab_methodology = tab_admin = None
if USER_IS_ADMIN and admin_tabs:
    if len(admin_tabs) == 2:
        tab_methodology, tab_admin = admin_tabs
    elif len(admin_tabs) == 1:
        # Solo se definió la pestaña de usuarios (compatibilidad defensiva)
        tab_admin = admin_tabs[0]

# ------------------------------------------------------------------------------
# 📊 Stock & Corrales (principal)
# ------------------------------------------------------------------------------
with tab_corrales:
    with card("📊 Stock, categorías y corrales", "Actualizá tipo de ración, categoría, cabezas y mezcla asignada por corral."):
        cat_df = load_catalog()
        mix_df = load_mixers()
        base = load_base()

        etapas_series = cat_df.get("etapa", pd.Series(dtype=str))
        tipos = sorted([t for t in etapas_series.dropna().astype(str).unique() if t.strip()]) or [
            "Terminación",
            "Recría",
        ]
        categorias = ["va", "nov"]
        pesos_lista = load_pesos()["peso_kg"].tolist()
        mix_clean = mix_df.dropna(subset=["mixer_id"]).copy()
        mix_clean["mixer_id"] = mix_clean["mixer_id"].astype(str)
        mixers = mix_clean["mixer_id"].tolist()
        mixer_cap_map = dict(zip(mix_clean["mixer_id"], mix_clean["capacidad_kg"]))

        if base.empty:
            base = pd.DataFrame({
                "nro_corral": list(range(1, 21)),
                "nombre_racion": ["" for _ in range(20)],
                "cod_racion": ["" for _ in range(20)],
                "tipo_racion": [""] * 20,
                "categ": ["va"] * 20,
                "PV_kg": [275] * 20,
                "CV_pct": [2.8] * 20,
                "AP_preten": [1.0] * 20,
                "nro_cab": [0] * 20,
                "mixer_id": [mixers[0] if mixers else ""] * 20,
                "capacidad_kg": [mixer_cap_map.get(mixers[0], 0) if mixers else 0] * 20,
                "kg_turno": [0.0] * 20,
                "turnos": [4] * 20,
                "meta_salida": [350] * 20,
            })

        racion_options = cat_df["nombre"].astype(str).tolist() if "nombre" in cat_df.columns else []

        colcfg = {
            "nro_corral": st.column_config.NumberColumn("n° de Corral", min_value=1, max_value=9999, step=1),
            "tipo_racion": st.column_config.TextColumn(
                "tipo de ración", help="Auto: se llena según la ración elegida", disabled=True
            ),
            "nombre_racion": st.column_config.SelectboxColumn(
                "nombre la ración",
                options=[""] + racion_options,
                help="Autocompleta tipo y puede pisar CV%",
            ),
            "categ": st.column_config.SelectboxColumn("categ", options=categorias),
            "PV_kg": st.column_config.SelectboxColumn("PV (kg)", options=pesos_lista),
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

        editable_cols = [
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

        display_cols = [
            "nro_corral",
            "tipo_racion",
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

        with st.form("form_base"):
            enriched = enrich_and_calc_base(base)
            grid = st.data_editor(
                enriched[display_cols],
                column_config=colcfg,
                column_order=display_cols,
                num_rows="dynamic",
                width="stretch",
                hide_index=True,
                key="grid_corrales",
            )
            c1, c2 = st.columns(2)
            save = c1.form_submit_button("💾 Guardar base", type="primary")
            refresh = c2.form_submit_button("🔄 Recargar")
            if save:
                out = enriched.copy()
                for col in editable_cols:
                    if col in grid.columns:
                        out[col] = grid[col]
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
                    mask_nonempty |= pd.to_numeric(out_to_save["nro_cab"], errors="coerce").fillna(0) > 0
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
# 🧮 Mixer
# ------------------------------------------------------------------------------
with tab_mixer:
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
                    subset["nro_cab"] = pd.to_numeric(
                        subset.get("nro_cab", 0), errors="coerce"
                    ).fillna(0).astype(int)
                    if "categ" in subset.columns:
                        subset["categ"] = subset["categ"].fillna("")
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
                    st.download_button(
                        f"⬇️ Exportar plan (Descarga {slot})",
                        data=export_df.to_csv(index=False).encode("utf-8"),
                        file_name=file_name,
                        mime="text/csv",
                        key=f"download_plan_{slot}",
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
                st.success(msg)
                st.toast(
                    "Backup de simulación registrado en mixer_sim_log.csv",
                    icon="🗂️",
                )
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
            st.download_button(
                "⬇️ Exportar plan consolidado (CSV)",
                data=consolidado.to_csv(index=False).encode("utf-8"),
                file_name=f"plan_mixers_consolidado_{fecha_str}.csv",
                mime="text/csv",
                type="primary",
                key="download_plan_consolidado",
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
            st.download_button(
                "⬇️ Descargar ZIP (todas las bases)",
                data=buffer,
                file_name=f"simulaciones_{username}_{ts}.zip",
                mime="application/zip",
                type="primary",
            )
        else:
            st.info("No se encontraron archivos en tu carpeta de usuario para exportar.")

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
        st.markdown("### Requerimientos energéticos (EM Mcal/día)")
        req_df = load_reqener()
        grid_req = st.data_editor(
            req_df,
            column_config={
                "peso": st.column_config.NumberColumn("PV (kg)", min_value=0.0, max_value=2000.0, step=0.5),
                "cat": st.column_config.TextColumn("Categoría"),
                "requerimiento_energetico": st.column_config.NumberColumn("Req. energético (Mcal EM/día)", min_value=0.0, max_value=50.0, step=0.1),
                "ap": st.column_config.NumberColumn("AP (kg/día)", min_value=0.0, max_value=10.0, step=0.1),
            },
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="param_reqener",
        )
        r1, r2 = st.columns(2)
        if r1.button("💾 Guardar requerimientos energéticos", type="primary"):
            save_reqener(grid_req); st.success("Requerimientos energéticos guardados."); st.toast("Req. EM actualizados.", icon="⚡"); rerun_with_cache_reset()
        if r2.button("🔄 Recargar requerimientos"):
            rerun_with_cache_reset()

        with st.expander("📐 Comparar con fórmula sintética de EM", expanded=False):
            eval_df = grid_req.copy()
            eval_df["EM_formula"] = eval_df.apply(
                lambda row: synthetic_em_requirement(
                    row.get("peso"), row.get("ap"), row.get("cat")
                ),
                axis=1,
            )
            em_tabla = pd.to_numeric(
                eval_df.get("requerimiento_energetico"), errors="coerce"
            )
            em_formula = pd.to_numeric(eval_df.get("EM_formula"), errors="coerce")
            eval_df["Delta_tabla_formula"] = (em_tabla - em_formula).round(3)

            display_cols = [
                "peso",
                "cat",
                "ap",
                "requerimiento_energetico",
                "EM_formula",
                "Delta_tabla_formula",
            ]

            st.dataframe(
                eval_df[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "peso": st.column_config.NumberColumn("PV (kg)", format="%.1f"),
                    "cat": st.column_config.TextColumn("Categoría"),
                    "ap": st.column_config.NumberColumn("AP (kg/día)", format="%.1f"),
                    "requerimiento_energetico": st.column_config.NumberColumn(
                        "Req. tabla (Mcal/día)", format="%.2f"
                    ),
                    "EM_formula": st.column_config.NumberColumn(
                        "EM fórmula sintética (Mcal/día)", format="%.2f"
                    ),
                    "Delta_tabla_formula": st.column_config.NumberColumn(
                        "Δ tabla - fórmula", format="%.2f"
                    ),
                },
            )
            st.caption(
                "La fórmula sintética usa PV^0.75 y AP según la categoría para estimar Mcal/día."
            )

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

        st.download_button(
            "⬇️ Exportar documentación técnica (Markdown)",
            data=md_content.encode("utf-8"),
            file_name="tecnologias_y_lenguajes.md",
            mime="text/markdown",
            disabled=not categorias,
        )

# ------------------------------------------------------------------------------
# 📐 Metodología y Cálculo (solo admin)
# ------------------------------------------------------------------------------
if USER_IS_ADMIN and tab_methodology is not None:
    with tab_methodology:
        st.subheader("📐 Metodología y Cálculo (solo admin)")
        md, meta = build_methodology_doc()
        st.markdown(md)

        st.download_button(
            "⬇️ Exportar metodología (Markdown)",
            data=md.encode("utf-8"),
            file_name="metodologia_y_calculo.md",
            mime="text/markdown",
            type="primary",
        )

        st.download_button(
            "⬇️ Exportar metadatos (JSON)",
            data=json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="metodologia_meta.json",
            mime="application/json",
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
                import bcrypt

                def save_user_store(store_dict):
                    AUTH_STORE.write_text(
                        yaml.safe_dump(store_dict, allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                    )
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
            params_cols[3].metric("CV MS (kg)", f"{consumo_ms:.2f}")

            colcfg = {
                "ingrediente": st.column_config.SelectboxColumn("Ingrediente (ORIGEN)", options=opciones_ingred),
                "pct_ms": st.column_config.NumberColumn("% inclusión (MS)", min_value=0.0, max_value=100.0, step=0.1)
            }

            rec_edit_cols = ["ingrediente","pct_ms"]
            grid_rec = st.data_editor(
                rec_r[rec_edit_cols],
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

                st.markdown("#### Métricas diarias de la ración")
                metrics_cols = st.columns(5)
                metrics_cols[0].metric("CV MS (kg)", f"{resultado['Consumo_MS_dia']:.2f}")
                metrics_cols[1].metric(
                    "Tal cual total (kg/día)", f"{resultado['asfed_total_kg_dia']:.2f}"
                )
                metrics_cols[2].metric("EM total (Mcal/día)", f"{resultado['EM_Mcal_dia']:.2f}")
                metrics_cols[3].metric("PB total (g/día)", f"{resultado['PB_g_dia']:.0f}")
                metrics_cols[4].metric("Costo/día", f"$ {resultado['costo_dia']:.2f}")

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
                    st.download_button(
                        "⬇️ Exportar cálculo (CSV)",
                        data=export_buffer.getvalue().encode("utf-8"),
                        file_name=f"racion_{rid}_calculo.csv",
                        mime="text/csv",
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

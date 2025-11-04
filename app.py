# app.py — JM P-Feedlot v0.26 (100% web) — UI mejorada (transiciones + menús + validaciones)
# Pestañas: 📊 Stock & Corrales | 🧾 Ajustes de raciones | 📦 Alimentos | 🧮 Mixer | ⚙️ Parámetros | ⬇️ Exportar
# Estructura:
#   app.py, calc_engine.py, requirements.txt
#   data/: alimentos.csv, raciones_base.csv, mixers.csv, pesos.csv, raciones_catalog.csv, raciones_recipes.csv
#         requerimientos_energeticos.csv, requerimiento_proteico.csv
from __future__ import annotations

import io
import os
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from calc_engine import Food, Ingredient, mixer_kg_by_ingredient
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
ADMIN_USERS = {"admin"}  # usuarios que verán la pestaña de administración

AUTH_DIR = GLOBAL_DATA_DIR / "auth"
AUTH_DIR.mkdir(parents=True, exist_ok=True)
AUTH_STORE = AUTH_DIR / "users.yaml"  # acá persistimos los usuarios editados por UI

# --- Columnas base (constantes) ---
ALIM_COLS = [
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

REQENER_COLS = ["peso", "cat", "requerimiento_energetico", "ap"]
REQPROT_COLS = ["peso", "cat", "ap", "req_proteico"]


def _num(x, default=0.0):
    try:
        val = float(pd.to_numeric(x, errors="coerce"))
        return default if pd.isna(val) else val
    except Exception:
        return default

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

st.sidebar.write(f"👤 {name} (@{username})")
authenticator.logout("Salir", "sidebar")
st.title("JM P-Feedlot v0.26 — Web")
st.caption("Stock corrales • Ajustes de raciones • Alimentos • Mixer • Parámetros • Export ZIP")

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

ALIM_PATH    = user_path("alimentos.csv")
BASE_PATH    = user_path("raciones_base.csv")
MIXERS_PATH  = user_path("mixers.csv")
PESOS_PATH   = user_path("pesos.csv")
CATALOG_PATH = user_path("raciones_catalog.csv")
RECIPES_PATH = user_path("raciones_recipes.csv")
REQENER_PATH = user_path("requerimientos_energeticos.csv")
REQPROT_PATH = user_path("requerimiento_proteico.csv")


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
        "kg_turno","AP_obt","turnos","meta_salida","dias_TERM","semanas_TERM","EFC_conv"
    ]).to_csv(BASE_PATH, index=False, encoding="utf-8")
if not REQENER_PATH.exists():
    pd.DataFrame(columns=REQENER_COLS).to_csv(REQENER_PATH, index=False, encoding="utf-8")
if not REQPROT_PATH.exists():
    pd.DataFrame(columns=REQPROT_COLS).to_csv(REQPROT_PATH, index=False, encoding="utf-8")

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
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in list(df.columns):
        c2 = str(c).strip().replace("\ufeff","")
        c2u = (c2.upper().replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U"))
        if c2u == "ORIGEN": rename_map[c] = "ORIGEN"
        elif c2u.strip() == "PRESENTACION": rename_map[c] = "PRESENTACION"
        elif c2u == "TIPO": rename_map[c] = "TIPO"
        elif c2u in ("MS","%MS","M.S."): rename_map[c] = "MS"
        elif c2u in ("TND","TND (%)","TND%","TND(%)"): rename_map[c] = "TND (%)"
        elif c2u in ("PB","%PB"): rename_map[c] = "PB"
        elif c2u in ("EE","%EE"): rename_map[c] = "EE"
        elif c2u in ("COEF ATC","COEFATC"): rename_map[c] = "COEF ATC"
        elif c2u in ("$/KG","PRECIO","PRECIO/KG","$XKG","PRECIO X KG"): rename_map[c] = "$/KG"
        elif c2u == "EM": rename_map[c] = "EM"
        elif c2u in ("ENP2","ENP"): rename_map[c] = "ENP2"
    df = df.rename(columns=rename_map)
    for col in ALIM_COLS:
        if col not in df.columns: df[col] = None
    df = df[ALIM_COLS]

    def _to_num(x, default=0.0):
        if pd.isna(x): return default
        s = str(x).strip()
        if not s: return default
        s = s.replace("\u00A0"," ").replace(" ","").replace("$","").replace("%","").replace(",", ".")
        try: return float(s)
        except: return default

    for c in ["MS","TND (%)","PB","EE","COEF ATC","$/KG","EM","ENP2"]:
        df[c] = df[c].map(lambda v: _to_num(v, 0.0))

    df.loc[df["MS"] <= 1.0, "MS"] = df.loc[df["MS"] <= 1.0, "MS"] * 100.0
    if (df["TND (%)"] <= 1.0).any() and (df["TND (%)"] > 0).any():
        df.loc[df["TND (%)"] <= 1.0, "TND (%)"] = df.loc[df["TND (%)"] <= 1.0, "TND (%)"] * 100.0

    for c in ["ORIGEN","PRESENTACION","TIPO"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df

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

@st.cache_data
def load_mixers() -> pd.DataFrame:
    try: df = pd.read_csv(MIXERS_PATH, encoding="utf-8-sig")
    except: df = pd.DataFrame({"mixer_id":[], "capacidad_kg":[]})
    df["capacidad_kg"] = pd.to_numeric(df["capacidad_kg"], errors="coerce").fillna(0).astype(float)
    return df

def save_mixers(df: pd.DataFrame):
    df.to_csv(MIXERS_PATH, index=False, encoding="utf-8")

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


def enrich_and_calc_base(df: pd.DataFrame) -> pd.DataFrame:
    cat_df = load_catalog()
    mix_df = load_mixers()
    nombre_to_id = dict(zip(cat_df["nombre"], cat_df["id"]))
    mixer_cap_map = dict(zip(mix_df["mixer_id"], mix_df["capacidad_kg"]))

    df = df.copy()
    df["cod_racion"] = df.apply(lambda r: nombre_to_id.get(str(r.get("nombre_racion", "")), ""), axis=1)
    df["capacidad_kg"] = df.apply(lambda r: mixer_cap_map.get(str(r.get("mixer_id", "")), 0), axis=1)

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

    def dias_term(r):
        try:
            delta = float(r["meta_salida"]) - float(r["PV_kg"])
            ap = max(float(r["AP_obt"]), 0.0001)
            d = max(delta / ap, 0.0)
            return int(round(d))
        except Exception:
            return 0

    df["dias_TERM"] = df.apply(dias_term, axis=1)
    df["semanas_TERM"] = (df["dias_TERM"] / 7.0).round(1)

    def efc(r):
        try:
            hd = max(float(r["nro_cab"]), 1.0)
            kg_dia_hd = (float(r["kg_turno_asfed_calc"]) * float(r["turnos"])) / hd
            ap = max(float(r["AP_obt"]), 0.0001)
            return round(kg_dia_hd / ap, 2)
        except Exception:
            return 0.0

    df["EFC_conv"] = df.apply(efc, axis=1)
    return df

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------
tab_corrales, tab_raciones, tab_alimentos, tab_mixer, tab_parametros, tab_export, tab_admin = st.tabs(
    [
        "📊 Stock & Corrales",
        "🧾 Ajustes de raciones",
        "📦 Alimentos",
        "🧮 Mixer",
        "⚙️ Parámetros",
        "⬇️ Exportar",
        "👤 Usuarios (Admin)",
    ]
)

# ------------------------------------------------------------------------------
# 📊 Stock & Corrales (principal)
# ------------------------------------------------------------------------------
with tab_corrales:
    with card("📊 Stock, categorías y corrales", "Actualizá tipo de ración, categoría, cabezas y mezcla asignada por corral."):
        cat_df = load_catalog()
        mix_df = load_mixers()
        base = load_base()

        tipos = ["Terminación", "Recría"]
        categorias = ["va", "nov"]
        pesos_lista = load_pesos()["peso_kg"].tolist()
        mixers = mix_df["mixer_id"].tolist()
        mixer_cap_map = dict(zip(mix_df["mixer_id"], mix_df["capacidad_kg"]))
        nombre_to_id = dict(zip(cat_df["nombre"], cat_df["id"]))

        if base.empty:
            base = pd.DataFrame({
                "tipo_racion":["Terminación"]*20,
                "nro_corral":list(range(1,21)),
                "cod_racion":["" for _ in range(20)],
                "nombre_racion":["" for _ in range(20)],
                "categ":["va"]*20,
                "PV_kg":[275]*20,
                "CV_pct":[2.8]*20,
                "AP_preten":[1.0]*20,
                "nro_cab":[0]*20,
                "mixer_id":[mixers[0] if mixers else ""]*20,
                "capacidad_kg":[mixer_cap_map.get(mixers[0],0) if mixers else 0]*20,
                "kg_turno":[0.0]*20,
                "AP_obt":[1.0]*20,
                "turnos":[4]*20,
                "meta_salida":[350]*20,
                "dias_TERM":[0]*20,
                "semanas_TERM":[0.0]*20,
                "EFC_conv":[0.0]*20,
            })

        colcfg = {
            "tipo_racion": st.column_config.SelectboxColumn("tipo de ración", options=tipos, required=True),
            "nro_corral": st.column_config.NumberColumn("n° de Corral", min_value=1, max_value=9999, step=1),
            "nombre_racion": st.column_config.SelectboxColumn(
                "nombre la ración", options=[""] + cat_df["nombre"].astype(str).tolist(), help="Autocompleta código"
            ),
            "categ": st.column_config.SelectboxColumn("categ", options=categorias),
            "PV_kg": st.column_config.SelectboxColumn("PV (kg)", options=pesos_lista),
            "CV_pct": st.column_config.NumberColumn("CV (%)", min_value=0.0, max_value=20.0, step=0.1),
            "AP_preten": st.column_config.NumberColumn("AP (kg) PRETEN", min_value=0.0, max_value=5.0, step=0.1),
            "nro_cab": st.column_config.NumberColumn("NRO CAB (und)", min_value=0, max_value=100000, step=1),
            "mixer_id": st.column_config.SelectboxColumn("Mixer", options=[""] + mixers, help="Trae capacidad"),
            "capacidad_kg": st.column_config.NumberColumn("capacidad (kg)", min_value=0, max_value=200000, step=10, disabled=True),
            "AP_obt": st.column_config.NumberColumn("AP OBT (kg/día)", min_value=0.0, max_value=5.0, step=0.01),
            "turnos": st.column_config.NumberColumn("turnos", min_value=1, max_value=24, step=1),
            "meta_salida": st.column_config.NumberColumn("META DE SALIDA (kg)", min_value=0, max_value=2000, step=5),
        }

        editable_cols = [
            "tipo_racion","nro_corral","nombre_racion","categ","PV_kg","CV_pct",
            "AP_preten","nro_cab","mixer_id","capacidad_kg","AP_obt","turnos","meta_salida",
        ]

        with st.form("form_base"):
            enriched = enrich_and_calc_base(base)
            grid = st.data_editor(
                enriched[editable_cols],
                column_config=colcfg,
                column_order=editable_cols,
                num_rows="dynamic",
                use_container_width=True,
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
                for col in ["kg_turno_calc","kg_turno_asfed_calc"]:
                    if col in out.columns:
                        out = out.drop(columns=[col])
                save_base(out)
                st.success("Base guardada.")
                st.toast("Base actualizada.", icon="📦")
                rerun_with_cache_reset()
            if refresh:
                rerun_with_cache_reset()

# ------------------------------------------------------------------------------
# 📦 Alimentos
# ------------------------------------------------------------------------------
with tab_alimentos:
    with card("📦 Catálogo de alimentos", "Editar y normalizar tu base de ingredientes"):
        col_fr, col_save = st.columns([1,1])
        if col_fr.button("🔄 Forzar recarga de catálogo"):
            rerun_with_cache_reset()

        alimentos_df = load_alimentos()
        with dropdown("Filtros avanzados"):
            c1, c2, c3 = st.columns(3)
            with c1: q = st.text_input("🔎 Buscar (ORIGEN contiene)", placeholder="maíz, afrechillo…")
            with c2: tipo = st.text_input("Filtrar TIPO (contiene)", placeholder="grano, silo…")
            with c3: ms_min = st.number_input("MS mínima (%)", 0.0, 100.0, 0.0, step=0.5)
            view = alimentos_df.copy()
            if q.strip(): view = view[view["ORIGEN"].str.contains(q, case=False, na=False)]
            if tipo.strip(): view = view[view["TIPO"].str.contains(tipo, case=False, na=False)]
            if ms_min>0: view = view[pd.to_numeric(view["MS"], errors="coerce").fillna(0)>=ms_min]
        grid_alim = st.data_editor(
            view,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="grid_alimentos",
            column_config={
                "ORIGEN": st.column_config.TextColumn("Origen", help="Nombre único del alimento (clave)."),
                "PRESENTACION": st.column_config.TextColumn("Presentación"),
                "TIPO": st.column_config.TextColumn("Tipo (grano, silo, subproducto)"),
                "MS": st.column_config.NumberColumn("MS (%)", min_value=0.0, max_value=100.0, step=0.1, help="Materia seca."),
                "TND (%)": st.column_config.NumberColumn("TND (%)", min_value=0.0, max_value=100.0, step=0.1),
                "PB": st.column_config.NumberColumn("PB (%)", min_value=0.0, max_value=100.0, step=0.1),
                "EE": st.column_config.NumberColumn("EE (%)", min_value=0.0, max_value=100.0, step=0.1),
                "COEF ATC": st.column_config.NumberColumn("Coef. ATC", step=0.01),
                "$/KG": st.column_config.NumberColumn("$ por kg", format="$ %.2f", step=0.01, help="Costo as-fed."),
                "EM": st.column_config.NumberColumn("EM (Mcal/kg MS)", step=0.01),
                "ENP2": st.column_config.NumberColumn("ENp (Mcal/kg MS)", step=0.01),
            },
        )

        st.write("**Estado del catálogo:**")
        ms_vals = pd.to_numeric(grid_alim.get("MS", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        cero_ms = int((ms_vals <= 0).sum())
        chip("MS > 0 en todos", cero_ms == 0)
        chip(f"Alimentos listados: {len(grid_alim)}", len(grid_alim) > 0)
        if cero_ms > 0:
            st.warning(f"Hay {cero_ms} alimentos con MS=0. Revisá esos valores antes de usarlos en recetas.")

        if col_save.button("💾 Guardar cambios del catálogo", type="primary"):
            edited = grid_alim.copy()
            edited = _normalize_columns(edited)

            base = _normalize_columns(alimentos_df.copy())
            base_idx = base.set_index("ORIGEN")
            edited_idx = edited.set_index("ORIGEN")

            base_idx.update(edited_idx)
            new_rows = edited_idx.index.difference(base_idx.index)
            base_idx = pd.concat([base_idx, edited_idx.loc[new_rows]], axis=0)

            out = base_idx.reset_index()
            save_alimentos(out)
            st.success("Catálogo guardado.")
            st.toast("Alimentos actualizados.", icon="🧾")
            rerun_with_cache_reset()

# ------------------------------------------------------------------------------
# 🧮 Mixer
# ------------------------------------------------------------------------------
with tab_mixer:
    with card("🧮 Cálculo de descarga de mixer (as-fed)", "Plan diario por tipo de ración"):
        st.markdown("### Descarga mixer (plan diario)")

        base_df = load_base()
        mixers_df = load_mixers()

        if base_df.empty:
            st.info("No hay corrales configurados en la base.")
        elif mixers_df.empty or mixers_df["mixer_id"].dropna().empty:
            st.warning("Definí mixers en la pestaña ⚙️ Parámetros para poder planificar la descarga.")
        else:
            base_calc = enrich_and_calc_base(base_df)
            tipos_racion = [t for t in base_calc["tipo_racion"].dropna().astype(str).unique() if t.strip()]
            if tipos_racion:
                mixer_options = mixers_df["mixer_id"].dropna().astype(str).tolist()

                combos = base_calc.copy()
                combos["turnos"] = pd.to_numeric(combos.get("turnos", 0), errors="coerce").fillna(0).astype(int)
                combos = combos[combos["nombre_racion"].astype(str).str.strip()!=""]
                combo_turnos = {}
                for (tipo, nombre), sub in combos.groupby(["tipo_racion","nombre_racion"]):
                    valid_turnos = sub["turnos"][sub["turnos"]>0]
                    if not valid_turnos.empty:
                        combo_turnos[(str(tipo), str(nombre))] = int(valid_turnos.mode().iloc[0])

                tipo_options = sorted(tipos_racion)
                opciones_por_tipo = {}
                for tipo in tipo_options:
                    subset = base_calc.loc[base_calc["tipo_racion"]==tipo, "nombre_racion"].dropna().astype(str)
                    opciones_por_tipo[str(tipo)] = sorted([n for n in subset.unique() if str(n).strip()])

                st.markdown("#### Planificar descargas individuales (hasta 3 cargas)")
                st.caption("Cada descarga se limita a una combinación de tipo de ración y ración específica.")

                for slot in range(1, 4):
                    st.markdown(f"##### Descarga {slot}")
                    col_mix, col_tipo, col_rac, col_turnos = st.columns((1.2, 1.1, 1.1, 0.8))
                    mixer_sel = col_mix.selectbox(
                        "Mixer", [""] + mixer_options, key=f"slot_mixer_{slot}",
                        help="Elegí el mixer que realizará esta descarga.",
                    )
                    tipo_sel = col_tipo.selectbox(
                        "Tipo de ración", [""] + tipo_options, key=f"slot_tipo_{slot}"
                    )
                    racion_opts = opciones_por_tipo.get(str(tipo_sel), [])
                    racion_sel = col_rac.selectbox(
                        "Ración", [""] + racion_opts, key=f"slot_racion_{slot}"
                    )
                    default_turnos = combo_turnos.get((str(tipo_sel), str(racion_sel)), 1)
                    turnos_val = col_turnos.number_input(
                        "Turnos", min_value=1, max_value=24, value=default_turnos, step=1,
                        key=f"slot_turnos_{slot}"
                    )

                    if mixer_sel and tipo_sel and racion_sel:
                        subset = base_calc[
                            (base_calc["tipo_racion"] == tipo_sel)
                            & (base_calc["nombre_racion"] == racion_sel)
                        ].copy()
                        if subset.empty:
                            st.info("No hay corrales asignados a esa combinación.")
                            continue

                        subset["kg_turno_asfed_calc"] = pd.to_numeric(
                            subset.get("kg_turno_asfed_calc", 0.0), errors="coerce"
                        ).fillna(0.0)
                        subset["nro_cab"] = pd.to_numeric(
                            subset.get("nro_cab", 0), errors="coerce"
                        ).fillna(0).astype(int)
                        cap_turno = float(
                            mixers_df.loc[mixers_df["mixer_id"] == mixer_sel, "capacidad_kg"].fillna(0).max()
                        )

                        kg_turno_total = float(subset["kg_turno_asfed_calc"].sum())
                        kg_dia_total = kg_turno_total * float(turnos_val)
                        over = kg_turno_total > cap_turno + 1e-6

                        cols_metrics = st.columns(2)
                        cols_metrics[0].metric("Kg por turno", f"{kg_turno_total:,.1f} kg")
                        cols_metrics[1].metric("Kg totales/día", f"{kg_dia_total:,.1f} kg")
                        if cap_turno > 0:
                            if over:
                                st.error(f"{tipo_sel} / {racion_sel}: {kg_turno_total:,.1f} kg por turno supera {cap_turno:,.0f} kg del mixer {mixer_sel}.")
                            else:
                                st.success(f"{tipo_sel} / {racion_sel}: {kg_turno_total:,.1f} kg por turno dentro de {cap_turno:,.0f} kg del mixer {mixer_sel}.")

                        st.caption(f"Turnos programados: {int(turnos_val)}")

                        def _categoria_val(row, categoria):
                            cat = str(row.get("categ","")).strip().lower()
                            if categoria=="va" and cat.startswith("va"): return row.get("nro_cab",0)
                            if categoria=="nov" and cat.startswith("nov"): return row.get("nro_cab",0)
                            return 0

                        corrales_df = pd.DataFrame({
                            "Corral": subset["nro_corral"],
                            "kg por turno": subset["kg_turno_asfed_calc"].round(1),
                            "vaquillonas": subset.apply(_categoria_val, categoria="va", axis=1),
                            "novillos": subset.apply(_categoria_val, categoria="nov", axis=1),
                        })
                        corrales_df["vaquillonas"] = pd.to_numeric(
                            corrales_df["vaquillonas"], errors="coerce"
                        ).fillna(0).astype(int)
                        corrales_df["novillos"] = pd.to_numeric(
                            corrales_df["novillos"], errors="coerce"
                        ).fillna(0).astype(int)
                        resumen = {
                            "Corral": "Total",
                            "kg por turno": round(corrales_df["kg por turno"].sum(), 1),
                            "vaquillonas": int(corrales_df["vaquillonas"].sum()),
                            "novillos": int(corrales_df["novillos"].sum()),
                        }
                        corrales_df = pd.concat([corrales_df, pd.DataFrame([resumen])], ignore_index=True)
                        st.dataframe(corrales_df, use_container_width=True, hide_index=True)

                        st.download_button(
                            f"⬇️ Exportar plan (Descarga {slot})",
                            data=corrales_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"plan_descarga{slot}_{tipo_sel}_{racion_sel}_{mixer_sel}.csv",
                            mime="text/csv",
                            key=f"download_plan_{slot}",
                        )
                    else:
                        st.caption("Seleccioná mixer, tipo y ración para generar el plan de esta descarga.")
            else:
                st.info("Configura tipos de ración en la base para generar el plan del mixer.")

        st.markdown("---")
        total_kg = st.number_input("Total del mixer (kg, as-fed)", 0.0, 50000.0, 5000.0, step=10.0)

        raciones = build_raciones_from_recipes()
        if not raciones:
            st.info("Definí recetas en la pestaña 🧾 Ajustes de raciones.")
        else:
            pick = st.selectbox("Ración a usar", [r["nombre"] for r in raciones])
            ra = next(r for r in raciones if r["nombre"] == pick)

            ings = []
            for x in ra["ingredientes"]:
                dm_pct = float(pd.to_numeric(x.get("MS", 100.0), errors="coerce") or 100.0)
                food = Food(
                    name=str(x.get("ORIGEN","Ingrediente")),
                    em=float(pd.to_numeric(x.get("EM", 0.0), errors="coerce") or 0.0),
                    pb=float(pd.to_numeric(x.get("PB", 0.0), errors="coerce") or 0.0),
                    dm=dm_pct,
                )
                ings.append( Ingredient(food=food, inclusion_pct=float(pd.to_numeric(x.get("inclusion_pct", 0.0), errors="coerce") or 0.0)) )

            if st.button("Calcular plan de carga", type="primary"):
                plan = mixer_kg_by_ingredient(ings, total_kg)
                if plan:
                    df_plan = pd.DataFrame({"Ingrediente": list(plan.keys()), "Kg (as-fed)": list(plan.values())})
                    st.dataframe(df_plan, use_container_width=True)
                    st.caption(f"Total calculado: {sum(plan.values()):.1f} kg (debería ≈ {total_kg:.1f} kg)")
                    st.toast("Plan de carga generado.", icon="✅")
                else:
                    st.warning("No hay ingredientes con inclusión > 0 o el total del mixer es 0.")

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
        alim_df = load_alimentos()
        grid_alim_p = st.data_editor(
            alim_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="param_alimentos",
        )
        c1, c2 = st.columns(2)
        if c1.button("💾 Guardar alimentos (parámetros)", type="primary"):
            save_alimentos(grid_alim_p); st.success("Alimentos guardados."); st.toast("Alimentos actualizados.", icon="🧾"); rerun_with_cache_reset()

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

        st.markdown("---")
        st.markdown("### Requerimientos proteicos (g PB/día)")
        reqprot_df = load_reqprot()
        grid_reqprot = st.data_editor(
            reqprot_df,
            column_config={
                "peso": st.column_config.NumberColumn("PV (kg)", min_value=0.0, max_value=2000.0, step=0.5),
                "cat": st.column_config.TextColumn("Categoría"),
                "ap": st.column_config.NumberColumn("AP (kg/día)", min_value=0.0, max_value=20.0, step=0.1),
                "req_proteico": st.column_config.NumberColumn("Req. proteico (g PB/día)", min_value=0.0, max_value=5000.0, step=1.0),
            },
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="param_reqprot",
        )
        rp1, rp2 = st.columns(2)
        if rp1.button("💾 Guardar requerimientos proteicos", type="primary"):
            save_reqprot(grid_reqprot); st.success("Requerimientos proteicos guardados."); st.toast("Req. PB actualizados.", icon="🧪"); rerun_with_cache_reset()
        if rp2.button("🔄 Recargar requerimientos proteicos"):
            rerun_with_cache_reset()

# ------------------------------------------------------------------------------
# 👤 Usuarios (Admin)
# ------------------------------------------------------------------------------
with tab_admin:
    if username not in ADMIN_USERS:
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
                    if new_user in ADMIN_USERS:
                        st.error("No podés eliminar un usuario admin definido en ADMIN_USERS.")
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
        cat_display = cat_display.set_index(["id","nombre","etapa","sexo"])

        editable_cat_cols = [c for c in ["pv_kg","cv_pct","corral_comparacion"] if c in cat_display.columns]
        grid_cat = st.data_editor(
            cat_display[editable_cat_cols],
            column_config={
                "pv_kg": st.column_config.NumberColumn("PV (kg)", min_value=0.0, max_value=1000.0, step=0.5),
                "cv_pct": st.column_config.NumberColumn("CV (%)", min_value=0.0, max_value=20.0, step=0.1),
                "corral_comparacion": st.column_config.NumberColumn("Corral de comparación", min_value=0.0, max_value=1000.0, step=1.0),
            },
            column_order=editable_cat_cols,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=False,
            key="grid_rac_catalog",
        )
        cat_updated = cat_display.copy()
        for col in editable_cat_cols:
            cat_updated[col] = grid_cat[col]
        cat_preview = cat_updated.reset_index()
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

            st.markdown("#### Vista previa (MS ponderada y factor as-fed)")
            df_view = grid_rec.copy()
            df_view = df_view[df_view["ingrediente"].astype(str).str.strip()!=""]
            if not df_view.empty:
                df_view = df_view.merge(
                    alimentos_norm[["ORIGEN","MS"]],
                    left_on="ingrediente",
                    right_on="ORIGEN",
                    how="left",
                )
                df_view = df_view.rename(columns={"MS":"MS_%"}).drop(columns=["ORIGEN"])
                df_view["MS_frac"] = pd.to_numeric(df_view["MS_%"], errors="coerce").fillna(100.0)/100.0
                w_ms = (df_view["pct_ms"]/100.0 * df_view["MS_frac"]).sum()
                w_ms = float(w_ms) if w_ms>0 else 1.0
                factor_asfed = 1.0 / w_ms
                m1, m2 = st.columns(2)
                m1.metric("MS ponderada", f"{w_ms*100:.1f} %")
                m2.metric("Factor as-fed", f"×{factor_asfed:.3f}")
                st.dataframe(df_view[["ingrediente","pct_ms","MS_%"]], use_container_width=True)
            else:
                st.info("Agregá ingredientes para ver la MS ponderada.")

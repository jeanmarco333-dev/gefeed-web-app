# app.py — JM P-Feedlot v0.26 (100% web)
# Requisitos: streamlit, pandas
# Estructura esperada:
#   app.py
#   calc_engine.py
#   requirements.txt
#   data/
#       alimentos.csv
#       raciones.json

import os
import json
import streamlit as st
import pandas as pd
from calc_engine import Food, Ingredient, analyze_ration, mixer_kg_by_ingredient

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="JM P-Feedlot v0.26 — Web", layout="wide")
st.title("JM P-Feedlot v0.26 — Web")
st.caption("Edición 100% web. Un usuario/licencia. Catálogo con columnas reales; Formulación EM/PB; Mixer as-fed.")

# --------------------------------------------------------------------------------------
# BLOQUE ROBUSTO DE DATOS
# --------------------------------------------------------------------------------------
import io

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

ALIM_PATH = os.path.join(DATA_DIR, "alimentos.csv")
RAC_PATH  = os.path.join(DATA_DIR, "raciones.json")

# Columnas oficiales del catálogo
ALIM_COLS = ["ORIGEN","PRESENTACION","TIPO","MS","TND (%)","PB","EE","COEF ATC","$/KG","EM","ENP2"]

# Si faltan archivos, crearlos
if not os.path.exists(ALIM_PATH):
    pd.DataFrame(columns=ALIM_COLS).to_csv(ALIM_PATH, index=False, encoding="utf-8")

if not os.path.exists(RAC_PATH):
    with open(RAC_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Limpia nombres y estandariza columnas
    rename_map = {}
    for c in list(df.columns):
        c2 = str(c).strip().replace("\ufeff","")  # BOM
        c2u = c2.upper()
        # normalizaciones frecuentes
        if c2u == "ORIGEN": rename_map[c] = "ORIGEN"
        elif c2u == "PRESENTACION": rename_map[c] = "PRESENTACION"
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

    # Asegurar todas en orden
    for col in ALIM_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[ALIM_COLS]

    # Conversión número (soporta coma decimal)
    def _to_num(x, default=0.0):
        if pd.isna(x): return default
        if isinstance(x, str):
            x = x.replace(",", ".")
        try:
            return float(x)
        except:
            return default

    for c in ["MS","TND (%)","PB","EE","COEF ATC","$/KG","EM","ENP2"]:
        df[c] = df[c].map(lambda v: _to_num(v, 0.0))

    for c in ["ORIGEN","PRESENTACION","TIPO"]:
        df[c] = df[c].fillna("").astype(str)

    return df

@st.cache_data
def load_alimentos() -> pd.DataFrame:
    # Intenta lectura flexible (coma y si no, punto y coma)
    try:
        df = pd.read_csv(ALIM_PATH, encoding="utf-8-sig")
    except Exception:
        df = pd.DataFrame(columns=ALIM_COLS)
    if df.shape[1] == 1:  # probablemente separador ;
        try:
            df = pd.read_csv(ALIM_PATH, sep=";", encoding="utf-8-sig")
        except:
            pass

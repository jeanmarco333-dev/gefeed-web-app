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
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

ALIM_PATH = os.path.join(DATA_DIR, "alimentos.csv")
RAC_PATH  = os.path.join(DATA_DIR, "raciones.json")

ALIM_COLS = ["ORIGEN","PRESENTACION","TIPO","MS","TND (%)","PB","EE","COEF ATC","$/KG","EM","ENP2"]

if not os.path.exists(ALIM_PATH):
    pd.DataFrame(columns=ALIM_COLS).to_csv(ALIM_PATH, index=False, encoding="utf-8")

if not os.path.exists(RAC_PATH):
    with open(RAC_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in list(df.columns):
        c2 = str(c).strip().replace("\ufeff","")  
        c2u = c2.upper()
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
    for col in ALIM_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[ALIM_COLS]
    def _to_num(x, default=0.0):
        if pd.isna(x): return default
        if isinstance(x, str): x = x.replace(",", ".")
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
    try:
        df = pd.read_csv(ALIM_PATH, encoding="utf-8-sig")
    except Exception:
        df = pd.DataFrame(columns=ALIM_COLS)
    if df.shape[1] == 1:
        try:
            df = pd.read_csv(ALIM_PATH, sep=";", encoding="utf-8-sig")
        except:
            pass
    return _normalize_columns(df)

def save_alimentos(df: pd.DataFrame):
    out = _normalize_columns(df.copy())
    out.to_csv(ALIM_PATH, index=False, encoding="utf-8")

def load_raciones() -> list:
    try:
        with open(RAC_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def save_raciones(data: list):
    with open(RAC_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --------------------------------------------------------------------------------------
# Cargar datos
# --------------------------------------------------------------------------------------
alimentos_df = load_alimentos()
raciones = load_raciones()

# --------------------------------------------------------------------------------------
# UI — Pestañas
# --------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📦 Alimentos", "🥣 Formulador", "🧮 Mixer", "⬇️ Exportar"])

# --------------------------------------------------------------------------------------
# 📦 Alimentos
# --------------------------------------------------------------------------------------
with tab1:
    st.subheader("Catálogo de alimentos")
    st.dataframe(alimentos_df, use_container_width=True)

    with st.expander("➕ Agregar alimento"):
        c1, c2, c3 = st.columns(3)
        origen = c1.text_input("ORIGEN")
        presentacion = c2.text_input("PRESENTACION")
        tipo = c3.text_input("TIPO")

        c4, c5, c6, c7 = st.columns(4)
        ms  = c4.number_input("MS (%)", 0.0, 100.0, 100.0)
        tnd = c5.number_input("TND (%)", 0.0, 100.0, 0.0)
        pb  = c6.number_input("PB (%)", 0.0, 100.0, 12.0)
        ee  = c7.number_input("EE (%)", 0.0, 100.0, 3.0)

        c8, c9, c10, c11 = st.columns(4)
        coef   = c8.number_input("COEF ATC", 0.0, 1.0, 0.9)
        precio = c9.number_input("$/KG", 0.0, 500000.0, 0.0)
        em     = c10.number_input("EM (Mcal/kg MS)", 0.0, 10.0, 2.8)
        enp2   = c11.number_input("ENP2 (Mcal/kg MS)", 0.0, 10.0, 2.2)

        if st.button("Agregar alimento"):
            if origen.strip():
                new_row = {
                    "ORIGEN": origen.strip(),
                    "PRESENTACION": presentacion.strip(),
                    "TIPO": tipo.strip(),
                    "MS": float(ms),
                    "TND (%)": float(tnd),
                    "PB": float(pb),
                    "EE": float(ee),
                    "COEF ATC": float(coef),
                    "$/KG": float(precio),
                    "EM": float(em),
                    "ENP2": float(enp2)
                }
                alimentos_df.loc[len(alimentos_df)] = new_row
                save_alimentos(alimentos_df)
                st.success(f"Agregado: {origen}")
                try:
                    st.cache_data.clear()
                except Exception:
                    pass
                st.rerun()
            else:
                st.error("El campo ORIGEN no puede estar vacío.")

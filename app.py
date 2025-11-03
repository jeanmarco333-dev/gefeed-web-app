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
    st.write("Columnas: ORIGEN, PRESENTACION, TIPO, MS, TND (%), PB, EE, COEF ATC, $/KG, EM, ENP2")

    st.dataframe(alimentos_df, use_container_width=True)

    with st.expander("➕ Agregar alimento"):
        c1, c2, c3 = st.columns(3)
        origen = c1.text_input("ORIGEN (nombre del alimento)")
        presentacion = c2.text_input("PRESENTACION")
        tipo = c3.text_input("TIPO (concentrado, voluminoso, etc.)")

        c4, c5, c6, c7 = st.columns(4)
        ms  = c4.number_input("MS (%)", 0.0, 100.0, 100.0, step=0.1)
        tnd = c5.number_input("TND (%)", 0.0, 100.0, 0.0, step=0.1)
        pb  = c6.number_input("PB (%)", 0.0, 100.0, 12.0, step=0.1)
        ee  = c7.number_input("EE (%)", 0.0, 100.0, 3.0, step=0.1)

        c8, c9, c10, c11 = st.columns(4)
        coef   = c8.number_input("COEF ATC", 0.0, 1.0, 0.9, step=0.01)
        precio = c9.number_input("$/KG", 0.0, 500000.0, 0.0, step=0.1)
        em     = c10.number_input("EM (Mcal/kg MS)", 0.0, 10.0, 2.8, step=0.01)
        enp2   = c11.number_input("ENP2 (Mcal/kg MS)", 0.0, 10.0, 2.2, step=0.01)

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
                st.experimental_rerun()
            else:
                st.error("El campo ORIGEN no puede estar vacío.")

    st.markdown("—")
    if st.button("💾 Guardar cambios del catálogo (si editaste en la tabla de arriba)"):
        save_alimentos(alimentos_df)
        st.success("Catálogo guardado.")

# --------------------------------------------------------------------------------------
# 🥣 Formulador (EM/PB y desvíos)
# --------------------------------------------------------------------------------------
with tab2:
    st.subheader("Formulación de ración (EM/PB y desvíos)")
    c1, c2, c3 = st.columns(3)
    target_em = c1.number_input("EM objetivo (Mcal/kg MS)", 0.0, 10.0, 2.6, step=0.01)
    target_pb = c2.number_input("PB objetivo (% MS)", 0.0, 100.0, 13.0, step=0.1)
    ration_name = c3.text_input("Nombre de la ración", "Ración Demo")

    st.markdown("### Ingredientes (% inclusión en base MS)")
    if alimentos_df.empty or alimentos_df["ORIGEN"].eq("").all():
        st.info("Cargá el catálogo de alimentos en la pestaña anterior.")
    else:
        # Editor con % inclusión
        editable = alimentos_df.copy()
        if "inclusion_pct" not in editable.columns:
            editable["inclusion_pct"] = 0.0

        st.caption("Asigná % inclusión a los alimentos que quieras usar (la suma ideal es 100% ±0,5).")
        grid = st.data_editor(
            editable[ALIM_COLS + ["inclusion_pct"]],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True
        )

        # Construir ingredientes con % > 0
        rows = grid[grid["inclusion_pct"] > 0]
        ings = []
        for _, r in rows.iterrows():
            dm_pct = float(pd.to_numeric(r.get("MS", 100.0), errors="coerce") or 100.0)  # MS = DM
            food = Food(
                name=str(r["ORIGEN"]),
                em=float(pd.to_numeric(r.get("EM", 0.0), errors="coerce") or 0.0),
                pb=float(pd.to_numeric(r.get("PB", 0.0), errors="coerce") or 0.0),
                dm=dm_pct
            )
            ings.append(Ingredient(food=food, inclusion_pct=float(r["inclusion_pct"])))

        col_a, col_b = st.columns([1,1])
        if col_a.button("Calcular EM/PB"):
            res = analyze_ration(ings, target_em, target_pb)
            sum_pct = float(rows["inclusion_pct"].sum())
            st.metric("EM calculada (Mcal/kg MS)", res.em)
            st.metric("PB calculada (% MS)", res.pb)
            st.metric("Desvío EM", res.dev_em)
            st.metric("Desvío PB", res.dev_pb)
            st.write(f"**Suma de inclusiones**: {sum_pct:.2f}%")
            if abs(sum_pct - 100) <= 0.5:
                st.success("✔ La suma de inclusiones está dentro de ±0,5%.")
            elif abs(sum_pct - 100) <= 1.5:
                st.warning("🟡 La suma de inclusiones se aleja un poco de 100%.")
            else:
                st.error("🔴 La suma de inclusiones está lejos de 100%.")

        if col_b.button("Guardar ración"):
            payload = {
                "name": ration_name,
                "target_em": float(target_em),
                "target_pb": float(target_pb),
                "ingredients": []
            }
            for _, r in rows.iterrows():
                payload["ingredients"].append({
                    "ORIGEN": str(r["ORIGEN"]),
                    "PRESENTACION": str(r.get("PRESENTACION", "")),
                    "TIPO": str(r.get("TIPO", "")),
                    "MS": float(pd.to_numeric(r.get("MS", 100.0), errors="coerce") or 100.0),
                    "TND (%)": float(pd.to_numeric(r.get("TND (%)", 0.0), errors="coerce") or 0.0),
                    "PB": float(pd.to_numeric(r.get("PB", 0.0), errors="coerce") or 0.0),
                    "EE": float(pd.to_numeric(r.get("EE", 0.0), errors="coerce") or 0.0),
                    "COEF ATC": float(pd.to_numeric(r.get("COEF ATC", 0.0), errors="coerce") or 0.0),
                    "$/KG": float(pd.to_numeric(r.get("$/KG", 0.0), errors="coerce") or 0.0),
                    "EM": float(pd.to_numeric(r.get("EM", 0.0), errors="coerce") or 0.0),
                    "ENP2": float(pd.to_numeric(r.get("ENP2", 0.0), errors="coerce") or 0.0),
                    "inclusion_pct": float(pd.to_numeric(r.get("inclusion_pct", 0.0), errors="coerce") or 0.0)
                })
            raciones.append(payload)
            save_raciones(raciones)
            st.success(f"Ración '{ration_name}' guardada.")

        if raciones:
            st.markdown("### Raciones guardadas")
            names = [r["name"] for r in raciones]
            pick = st.selectbox("Seleccioná una ración para ver", names)
            if pick:
                ra = next(r for r in raciones if r["name"] == pick)
                st.json(ra)

# --------------------------------------------------------------------------------------
# 🧮 Mixer (as-fed con %MS)
# --------------------------------------------------------------------------------------
with tab3:
    st.subheader("Cálculo de descarga de mixer (as-fed)")
    total_kg = st.number_input("Total del mixer (kg, as-fed)", 0.0, 50000.0, 5000.0, step=10.0)

    if not raciones:
        st.info("Guardá al menos una ración en la pestaña Formulador.")
    else:
        pick = st.selectbox("Ración a usar", [r["name"] for r in raciones])
        ra = next(r for r in raciones if r["name"] == pick)

        # Reconstituir ingredientes para el motor (DM = MS)
        ings = []
        for x in ra["ingredients"]:
            dm_pct = float(pd.to_numeric(x.get("MS", 100.0), errors="coerce") or 100.0)
            food = Food(
                name=str(x.get("ORIGEN", "Ingrediente")),
                em=float(pd.to_numeric(x.get("EM", 0.0), errors="coerce") or 0.0),
                pb=float(pd.to_numeric(x.get("PB", 0.0), errors="coerce") or 0.0),
                dm=dm_pct
            )
            ings.append(Ingredient(food=food, inclusion_pct=float(pd.to_numeric(x.get("inclusion_pct", 0.0), errors="coerce") or 0.0)))

        if st.button("Calcular plan de carga"):
            plan = mixer_kg_by_ingredient(ings, total_kg)
            if plan:
                df_plan = pd.DataFrame({"Ingrediente": list(plan.keys()), "Kg (as-fed)": list(plan.values())})
                st.dataframe(df_plan, use_container_width=True)
                st.caption(f"Total calculado: {sum(plan.values()):.1f} kg (debería ≈ {total_kg:.1f} kg)")
            else:
                st.warning("No hay ingredientes con inclusión > 0 o el total del mixer es 0.")

# --------------------------------------------------------------------------------------
# ⬇️ Exportar
# --------------------------------------------------------------------------------------
with tab4:
    st.subheader("Exportación")
    col1, col2 = st.columns(2)

    export_alim = alimentos_df[ALIM_COLS].to_csv(index=False).encode("utf-8")
    col1.download_button("⬇️ Descargar alimentos.csv", data=export_alim, file_name="alimentos.csv", mime="text/csv")

    if raciones:
        pick = col2.selectbox("Ración a exportar (CSV)", [r["name"] for r in raciones])
        ra = next(r for r in raciones if r["name"] == pick)
        export_df = pd.DataFrame(ra["ingredients"])
        export_csv = export_df.to_csv(index=False).encode("utf-8")
        col2.download_button("⬇️ Descargar ración.csv", data=export_csv, file_name=f"{pick}.csv", mime="text/csv")
    else:
        col2.info("No hay raciones guardadas aún.")

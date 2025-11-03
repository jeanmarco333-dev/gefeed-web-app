# app.py — JM P-Feedlot v0.26 (100% web, 1 usuario/licencia)
# Requisitos: streamlit, pandas
# Estructura esperada del repo (raíz):
#   app.py
#   calc_engine.py
#   requirements.txt  ->  (streamlit, pandas)
#   data/
#       alimentos.csv     (encabezados)
#       raciones.json     ([])

import os
import json
import streamlit as st
import pandas as pd
from calc_engine import Food, Ingredient, analyze_ration, mixer_kg_by_ingredient

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="JM P-Feedlot v0.26 — Web", layout="wide")

# Carpeta de datos (crea si no existe)
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

ALIM_PATH = os.path.join(DATA_DIR, "alimentos.csv")
RAC_PATH  = os.path.join(DATA_DIR, "raciones.json")

# Columnas oficiales del catálogo
ALIM_COLS = [
    "ORIGEN", "PRESENTACION", "TIPO", "MS", "TND (%)", "PB", "EE", "COEF ATC", "$/KG", "EM", "ENP2"
]

# Si no existen archivos, crearlos vacíos con encabezados/estructura
if not os.path.exists(ALIM_PATH):
    pd.DataFrame(columns=ALIM_COLS).to_csv(ALIM_PATH, index=False)

if not os.path.exists(RAC_PATH):
    with open(RAC_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------
@st.cache_data
def load_alimentos() -> pd.DataFrame:
    try:
        df = pd.read_csv(ALIM_PATH)
    except Exception:
        df = pd.DataFrame(columns=ALIM_COLS)
    # Asegurar columnas faltantes
    for c in ALIM_COLS:
        if c not in df.columns:
            df[c] = None
    # Tipos numéricos con defaults
    for num_col, default in [("MS", 100.0), ("TND (%)", 0.0), ("PB", 0.0), ("EE", 0.0),
                             ("COEF ATC", 0.0), ("$/KG", 0.0), ("EM", 0.0), ("ENP2", 0.0)]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(default)
    # Texto
    for txt_col in ["ORIGEN", "PRESENTACION", "TIPO"]:
        df[txt_col] = df[txt_col].fillna("").astype(str)
    return df

def save_alimentos(df: pd.DataFrame):
    # Guardar solo columnas oficiales, en orden
    out = df.copy()
    # Asegura el orden y la existencia de las columnas
    for c in ALIM_COLS:
        if c not in out.columns:
            out[c] = None
    out = out[ALIM_COLS]
    out.to_csv(ALIM_PATH, index=False)

def load_raciones() -> list:
    try:
        with open(RAC_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_raciones(data: list):
    with open(RAC_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.title("JM P-Feedlot v0.26 — Web")
st.caption("Edición 100% web (un usuario/licencia). Catálogo con columnas reales. Formulación EM/PB y Mixer as-fed.")

alimentos_df = load_alimentos()
raciones = load_raciones()

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
        # Editor simple con columna de % inclusión
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
            # Mapear MS -> DM para el motor (DM = MS)
            dm_pct = float(pd.to_numeric(r.get("MS", 100.0), errors="coerce") or 100.0)
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

    # Exportar catálogo de alimentos
    export_alim = alimentos_df[ALIM_COLS].to_csv(index=False).encode("utf-8")
    col1.download_button("⬇️ Descargar alimentos.csv", data=export_alim, file_name="alimentos.csv", mime="text/csv")

    # Exportar una ración
    if raciones:
        pick = col2.selectbox("Ración a exportar (CSV)", [r["name"] for r in raciones])
        ra = next(r for r in raciones if r["name"] == pick)
        export_df = pd.DataFrame(ra["ingredients"])
        export_csv = export_df.to_csv(index=False).encode("utf-8")
        col2.download_button("⬇️ Descargar ración.csv", data=export_csv, file_name=f"{pick}.csv", mime="text/csv")
    else:
        col2.info("No hay raciones guardadas aún.")

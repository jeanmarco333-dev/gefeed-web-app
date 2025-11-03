
import streamlit as st
import pandas as pd
import json
from calc_engine import Food, Ingredient, analyze_ration, mixer_kg_by_ingredient

st.set_page_config(page_title="JM P-Feedlot v0.26 (Web)", layout="wide")
st.title("JM P-Feedlot v0.26 — 100% Web")

ALIM_PATH = "data/alimentos.csv"
RAC_PATH = "data/raciones.json"

@st.cache_data
def load_alimentos():
    try:
        df = pd.read_csv(ALIM_PATH)
    except Exception:
        df = pd.DataFrame(columns=["name", "em", "pb", "dm"])
    if "dm" not in df.columns:
        df["dm"] = 100.0
    return df

def save_alimentos(df: pd.DataFrame):
    df.to_csv(ALIM_PATH, index=False)

def load_raciones():
    try:
        with open(RAC_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_raciones(data):
    with open(RAC_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

alimentos_df = load_alimentos()
raciones = load_raciones()

tab1, tab2, tab3, tab4 = st.tabs(["📦 Alimentos", "🥣 Formulador", "🧮 Mixer", "⬇️ Exportar"])

with tab1:
    st.subheader("Catálogo de alimentos")
    st.dataframe(alimentos_df, use_container_width=True)
    with st.expander("➕ Agregar alimento"):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("Nombre", "")
        em = c2.number_input("EM (Mcal/kg MS)", 0.0, 10.0, 2.8, step=0.01)
        pb = c3.number_input("PB (% MS)", 0.0, 100.0, 12.0, step=0.1)
        dm = c4.number_input("DM (%)", 0.0, 100.0, 100.0, step=0.1)
        if st.button("Agregar"):
            if name.strip():
                alimentos_df.loc[len(alimentos_df)] = [name.strip(), em, pb, dm]
                save_alimentos(alimentos_df); st.success(f"Agregado: {name}")
                st.experimental_rerun()
            else:
                st.error("El nombre no puede estar vacío.")

    if st.button("💾 Guardar cambios del catálogo"):
        save_alimentos(alimentos_df); st.success("Catálogo guardado.")

with tab2:
    st.subheader("Formulación de ración (EM/PB y desvíos)")
    c1, c2, c3 = st.columns(3)
    target_em = c1.number_input("EM objetivo", 0.0, 5.0, 2.6, step=0.01)
    target_pb = c2.number_input("PB objetivo", 0.0, 50.0, 13.0, step=0.1)
    ration_name = c3.text_input("Nombre de la ración", "Ración Demo")

    st.caption("Asigná % de inclusión a los alimentos (ideal suma 100%).")
    if alimentos_df.empty:
        st.info("Cargá alimentos en la pestaña anterior.")
    else:
        editable = alimentos_df.copy()
        editable["inclusion_pct"] = 0.0
        grid = st.data_editor(editable, num_rows="dynamic", use_container_width=True, hide_index=True)
        rows = grid[grid["inclusion_pct"] > 0]
        ings = [Ingredient(food=Food(r["name"], float(r["em"]), float(r["pb"]), float(r.get("dm", 100.0))), inclusion_pct=float(r["inclusion_pct"])) for _, r in rows.iterrows()]
        col_a, col_b = st.columns([1,1])
        if col_a.button("Calcular EM/PB"):
            res = analyze_ration(ings, target_em, target_pb)
            sum_pct = sum(rows["inclusion_pct"])
            st.metric("EM (calc)", res.em); st.metric("PB (calc)", res.pb)
            st.metric("Desvío EM", res.dev_em); st.metric("Desvío PB", res.dev_pb)
            st.write(f"Suma de inclusiones: {sum_pct:.2f}%")
            if abs(sum_pct - 100) <= 0.5: st.success("✔ Dentro de ±0,5%")
            else: st.warning("⚠ Alejado de 100%")

        if col_b.button("Guardar ración"):
            payload = {
                "name": ration_name,
                "target_em": target_em,
                "target_pb": target_pb,
                "ingredients": [
                    {"name": r["name"], "em": float(r["em"]), "pb": float(r["pb"]), "dm": float(r.get("dm", 100.0)), "inclusion_pct": float(r["inclusion_pct"])}
                    for _, r in rows.iterrows()
                ]
            }
            raciones.append(payload); save_raciones(raciones)
            st.success(f"Ración '{ration_name}' guardada.")

        if raciones:
            st.markdown("### Raciones guardadas")
            names = [r["name"] for r in raciones]
            pick = st.selectbox("Ver ración", names)
            if pick:
                ra = next(r for r in raciones if r["name"] == pick)
                st.json(ra)

with tab3:
    st.subheader("Descarga de mixer")
    total_kg = st.number_input("Total del mixer (kg, as-fed)", 0.0, 50000.0, 5000.0, step=10.0)
    if not raciones:
        st.info("Guardá primero una ración.")
    else:
        pick = st.selectbox("Ración", [r["name"] for r in raciones])
        ra = next(r for r in raciones if r["name"] == pick)
        ings = [Ingredient(food=Food(x["name"], x["em"], x["pb"], x.get("dm", 100.0)), inclusion_pct=x["inclusion_pct"]) for x in ra["ingredients"]]
        plan = mixer_kg_by_ingredient(ings, total_kg)
        if st.button("Calcular plan de carga"):
            st.dataframe(pd.DataFrame({"Ingrediente": list(plan.keys()), "Kg": list(plan.values())}))

with tab4:
    st.subheader("Exportar CSV")
    col1, col2 = st.columns(2)
    if col1.download_button("Descargar alimentos.csv", data=alimentos_df.to_csv(index=False).encode("utf-8"), file_name="alimentos.csv"):
        st.success("Descarga iniciada.")
    if raciones:
        pick = col2.selectbox("Ración a exportar", [r["name"] for r in raciones])
        ra = next(r for r in raciones if r["name"] == pick)
        export_df = pd.DataFrame(ra["ingredients"])
        export_csv = export_df.to_csv(index=False).encode("utf-8")
        if col2.download_button("Descargar ración.csv", data=export_csv, file_name=f"{pick}.csv"):
            st.success("Descarga iniciada.")

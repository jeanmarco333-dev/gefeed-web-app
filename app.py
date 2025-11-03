# app.py — JM P-Feedlot v0.26 (100% web)
# Requisitos: streamlit, pandas
# Estructura esperada:
#   app.py
#   calc_engine.py
#   requirements.txt
#   data/
#       alimentos.csv
#       raciones.json
#       raciones_base.csv
#       mixers.csv
#       pesos.csv

import os
import json
import io
import zipfile
import datetime
import streamlit as st
import pandas as pd
from calc_engine import Food, Ingredient, analyze_ration, mixer_kg_by_ingredient

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="JM P-Feedlot v0.26 — Web", layout="wide")
st.title("JM P-Feedlot v0.26 — Web")
st.caption("Catálogo con columnas reales • Formulación EM/PB • Mixer as-fed • Corrales • Parámetros • Exportación ZIP")

# --------------------------------------------------------------------------------------
# Paths base
# --------------------------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

ALIM_PATH = os.path.join(DATA_DIR, "alimentos.csv")
RAC_PATH  = os.path.join(DATA_DIR, "raciones.json")
BASE_PATH = os.path.join(DATA_DIR, "raciones_base.csv")
MIXERS_PATH = os.path.join(DATA_DIR, "mixers.csv")
PESOS_PATH  = os.path.join(DATA_DIR, "pesos.csv")
CATALOG_PATH= os.path.join(DATA_DIR, "raciones_catalog.csv")

ALIM_COLS = ["ORIGEN","PRESENTACION","TIPO","MS","TND (%)","PB","EE","COEF ATC","$/KG","EM","ENP2"]

# Crear archivos mínimos si faltan
if not os.path.exists(ALIM_PATH):
    pd.DataFrame(columns=ALIM_COLS).to_csv(ALIM_PATH, index=False, encoding="utf-8")
if not os.path.exists(RAC_PATH):
    with open(RAC_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
if not os.path.exists(MIXERS_PATH):
    pd.DataFrame({"mixer_id":["MX-4200","MX-6000"], "capacidad_kg":[4200,6000]}).to_csv(MIXERS_PATH, index=False, encoding="utf-8")
if not os.path.exists(PESOS_PATH):
    pd.DataFrame({"peso_kg":[150,162.5,175,187.5,200,212.5,225,237.5,250,262.5,275,287.5,300,312.5,325,337.5,350,362.5,375,387.5,400,412.5,425,437.5,450]}).to_csv(PESOS_PATH, index=False, encoding="utf-8")
if not os.path.exists(CATALOG_PATH):
    pd.DataFrame({"cod":[2,3],"nombre":["R-JOSE","term"]}).to_csv(CATALOG_PATH, index=False, encoding="utf-8")
if not os.path.exists(BASE_PATH):
    pd.DataFrame(columns=["tipo_racion","nro_corral","cod_racion","nombre_racion","categ",
                          "PV_kg","CV_pct","AP_preten","nro_cab","mixer_id","capacidad_kg",
                          "kg_turno","AP_obt","turnos","meta_salida","dias_TERM","semanas_TERM","EFC_conv"]
                ).to_csv(BASE_PATH, index=False, encoding="utf-8")

# --------------------------------------------------------------------------------------
# Normalización robusta
# --------------------------------------------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalizar nombres
    rename_map = {}
    for c in list(df.columns):
        c2 = str(c).strip().replace("\ufeff","")
        c2u = (c2.upper()
               .replace("Á","A").replace("É","E").replace("Í","I").replace("Ó","O").replace("Ú","U"))
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

    # asegurar columnas
    for col in ALIM_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[ALIM_COLS]

    # helper numérico
    def _to_num(x, default=0.0):
        if pd.isna(x): return default
        s = str(x).strip()
        if not s: return default
        s = s.replace("\u00A0"," ").replace(" ","")
        s = s.replace("$","").replace("%","")
        s = s.replace(",", ".")
        try: return float(s)
        except: return default

    # convertir numéricas
    for c in ["MS","TND (%)","PB","EE","COEF ATC","$/KG","EM","ENP2"]:
        df[c] = df[c].map(lambda v: _to_num(v, 0.0))

    # escalar MS si viene como fracción
    df.loc[df["MS"] <= 1.0, "MS"] = df.loc[df["MS"] <= 1.0, "MS"] * 100.0
    # TND en fracción
    if (df["TND (%)"] <= 1.0).any() and (df["TND (%)"] > 0).any():
        df.loc[df["TND (%)"] <= 1.0, "TND (%)"] = df.loc[df["TND (%)"] <= 1.0, "TND (%)"] * 100.0

    # textos
    for c in ["ORIGEN","PRESENTACION","TIPO"]:
        df[c] = df[c].fillna("").astype(str).str.strip()

    return df

# --------------------------------------------------------------------------------------
# IO helpers con caché
# --------------------------------------------------------------------------------------
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

@st.cache_data
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

@st.cache_data
def load_mixers() -> pd.DataFrame:
    try:
        df = pd.read_csv(MIXERS_PATH, encoding="utf-8-sig")
    except:
        df = pd.DataFrame({"mixer_id":[], "capacidad_kg":[]})
    df["capacidad_kg"] = pd.to_numeric(df["capacidad_kg"], errors="coerce").fillna(0).astype(float)
    return df

def save_mixers(df: pd.DataFrame):
    df.to_csv(MIXERS_PATH, index=False, encoding="utf-8")

@st.cache_data
def load_pesos() -> pd.DataFrame:
    try:
        df = pd.read_csv(PESOS_PATH, encoding="utf-8-sig")
    except:
        df = pd.DataFrame({"peso_kg":[]})
    df["peso_kg"] = pd.to_numeric(df["peso_kg"], errors="coerce").fillna(0).astype(float)
    df = df[df["peso_kg"]>0].drop_duplicates().sort_values("peso_kg").reset_index(drop=True)
    return df

def save_pesos(df: pd.DataFrame):
    out = df.copy()
    out["peso_kg"] = pd.to_numeric(out["peso_kg"], errors="coerce")
    out = out.dropna().sort_values("peso_kg")
    out.to_csv(PESOS_PATH, index=False, encoding="utf-8")

@st.cache_data
def load_catalog() -> pd.DataFrame:
    try:
        df = pd.read_csv(CATALOG_PATH, encoding="utf-8-sig")
    except:
        df = pd.DataFrame({"cod":[], "nombre":[]})
    return df

def save_catalog(df: pd.DataFrame):
    df.to_csv(CATALOG_PATH, index=False, encoding="utf-8")

@st.cache_data
def load_base() -> pd.DataFrame:
    try:
        df = pd.read_csv(BASE_PATH, encoding="utf-8-sig")
    except:
        df = pd.DataFrame()
    return df

def save_base(df: pd.DataFrame):
    df.to_csv(BASE_PATH, index=False, encoding="utf-8")

# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📦 Alimentos", "🥣 Formulador", "🧮 Mixer", "⬇️ Exportar", "📊 Corrales", "⚙️ Parámetros"]
)

# --------------------------------------------------------------------------------------
# 📦 Alimentos (editable)
# --------------------------------------------------------------------------------------
with tab1:
    st.subheader("Catálogo de alimentos")
    st.caption("Editá directamente en la tabla. Luego tocá “Guardar cambios”.")
    col_fr, col_save = st.columns([1,1])
    if col_fr.button("🔄 Forzar recarga de catálogo"):
        try: st.cache_data.clear()
        except: pass
        st.rerun()

    alimentos_df = load_alimentos()
    grid_alim = st.data_editor(
        alimentos_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="grid_alimentos"
    )

    if col_save.button("💾 Guardar cambios del catálogo"):
        save_alimentos(grid_alim)
        st.success("Catálogo guardado.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()

    with st.expander("➕ Agregar alimento (formulario)"):
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
                grid_alim = pd.concat([grid_alim, pd.DataFrame([new_row])], ignore_index=True)
                save_alimentos(grid_alim)
                st.success(f"Agregado: {origen}")
                try: st.cache_data.clear()
                except: pass
                st.rerun()
            else:
                st.error("El campo ORIGEN no puede estar vacío.")

# --------------------------------------------------------------------------------------
# 🥣 Formulador (EM/PB y desvíos)
# --------------------------------------------------------------------------------------
with tab2:
    st.subheader("Formulación de ración (EM/PB)")
    c1, c2, c3 = st.columns(3)
    target_em = c1.number_input("EM objetivo (Mcal/kg MS)", 0.0, 10.0, 2.6, step=0.01)
    target_pb = c2.number_input("PB objetivo (% MS)", 0.0, 100.0, 13.0, step=0.1)
    ration_name = c3.text_input("Nombre de la ración", "Ración Demo")

    st.markdown("### Ingredientes (% inclusión en base MS)")
    alimentos_df = load_alimentos()
    if alimentos_df.empty or alimentos_df["ORIGEN"].eq("").all():
        st.info("Cargá el catálogo de alimentos en la pestaña Alimentos/Parámetros.")
    else:
        editable = alimentos_df.copy()
        if "inclusion_pct" not in editable.columns:
            editable["inclusion_pct"] = 0.0

        st.caption("Asigná % inclusión (ideal suma 100% ±0,5).")
        grid = st.data_editor(
            editable[ALIM_COLS + ["inclusion_pct"]],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="grid_formulador"
        )

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
            ings.append(Ingredient(food=food, inclusion_pct=float(pd.to_numeric(r.get("inclusion_pct", 0.0), errors="coerce") or 0.0)))

        col_a, col_b = st.columns([1,1])
        if col_a.button("Calcular EM/PB"):
            res = analyze_ration(ings, target_em, target_pb)
            sum_pct = float(rows["inclusion_pct"].sum())
            st.metric("EM calculada (Mcal/kg MS)", res.em)
            st.metric("PB calculada (% MS)", res.pb)
            st.metric("Desvío EM", res.dev_em)
            st.metric("Desvío PB", res.dev_pb)
            st.write(f"**Suma de inclusiones**: {sum_pct:.2f}%")

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
            rs = load_raciones()
            rs.append(payload)
            save_raciones(rs)
            st.success(f"Ración '{ration_name}' guardada.")
            try: st.cache_data.clear()
            except: pass
            st.rerun()

# --------------------------------------------------------------------------------------
# 🧮 Mixer (as-fed con %MS)
# --------------------------------------------------------------------------------------
with tab3:
    st.subheader("Cálculo de descarga de mixer (as-fed)")
    total_kg = st.number_input("Total del mixer (kg, as-fed)", 0.0, 50000.0, 5000.0, step=10.0)

    raciones = load_raciones()
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
            ings.append(Ingredient(
                food=food,
                inclusion_pct=float(pd.to_numeric(x.get("inclusion_pct", 0.0), errors="coerce") or 0.0)
            ))

        if st.button("Calcular plan de carga"):
            plan = mixer_kg_by_ingredient(ings, total_kg)
            if plan:
                df_plan = pd.DataFrame({"Ingrediente": list(plan.keys()), "Kg (as-fed)": list(plan.values())})
                st.dataframe(df_plan, use_container_width=True)
                st.caption(f"Total calculado: {sum(plan.values()):.1f} kg (debería ≈ {total_kg:.1f} kg)")
            else:
                st.warning("No hay ingredientes con inclusión > 0 o el total del mixer es 0.")

# --------------------------------------------------------------------------------------
# ⬇️ Exportar (ZIP con TODO)
# --------------------------------------------------------------------------------------
with tab4:
    st.subheader("⬇️ Exportar datos y simulaciones")
    st.caption("Descargá todas las bases en un único ZIP (alimentos, raciones, corrales, mixers, pesos, catálogo).")

    files_to_zip = []
    for fname in ["alimentos.csv", "raciones.json", "raciones_base.csv", "mixers.csv", "pesos.csv", "raciones_catalog.csv"]:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            files_to_zip.append(fpath)

    if files_to_zip:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files_to_zip:
                zf.write(f, arcname=os.path.basename(f))
        buffer.seek(0)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        st.download_button(
            "⬇️ Descargar ZIP (todas las bases)",
            data=buffer,
            file_name=f"simulaciones_{ts}.zip",
            mime="application/zip"
        )
    else:
        st.info("No se encontraron archivos en /data para exportar.")

# --------------------------------------------------------------------------------------
# 📊 Corrales / Raciones Base
# --------------------------------------------------------------------------------------
with tab5:
    st.subheader("📊 Base de Corrales y Raciones")
    cat_df = load_catalog()
    mix_df = load_mixers()
    base   = load_base()

    # Choices
    tipos = ["Terminación","Recría"]
    categorias = ["va","nov"]
    pesos_lista = load_pesos()["peso_kg"].tolist()
    mixers = mix_df["mixer_id"].tolist()
    mixer_cap_map = dict(zip(mix_df["mixer_id"], mix_df["capacidad_kg"]))
    nombre_to_cod = dict(zip(cat_df["nombre"], cat_df["cod"]))

    # Si base vacía, inicializar 1..20
    if base.empty:
        base = pd.DataFrame({
            "tipo_racion": ["Terminación"]*20,
            "nro_corral": list(range(1,21)),
            "cod_racion": ["" for _ in range(20)],
            "nombre_racion": ["" for _ in range(20)],
            "categ": ["va"]*20,
            "PV_kg": [275]*20,
            "CV_pct": [2.8]*20,
            "AP_preten": [1.0]*20,
            "nro_cab": [0]*20,
            "mixer_id": [mixers[0] if mixers else ""]*20,
            "capacidad_kg": [mixer_cap_map.get(mixers[0],0) if mixers else 0]*20,
            "kg_turno": [0.0]*20,
            "AP_obt": [1.0]*20,
            "turnos": [4]*20,
            "meta_salida": [350]*20,
            "dias_TERM": [0]*20,
            "semanas_TERM":[0.0]*20,
            "EFC_conv": [0.0]*20
        })

    colcfg = {
        "tipo_racion": st.column_config.SelectboxColumn("tipo de ración", options=tipos, required=True),
        "nro_corral": st.column_config.NumberColumn("n° de Corral", min_value=1, max_value=9999, step=1),
        "nombre_racion": st.column_config.SelectboxColumn("nombre la ración", options=[""]+cat_df["nombre"].astype(str).tolist(), help="Autocompleta código"),
        "categ": st.column_config.SelectboxColumn("categ", options=categorias, help="va / nov"),
        "PV_kg": st.column_config.SelectboxColumn("PV (kg)", options=pesos_lista),
        "CV_pct": st.column_config.NumberColumn("CV (%)", min_value=0.0, max_value=20.0, step=0.1),
        "AP_preten": st.column_config.NumberColumn("AP (kg) PRETEN", min_value=0.0, max_value=5.0, step=0.1),
        "nro_cab": st.column_config.NumberColumn("NRO CAB (und)", min_value=0, max_value=100000, step=1),
        "mixer_id": st.column_config.SelectboxColumn("Mixer", options=[""]+mixers, help="Trae capacidad"),
        "capacidad_kg": st.column_config.NumberColumn("capacidad (kg)", min_value=0, max_value=200000, step=10),
        "kg_turno": st.column_config.NumberColumn("kg por turno (editable)", min_value=0.0, max_value=200000.0, step=1.0),
        "AP_obt": st.column_config.NumberColumn("AP OBT (kg/día)", min_value=0.0, max_value=5.0, step=0.01),
        "turnos": st.column_config.NumberColumn("turnos", min_value=1, max_value=24, step=1),
        "meta_salida": st.column_config.NumberColumn("META DE SALIDA (kg)", min_value=0, max_value=2000, step=5),
        "dias_TERM": st.column_config.NumberColumn("Días-TERM (calc)", disabled=True),
        "semanas_TERM": st.column_config.NumberColumn("Semanas-TERM (calc)", disabled=True),
        "EFC_conv": st.column_config.NumberColumn("EFC (calc)", disabled=True),
    }

    st.caption("“nombre la ración” autocompleta ‘cod ración’. ‘capacidad’ viene de mixers.csv. Los campos de cálculo se actualizan al Guardar.")

    def enrich_and_calc(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # autocompletar cod desde nombre
        df["cod_racion"] = df.apply(lambda r: nombre_to_cod.get(str(r.get("nombre_racion","")), ""), axis=1)
        # capacidad por mixer
        df["capacidad_kg"] = df.apply(lambda r: mixer_cap_map.get(str(r.get("mixer_id","")), 0), axis=1)

        # sugerencia kg_turno (podés reemplazar por tu fórmula real)
        def kg_turno_calc(r):
            try:
                return round( (float(r["PV_kg"])*(float(r["CV_pct"])/100.0) * float(r["nro_cab"])) / max(float(r["turnos"]),1.0), 1 )
            except:
                return 0.0
        df["kg_turno_calc"] = df.apply(kg_turno_calc, axis=1)
        df["kg_turno"] = df.apply(lambda r: r["kg_turno"] if float(r.get("kg_turno",0) or 0)>0 else r["kg_turno_calc"], axis=1)

        # Días / Semanas a meta
        def dias_term(r):
            try:
                delta = float(r["meta_salida"]) - float(r["PV_kg"])
                ap = max(float(r["AP_obt"]), 0.0001)
                d = max(delta/ap, 0.0)
                return int(round(d))
            except:
                return 0
        df["dias_TERM"] = df.apply(dias_term, axis=1)
        df["semanas_TERM"] = (df["dias_TERM"] / 7.0).round(1)

        # EFC ≈ (kg alimento / hd / día) / AP_obt
        def efc(r):
            try:
                hd = max(float(r["nro_cab"]),1.0)
                kg_dia_hd = (float(r["kg_turno"]) * float(r["turnos"])) / hd
                ap = max(float(r["AP_obt"]), 0.0001)
                return round(kg_dia_hd / ap, 2)
            except:
                return 0.0
        df["EFC_conv"] = df.apply(efc, axis=1)

        return df

    grid = st.data_editor(
        enrich_and_calc(base),
        column_config=colcfg,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="grid_corrales"
    )

    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("💾 Guardar base"):
        out = grid.copy()
        if "kg_turno_calc" in out.columns:
            out = out.drop(columns=["kg_turno_calc"])
        save_base(out)
        st.success("Base guardada.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()

    if c2.button("🔄 Recargar (desde CSV)"):
        try: st.cache_data.clear()
        except: pass
        st.rerun()

# --------------------------------------------------------------------------------------
# ⚙️ Parámetros técnicos (alimentos, mixers, pesos)
# --------------------------------------------------------------------------------------
with tab6:
    st.subheader("⚙️ Parámetros técnicos")
    st.caption("Mantené las bases: alimentos, mixers y lista de PV (kg).")

    # Alimentos
    st.markdown("### Catálogo de alimentos")
    alim_df = load_alimentos()
    grid_alim_p = st.data_editor(
        alim_df,
        num_rows="dynamic", use_container_width=True, hide_index=True, key="param_alimentos"
    )
    c1, c2 = st.columns(2)
    if c1.button("💾 Guardar alimentos (parámetros)"):
        save_alimentos(grid_alim_p)
        st.success("Alimentos guardados.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()

    st.markdown("---")
    # Mixers
    st.markdown("### Mixers (capacidad)")
    mix_df = load_mixers()
    grid_mix = st.data_editor(
        mix_df,
        column_config={
            "mixer_id": st.column_config.TextColumn("Mixer ID"),
            "capacidad_kg": st.column_config.NumberColumn("Capacidad (kg)", min_value=0, step=10)
        },
        num_rows="dynamic", use_container_width=True, hide_index=True, key="param_mixers"
    )
    d1, d2 = st.columns(2)
    if d1.button("💾 Guardar mixers"):
        save_mixers(grid_mix)
        st.success("Mixers guardados.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()

    st.markdown("---")
    # Pesos
    st.markdown("### PV (kg) — lista de opciones")
    pesos_df = load_pesos()
    grid_pes = st.data_editor(
        pesos_df,
        column_config={"peso_kg": st.column_config.NumberColumn("PV (kg)", min_value=1.0, max_value=2000.0, step=0.5)},
        num_rows="dynamic", use_container_width=True, hide_index=True, key="param_pesos"
    )
    p1, p2 = st.columns(2)
    if p1.button("💾 Guardar PV (kg)"):
        save_pesos(grid_pes)
        st.success("Lista de PV guardada.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()

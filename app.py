# app.py — JM P-Feedlot v0.26 (100% web)
# Pestañas: 📦 Alimentos | 🧮 Mixer | ⬇️ Exportar | 📊 Corrales | ⚙️ Parámetros | 🧾 Creador/Editor de raciones
# Estructura:
#   app.py, calc_engine.py, requirements.txt
#   data/: alimentos.csv, raciones_base.csv, mixers.csv, pesos.csv, raciones_catalog.csv, raciones_recipes.csv

import os, io, zipfile, datetime
import streamlit as st
import pandas as pd
from calc_engine import Food, Ingredient, mixer_kg_by_ingredient

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
st.set_page_config(page_title="JM P-Feedlot v0.26 — Web", layout="wide")
st.title("JM P-Feedlot v0.26 — Web")
st.caption("Catálogo • Mixer as-fed • Corrales • Parámetros • Raciones • Export ZIP")

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

ALIM_PATH = os.path.join(DATA_DIR, "alimentos.csv")
BASE_PATH = os.path.join(DATA_DIR, "raciones_base.csv")
MIXERS_PATH = os.path.join(DATA_DIR, "mixers.csv")
PESOS_PATH  = os.path.join(DATA_DIR, "pesos.csv")
CATALOG_PATH= os.path.join(DATA_DIR, "raciones_catalog.csv")
RECIPES_PATH= os.path.join(DATA_DIR, "raciones_recipes.csv")
REQENER_PATH= os.path.join(DATA_DIR, "requerimientos_energeticos.csv")
REQPROT_PATH= os.path.join(DATA_DIR, "requerimiento_proteico.csv")

ALIM_COLS = ["ORIGEN","PRESENTACION","TIPO","MS","TND (%)","PB","EE","COEF ATC","$/KG","EM","ENP2"]
REQENER_COLS = ["peso","cat","requerimiento_energetico","ap"]
REQPROT_COLS = ["peso","cat","ap","req_proteico"]

# Crear archivos mínimos si faltan
if not os.path.exists(ALIM_PATH):
    pd.DataFrame(columns=ALIM_COLS).to_csv(ALIM_PATH, index=False, encoding="utf-8")
if not os.path.exists(MIXERS_PATH):
    pd.DataFrame({"mixer_id":["MX-4200","MX-6000"], "capacidad_kg":[4200,6000]}).to_csv(MIXERS_PATH, index=False, encoding="utf-8")
if not os.path.exists(PESOS_PATH):
    pd.DataFrame({"peso_kg":[150,162.5,175,187.5,200,212.5,225,237.5,250,262.5,275,287.5,300,312.5,325,337.5,350,362.5,375,387.5,400,412.5,425,437.5,450]}).to_csv(PESOS_PATH, index=False, encoding="utf-8")
if not os.path.exists(CATALOG_PATH):
    pd.DataFrame({"id":[1,2,3],"nombre":["R-JOSE","term","R-DTTE"],"etapa":["RECRIA","RECRIA","RECRIA"]}).to_csv(CATALOG_PATH, index=False, encoding="utf-8")
if not os.path.exists(RECIPES_PATH):
    pd.DataFrame(columns=["id_racion","nombre_racion","ingrediente","pct_ms"]).to_csv(RECIPES_PATH, index=False, encoding="utf-8")
if not os.path.exists(BASE_PATH):
    pd.DataFrame(columns=[
        "tipo_racion","nro_corral","cod_racion","nombre_racion","categ",
        "PV_kg","CV_pct","AP_preten","nro_cab","mixer_id","capacidad_kg",
        "kg_turno","AP_obt","turnos","meta_salida","dias_TERM","semanas_TERM","EFC_conv"
    ]).to_csv(BASE_PATH, index=False, encoding="utf-8")
if not os.path.exists(REQENER_PATH):
    pd.DataFrame(columns=["peso","cat","requerimiento_energetico","ap"]).to_csv(REQENER_PATH, index=False, encoding="utf-8")
if not os.path.exists(REQPROT_PATH):
    pd.DataFrame(columns=["peso","cat","ap","req_proteico"]).to_csv(REQPROT_PATH, index=False, encoding="utf-8")

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
    out.dropna().sort_values("peso_kg").to_csv(PESOS_PATH, index=False, encoding="utf-8")

@st.cache_data
def load_reqener() -> pd.DataFrame:
    try:
        df = pd.read_csv(REQENER_PATH, encoding="utf-8-sig")
    except Exception:
        df = pd.DataFrame(columns=REQENER_COLS)

    for col in REQENER_COLS:
        if col not in df.columns:
            df[col] = None
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
        if cname in ("peso", "peso_kg", "pv", "pv_kg"):
            rename_map[col] = "peso"
        elif cname in ("cat", "categoria", "categoría"):
            rename_map[col] = "cat"
        elif cname in ("ap", "ap_kg_dia", "ap_kg/dia", "ap_kg-dia"):
            rename_map[col] = "ap"
        elif cname in ("req_proteico", "requerimiento_proteico", "proteina", "proteína"):
            rename_map[col] = "req_proteico"
    df = df.rename(columns=rename_map)

    for col in REQPROT_COLS:
        if col not in df.columns:
            df[col] = None
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
        return pd.DataFrame({"id":[], "nombre":[], "etapa":[]})

    # Normalizar encabezados (maneja archivos con mayúsculas/espacios)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Asegurar columnas esperadas
    if "id" not in df.columns:
        df["id"] = pd.Series(dtype="int64")
    if "nombre" not in df.columns:
        df["nombre"] = pd.Series(dtype="object")
    if "etapa" not in df.columns:
        df["etapa"] = pd.Series(dtype="object")

    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int)
    df["nombre"] = df["nombre"].fillna("").astype(str)
    df["etapa"] = df["etapa"].fillna("").astype(str)

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

    lookup = {}
    for _, row in alimentos.iterrows():
        nombre = str(row.get("ORIGEN", "")).strip()
        if not nombre:
            continue
        lookup[nombre.lower()] = row

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
                "inclusion_pct": _num(ing.get("pct_ms", 0.0), 0.0)
            })

        ingredientes = [i for i in ingredientes if i["inclusion_pct"] > 0]
        if not ingredientes:
            continue

        raciones.append({"id": rid, "nombre": nombre, "ingredientes": ingredientes})

    return raciones

@st.cache_data
def load_base() -> pd.DataFrame:
    try: return pd.read_csv(BASE_PATH, encoding="utf-8-sig")
    except: return pd.DataFrame()

def save_base(df: pd.DataFrame):
    df.to_csv(BASE_PATH, index=False, encoding="utf-8")

# ------------------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📦 Alimentos", "🧮 Mixer", "⬇️ Exportar", "📊 Corrales", "⚙️ Parámetros", "🧾 Creador/Editor de raciones"]
)

# ------------------------------------------------------------------------------
# 📦 Alimentos
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Catálogo de alimentos")
    col_fr, col_save = st.columns([1,1])
    if col_fr.button("🔄 Forzar recarga de catálogo"):
        try: st.cache_data.clear()
        except: pass
        st.rerun()

    alimentos_df = load_alimentos()
    grid_alim = st.data_editor(alimentos_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="grid_alimentos")

    if col_save.button("💾 Guardar cambios del catálogo"):
        save_alimentos(grid_alim)
        st.success("Catálogo guardado.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()

# ------------------------------------------------------------------------------
# 🧮 Mixer
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Cálculo de descarga de mixer (as-fed)")
    total_kg = st.number_input("Total del mixer (kg, as-fed)", 0.0, 50000.0, 5000.0, step=10.0)

    raciones = build_raciones_from_recipes()
    if not raciones:
        st.info("Definí recetas en la pestaña 🧾 Creador/Editor de raciones.")
    else:
        pick = st.selectbox("Ración a usar", [r["nombre"] for r in raciones])
        ra = next(r for r in raciones if r["nombre"] == pick)

        ings = []
        for x in ra["ingredientes"]:
            dm_pct = float(pd.to_numeric(x.get("MS", 100.0), errors="coerce") or 100.0)
            food = Food(name=str(x.get("ORIGEN", "Ingrediente")),
                        em=float(pd.to_numeric(x.get("EM", 0.0), errors="coerce") or 0.0),
                        pb=float(pd.to_numeric(x.get("PB", 0.0), errors="coerce") or 0.0),
                        dm=dm_pct)
            ings.append(Ingredient(food=food, inclusion_pct=float(pd.to_numeric(x.get("inclusion_pct", 0.0), errors="coerce") or 0.0)))

        if st.button("Calcular plan de carga"):
            plan = mixer_kg_by_ingredient(ings, total_kg)
            if plan:
                df_plan = pd.DataFrame({"Ingrediente": list(plan.keys()), "Kg (as-fed)": list(plan.values())})
                st.dataframe(df_plan, use_container_width=True)
                st.caption(f"Total calculado: {sum(plan.values()):.1f} kg (debería ≈ {total_kg:.1f} kg)")
            else:
                st.warning("No hay ingredientes con inclusión > 0 o el total del mixer es 0.")

# ------------------------------------------------------------------------------
# ⬇️ Exportar
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("⬇️ Exportar datos y simulaciones")
    files_to_zip = []
    for fname in ["alimentos.csv","raciones_base.csv","mixers.csv","pesos.csv","raciones_catalog.csv","raciones_recipes.csv"]:
        f = os.path.join(DATA_DIR, fname)
        if os.path.exists(f): files_to_zip.append(f)
    if files_to_zip:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files_to_zip:
                zf.write(f, arcname=os.path.basename(f))
        buffer.seek(0)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        st.download_button("⬇️ Descargar ZIP (todas las bases)", data=buffer, file_name=f"simulaciones_{ts}.zip", mime="application/zip")
    else:
        st.info("No se encontraron archivos en /data para exportar.")

# ------------------------------------------------------------------------------
# 📊 Corrales
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("📊 Base de Corrales y Raciones")
    cat_df = load_catalog()
    mix_df = load_mixers()
    base   = load_base()

    tipos = ["Terminación","Recría"]
    categorias = ["va","nov"]
    pesos_lista = load_pesos()["peso_kg"].tolist()
    mixers = mix_df["mixer_id"].tolist()
    mixer_cap_map = dict(zip(mix_df["mixer_id"], mix_df["capacidad_kg"]))
    nombre_to_id = dict(zip(cat_df["nombre"], cat_df["id"]))

    if base.empty:
        base = pd.DataFrame({
            "tipo_racion": ["Terminación"]*20,
            "nro_corral": list(range(1,21)),
            "cod_racion": ["" for _ in range(20)],
            "nombre_racion": ["" for _ in range(20)],
            "categ": ["va"]*20,
            "PV_kg": [275]*20, "CV_pct": [2.8]*20, "AP_preten": [1.0]*20,
            "nro_cab": [0]*20, "mixer_id": [mixers[0] if mixers else ""]*20,
            "capacidad_kg": [mixer_cap_map.get(mixers[0],0) if mixers else 0]*20,
            "kg_turno": [0.0]*20, "AP_obt": [1.0]*20, "turnos": [4]*20,
            "meta_salida": [350]*20, "dias_TERM": [0]*20, "semanas_TERM":[0.0]*20, "EFC_conv": [0.0]*20
        })

    colcfg = {
        "tipo_racion": st.column_config.SelectboxColumn("tipo de ración", options=tipos, required=True),
        "nro_corral": st.column_config.NumberColumn("n° de Corral", min_value=1, max_value=9999, step=1),
        "nombre_racion": st.column_config.SelectboxColumn("nombre la ración", options=[""]+cat_df["nombre"].astype(str).tolist(), help="Autocompleta código"),
        "categ": st.column_config.SelectboxColumn("categ", options=categorias),
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

    def enrich_and_calc(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["cod_racion"] = df.apply(lambda r: nombre_to_id.get(str(r.get("nombre_racion","")), ""), axis=1)
        df["capacidad_kg"] = df.apply(lambda r: mixer_cap_map.get(str(r.get("mixer_id","")), 0), axis=1)

        # Sugerencia kg_turno (ajustá a tu fórmula si querés)
        def kg_turno_calc(r):
            try:
                return round( (float(r["PV_kg"])*(float(r["CV_pct"])/100.0) * float(r["nro_cab"])) / max(float(r["turnos"]),1.0), 1 )
            except: return 0.0
        df["kg_turno_calc"] = df.apply(kg_turno_calc, axis=1)
        df["kg_turno"] = df.apply(lambda r: r["kg_turno"] if float(r.get("kg_turno",0) or 0)>0 else r["kg_turno_calc"], axis=1)

        # Ajuste por receta (MS ponderada → as-fed)
        recipes = load_recipes()
        alimentos = load_alimentos()[["ORIGEN","MS"]]
        ms_map = {}
        for nombre in df["nombre_racion"].dropna().unique():
            sub = recipes[recipes["nombre_racion"]==nombre]
            if not sub.empty:
                sub = sub.merge(alimentos, left_on="ingrediente", right_on="ORIGEN", how="left")
                sub["MS_frac"] = pd.to_numeric(sub["MS"], errors="coerce").fillna(100.0)/100.0
                w = (pd.to_numeric(sub["pct_ms"], errors="coerce").fillna(0.0)/100.0 * sub["MS_frac"]).sum()
                ms_map[nombre] = float(w) if w>0 else 1.0
        def kg_turno_asfed(r):
            try:
                ms = ms_map.get(str(r["nombre_racion"]), 1.0)
                base = float(r["kg_turno"])
                return round(base / max(ms, 1e-6), 1)
            except: return 0.0
        df["kg_turno_asfed_calc"] = df.apply(kg_turno_asfed, axis=1)

        # Días / Semanas a meta
        def dias_term(r):
            try:
                delta = float(r["meta_salida"]) - float(r["PV_kg"])
                ap = max(float(r["AP_obt"]), 0.0001)
                d = max(delta/ap, 0.0)
                return int(round(d))
            except: return 0
        df["dias_TERM"] = df.apply(dias_term, axis=1)
        df["semanas_TERM"] = (df["dias_TERM"] / 7.0).round(1)

        # EFC ≈ (kg alimento / hd / día) / AP_obt — usando as-fed calculado
        def efc(r):
            try:
                hd = max(float(r["nro_cab"]),1.0)
                kg_dia_hd = (float(r["kg_turno_asfed_calc"]) * float(r["turnos"])) / hd
                ap = max(float(r["AP_obt"]), 0.0001)
                return round(kg_dia_hd / ap, 2)
            except: return 0.0
        df["EFC_conv"] = df.apply(efc, axis=1)
        return df

    grid = st.data_editor(
        enrich_and_calc(base),
        column_config=colcfg,
        num_rows="dynamic", use_container_width=True, hide_index=True, key="grid_corrales"
    )

    c1, c2, c3 = st.columns([1,1,1])
    if c1.button("💾 Guardar base"):
        out = grid.copy()
        for col in ["kg_turno_calc","kg_turno_asfed_calc"]:
            if col in out.columns: out = out.drop(columns=[col])
        save_base(out); st.success("Base guardada.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()
    if c2.button("🔄 Recargar (desde CSV)"):
        try: st.cache_data.clear()
        except: pass
        st.rerun()

# ------------------------------------------------------------------------------
# ⚙️ Parámetros
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("⚙️ Parámetros técnicos")

    st.markdown("### Catálogo de alimentos")
    alim_df = load_alimentos()
    grid_alim_p = st.data_editor(alim_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="param_alimentos")
    c1, c2 = st.columns(2)
    if c1.button("💾 Guardar alimentos (parámetros)"):
        save_alimentos(grid_alim_p); st.success("Alimentos guardados.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()

    st.markdown("---")
    st.markdown("### Mixers (capacidad)")
    mix_df = load_mixers()
    grid_mix = st.data_editor(
        mix_df,
        column_config={"mixer_id": st.column_config.TextColumn("Mixer ID"),
                       "capacidad_kg": st.column_config.NumberColumn("Capacidad (kg)", min_value=0, step=10)},
        num_rows="dynamic", use_container_width=True, hide_index=True, key="param_mixers")
    if c2.button("💾 Guardar mixers"):
        save_mixers(grid_mix); st.success("Mixers guardados.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()

    st.markdown("---")
    st.markdown("### PV (kg) — lista de opciones")
    pesos_df = load_pesos()
    grid_pes = st.data_editor(
        pesos_df,
        column_config={"peso_kg": st.column_config.NumberColumn("PV (kg)", min_value=1.0, max_value=2000.0, step=0.5)},
        num_rows="dynamic", use_container_width=True, hide_index=True, key="param_pesos")
    p1, p2 = st.columns(2)
    if p1.button("💾 Guardar PV (kg)"):
        save_pesos(grid_pes); st.success("Lista de PV guardada.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()

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
        num_rows="dynamic", use_container_width=True, hide_index=True, key="param_reqener"
    )
    r1, r2 = st.columns(2)
    if r1.button("💾 Guardar requerimientos energéticos"):
        save_reqener(grid_req); st.success("Requerimientos energéticos guardados.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()
    if r2.button("🔄 Recargar requerimientos"):
        try: st.cache_data.clear()
        except: pass
        st.rerun()

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
        num_rows="dynamic", use_container_width=True, hide_index=True, key="param_reqprot"
    )
    rp1, rp2 = st.columns(2)
    if rp1.button("💾 Guardar requerimientos proteicos"):
        save_reqprot(grid_reqprot); st.success("Requerimientos proteicos guardados.")
        try: st.cache_data.clear()
        except: pass
        st.rerun()
    if rp2.button("🔄 Recargar requerimientos proteicos"):
        try: st.cache_data.clear()
        except: pass
        st.rerun()

# ------------------------------------------------------------------------------
# 🧾 Creador/Editor de raciones (catálogo + recetas)
# ------------------------------------------------------------------------------
with tab6:
    st.subheader("🧾 Creador/Editor de raciones")
    st.caption("Definí hasta 6 ingredientes por ración (suma 100% MS).")

    cat = load_catalog()
    rec = load_recipes()
    alimentos_df = load_alimentos()
    opciones_ingred = [""] + sorted(alimentos_df["ORIGEN"].dropna().astype(str).unique().tolist())

    st.markdown("### Catálogo de raciones")
    grid_cat = st.data_editor(
        cat,
        column_config={
            "id": st.column_config.NumberColumn("ID", min_value=1, step=1),
            "nombre": st.column_config.TextColumn("NOMBRE"),
            "etapa": st.column_config.SelectboxColumn("ETAPA", options=["","RECRIA","terminacion"])
        },
        num_rows="dynamic", use_container_width=True, hide_index=True, key="grid_rac_catalog"
    )
    c1, c2 = st.columns(2)
    if c1.button("💾 Guardar catálogo"):
        if grid_cat["id"].duplicated().any():
            st.error("IDs duplicados en el catálogo.")
        else:
            save_catalog(grid_cat); st.success("Catálogo guardado.")
            try: st.cache_data.clear()
            except: pass
            st.rerun()

    st.markdown("---")
    st.markdown("### Receta por ración (máx. 6 ingredientes)")
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

        grid_rec = st.data_editor(
            rec_r[["id_racion","nombre_racion","ingrediente","pct_ms"]],
            column_config=colcfg, use_container_width=True, hide_index=True, num_rows=6, key="grid_rac_recipe"
        )

        total_pct = float(pd.to_numeric(grid_rec["pct_ms"], errors="coerce").fillna(0.0).sum())
        st.write(f"**Suma % (MS):** {total_pct:.1f}%")
        if abs(total_pct-100) <= 0.5: st.success("✔ 100% ±0,5")
        else: st.warning("⚠️ Lo ideal es 100% (±0,5).")

        if st.button("💾 Guardar receta de esta ración"):
            out = grid_rec.copy()
            out["id_racion"] = rid; out["nombre_racion"] = pick
            out = out[out["ingrediente"].astype(str).str.strip()!=""]
            rec2 = load_recipes(); rec2 = rec2[rec2["id_racion"]!=rid]
            rec2 = pd.concat([rec2, out], ignore_index=True)
            save_recipes(rec2); st.success("Receta guardada.")
            try: st.cache_data.clear()
            except: pass
            st.rerun()

        st.markdown("#### Vista previa (MS ponderada y factor as-fed)")
        df_view = grid_rec.copy()
        df_view = df_view[df_view["ingrediente"].astype(str).str.strip()!=""]
        if not df_view.empty:
            df_view = df_view.merge(alimentos_df[["ORIGEN","MS"]], left_on="ingrediente", right_on="ORIGEN", how="left")
            df_view = df_view.rename(columns={"MS":"MS_%"}).drop(columns=["ORIGEN"])
            df_view["MS_frac"] = pd.to_numeric(df_view["MS_%"], errors="coerce").fillna(100.0)/100.0
            w_ms = (df_view["pct_ms"]/100.0 * df_view["MS_frac"]).sum()
            w_ms = float(w_ms) if w_ms>0 else 1.0
            factor_asfed = 1.0 / w_ms
            st.write(f"**MS ponderada:** {w_ms*100:.1f}%  →  **Factor as-fed:** ×{factor_asfed:.3f}")
            st.dataframe(df_view[["ingrediente","pct_ms","MS_%"]], use_container_width=True)
        else:
            st.info("Agregá ingredientes para ver la MS ponderada.")

# calc_engine.py — Motor de cálculo (EM/PB y mixer as-fed)
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd

@dataclass
class Food:
    name: str
    em: float   # Mcal/kg MS
    pb: float   # % MS
    dm: float = 100.0  # % MS (DM = MS)

@dataclass
class Ingredient:
    food: Food
    inclusion_pct: float  # % de la ración (base MS)

@dataclass
class RationResult:
    em: float
    pb: float
    dev_em: float
    dev_pb: float

def analyze_ration(ingredients: List[Ingredient], target_em: float, target_pb: float) -> RationResult:
    if not ingredients:
        return RationResult(0.0, 0.0, -target_em, -target_pb)
    total_pct = sum(i.inclusion_pct for i in ingredients) or 1.0
    norm = 100.0 / total_pct
    em = pb = 0.0
    for ing in ingredients:
        share = ing.inclusion_pct * norm / 100.0
        em += share * float(ing.food.em or 0.0)
        pb += share * float(ing.food.pb or 0.0)
    em = round(em, 3)
    pb = round(pb, 3)
    return RationResult(em=em, pb=pb, dev_em=round(em - target_em, 3), dev_pb=round(pb - target_pb, 3))

def mixer_kg_by_ingredient(ingredients: List[Ingredient], mixer_total_kg_as_fed: float) -> Dict[str, float]:
    if not ingredients or mixer_total_kg_as_fed <= 0:
        return {}
    total_pct = sum(i.inclusion_pct for i in ingredients) or 1.0
    norm = 100.0 / total_pct
    shares_ms = [(i.food.name, (i.inclusion_pct * norm) / 100.0, float(i.food.dm or 100.0)/100.0) for i in ingredients]
    asfed = {n: (s / d if d > 0 else s) for (n, s, d) in shares_ms}
    total_asfed = sum(asfed.values()) or 1.0
    factor = mixer_total_kg_as_fed / total_asfed
    return {n: round(v * factor, 1) for n, v in asfed.items()}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        val = float(pd.to_numeric(value, errors="coerce"))
    except Exception:
        return default
    return default if pd.isna(val) else val


def ration_split_from_pv_cv(
    rec_df: pd.DataFrame,
    alimentos_df: pd.DataFrame,
    pv_kg: float,
    cv_pct: float,
) -> dict:
    """Calcula la distribución diaria de MS/as-fed/EM/PB/costo."""

    consumo_ms = _safe_float(pv_kg) * (_safe_float(cv_pct) / 100.0)

    resultado = {
        "Consumo_MS_dia": consumo_ms,
        "asfed_total_kg_dia": 0.0,
        "EM_Mcal_dia": 0.0,
        "PB_g_dia": 0.0,
        "costo_dia": 0.0,
        "detalle": [],
    }

    if rec_df is None or rec_df.empty:
        return resultado

    ingredientes = rec_df.copy()
    ingredientes["ingrediente"] = (
        ingredientes.get("ingrediente", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    ingredientes["pct_ms"] = pd.to_numeric(
        ingredientes.get("pct_ms", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    ingredientes = ingredientes[ingredientes["ingrediente"] != ""]

    if ingredientes.empty:
        return resultado

    lookup = {}
    if alimentos_df is not None and not alimentos_df.empty:
        alim_work = alimentos_df.copy()
        if "ORIGEN" not in alim_work.columns:
            alim_work.columns = [str(c).strip() for c in alim_work.columns]
        for _, row in alim_work.iterrows():
            nombre = str(row.get("ORIGEN", "")).strip()
            if not nombre:
                continue
            lookup[nombre.lower()] = row

    total_pct = ingredientes["pct_ms"].sum() or 1.0

    detalle = []
    asfed_total = em_total = pb_total = costo_total = 0.0

    for _, row in ingredientes.iterrows():
        nombre = str(row["ingrediente"]).strip()
        pct_ms = float(row["pct_ms"])
        inclusion = pct_ms / total_pct

        ref = lookup.get(nombre.lower()) or {}

        ms_pct = _safe_float(ref.get("MS", 0.0))
        ms_frac = ms_pct / 100.0 if ms_pct > 0 else 0.0
        em_val = _safe_float(ref.get("EM", 0.0))
        pb_pct = _safe_float(ref.get("PB", 0.0))
        precio = _safe_float(ref.get("$/KG", 0.0))

        ms_kg = consumo_ms * inclusion
        asfed_kg = ms_kg / (ms_frac if ms_frac > 0 else 1e-6)
        em_mcal = ms_kg * em_val
        pb_g = ms_kg * (pb_pct / 100.0) * 1000.0
        costo = asfed_kg * precio

        asfed_total += asfed_kg
        em_total += em_mcal
        pb_total += pb_g
        costo_total += costo

        detalle.append(
            {
                "ingrediente": nombre,
                "pct_ms": pct_ms,
                "MS_%_alimento": ms_pct,
                "MS_kg": ms_kg,
                "asfed_kg": asfed_kg,
                "EM_Mcal": em_mcal,
                "PB_g": pb_g,
                "precio": precio,
                "costo_dia": costo,
                "missing_ms": ms_pct <= 0,
                "missing_em": em_val <= 0,
                "missing_pb": pb_pct <= 0,
                "missing_precio": precio <= 0,
            }
        )

    resultado["detalle"] = detalle
    resultado["asfed_total_kg_dia"] = asfed_total
    resultado["EM_Mcal_dia"] = em_total
    resultado["PB_g_dia"] = pb_total
    resultado["costo_dia"] = costo_total

    return resultado

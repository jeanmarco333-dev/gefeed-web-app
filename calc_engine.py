# calc_engine.py — Motor de cálculo (EM/PB y mixer as-fed)
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
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


def sugerencia_balance(em_calc: float, em_req: float, pb_calc: float, pb_req: float) -> list[str]:
    tips: list[str] = []
    if em_calc < em_req:
        tips.append("⚡ Aumentar energía (maíz, sorgo, descarte pop).")
    if pb_calc < pb_req:
        tips.append("💪 Aumentar proteína (soja, garbanzo, PD Creston).")
    if not tips:
        tips.append("✅ Ración equilibrada según requerimientos actuales.")
    return tips


def _safe_num(x, default: float = 0.0) -> float:
    try:
        v = float(pd.to_numeric(x, errors="coerce"))
        return default if np.isnan(v) else v
    except Exception:
        return default


def optimize_ration(
    alimentos_df: pd.DataFrame,
    disponibles: list[str],
    consumo_ms_dia: float,
    em_req_mcal_dia: float,
    pb_req_g_dia: float,
    max_ingredientes: int = 6,
    pesos_obj: dict | None = None,
    bounds: dict | None = None,
    aplicar_coef_atc: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Optimiza una ración heurística por descenso proyectado."""

    if pesos_obj is None:
        pesos_obj = {"em": 1.0, "pb": 1.0, "cost": 0.15}
    if bounds is None:
        bounds = {"min_pct_ms": {}, "max_pct_ms": {}}

    pool = alimentos_df.copy()
    pool["name"] = pool.get("ORIGEN", "").astype(str)
    if disponibles:
        disponibles_l = [n.lower() for n in disponibles]
        pool = pool[pool["name"].str.lower().isin(disponibles_l)]

    for col in ["MS", "EM", "PB", "$/KG", "COEF ATC"]:
        if col not in pool.columns:
            pool[col] = np.nan

    ms_clean = pool["MS"].apply(_safe_num)
    ms_clean = np.clip(ms_clean, 1.0, 100.0)
    pool["MS_frac"] = ms_clean / 100.0
    pool["em_mcal_per_kgMS"] = pool["EM"].apply(_safe_num)
    pb_clean = np.clip(pool["PB"].apply(_safe_num), 0.0, 100.0)
    pool["pb_g_per_kgMS"] = pb_clean * 10.0

    if pool.empty or (
        pool["em_mcal_per_kgMS"].fillna(0).sum() == 0
        and pool["pb_g_per_kgMS"].fillna(0).sum() == 0
    ):
        return pd.DataFrame(), {"status": "error", "msg": "No hay alimentos válidos para optimizar."}

    top_energy = pool.sort_values("em_mcal_per_kgMS", ascending=False).head(max_ingredientes // 2)
    top_prot = pool.sort_values("pb_g_per_kgMS", ascending=False).head(max_ingredientes // 2)
    candidates = pd.concat([top_energy, top_prot]).drop_duplicates(subset=["name"]).head(max_ingredientes)
    if len(candidates) < max_ingredientes:
        rest = pool[~pool["name"].isin(candidates["name"])]
        cheap = rest.sort_values("$/KG", ascending=True).head(max_ingredientes - len(candidates))
        candidates = pd.concat([candidates, cheap]).drop_duplicates(subset=["name"]).head(max_ingredientes)

    em = candidates["em_mcal_per_kgMS"].to_numpy(dtype=float)
    pb = candidates["pb_g_per_kgMS"].to_numpy(dtype=float)
    cost_asfed = candidates["$/KG"].apply(_safe_num).to_numpy(dtype=float)
    ms_frac = np.clip(candidates["MS_frac"].to_numpy(dtype=float), 1e-6, 1.0)

    consumo_ms_dia = float(consumo_ms_dia)
    if consumo_ms_dia <= 0:
        return pd.DataFrame(), {"status": "error", "msg": "El consumo de MS debe ser mayor a 0."}

    em_target_perkgMS = em_req_mcal_dia / max(consumo_ms_dia, 1e-6)
    pb_target_perkgMS = pb_req_g_dia / max(consumo_ms_dia, 1e-6)

    n = len(candidates)
    if n == 0:
        return pd.DataFrame(), {"status": "error", "msg": "No hay candidatos suficientes para optimizar."}
    x = np.ones(n, dtype=float) / n

    lb = np.zeros(n)
    ub = np.ones(n)
    min_bounds = bounds.get("min_pct_ms", {}) if isinstance(bounds, dict) else {}
    max_bounds = bounds.get("max_pct_ms", {}) if isinstance(bounds, dict) else {}
    for i, name in enumerate(candidates["name"]):
        if isinstance(min_bounds, dict):
            lb[i] = max(0.0, float(min_bounds.get(str(name), 0.0)) / 100.0)
        if isinstance(max_bounds, dict):
            ub[i] = min(1.0, float(max_bounds.get(str(name), 100.0)) / 100.0)

    def project_simplex_with_bounds(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        y = np.clip(y, lo, hi)
        for _ in range(50):
            s = y.sum()
            if abs(s - 1.0) < 1e-6:
                break
            y = np.clip(y + (1.0 - s) / len(y), lo, hi)
        s = y.sum()
        if s > 0:
            y = y / s
        return y

    cost_per_kgMS = cost_asfed / ms_frac
    w_em = float(pesos_obj.get("em", 1.0))
    w_pb = float(pesos_obj.get("pb", 1.0))
    w_cst = float(pesos_obj.get("cost", 0.15))

    lr = 0.1
    for _ in range(800):
        em_err = (em @ x) - em_target_perkgMS
        pb_err = (pb @ x) - pb_target_perkgMS
        grad = 2 * w_em * em_err * em + 2 * w_pb * pb_err * pb + w_cst * cost_per_kgMS
        x = project_simplex_with_bounds(x - lr * grad, lb, ub)

    pct_ms = x * 100.0
    pct_ms = np.maximum(pct_ms, 0.0)
    pct_sum = pct_ms.sum()
    if pct_sum <= 0:
        return pd.DataFrame(), {"status": "error", "msg": "La optimización devolvió una ración vacía."}
    pct_ms = pct_ms / pct_sum * 100.0

    ms_kg = pct_ms / 100.0 * consumo_ms_dia
    asfed_kg = ms_kg / ms_frac
    if aplicar_coef_atc:
        coef = candidates["COEF ATC"].apply(_safe_num).to_numpy(dtype=float)
        coef = np.where(coef <= 0, 1.0, coef)
        asfed_kg = asfed_kg * coef
    em_mcal = ms_kg * em
    pb_g = ms_kg * pb
    costo = asfed_kg * cost_asfed

    df = pd.DataFrame(
        {
            "ingrediente": candidates["name"].to_list(),
            "pct_ms": np.round(pct_ms, 2),
            "ms_kg_dia": np.round(ms_kg, 3),
            "asfed_kg_dia": np.round(asfed_kg, 3),
            "em_mcal": np.round(em_mcal, 3),
            "pb_g": np.round(pb_g, 1),
            "costo": np.round(costo, 2),
            "MS_%": np.round(ms_frac * 100, 1),
            "$/KG": np.round(cost_asfed, 2),
        }
    )

    em_calc = float(em_mcal.sum())
    pb_calc = float(pb_g.sum())
    cost_total = float(costo.sum())
    asfed_total = float(asfed_kg.sum())

    resumen = {
        "em_calc": em_calc,
        "pb_calc": pb_calc,
        "cost_total": cost_total,
        "asfed_total_kg_dia": asfed_total,
        "status": "ok",
        "msg": "Optimización heurística finalizada.",
    }
    return df.sort_values("pct_ms", ascending=False).reset_index(drop=True), resumen

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
def ration_split_from_pv_cv(
    rec_df: pd.DataFrame,
    alimentos_df: pd.DataFrame,
    pv_kg: float,
    cv_pct: float,
) -> dict:
    """Calcula la distribución diaria de MS/as-fed/EM/PB/costo."""

    def _num(x, default=0.0):
        try:
            v = float(pd.to_numeric(x, errors="coerce"))
            return default if pd.isna(v) else v
        except Exception:
            return default

    def _txt(x):
        if pd.isna(x):
            return ""
        return str(x)

    pv_clean = _num(pv_kg, 0.0)
    cv_clean = _num(cv_pct, 0.0)
    consumo_ms = pv_clean * (cv_clean / 100.0)

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
            origen = _txt(row.get("ORIGEN")).strip()
            if not origen:
                continue
            lookup[origen.lower()] = row.to_dict()

    total_pct = ingredientes["pct_ms"].sum() or 1.0

    detalle = []
    asfed_total = em_total = pb_total = costo_total = 0.0

    for _, row in ingredientes.iterrows():
        nombre = _txt(row.get("ingrediente")).strip()
        if not nombre:
            continue

        incl_pct = _num(row.get("pct_ms", 0.0), 0.0)
        inclusion = incl_pct / total_pct if total_pct else 0.0

        ref = lookup.get(nombre.lower(), {})

        ms_pct = _num(ref.get("MS", 100.0), 100.0)
        em_val = _num(ref.get("EM", 0.0), 0.0)
        pb_pct = _num(ref.get("PB", 0.0), 0.0)
        precio = _num(ref.get("$/KG", 0.0), 0.0)

        ms_frac = ms_pct / 100.0 if ms_pct > 0 else 0.0
        ms_kg = consumo_ms * inclusion
        asfed_kg = ms_kg / ms_frac if ms_frac > 1e-6 else 0.0
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
                "pct_ms": incl_pct,
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

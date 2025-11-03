# calc_engine.py — Motor de cálculo (EM/PB y mixer as-fed)
from dataclasses import dataclass
from typing import List, Dict

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

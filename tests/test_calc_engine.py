from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calc_engine import calculate_gmd


def test_calculate_gmd_accepts_comma_decimal_input():
    assert calculate_gmd(ap_kg_dia="1,25") == pytest.approx(1.25)


def test_calculate_gmd_falls_back_to_weights_with_commas():
    result = calculate_gmd(peso_inicial="250,5", peso_final="310", dias="30")
    assert result == pytest.approx((310 - 250.5) / 30)


def test_calculate_gmd_returns_none_for_invalid_inputs():
    assert (
        calculate_gmd(ap_kg_dia="nope", peso_inicial="a", peso_final=None, dias="")
        is None
    )

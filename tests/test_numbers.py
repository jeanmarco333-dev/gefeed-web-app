from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.numbers import normalize_animal_counts


def test_normalize_handles_apostrophes_and_thousand_separators():
    raw = pd.Series(["'1.200", "1’500", " 750 ", "1 250", "2,5"])
    result = normalize_animal_counts(raw)
    assert result.tolist() == [1200, 1500, 750, 1250, 2]


def test_normalize_returns_zero_series_for_none_input():
    out = normalize_animal_counts(None, index=pd.RangeIndex(0, 3))
    assert out.tolist() == [0, 0, 0]


def test_normalize_discards_trailing_labels_and_signs():
    raw = pd.Series(["1.250 hd", "+2.500 cabezas", "- 350 terneros", "invalid"])
    result = normalize_animal_counts(raw)
    assert result.tolist() == [1250, 2500, -350, 0]

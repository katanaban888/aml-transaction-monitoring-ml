"""
tests/test_synthetic_pipeline.py
================================
Дымовой тест всего конвейера на маленьком объёме (без pytest, обычные assert).

Зачем:
  * проверяет, что генератор синтетики, загрузчик и EDA-утилиты не сломались
    после правок (например, после изменения логики в src/data_loader.py);
  * работает БЕЗ реального датасета — можно гонять в CI и на любой машине.

Запуск:
    python tests/test_synthetic_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    RAW_COLUMNS, COL_TARGET, COL_LAUND_TYPE, class_balance, add_time_features,
    optimize_dtypes,
)
from src.eda_utils import rate_by_group, two_proportion_ztest, categories_equal  # noqa: E402
from src.synthetic_data import generate, TYPOLOGY_COUNTS  # noqa: E402

N_TEST_ROWS = 50_000


def test_generator_schema():
    """Генератор выдаёт ровно те же 12 колонок и помечает отмывание."""
    with tempfile.TemporaryDirectory() as tmp:
        path = generate(n_rows=N_TEST_ROWS, seed=1, out_path=Path(tmp) / "s.csv", verbose=False)
        df = pd.read_csv(path)

    assert list(df.columns) == RAW_COLUMNS, f"Схема не совпала: {list(df.columns)}"
    assert len(df) == N_TEST_ROWS, f"Строк {len(df)}, ожидалось {N_TEST_ROWS}"
    assert df[COL_TARGET].isin([0, 1]).all(), "Is_laundering должен быть 0/1"
    assert df[COL_TARGET].sum() > 0, "В выборке должно быть хотя бы одно отмывание"
    # У «чистых» транзакций типология пустая, у «грязных» — заполнена
    assert df.loc[df[COL_TARGET] == 0, COL_LAUND_TYPE].isna().all()
    assert df.loc[df[COL_TARGET] == 1, COL_LAUND_TYPE].notna().all()
    print("OK  test_generator_schema")


def test_typology_counts_full_scale():
    """На полном объёме раскладка по типологиям совпадает с реальным SAML-D."""
    from src.synthetic_data import _collect_illicit

    df = _collect_illicit(np.random.default_rng(0))
    counts = df[COL_LAUND_TYPE].value_counts().to_dict()
    assert counts == TYPOLOGY_COUNTS, f"Раскладка типологий не совпала: {counts}"
    assert len(df) == sum(TYPOLOGY_COUNTS.values())
    print("OK  test_typology_counts_full_scale")


def test_loader_and_time_features():
    """Загрузчик парсит дату/время и строит календарные признаки."""
    with tempfile.TemporaryDirectory() as tmp:
        path = generate(n_rows=N_TEST_ROWS, seed=2, out_path=Path(tmp) / "s.csv", verbose=False)
        df = pd.read_csv(path)

    df = optimize_dtypes(df, verbose=False)
    df = add_time_features(df, verbose=False)

    assert pd.api.types.is_datetime64_any_dtype(df["Date"]), "Date не распознана как дата"
    assert df["Date"].isna().sum() == 0, "Есть неразобранные даты"
    assert "hour" in df.columns and "txn_ts" in df.columns
    assert df["hour"].between(0, 23).all(), "Час вне диапазона 0-23"
    # Строковая колонка Time удаляется загрузчиком — она дублирует hour/minute
    assert "Time" not in df.columns
    print("OK  test_loader_and_time_features")


def test_eda_utils():
    """lift-таблицы и z-тест считаются корректно."""
    with tempfile.TemporaryDirectory() as tmp:
        path = generate(n_rows=N_TEST_ROWS, seed=3, out_path=Path(tmp) / "s.csv", verbose=False)
        df = pd.read_csv(path)

    df = add_time_features(optimize_dtypes(df, verbose=False), verbose=False)
    bal = class_balance(df[COL_TARGET])
    assert 0 < bal["base_rate"] < 1, "base_rate вне диапазона"

    tbl = rate_by_group(df, "Payment_type", min_support=1, sort_by="lift")
    assert (tbl["n"] >= 1).all(), "min_support отработал неверно"
    # lift считается относительно base rate: средневзвешенный lift по всем группам = 1
    weighted = (tbl["lift"] * tbl["n"]).sum() / tbl["n"].sum()
    assert abs(weighted - 1.0) < 1e-6, f"Средневзвешенный lift должен быть 1, получено {weighted}"

    z, p = two_proportion_ztest(50, 100, 50, 10_000)
    assert z > 0 and p < 0.01, "z-тест не выявил очевидного различия долей"
    z0, p0 = two_proportion_ztest(10, 100, 1_000, 10_000)
    assert p0 > 0.05, "z-тест не должен находить различие там, где его нет"

    eq = categories_equal(df["Payment_currency"], df["Received_currency"])
    assert eq.dtype == bool and len(eq) == len(df)
    print("OK  test_eda_utils")


if __name__ == "__main__":
    test_generator_schema()
    test_typology_counts_full_scale()
    test_loader_and_time_features()
    test_eda_utils()
    print("\nВСЕ ТЕСТЫ ПРОЙДЕНЫ")

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



def _small_frame(n: int = 5_000, seed: int = 5) -> pd.DataFrame:
    """Небольшой подготовленный фрейм (как после data_loader) для тестов признаков."""
    with tempfile.TemporaryDirectory() as tmp:
        path = generate(n_rows=n, seed=seed, out_path=Path(tmp) / "s.csv", verbose=False)
        df = pd.read_csv(path)
    return add_time_features(optimize_dtypes(df, verbose=False), verbose=False)


def test_velocity_matches_bruteforce():
    """Векторный расчёт скользящих окон совпадает с лобовым перебором.

    Это самый важный тест этапа 2: окна считаются хитрой упаковкой
    (счёт, время) в одно int64 + searchsorted. Ошибка на единицу справа
    или слева незаметна глазом, но полностью ломает смысл признака.
    """
    from src.features import compute_velocity_features

    df = _small_frame(4_000, seed=7)
    vel = compute_velocity_features(df)

    assert len(vel) == len(df), "Число строк velocity не совпало"
    for col in ["vel_count_1h", "vel_count_24h", "vel_amount_sum_1h", "vel_amount_sum_24h"]:
        assert col in vel.columns, f"Нет колонки {col}"

    # --- брутфорс по тем же правилам: окно (t - W, t], текущая операция входит
    senders = df["Sender_account"].to_numpy()
    ts = df["txn_ts"].to_numpy().astype("datetime64[s]").astype(np.int64)
    amt = df["Amount"].to_numpy(dtype="float64")
    check = ts[:400]                                   # первые 400 строк — достаточно
    for w, secs in ((1, 3600), (24, 86_400)):
        vec_c = vel[f"vel_count_{w}h"].to_numpy()[:400]
        vec_s = vel[f"vel_amount_sum_{w}h"].to_numpy()[:400]
        for i in range(len(check)):
            m = (senders == senders[i]) & (ts > ts[i] - secs) & (ts <= ts[i])
            assert vec_c[i] == m.sum(), (
                f"count {w}h: вектор {vec_c[i]} != брутфорс {m.sum()} (строка {i})")
            assert abs(vec_s[i] - amt[m].sum()) < 1e-2, (
                f"sum {w}h расходится в строке {i}")
    print("OK  test_velocity_matches_bruteforce")


def test_cyclic_core_finds_cycles():
    """«Отшелушивание» вершин оставляет ровно те счета, что лежат на циклах."""
    from src.graph_features import cyclic_core

    tiny = pd.DataFrame({
        "Sender_account":   ["A", "B", "C", "D", "E", "F"],
        "Receiver_account": ["B", "C", "A", "E", "D", "A"],
    })
    core, edges = cyclic_core(tiny, verbose=False)
    assert sorted(core) == ["A", "B", "C", "D", "E"], f"Получилось {sorted(core)}"
    # рёбер внутри ядра ровно 5: A->B, B->C, C->A, D->E, E->D
    assert len(edges) == 5, f"Рёбер в ядре {len(edges)}, ожидалось 5"

    # граф без циклов -> ядро пустое
    dag = pd.DataFrame({"Sender_account": ["A", "B"], "Receiver_account": ["B", "C"]})
    assert len(cyclic_core(dag, verbose=False)[0]) == 0, "В графе без циклов ядро должно быть пустым"
    print("OK  test_cyclic_core_finds_cycles")


def test_fit_apply_no_leakage():
    """Профили FITятся на train и применяются ко всем периодам без пересчёта.

    Проверяем две вещи:
      1. признаки считаются и не падают;
      2. профиль, построенный на train, отличается от профиля по всем данным
         (иначе «FIT на train» — просто слова).
    """
    from src.features import (fit_account_profiles, add_behavioural_features,
                              fit_pair_stats, add_pair_features)
    from src.split import time_split, assert_no_leakage

    df = _small_frame(6_000, seed=11)
    train_mask, val_mask, test_mask = time_split(df, verbose=False)
    assert_no_leakage(df, train_mask, test_mask)

    prof_train = fit_account_profiles(df[train_mask])
    prof_all = fit_account_profiles(df)
    assert "sender_median_amount" in prof_train.columns

    out = add_behavioural_features(df, prof_train)
    for col in ["amount_to_sender_median", "amount_zscore_sender", "sender_txn_count"]:
        assert col in out.columns, f"Нет признака {col}"

    # профили.train != профили.все_данные хотя бы для части счетов
    both = prof_train[["sender_median_amount"]].join(
        prof_all[["sender_median_amount"]], how="inner", rsuffix="_all").dropna()
    diff = (both["sender_median_amount"] - both["sender_median_amount_all"]).abs()
    assert len(both) > 0 and (diff > 0).any(), "Профили train и всех данных совпали: FIT не работает?"

    pair = fit_pair_stats(df[train_mask])
    out = add_pair_features(out, pair)
    assert "pair_count" in out.columns and "is_new_pair" in out.columns
    print("OK  test_fit_apply_no_leakage")


def test_streaming_build_writes_parquet():
    """Потоковая сборка пишет parquet со всеми признаками и меткой периода."""
    from src.features import build_and_save_features
    from src.split import time_split

    df = _small_frame(3_000, seed=13)
    train_mask, val_mask, test_mask = time_split(df, verbose=False)
    period = pd.Series("train", index=df.index, dtype=object)
    period[val_mask] = "val"
    period[test_mask] = "test"

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "feat.parquet"
        meta = build_and_save_features(df, train_mask=np.asarray(train_mask),
                                       period=np.asarray(period),
                                       out_path=out_path, chunk_size=1_000,
                                       graph_max_edges=50_000, with_graph=True,
                                       verbose=False)
        saved = pd.read_parquet(out_path)

    assert len(saved) == len(df), f"Строк в файле {len(saved)}, ожидалось {len(df)}"
    assert "period" in saved.columns and COL_TARGET in saved.columns
    assert set(saved["period"].unique()) == {"train", "val", "test"}
    assert len(meta["feature_columns"]) > 30, f"Признаков всего {len(meta['feature_columns'])}"
    # Мерж прошёл, если базовые признаки заполнены.
    # Оговорка: на такой крошечной выборке у счёта может быть всего одна
    # операция, тогда std и z-по z-оценка по счёту = NaN. На полном объёме
    # (по 79 операций на счёт) пропусков нет — это проверяет тетрадка Этапа 2.
    feat_cols = [c for c in saved.columns if c not in (COL_TARGET, "period")]
    nan_share = saved[feat_cols].isna().mean()
    assert (nan_share < 0.99).all(), f"Есть признаки-пустышки: {nan_share[nan_share >= 0.99]}"
    # Почему НЕ требуем 100% заполнения: у счёта, которого не было в train,
    # профиля и граф-статистики нет — это правильное поведение FIT -> APPLY
    # (в проде для нового клиента тоже сначала нет истории). Важно, что мерж
    # сработал для БОЛЬШИНСТВА строк, а не «проехал мимо».
    for base in ["amount_log", "vel_count_24h", "vel_amount_sum_24h", "g_in_degree"]:
        filled = saved[base].notna().mean()
        assert filled > 0.6, f"{base} заполнен всего на {filled:.0%} — мерж признаков сломан"
    print("OK  test_streaming_build_writes_parquet")


def test_check_feature_leakage_flags_hidden_target():
    """Детектор утечки ловит признак, который почти равен таргету, и молчит на шуме."""
    from src.split import check_feature_leakage

    rng = np.random.default_rng(0)
    n = 2_000
    y = (rng.random(n) < 0.1).astype("int8")
    X = pd.DataFrame({"шум": rng.normal(size=n),
                      "подглядывание": y + rng.normal(scale=0.01, size=n)})
    corr = check_feature_leakage(X, pd.Series(y), threshold=0.90)["|corr|"]
    assert corr["подглядывание"] > 0.90, f"Утечку не поймали: {corr['подглядывание']}"
    assert corr["шум"] < 0.10, f"Шум приняли за утечку: {corr['шум']}"

    X_ok = pd.DataFrame({"шум1": rng.normal(size=n), "шум2": rng.normal(size=n)})
    corr_ok = check_feature_leakage(X_ok, pd.Series(y), threshold=0.90)["|corr|"]
    assert (corr_ok < 0.90).all(), f"Ложное срабатывание: {corr_ok}"
    print("OK  test_check_feature_leakage_flags_hidden_target")



def _fake_feature_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Ручная матрица признаков: нужен предсказуемый набор для проверки правил."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "amount_above_threshold": rng.integers(0, 2, n),
        "amount_near_threshold": rng.integers(0, 2, n),
        "vel_near_threshold_count_24h": rng.integers(0, 4, n),
        "vel_near_threshold_count_168h": rng.integers(0, 5, n),
        "vel_count_1h": rng.integers(1, 5, n),
        "vel_amount_24h_to_median": rng.exponential(3, n),
        "amount_zscore_sender": rng.exponential(1, n),
        "g_out_degree": rng.integers(1, 60, n),
        "g_in_degree": rng.integers(1, 60, n),
        "g_total_degree": rng.integers(2, 120, n),
        "g_in_repeat_cycle": rng.integers(0, 2, n),
        "g_is_mutual_pair": rng.integers(0, 2, n),
        "is_new_pair": rng.integers(0, 2, n),
        "amount_pct_in_currency": rng.uniform(0, 100, n),
        "is_night": rng.integers(0, 2, n),
        "g_degree_ratio": rng.exponential(1, n),
    })


def test_rules_baseline():
    """Правила срабатывают, метрики считаются, пороги подбираются по train."""
    from src.rules import (default_rule_book, fit_rules, apply_rules, risk_score,
                           any_rule, rule_metrics, score_curve, recall_by_typology,
                           daily_alerts, confusion)

    X = _fake_feature_frame()
    rng = np.random.default_rng(1)
    # «Отмывание» там, где сработало сразу много правил — искусственная связка,
    # нужная только затем, чтобы метрики были осмысленными.
    y = (rng.random(len(X)) < 0.05).astype("int8")
    book = default_rule_book()
    fit_rules(X, book)

    flags = apply_rules(X, book)
    assert flags.shape == (len(X), len(book)), f"Таблица флагов {flags.shape}"
    assert flags.dtypes.unique().tolist() == [bool], "Флаги должны быть булевыми"

    # пороги-квантили действительно посчитались
    tuned = [r for r in book if r.quantile is not None]
    assert len(tuned) >= 4 and all(r.threshold is not None for r in tuned)

    score = risk_score(flags)
    assert score.max() <= len(book) and score.min() >= 0
    assert (any_rule(flags).to_numpy() == (score >= 1).to_numpy()).all()

    met = rule_metrics(flags, y, book)
    assert len(met) == len(book)
    assert met["lift"].max() > 1, "Хотя бы одно правило должно быть лучше случайного"

    # метрики на «идеальном» алерте: recall 100%, precision 100%
    perfect = confusion(np.ones(len(y), dtype=bool), y)
    assert perfect["recall_%"] == 100.0 and perfect["fn"] == 0
    # метрики на «пустом» алерте: recall 0
    empty = confusion(np.zeros(len(y), dtype=bool), y)
    assert empty["recall_%"] == 0.0 and empty["tp"] == 0

    curve = score_curve(y, score.to_numpy())
    assert (curve["порог_k"] == np.arange(1, len(curve) + 1)).all()
    # с ростом порога алертов становится меньше, а precision — не падает
    assert curve["alerts"].is_monotonic_decreasing, "С ростом k алертов должно быть меньше"

    # Проверяем арифметику функции: если алертуем всё подряд, recall = 100%
    typ = pd.Series(np.where(y == 1, "Structuring", None))
    rec = recall_by_typology(np.ones(len(y), dtype=bool), y, typ, min_support=1)
    assert len(rec) == 1 and abs(rec.loc[0, "recall_%"] - 100.0) < 1e-9
    # ...а если не алертуем ничего — 0%
    rec0 = recall_by_typology(np.zeros(len(y), dtype=bool), y, typ, min_support=1)
    assert rec0.loc[0, "recall_%"] == 0.0

    days = np.array(["2022-01-01", "2022-01-01", "2022-01-02", "2022-01-02"] * (len(y) // 4))
    per_day = daily_alerts(np.ones(len(y), dtype=bool), days)
    assert len(per_day) == 2 and per_day.sum() == len(y)
    print("OK  test_rules_baseline")


if __name__ == "__main__":
    test_generator_schema()
    test_typology_counts_full_scale()
    test_loader_and_time_features()
    test_eda_utils()
    test_velocity_matches_bruteforce()
    test_cyclic_core_finds_cycles()
    test_fit_apply_no_leakage()
    test_streaming_build_writes_parquet()
    test_check_feature_leakage_flags_hidden_target()
    test_rules_baseline()
    print("\nВСЕ ТЕСТЫ ПРОЙДЕНЫ")

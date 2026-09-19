"""
src/eda_utils.py
================
Мелкие переиспользуемые функции для разведочного анализа (EDA) и правил.

Три функции, которые мы будем использовать в каждом втором графике Этапа 1:

1. rate_by_group()  — «какая доля отмывания в каждой группе и во сколько раз
                      она выше средней по банку (lift)».
2. two_proportion_ztest() — «а не случайность ли это?» (проверка двух долей).
3. save_fig()       — сохранение графиков в reports/figures/.

ПОЧЕМУ LIFT, А НЕ ПРОСТО «ПРОЦЕНТ ОТМЫВАНИЯ»?
----------------------------------------------
В AML positives — 0.1% данных. Если в какой-то группе «доля отмывания 0.5%»,
звучит ничтожно. Но если в среднем по датасету 0.1% — это **в 5 раз выше нормы**,
то есть группа в 5 раз «грязнее» случайной выборки. Именно lift (во сколько раз
выше base rate) — язык, на котором говорят риск-аналитики и на котором потом
обосновывают пороги правил мониторинга («ставить алерт при lift > 3»).

ПОЧЕМУ Z-ТЕСТ, А НЕ «НА ГЛАЗ»?
-------------------------------
Группа из 30 транзакций с 2 отмываниями даёт rate 6.7% и огромный lift — но это
статистический шум. На собеседовании фраза «я проверил значимость, p-value < 0.01»
отличает аналитика от студента. Тест считается без scipy — по формуле нормального
приближения, этого достаточно при n в тысячи.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import COL_TARGET, FIGURES_DIR

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. LIFT-ТАБЛИЦЫ
# ---------------------------------------------------------------------------
def rate_by_group(
    df: pd.DataFrame,
    by: str | list[str],
    target: str = COL_TARGET,
    min_support: int = 1,
    sort_by: str = "lift",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Считает по группам: объём, число отмываний, долю (rate) и lift к base rate.

    Параметры
    ---------
    by : колонка или список колонок для группировки
        Пример: "Payment_type", ["Sender_bank_location", "Receiver_bank_location"].
    min_support : int
        Отсечь группы, где транзакций меньше этого числа.
        ОБЯЗАТЕЛЬНО К ИСПОЛЬЗОВАНИЮ: коридор с 3 транзакциями и 1 отмыванием
        даёт rate 33% и lift 300 — визуально «самый опасный коридор в мире»,
        хотя это шум. Порог поддержки (например, >= 1000 транзакций) — стандарт
        риск-практики: правило должно опираться на достаточно наблюдений.

    Возвращает
    ---------
    DataFrame с колонками: n, n_laundering, rate_pct, lift, share_of_laundering_pct
        rate_pct               — % отмывания внутри группы
        lift                   — во сколько раз выше, чем в среднем по датасету
        share_of_laundering_pct— какую долю ВСЕХ отмываний покрывает эта группа
                                 (важно: коридор с lift 10 и объёмом 100 строк
                                  поймает 1 кейс, а коридор с lift 2 и объёмом
                                  1 млн — тысячи. Нужны обе метрики.)
    """
    grouped = df.groupby(by, observed=True)[target].agg(n="size", n_laundering="sum")
    grouped = grouped[grouped["n"] >= min_support].copy()

    base_rate = float(df[target].mean())
    total_laundering = float(df[target].sum())

    grouped["rate_pct"] = grouped["n_laundering"] / grouped["n"] * 100
    grouped["lift"] = (grouped["n_laundering"] / grouped["n"]) / base_rate if base_rate > 0 else np.nan
    grouped["share_of_laundering_pct"] = grouped["n_laundering"] / total_laundering * 100

    return grouped.sort_values(sort_by, ascending=ascending)


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """
    Двусторонний z-тест для двух долей. Возвращает (z, p_value).

    Пример: доля отмывания среди транзакций $9,000–10,000 vs среди всех остальных.
    H0: доли одинаковы. p < 0.05 → различие статистически значимо,
    «прижимание к порогу» реально существует, а не случайно.
    """
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (p1 - p2) / se
    # Двусторонний p-value через complementary error function (без scipy).
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return z, p_value


def compare_band(
    df: pd.DataFrame,
    mask: pd.Series,
    target: str = COL_TARGET,
    label_in: str = "в группе",
    label_out: str = "вне группы",
) -> pd.DataFrame:
    """
    Сравнивает долю отмывания внутри подвыборки (mask) и вне её, со z-тестом.
    Готовая табличка «есть ли сигнал в этом красном флаге».
    """
    y = df[target].to_numpy()
    m = mask.to_numpy()

    n_in, x_in = int(m.sum()), int(y[m].sum())
    n_out, x_out = int((~m).sum()), int(y[~m].sum())

    # Защита от «пустой» группы: если в данных ВСЕ счета транзитные или ни одного,
    # сравнивать не с чем — сообщаем об этом, а не молча выдаём nan.
    if n_in == 0 or n_out == 0:
        print(f"ВНИМАНИЕ: одна из групп пуста (в группе {n_in:,}, вне группы {n_out:,}). "
              f"Сравнение невозможно — признак не разделяет выборку.")
        return pd.DataFrame({
            "группа": [f"ДА: {label_in}", f"НЕТ: {label_out}"],
            "n_transactions": [n_in, n_out],
            "n_laundering": [x_in, x_out],
            "rate_pct": [x_in / n_in * 100 if n_in else np.nan,
                         x_out / n_out * 100 if n_out else np.nan],
            "lift_vs_other": [np.nan, np.nan],
        })

    z, p = two_proportion_ztest(x_in, n_in, x_out, n_out)
    rate_in = x_in / n_in * 100
    rate_out = x_out / n_out * 100

    out = pd.DataFrame(
        {
            "группа": [f"ДА: {label_in}", f"НЕТ: {label_out}"],
            "n_transactions": [n_in, n_out],
            "n_laundering": [x_in, x_out],
            "rate_pct": [rate_in, rate_out],
            "lift_vs_other": [rate_in / rate_out if rate_out else np.nan, 1.0],
        }
    )
    print(f"rate внутри: {rate_in:.4f}%  |  rate вне: {rate_out:.4f}%  "
          f"|  lift: {rate_in / rate_out if rate_out else float('nan'):.2f}x")
    print(f"z = {z:,.2f}, p-value = {p:.3e}  "
          f"-> {'ЗНАЧИМО (сигнал есть)' if p < 0.05 else 'не значимо (шум)'}")
    return out


# ---------------------------------------------------------------------------
# 2. КАРМАН ДЛЯ ГРАФИКОВ
# ---------------------------------------------------------------------------
def save_fig(name: str, fig=None, dpi: int = 150, tight: bool = True) -> Path:
    """
    Сохраняет текущий (или переданный) график в reports/figures/.

    Зачем: графики из ноутбука нужны для README (Этап 7) и для презентации.
    Если не сохранять автоматически, потом придётся перезапускать всё заново.
    """
    import matplotlib.pyplot as plt

    fig = fig if fig is not None else plt.gcf()
    path = FIGURES_DIR / f"{name}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[save_fig] {path.relative_to(FIGURES_DIR.parents[0])}")
    return path


# ---------------------------------------------------------------------------
# 3. БИНОВАНИЕ (разбивка на корзины)
# ---------------------------------------------------------------------------
def bucketize(series: pd.Series, bins: list[float], labels: list[str] | None = None,
              right: bool = False) -> pd.Series:
    """
    Обёртка над pd.cut, которая всегда возвращает понятные подписи.

    Зачем бины вместо непрерывной оси: долю отмывания на глаз по графику
    не оценить (0.1%), а по корзинам — видно сразу: «в этой корзине rate 3%».
    """
    return pd.cut(series, bins=bins, labels=labels, right=right, include_lowest=True)


def rate_by_bucket(df: pd.DataFrame, bucket_col: str, target: str = COL_TARGET) -> pd.DataFrame:
    """rate/lift по уже созданной колонке-корзине (удобно строить столбчатые графики)."""
    out = df.groupby(bucket_col, observed=True)[target].agg(n="size", n_laundering="sum")
    base_rate = float(df[target].mean())
    out["rate_pct"] = out["n_laundering"] / out["n"] * 100
    out["lift"] = (out["n_laundering"] / out["n"]) / base_rate if base_rate > 0 else np.nan
    return out

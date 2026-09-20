"""
src/split.py
============
Разбиение данных на train / validation / test **по времени** и защита от утечки.

ПОЧЕМУ ИМЕННО ПО ВРЕМЕНИ (это первое, о чём спрашивают на AML-собеседовании)
---------------------------------------------------------------------------
Если перемешать транзакции случайно, то:
  * в train попадут операции, которые произошли ПОСЛЕ операций из test;
  * любые «исторические» признаки (средняя сумма счёта, частота операций)
    будут подсчитаны с использованием будущего — модель «увидит будущее»;
  * метрики окажутся завышенными на 10-30 пунктов PR-AUC, и в продакшене
    модель внезапно станет никуда не годной.

Реальный мониторинг работает именно так: модель обучается на истории и каждый
день скорит НОВЫЕ транзакции. Значит и проверять надо так же:
    train = прошлое  ->  val = ближайшее будущее  ->  test = следующее будущее.

Дополнительно здесь лежит «принцип FIT/APPLY»:
    профили счетов (средние суммы, степень, частота) считаются ТОЛЬКО на train
    и применяются ко всем периодам. Иначе профиль счёта в train будет знать
    о том, сколько денег счёт прокинул в test — это тоже утечка, только хитрая.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    time_col: str = "txn_ts",
    verbose: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Режет данные по времени: первые train_frac — train, дальше val_frac — val,
    остаток — test. Возвращает три булевы маски.

    Границы считаются по КВАНТИЛЯМ времени (а не по номеру строки), потому что
    транзакций в разные дни бывает разное количество.
    """
    t = df[time_col]
    q_train = t.quantile(train_frac)
    q_val = t.quantile(train_frac + val_frac)

    train_mask = t <= q_train
    val_mask = (t > q_train) & (t <= q_val)
    test_mask = t > q_val

    if verbose:
        base = df["Is_laundering"].mean() if "Is_laundering" in df else float("nan")
        print(f"[time_split] train: {int(train_mask.sum()):,} строк | "
              f"{t[train_mask].min().date()} .. {t[train_mask].max().date()}")
        print(f"[time_split] val:   {int(val_mask.sum()):,} строк | "
              f"{t[val_mask].min().date()} .. {t[val_mask].max().date()}")
        print(f"[time_split] test:  {int(test_mask.sum()):,} строк | "
              f"{t[test_mask].min().date()} .. {t[test_mask].max().date()}")
        if "Is_laundering" in df:
            for name, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
                print(f"    {name:5s}: отмывания = {int(df.loc[m, 'Is_laundering'].sum()):,} "
                      f"({df.loc[m, 'Is_laundering'].mean() * 100:.4f}% при базовом "
                      f"{base * 100:.4f}%)")
    return train_mask, val_mask, test_mask


def assert_no_leakage(df: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series,
                      time_col: str = "txn_ts") -> None:
    """
    Проверяет, что train и test не перемешаны по времени.
    Падает с ошибкой, если границы нарушены — это дешёвая страховка от
    случайного random_split, который потом испортит все метрики Этапа 4.
    """
    t = df[time_col]
    max_train, min_test = t[train_mask].max(), t[test_mask].min()
    if max_train > min_test:
        raise AssertionError(
            f"УТЕЧКА ВРЕМЕНИ: последняя транзакция train ({max_train}) "
            f"позже первой транзакции test ({min_test}). "
            f"Используй time_split(), а не случайное разбиение."
        )
    print(f"[assert_no_leakage] OK: train заканчивается {max_train}, "
          f"test начинается {min_test}")


def check_feature_leakage(X: pd.DataFrame, y: pd.Series, threshold: float = 0.90,
                          top_n: int = 10) -> pd.DataFrame:
    """
    Дёшево ловит «слишком хорошие» признаки: если один признак в одиночку даёт
    корреляцию с таргетом выше threshold — почти наверняка в нём есть утечка
    (например, посчитан по всему датасету, включая будущее).

    Возвращает топ самых подозрительных признаков.
    """
    corrs = {}
    for col in X.columns:
        s = X[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        # nan не должны ломать расчёт
        mask = s.notna()
        if mask.sum() < 100 or s.nunique(dropna=True) < 2:
            continue
        corrs[col] = float(np.corrcoef(s[mask].to_numpy(), y[mask].to_numpy())[0, 1])

    out = (pd.Series(corrs).abs().sort_values(ascending=False).head(top_n)
           .rename("|corr|").to_frame())
    suspicious = out[out["|corr|"] > threshold]
    if len(suspicious):
        print(f"[check_feature_leakage] ВНИМАНИЕ: {len(suspicious)} признаков с "
              f"|corr| > {threshold} — проверь, не подсмотрено ли будущее:")
        print(suspicious)
    else:
        print(f"[check_feature_leakage] OK: признаков с |corr| > {threshold} нет. "
              f"Максимальная |corr| = {out['|corr|'].max():.3f}")
    return out

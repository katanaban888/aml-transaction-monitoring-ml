"""
tests/make_synthetic_saml_d.py
==============================
Генератор НЕБОЛЬШОЙ синтетической копии SAML-D той же схемы (12 колонок).

ЗАЧЕМ ЭТО НУЖНО (и почему это НЕ часть основного пайплайна):
1. Быстро тестировать код (EDA, фичи, правила) не читая CSV на 9.5 млн строк:
   прогнал на 200 тысячах за секунды — значит код синтаксически и логически жив.
2. Прогонять юнит-тесты и CI без тяжёлых данных.
3. Проверять код в песочнице/на другом компьютере, где датасета нет.

Реальные данные ВСЕГДА берутся из data/raw/ — этот генератор их не заменяет.
Запуск:  python tests/make_synthetic_saml_d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Чтобы скрипт можно было запускать из любого места: python tests/make_synthetic_saml_d.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import RAW_DIR  # noqa: E402

N_ROWS = 200_000
RANDOM_STATE = 42

COUNTRIES = ["USA", "United Kingdom", "Germany", "France", "Italy", "Spain",
             "China", "India", "UAE", "Turkey", "Russia", "Brazil", "Mexico",
             "Nigeria", "Japan", "Switzerland", "Cyprus", "Malta"]
CURRENCIES = ["US Dollar", "Euro", "British Pound", "Yuan", "Rupee",
              "UAE Dirham", "Swiss Franc", "Yen", "Ruble", "Lira"]
PAYMENT_TYPES = ["Credit card", "Debit card", "ACH", "Wire", "Cash Deposit",
                 "Cash Withdrawal", "Cheque", "Cross-border", "Card"]
LAUNDERING_TYPES = ["Structuring", "Smurfing", "Cash_Withdrawal", "Deposit-Send",
                    "Fan_In", "Fan_Out", "Layered_Fan_In", "Layered_Fan_Out",
                    "Bipartite", "Stacked Bipartite", "Cycle", "Gather-Scatter",
                    "Scatter-Gather", "Single_large", "Over-Invoicing",
                    "Behavioural_Change_1", "Behavioural_Change_2"]


def make_synthetic(n_rows: int = N_ROWS, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    n_accounts = max(50, n_rows // 20)
    accounts = [f"ACC_{i:07d}" for i in range(n_accounts)]

    senders = rng.choice(accounts, size=n_rows)
    receivers = rng.choice(accounts, size=n_rows)
    # убираем случайные самопереводы
    same = senders == receivers
    receivers[same] = rng.choice(accounts, size=int(same.sum()))

    # Даты: 180 дней
    start = pd.Timestamp("2022-01-01")
    day_offsets = rng.integers(0, 180, size=n_rows)
    hour = rng.integers(0, 24, size=n_rows)
    minute = rng.integers(0, 60, size=n_rows)
    second = rng.integers(0, 60, size=n_rows)
    ts = start + pd.to_timedelta(day_offsets, unit="D") \
        + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")

    # Суммы: логнормальное распределение — как настоящие платежи
    amount = np.round(np.exp(rng.normal(6.5, 1.6, size=n_rows)), 2)

    pay_cur = rng.choice(CURRENCIES, size=n_rows, p=[0.35, 0.2, 0.12, 0.06, 0.05,
                                                     0.05, 0.05, 0.04, 0.04, 0.04])
    rec_cur = pay_cur.copy()
    mismatch = rng.random(n_rows) < 0.08
    rec_cur[mismatch] = rng.choice(CURRENCIES, size=int(mismatch.sum()))

    loc_weights = np.array([3, 2, 2, 1.5, 1.5, 1.2, 1.2, 1.2, 1.2, 1, 1, 1, 1,
                            0.8, 0.9, 0.8, 0.5, 0.5])
    sender_loc = rng.choice(COUNTRIES, size=n_rows, p=loc_weights / loc_weights.sum())
    receiver_loc = rng.choice(COUNTRIES, size=n_rows)

    pay_type = rng.choice(PAYMENT_TYPES, size=n_rows)

    df = pd.DataFrame({
        "Time": [f"{h:02d}:{m:02d}:{s:02d}" for h, m, s in zip(hour, minute, second)],
        "Date": ts.strftime("%Y-%m-%d"),
        "Sender_account": senders,
        "Receiver_account": receivers,
        "Amount": amount,   # float64; в float32 приведёт src/data_loader.py при чтении
        "Payment_currency": pay_cur,
        "Received_currency": rec_cur,
        "Sender_bank_location": sender_loc,
        "Receiver_bank_location": receiver_loc,
        "Payment_type": pay_type,
        "Is_laundering": np.zeros(n_rows, dtype="int8"),
        "Laundering_type": "None",
    })

    # ---- Внедряем "отмывание": ~0.6% строк (в реале 0.1%, здесь больше
    # ---- для наглядности графиков на маленькой выборке) -------------------
    n_illicit = int(n_rows * 0.006)
    idx = rng.choice(n_rows, size=n_illicit, replace=False)

    types_draw = rng.choice(LAUNDERING_TYPES, size=n_illicit)
    df.loc[idx, "Is_laundering"] = 1
    df.loc[idx, "Laundering_type"] = types_draw

    # Structuring — суммы чуть ниже порога 10,000 (ключевая типология!)
    struct_mask = types_draw == "Structuring"
    df.loc[idx[struct_mask], "Amount"] = np.round(
        rng.uniform(8_800, 9_950, size=int(struct_mask.sum())), 2)

    # Single_large — одна огромная транзакция
    large_mask = types_draw == "Single_large"
    df.loc[idx[large_mask], "Amount"] = np.round(
        rng.uniform(500_000, 5_000_000, size=int(large_mask.sum())), 2)

    # Smurfing — много мелких одинаковых сумм ночью
    smurf_mask = types_draw == "Smurfing"
    smurf_idx = idx[smurf_mask]
    df.loc[smurf_idx, "Amount"] = np.round(rng.uniform(300, 900, size=len(smurf_idx)), 2)
    df.loc[smurf_idx, "Time"] = [f"{h:02d}:{m:02d}:{s:02d}"
                                 for h, m, s in zip(rng.integers(0, 5, size=len(smurf_idx)),
                                                    rng.integers(0, 60, size=len(smurf_idx)),
                                                    rng.integers(0, 60, size=len(smurf_idx)))]
    return df


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "SAML-D_SYNTHETIC_SAMPLE.csv"
    df = make_synthetic()
    df.to_csv(out, index=False)
    print(f"Создан файл: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"Строк: {len(df):,}, отмывания: {int(df['Is_laundering'].sum()):,} "
          f"({df['Is_laundering'].mean() * 100:.3f}%)")

"""
src/synthetic_data.py
=====================
Генератор датасета, ПОХОЖЕГО на SAML-D, со всеми 17 типологиями отмывания.

ЗАЧЕМ ЭТОТ МОДУЛЬ (прочитай обязательно):
-----------------------------------------
1. Реальный SAML-D (9.5 млн строк, ~1 ГБ) лежит только на машине разработчика:
   данные с Kaggle нельзя перекладывать в git. Из-за этого код нельзя протестировать
   в CI, на другой машине или в облаке.
2. Здесь мы генерируем датасет ТОЙ ЖЕ СХЕМЫ и ТОГО ЖЕ МАСШТАБА (9 504 852 строки,
   9 873 отмывания, те же 17 типологий с теми же count'ами), причём каждая типология
   генерируется по своей «физике»:
       Structuring      = серия платежей чуть ниже порога $10 000
       Smurfing         = много мелких платежей за короткое время
       Fan_In / Fan_Out = звезда на графе (сбор / раскидывание)
       Cycle            = деньги идут по кругу A -> B -> C -> ... -> A
       Bipartite        = две группы счетов, связанные «все со всеми»
       Gather-Scatter   = много -> хаб -> много
       Scatter-Gather   = хаб -> много мулов -> финальный счёт
       Single_large     = одна огромная транзакция
       Over-Invoicing   = завышенный инвойс через границу
       Behavioural_Change_1/2 = счёт резко меняет сумму / контрагентов
3. Это позволяет гонять весь пайплайн (EDA -> фичи -> правила -> ML -> дашборд)
  端到端, не имея доступа к реальным данным, и сравнивать поведение кода.

ВАЖНО: цифры, полученные на этом датасете, — ОРИЕНТИР, а не реальные показатели.
В отчётах они подписываются как «synthetic reference run».

ЗАПУСК:
    python src/synthetic_data.py                  # полный объём (~9.5 млн строк)
    python src/synthetic_data.py --n-rows 500000  # быстрая версия для отладки
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import gc
import pandas as pd

from src.data_loader import RAW_DIR, RAW_COLUMNS

# ---------------------------------------------------------------------------
# ПАРАМЕТРЫ, СОВПАДАЮЩИЕ С РЕАЛЬНЫМ SAML-D
# ---------------------------------------------------------------------------
N_TOTAL = 9_504_852          # всего транзакций в реальном датасете
N_ILLICIT = 9_873            # всего отмываний (0.1039%)

# Раскладка по типологиям — 1 в 1 как в реальном SAML-D
TYPOLOGY_COUNTS = {
    "Structuring": 1870,
    "Cash_Withdrawal": 1334,
    "Deposit-Send": 945,
    "Smurfing": 932,
    "Layered_Fan_In": 656,
    "Layered_Fan_Out": 529,
    "Stacked Bipartite": 506,
    "Behavioural_Change_1": 394,
    "Bipartite": 383,
    "Cycle": 382,
    "Fan_In": 364,
    "Gather-Scatter": 354,
    "Behavioural_Change_2": 345,
    "Scatter-Gather": 338,
    "Single_large": 250,
    "Fan_Out": 237,
    "Over-Invoicing": 54,
}

N_ACCOUNTS = 120_000         # размер «банка»
N_DAYS = 365
START_DATE = "2022-01-01"

# Страны (включая «высокорисковые» из описания SAML-D: Mexico, Turkey, Morocco, UAE)
COUNTRIES = np.array([
    "USA", "United Kingdom", "Germany", "France", "Italy", "Spain", "China",
    "India", "UAE", "Turkey", "Russia", "Brazil", "Mexico", "Morocco",
    "Japan", "Switzerland", "Cyprus", "Malta",
])
COUNTRY_W = np.array([0.20, 0.10, 0.08, 0.06, 0.05, 0.04, 0.07,
                      0.06, 0.05, 0.05, 0.04, 0.04, 0.05, 0.03,
                      0.04, 0.02, 0.01, 0.01])
COUNTRY_W = COUNTRY_W / COUNTRY_W.sum()

CURRENCY_BY_COUNTRY = {
    "USA": "US Dollar", "United Kingdom": "British Pound", "Germany": "Euro",
    "France": "Euro", "Italy": "Euro", "Spain": "Euro", "China": "Yuan",
    "India": "Rupee", "UAE": "UAE Dirham", "Turkey": "Lira", "Russia": "Ruble",
    "Brazil": "Brazilian Real", "Mexico": "Mexican Peso", "Morocco": "Moroccan Dirham",
    "Japan": "Yen", "Switzerland": "Swiss Franc", "Cyprus": "Euro", "Malta": "Euro",
}
CURRENCIES = np.array([
    "US Dollar", "Euro", "British Pound", "Yuan", "Rupee", "UAE Dirham",
    "Swiss Franc", "Yen", "Ruble", "Lira", "Brazilian Real", "Mexican Peso",
    "Moroccan Dirham",
])

PAYMENT_TYPES = np.array([
    "Credit card", "Debit card", "ACH", "Wire", "Cash Deposit",
    "Cash Withdrawal", "Cheque", "Cross-border", "Card",
])
PAYMENT_W = np.array([0.16, 0.20, 0.14, 0.12, 0.10, 0.09, 0.06, 0.06, 0.07])
PAYMENT_W = PAYMENT_W / PAYMENT_W.sum()

# Профиль «рабочего дня»: ночью транзакций мало, пик в обед
HOUR_W = np.array([
    0.010, 0.006, 0.005, 0.004, 0.005, 0.010,   # 00-05
    0.020, 0.035, 0.055, 0.070, 0.075, 0.080,   # 06-11
    0.075, 0.070, 0.075, 0.080, 0.085, 0.075,   # 12-17
    0.055, 0.040, 0.030, 0.020, 0.015, 0.011,   # 18-23
])
HOUR_W = HOUR_W / HOUR_W.sum()

# Структура реальной платёжной сети: счёт платит в основном ОДНИМ И ТЕМ ЖЕ
# контрагентам (магазины, аренда, зарплата, свои счета), а не случайным людям.
# Без этого граф получается случайно-плотным: у каждой вершины огромное
# «циклическое ядро», и граф-признаки перестают что-либо разделять.
N_COUNTERPARTIES = 12        # сколько «постоянных» контрагентов у счёта
REPEAT_SHARE = 0.60          # какая доля переводов идёт постоянным контрагентам

CROSS_BORDER_RATE = 0.30     # доля транзакций между разными странами
CURRENCY_MISMATCH_RATE = 0.08

_cp_rng = np.random.default_rng(7)
CP_LUT = _cp_rng.integers(0, N_ACCOUNTS, size=(N_ACCOUNTS, N_COUNTERPARTIES))

_TIME_LUT = np.array([f"{h:02d}:{m:02d}:{s:02d}"
                      for h in range(24) for m in range(60) for s in range(60)])
_DATE_LUT = pd.date_range(START_DATE, periods=N_DAYS).strftime("%Y-%m-%d").to_numpy()
_ACC_LUT = np.array([f"ACC_{i:07d}" for i in range(N_ACCOUNTS)])


# ---------------------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------------
def _pick_countries(rng, n: int, sender_idx: np.ndarray | None = None):
    """Возвращает (sender_country_idx, receiver_country_idx)."""
    s_loc = rng.choice(len(COUNTRIES), size=n, p=COUNTRY_W)
    r_loc = s_loc.copy()
    cross = rng.random(n) < CROSS_BORDER_RATE
    r_loc[cross] = rng.choice(len(COUNTRIES), size=int(cross.sum()), p=COUNTRY_W)
    return s_loc, r_loc


def _currencies(rng, s_loc: np.ndarray, r_loc: np.ndarray):
    """Валюта платежа по стране отправителя, валюта получения — по стране получателя."""
    pay_cur = np.array([list(CURRENCIES).index(CURRENCY_BY_COUNTRY[COUNTRIES[i]])
                        for i in s_loc])
    rec_cur = np.array([list(CURRENCIES).index(CURRENCY_BY_COUNTRY[COUNTRIES[i]])
                        for i in r_loc])
    mismatch = rng.random(len(s_loc)) < CURRENCY_MISMATCH_RATE
    if mismatch.any():
        rec_cur[mismatch] = rng.choice(len(CURRENCIES), size=int(mismatch.sum()))
    return pay_cur, rec_cur


def _day_second(rng, n: int, night_share: float = 0.0):
    """Случайные метки времени внутри дня: (день, секунда_от_полуночи)."""
    day = rng.integers(0, N_DAYS, size=n)
    if night_share > 0:
        # ночные часы 00:00-05:59 задаём явно (для Smurfing/Cash_Withdrawal)
        hour = np.where(rng.random(n) < night_share,
                        rng.integers(0, 6, size=n),
                        rng.choice(24, size=n, p=HOUR_W))
    else:
        hour = rng.choice(24, size=n, p=HOUR_W)
    minute = rng.integers(0, 60, size=n)
    second = rng.integers(0, 60, size=n)
    return day, (hour * 3600 + minute * 60 + second).astype("int64")


class _Collector:
    """Накапливает сгенерированные транзакции (все массивы — числовые коды)."""

    def __init__(self):
        self.parts = []

    def add(self, sender, receiver, amount, day, sec, pay_idx, s_loc, r_loc,
            pay_cur, rec_cur, laundering_type):
        # Серии платежей (smurfing/structuring) строятся как «старт + случайные
        # приращения», поэтому метка времени может вылезти за границы суток.
        # Обрезаем, иначе индексация таблицы времени упадёт (86400 значений).
        self.parts.append(dict(
            Sender_account=sender.astype("int64"),
            Receiver_account=receiver.astype("int64"),
            Amount=np.round(amount, 2),
            day=np.clip(day, 0, N_DAYS - 1).astype("int64"),
            sec=np.clip(sec, 0, 86_399).astype("int64"),
            Payment_type=pay_idx.astype("int64"),
            Sender_bank_location=s_loc.astype("int64"),
            Receiver_bank_location=r_loc.astype("int64"),
            Payment_currency=pay_cur.astype("int64"),
            Received_currency=rec_cur.astype("int64"),
            Laundering_type=laundering_type,
        ))

    def frame(self, rng) -> pd.DataFrame:
        df = pd.concat([pd.DataFrame(p) for p in self.parts], ignore_index=True)
        df["Is_laundering"] = 1
        return df


# ---------------------------------------------------------------------------
# ГЕНЕРАТОРЫ ОТДЕЛЬНЫХ ТИПОЛОГИЙ
# ---------------------------------------------------------------------------
def _gen_structuring(rng, col: _Collector, total: int):
    """Серия платежей ЧУТЬ НИЖЕ порога $10 000 (классический structuring)."""
    while col.rows < total:
        n = min(rng.integers(5, 12), total - col.rows)
        sender = np.full(n, rng.integers(0, N_ACCOUNTS))
        receiver = np.full(n, rng.integers(0, N_ACCOUNTS))
        amount = rng.uniform(8_700, 9_950, n)
        day = np.full(n, rng.integers(0, N_DAYS))
        start = rng.integers(9 * 3600, 15 * 3600)
        sec = start + np.sort(rng.integers(0, 6 * 3600, n))
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.05, 0.05, 0.2, 0.3, 0.25, 0.0, 0.05, 0.05, 0.05])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender, receiver, amount, day, sec, pay, s_loc, r_loc, pay_cur, rec_cur,
                "Structuring")


def _gen_smurfing(rng, col: _Collector, total: int):
    """Много мелких платежей за короткое время, часто ночью."""
    while col.rows < total:
        n = min(rng.integers(8, 16), total - col.rows)
        sender = np.full(n, rng.integers(0, N_ACCOUNTS))
        receiver = np.full(n, rng.integers(0, N_ACCOUNTS))
        amount = rng.uniform(300, 950, n)
        day, sec = _day_second(rng, 1, night_share=0.6)
        day = np.full(n, day[0])
        start = sec[0]
        sec = start + np.sort(rng.integers(0, 90 * 60, n))
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.2, 0.25, 0.1, 0.1, 0.15, 0.05, 0.05, 0.05, 0.05])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender, receiver, amount, day, sec, pay, s_loc, r_loc, pay_cur, rec_cur,
                "Smurfing")


def _gen_simple(rng, col: _Collector, total: int, name: str,
                pay_name: str = "Cash Withdrawal", night: float = 0.35,
                lo: float = 1_000, hi: float = 9_800):
    """Одиночные транзакции заданного типа (Cash_Withdrawal, Deposit-Send, Single_large...)."""
    pay_idx = int(np.where(PAYMENT_TYPES == pay_name)[0][0])
    n = total
    sender = rng.integers(0, N_ACCOUNTS, n)
    receiver = rng.integers(0, N_ACCOUNTS, n)
    amount = rng.uniform(lo, hi, n)
    day, sec = _day_second(rng, n, night_share=night)
    pay = np.full(n, pay_idx)
    s_loc, r_loc = _pick_countries(rng, n)
    pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
    col.add(sender, receiver, amount, day, sec, pay, s_loc, r_loc, pay_cur, rec_cur, name)


def _gen_fan_out(rng, col: _Collector, total: int):
    """Fan-Out: один счёт раскидывает деньги многим получателям."""
    while col.rows < total:
        n = min(rng.integers(12, 30), total - col.rows)
        sender = np.full(n, rng.integers(0, N_ACCOUNTS))
        receiver = rng.choice(N_ACCOUNTS, size=n, replace=False)
        amount = rng.uniform(7_000, 9_900, n)
        day = np.full(n, rng.integers(0, N_DAYS))
        start = rng.integers(9 * 3600, 16 * 3600)
        sec = start + np.sort(rng.integers(0, 5 * 3600, n))
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.05, 0.05, 0.2, 0.35, 0.1, 0.0, 0.05, 0.15, 0.05])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender, receiver, amount, day, sec, pay, s_loc, r_loc, pay_cur, rec_cur,
                "Fan_Out")


def _gen_fan_in(rng, col: _Collector, total: int):
    """Fan-In: много отправителей собирают деньги на один счёт."""
    while col.rows < total:
        n = min(rng.integers(15, 35), total - col.rows)
        receiver = np.full(n, rng.integers(0, N_ACCOUNTS))
        sender = rng.choice(N_ACCOUNTS, size=n, replace=False)
        amount = rng.uniform(3_000, 9_500, n)
        day = np.full(n, rng.integers(0, N_DAYS))
        start = rng.integers(8 * 3600, 15 * 3600)
        sec = start + np.sort(rng.integers(0, 8 * 3600, n))
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.05, 0.05, 0.2, 0.3, 0.15, 0.0, 0.05, 0.15, 0.05])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender, receiver, amount, day, sec, pay, s_loc, r_loc, pay_cur, rec_cur,
                "Fan_In")


def _gen_layered_fan_out(rng, col: _Collector, total: int):
    """Layered Fan-Out: A -> несколько «мулов» -> каждый мул раскидывает дальше."""
    while col.rows < total:
        hub = rng.integers(0, N_ACCOUNTS)
        mules = rng.choice(N_ACCOUNTS, size=3, replace=False)
        tails = rng.choice(N_ACCOUNTS, size=12, replace=False)
        amount = rng.uniform(25_000, 60_000)
        day = rng.integers(0, N_DAYS - 3)
        s1 = np.full(len(mules), hub)
        a1 = np.full(len(mules), amount)
        d1 = np.full(len(mules), day)
        sec1 = np.sort(rng.integers(9 * 3600, 12 * 3600, len(mules)))
        s2 = np.repeat(mules, 4)
        a2 = rng.uniform(6_000, 9_900, len(s2))
        d2 = np.full(len(s2), day + rng.integers(1, 3))
        sec2 = np.sort(rng.integers(9 * 3600, 17 * 3600, len(s2)))
        sender = np.concatenate([s1, s2])
        receiver = np.concatenate([mules, tails[:len(s2)]])
        amount_all = np.concatenate([a1, a2])
        day_all = np.concatenate([d1, d2])
        sec_all = np.concatenate([sec1, sec2])
        n = len(sender)
        keep = min(n, total - col.rows)
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.05, 0.05, 0.15, 0.4, 0.1, 0.0, 0.05, 0.15, 0.05])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender[:keep], receiver[:keep], amount_all[:keep], day_all[:keep],
                sec_all[:keep], pay[:keep], s_loc[:keep], r_loc[:keep],
                pay_cur[:keep], rec_cur[:keep], "Layered_Fan_Out")


def _gen_layered_fan_in(rng, col: _Collector, total: int):
    """Layered Fan-In: много -> мулы -> финальный счёт."""
    while col.rows < total:
        final = rng.integers(0, N_ACCOUNTS)
        mules = rng.choice(N_ACCOUNTS, size=3, replace=False)
        sources = rng.choice(N_ACCOUNTS, size=12, replace=False)
        day = rng.integers(0, N_DAYS - 3)
        s1 = sources
        a1 = rng.uniform(5_000, 9_500, len(s1))
        d1 = np.full(len(s1), day)
        sec1 = np.sort(rng.integers(8 * 3600, 15 * 3600, len(s1)))
        s2 = np.repeat(mules, 1)
        a2 = rng.uniform(18_000, 40_000, len(s2))
        d2 = np.full(len(s2), day + rng.integers(1, 3))
        sec2 = np.sort(rng.integers(9 * 3600, 16 * 3600, len(s2)))
        sender = np.concatenate([s1, s2])
        receiver = np.concatenate([np.repeat(mules, 4)[:len(s1)], s2])
        amount_all = np.concatenate([a1, a2])
        day_all = np.concatenate([d1, d2])
        sec_all = np.concatenate([sec1, sec2])
        n = len(sender)
        keep = min(n, total - col.rows)
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.05, 0.05, 0.2, 0.35, 0.1, 0.0, 0.05, 0.15, 0.05])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender[:keep], receiver[:keep], amount_all[:keep], day_all[:keep],
                sec_all[:keep], pay[:keep], s_loc[:keep], r_loc[:keep],
                pay_cur[:keep], rec_cur[:keep], "Layered_Fan_In")


def _gen_bipartite(rng, col: _Collector, total: int):
    """Bipartite: две группы счетов, связанные «все со всеми»."""
    while col.rows < total:
        g1 = rng.choice(N_ACCOUNTS, size=rng.integers(4, 8), replace=False)
        g2 = rng.choice(N_ACCOUNTS, size=rng.integers(4, 8), replace=False)
        senders, receivers = np.meshgrid(g1, g2)
        sender, receiver = senders.ravel(), receivers.ravel()
        n = len(sender)
        keep = min(n, total - col.rows)
        amount = rng.uniform(4_000, 9_800, n)
        day, sec = _day_second(rng, n)
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.05, 0.05, 0.2, 0.3, 0.1, 0.0, 0.05, 0.2, 0.05])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender[:keep], receiver[:keep], amount[:keep], day[:keep], sec[:keep],
                pay[:keep], s_loc[:keep], r_loc[:keep], pay_cur[:keep], rec_cur[:keep],
                "Bipartite")


def _gen_stacked_bipartite(rng, col: _Collector, total: int):
    """Stacked Bipartite: две bipartite-структуры, поставленные друг на друга."""
    while col.rows < total:
        g1 = rng.choice(N_ACCOUNTS, size=5, replace=False)
        g2 = rng.choice(N_ACCOUNTS, size=5, replace=False)
        g3 = rng.choice(N_ACCOUNTS, size=5, replace=False)
        s1, r1 = np.meshgrid(g1, g2)
        s2, r2 = np.meshgrid(g2, g3)
        sender = np.concatenate([s1.ravel(), s2.ravel()])
        receiver = np.concatenate([r1.ravel(), r2.ravel()])
        n = len(sender)
        keep = min(n, total - col.rows)
        amount = rng.uniform(3_500, 9_500, n)
        day, sec = _day_second(rng, n)
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.05, 0.05, 0.2, 0.3, 0.1, 0.0, 0.05, 0.2, 0.05])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender[:keep], receiver[:keep], amount[:keep], day[:keep], sec[:keep],
                pay[:keep], s_loc[:keep], r_loc[:keep], pay_cur[:keep], rec_cur[:keep],
                "Stacked Bipartite")


def _gen_cycle(rng, col: _Collector, total: int):
    """Cycle: деньги идут по кругу и возвращаются (A -> B -> C -> ... -> A)."""
    while col.rows < total:
        k = int(rng.integers(5, 10))
        ring = rng.choice(N_ACCOUNTS, size=k, replace=False)
        sender = ring
        receiver = np.roll(ring, -1)
        amount = rng.uniform(20_000, 80_000) * (0.97 ** np.arange(k))  # «комиссия» на каждом шаге
        day = rng.integers(0, N_DAYS - k)
        day = day + np.arange(k)
        sec = np.sort(rng.integers(9 * 3600, 17 * 3600, k))
        n = k
        keep = min(n, total - col.rows)
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.0, 0.0, 0.15, 0.5, 0.05, 0.0, 0.05, 0.25, 0.0])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender[:keep], receiver[:keep], amount[:keep], day[:keep], sec[:keep],
                pay[:keep], s_loc[:keep], r_loc[:keep], pay_cur[:keep], rec_cur[:keep],
                "Cycle")


def _gen_gather_scatter(rng, col: _Collector, total: int):
    """Gather-Scatter: много -> один хаб -> много."""
    while col.rows < total:
        hub = rng.integers(0, N_ACCOUNTS)
        k = int(rng.integers(8, 14))
        sources = rng.choice(N_ACCOUNTS, size=k, replace=False)
        targets = rng.choice(N_ACCOUNTS, size=k, replace=False)
        day = rng.integers(0, N_DAYS - 2)
        s1, a1 = sources, rng.uniform(4_000, 9_500, k)
        d1 = np.full(k, day)
        sec1 = np.sort(rng.integers(8 * 3600, 13 * 3600, k))
        s2 = np.full(k, hub)
        a2 = rng.uniform(4_000, 9_500, k)
        d2 = np.full(k, day + 1)
        sec2 = np.sort(rng.integers(13 * 3600, 19 * 3600, k))
        sender = np.concatenate([s1, s2])
        receiver = np.concatenate([np.full(k, hub), targets])
        amount = np.concatenate([a1, a2])
        day_all = np.concatenate([d1, d2])
        sec_all = np.concatenate([sec1, sec2])
        n = len(sender)
        keep = min(n, total - col.rows)
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.05, 0.05, 0.2, 0.35, 0.1, 0.0, 0.05, 0.15, 0.05])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender[:keep], receiver[:keep], amount[:keep], day_all[:keep], sec_all[:keep],
                pay[:keep], s_loc[:keep], r_loc[:keep], pay_cur[:keep], rec_cur[:keep],
                "Gather-Scatter")


def _gen_scatter_gather(rng, col: _Collector, total: int):
    """Scatter-Gather: один -> много мулов -> один финальный счёт."""
    while col.rows < total:
        hub = rng.integers(0, N_ACCOUNTS)
        final = rng.integers(0, N_ACCOUNTS)
        k = int(rng.integers(8, 14))
        mules = rng.choice(N_ACCOUNTS, size=k, replace=False)
        day = rng.integers(0, N_DAYS - 2)
        s1 = np.full(k, hub)
        a1 = rng.uniform(4_000, 9_800, k)
        d1 = np.full(k, day)
        sec1 = np.sort(rng.integers(9 * 3600, 14 * 3600, k))
        s2 = mules
        a2 = rng.uniform(4_000, 9_800, k)
        d2 = np.full(k, day + 1)
        sec2 = np.sort(rng.integers(14 * 3600, 20 * 3600, k))
        sender = np.concatenate([s1, s2])
        receiver = np.concatenate([mules, np.full(k, final)])
        amount = np.concatenate([a1, a2])
        day_all = np.concatenate([d1, d2])
        sec_all = np.concatenate([sec1, sec2])
        n = len(sender)
        keep = min(n, total - col.rows)
        pay = rng.choice(len(PAYMENT_TYPES), size=n,
                         p=np.array([0.05, 0.05, 0.2, 0.35, 0.1, 0.0, 0.05, 0.15, 0.05])
                         / 1.0)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender[:keep], receiver[:keep], amount[:keep], day_all[:keep], sec_all[:keep],
                pay[:keep], s_loc[:keep], r_loc[:keep], pay_cur[:keep], rec_cur[:keep],
                "Scatter-Gather")


def _gen_behavioural_change_1(rng, col: _Collector, total: int):
    """Behavioural_Change_1: счёт резко меняет РАЗМЕР операций (скачок суммы)."""
    while col.rows < total:
        acc = rng.integers(0, N_ACCOUNTS)
        k = int(rng.integers(3, 6))          # обычная история
        n = k + 1                             # + одна аномальная
        keep = min(n, total - col.rows)
        receiver = rng.choice(N_ACCOUNTS, size=n, replace=False)
        base = rng.uniform(500, 3_000)
        amount = np.concatenate([rng.uniform(base * 0.7, base * 1.3, k),
                                 [base * rng.uniform(15, 40)]])   # резкий скачок
        day = rng.integers(0, N_DAYS - 10)
        day = np.concatenate([rng.integers(day, day + 8, k), [day + 9]])
        sec = rng.integers(9 * 3600, 18 * 3600, n)
        pay = rng.choice(len(PAYMENT_TYPES), size=n, p=PAYMENT_W)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(np.full(n, acc)[:keep], receiver[:keep], amount[:keep], day[:keep],
                sec[:keep], pay[:keep], s_loc[:keep], r_loc[:keep], pay_cur[:keep],
                rec_cur[:keep], "Behavioural_Change_1")


def _gen_behavioural_change_2(rng, col: _Collector, total: int):
    """Behavioural_Change_2: счёт резко меняет КОНТРАГЕНТОВ и частоту операций."""
    while col.rows < total:
        acc = rng.integers(0, N_ACCOUNTS)
        n = int(rng.integers(4, 9))
        keep = min(n, total - col.rows)
        receiver = rng.choice(N_ACCOUNTS, size=n, replace=False)
        amount = rng.uniform(2_000, 9_000, n)
        day = np.full(n, rng.integers(0, N_DAYS))
        start = rng.integers(9 * 3600, 12 * 3600)
        sec = start + np.sort(rng.integers(0, 3 * 3600, n))   # залп за пару часов
        pay = rng.choice(len(PAYMENT_TYPES), size=n, p=PAYMENT_W)
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(np.full(n, acc)[:keep], receiver[:keep], amount[:keep], day[:keep],
                sec[:keep], pay[:keep], s_loc[:keep], r_loc[:keep], pay_cur[:keep],
                rec_cur[:keep], "Behavioural_Change_2")


def _gen_over_invoicing(rng, col: _Collector, total: int):
    """Over-Invoicing: завышенный инвойс через границу (integration)."""
    pay_idx = int(np.where(PAYMENT_TYPES == "Cross-border")[0][0])
    n = total
    sender = rng.integers(0, N_ACCOUNTS, n)
    receiver = rng.integers(0, N_ACCOUNTS, n)
    amount = rng.uniform(150_000, 2_000_000, n)
    day, sec = _day_second(rng, n)
    pay = np.full(n, pay_idx)
    s_loc = rng.choice(len(COUNTRIES), size=n, p=COUNTRY_W)
    r_loc = rng.choice(len(COUNTRIES), size=n, p=COUNTRY_W)
    pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
    col.add(sender, receiver, amount, day, sec, pay, s_loc, r_loc, pay_cur, rec_cur,
            "Over-Invoicing")


def _gen_deposit_send(rng, col: _Collector, total: int):
    """Deposit-Send: наличные кладут на счёт и сразу отправляют дальше (placement)."""
    deposit_idx = int(np.where(PAYMENT_TYPES == "Cash Deposit")[0][0])
    wire_idx = int(np.where(PAYMENT_TYPES == "Wire")[0][0])
    while col.rows < total:
        n = min(2, total - col.rows)
        acc = rng.integers(0, N_ACCOUNTS)
        target = rng.integers(0, N_ACCOUNTS)
        cash_in = rng.uniform(15_000, 90_000)
        day = rng.integers(0, N_DAYS - 3)
        sender = np.array([rng.integers(0, N_ACCOUNTS), acc])[:n]
        receiver = np.array([acc, target])[:n]
        amount = np.array([cash_in, cash_in * rng.uniform(0.93, 0.99)])[:n]
        day_all = np.array([day, day + rng.integers(0, 3)])[:n]
        sec = rng.integers(9 * 3600, 17 * 3600, n)
        pay = np.array([deposit_idx, wire_idx])[:n]
        s_loc, r_loc = _pick_countries(rng, n)
        pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)
        col.add(sender, receiver, amount, day_all, sec, pay, s_loc, r_loc, pay_cur, rec_cur,
                "Deposit-Send")


# ---------------------------------------------------------------------------
# СБОРКА ДАТАСЕТА
# ---------------------------------------------------------------------------
def _run(gen, rng, total: int, *args) -> pd.DataFrame:
    """
    Запускает один генератор типологии и следит, чтобы он дал РОВНО total строк.

    ВАЖНО: счётчик строк у каждого генератора СВОЙ. Раньше он был общим на все
    типологии, из-за чего после Structuring (1870 строк) все остальные генераторы
    считали, что «норма уже выполнена», и не создавали ни одной транзакции.
    """
    col = _Collector()
    col.rows = 0
    original_add = col.add

    def add(*a, **kw):
        original_add(*a, **kw)
        col.rows = sum(len(part["Sender_account"]) for part in col.parts)

    col.add = add
    gen(rng, col, total, *args)
    df = col.frame(rng)
    assert len(df) == total, f"{gen.__name__}: ожидалось {total}, получено {len(df)}"
    return df


def _collect_illicit(rng) -> pd.DataFrame:
    """Генерирует все отмывочные транзакции по типологиям (ровно N_ILLICIT строк)."""
    parts = [
        _run(_gen_structuring, rng, TYPOLOGY_COUNTS["Structuring"]),
        _run(_gen_smurfing, rng, TYPOLOGY_COUNTS["Smurfing"]),
        _run(_gen_simple, rng, TYPOLOGY_COUNTS["Cash_Withdrawal"], "Cash_Withdrawal",
             "Cash Withdrawal", 0.35, 1_000, 9_800),
        _run(_gen_deposit_send, rng, TYPOLOGY_COUNTS["Deposit-Send"]),
        _run(_gen_layered_fan_out, rng, TYPOLOGY_COUNTS["Layered_Fan_Out"]),
        _run(_gen_layered_fan_in, rng, TYPOLOGY_COUNTS["Layered_Fan_In"]),
        _run(_gen_stacked_bipartite, rng, TYPOLOGY_COUNTS["Stacked Bipartite"]),
        _run(_gen_behavioural_change_1, rng, TYPOLOGY_COUNTS["Behavioural_Change_1"]),
        _run(_gen_bipartite, rng, TYPOLOGY_COUNTS["Bipartite"]),
        _run(_gen_cycle, rng, TYPOLOGY_COUNTS["Cycle"]),
        _run(_gen_fan_in, rng, TYPOLOGY_COUNTS["Fan_In"]),
        _run(_gen_gather_scatter, rng, TYPOLOGY_COUNTS["Gather-Scatter"]),
        _run(_gen_behavioural_change_2, rng, TYPOLOGY_COUNTS["Behavioural_Change_2"]),
        _run(_gen_scatter_gather, rng, TYPOLOGY_COUNTS["Scatter-Gather"]),
        _run(_gen_simple, rng, TYPOLOGY_COUNTS["Single_large"], "Single_large",
             "Wire", 0.15, 500_000, 5_000_000),
        _run(_gen_fan_out, rng, TYPOLOGY_COUNTS["Fan_Out"]),
        _run(_gen_over_invoicing, rng, TYPOLOGY_COUNTS["Over-Invoicing"]),
    ]
    df = pd.concat(parts, ignore_index=True)

    # Чиним «самопереводы» (отправитель и получатель случайно совпали)
    same = df["Sender_account"] == df["Receiver_account"]
    if same.any():
        df.loc[same, "Receiver_account"] = (df.loc[same, "Receiver_account"] + 1) % N_ACCOUNTS
    return df


def _normal_chunk(rng, n: int) -> pd.DataFrame:
    """Генерирует порцию ОБЫЧНЫХ (легальных) транзакций."""
    sender = rng.integers(0, N_ACCOUNTS, n)
    random_receiver = rng.integers(0, N_ACCOUNTS, n)
    # постоянные контрагенты: у каждого счёта свой небольшой «круг общения»
    usual_receiver = CP_LUT[sender, rng.integers(0, N_COUNTERPARTIES, n)]
    repeat = rng.random(n) < REPEAT_SHARE
    receiver = np.where(repeat, usual_receiver, random_receiver)
    same = sender == receiver
    if same.any():
        receiver[same] = (receiver[same] + 1) % N_ACCOUNTS

    amount = np.exp(rng.normal(6.2, 1.5, n)).clip(1, 5_000_000)
    day, sec = _day_second(rng, n)
    pay = rng.choice(len(PAYMENT_TYPES), size=n, p=PAYMENT_W)
    s_loc, r_loc = _pick_countries(rng, n)
    pay_cur, rec_cur = _currencies(rng, s_loc, r_loc)

    return pd.DataFrame({
        "Sender_account": sender, "Receiver_account": receiver,
        "Amount": np.round(amount, 2), "day": day, "sec": sec,
        "Payment_type": pay, "Sender_bank_location": s_loc,
        "Receiver_bank_location": r_loc, "Payment_currency": pay_cur,
        "Received_currency": rec_cur, "Is_laundering": 0, "Laundering_type": np.array([None] * n, dtype=object),
    })


def _codes_to_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Превращает числовые коды в итоговые колонки SAML-D."""
    out = pd.DataFrame({
        "Time": _TIME_LUT[df["sec"].to_numpy()],
        "Date": _DATE_LUT[df["day"].to_numpy()],
        "Sender_account": _ACC_LUT[df["Sender_account"].to_numpy()],
        "Receiver_account": _ACC_LUT[df["Receiver_account"].to_numpy()],
        "Amount": df["Amount"].to_numpy(),
        "Payment_currency": CURRENCIES[df["Payment_currency"].to_numpy()],
        "Received_currency": CURRENCIES[df["Received_currency"].to_numpy()],
        "Sender_bank_location": COUNTRIES[df["Sender_bank_location"].to_numpy()],
        "Receiver_bank_location": COUNTRIES[df["Receiver_bank_location"].to_numpy()],
        "Payment_type": PAYMENT_TYPES[df["Payment_type"].to_numpy()],
        "Is_laundering": df["Is_laundering"].to_numpy().astype("int8"),
        "Laundering_type": df["Laundering_type"].to_numpy()
                           if "Laundering_type" in df else None,
    })
    return out[RAW_COLUMNS]


def generate(n_rows: int = N_TOTAL, seed: int = 42, chunk: int = 1_000_000,
             out_path=None, verbose: bool = True) -> Path:
    """
    Генерирует CSV той же схемы, что и SAML-D.

    Пишет по частям (chunk), чтобы не держать 9.5 млн строк в памяти сразу.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(out_path) if out_path else RAW_DIR / "SAML-D_synthetic_full.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale = n_rows / N_TOTAL
    illicit = _collect_illicit(rng)
    if scale < 1.0:                       # пропорционально урезаем «грязные» транзакции
        keep = max(1, int(len(illicit) * scale))
        illicit = illicit.sample(n=keep, random_state=seed)

    n_normal = max(0, n_rows - len(illicit))
    if verbose:
        print(f"[synthetic] всего строк: {n_rows:,} | отмываний: {len(illicit):,} "
              f"({len(illicit) / n_rows * 100:.4f}%) | обычных: {n_normal:,}")

    # ВАЖНО: «грязные» транзакции НЕЛЬЗЯ дописывать в конец файла. Если это
    # сделать, то любые операции, завязанные на порядок строк (head(n),
    # разбиение «первые 80% / остальное», батчи при обучении), окажутся
    # полностью «чистыми» или полностью «грязными» — модель выучит порядок
    # строк, а не отмывание. Поэтому вкатываем illicit порциями внутрь потока
    # и перемешиваем каждую порцию.
    # Сначала прикидываем размеры порций обычных транзакций...
    sizes, remaining = [], n_normal
    while remaining > 0:
        n = min(chunk, remaining)
        sizes.append(n)
        remaining -= n
    n_chunks = len(sizes)
    illicit = illicit.sample(frac=1.0, random_state=seed)      # сам блок тоже мешаем

    # ...и распределяем «грязные» ПРОПОРЦИОНАЛЬНО размеру порции. Если делить
    # поровну, последняя (неполная) порция окажется вдвое «грязнее» остальных,
    # и в конце файла снова появится сгусток отмываний.
    shares = np.round(np.array(sizes, dtype=float) / n_normal * len(illicit)).astype(int)
    shares[-1] = len(illicit) - shares[:-1].sum()             # чтобы сумма сошлась точно
    offs = np.concatenate(([0], np.cumsum(shares)))

    first = True
    written = 0
    for i in range(n_chunks):
        n = min(chunk, n_normal - written)
        if n <= 0:
            break
        part = _codes_to_frame(_normal_chunk(rng, n))
        ill_part = _codes_to_frame(illicit.iloc[offs[i]:offs[i + 1]])
        # перемешиваем порцию: внутри неё «грязные» строки распределены случайно
        combined = pd.concat([part, ill_part], ignore_index=True)
        combined = combined.sample(frac=1.0, random_state=seed + i)
        combined.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False
        written += n
        if verbose:
            print(f"  ... записано транзакций: {written + len(ill_part):,} "
                  f"/ {n_rows:,}", end="\r")
        del part, ill_part, combined
        gc.collect()

    if verbose:
        print(f"\n[synthetic] готово: {out_path} ({out_path.stat().st_size / 1e6:,.0f} MB)")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор SAML-D-подобного датасета")
    parser.add_argument("--n-rows", type=int, default=N_TOTAL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    generate(n_rows=args.n_rows, seed=args.seed, out_path=args.out)

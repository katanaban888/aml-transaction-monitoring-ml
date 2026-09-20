"""
src/features.py
===============
Признаки для детекции отмывания: транзакционные, поведенческие, velocity, парные.

ГЛАВНОЕ ПРАВИЛО ЭТОГО МОДУЛЯ (и всего проекта)
----------------------------------------------
Ни один признак не смеет заглядывать в будущее. Отсюда два механизма:

1. **FIT / APPLY.** Профили счетов (средняя сумма, число контрагентов, степень
   в графе, статистика пары) считаются ТОЛЬКО на train-периоде и затем
   «приклеиваются» к val/test как есть. Если посчитать профиль по всему
   датасету, модель в train узнает, сколько денег счёт прокинет в будущем —
   метрики взлетят, а в продакшене всё развалится.

2. **Скользящие окна только назад.** Velocity-признаки считаются по операциям,
   которые уже произошли к моменту текущей транзакции (окно «последние 1/24/168
   часов»), а не по всему дню целиком.

БИЗНЕС-ЛОГИКА ПРИЗНАКОВ (почему именно такие)
---------------------------------------------
* **Сумма и её отношение к порогу** — стадия placement: structuring живёт
  в полосе 90-100% от $10 000.
* **Сумма относительно обычного поведения счёта** — behavioural profiling:
  счёт, который всегда платил по $500 и вдруг отправил $20 000, подозрителен
  независимо от абсолютной суммы.
* **Velocity (частота и объём за окно)** — smurfing/structuring: серия
  однотипных платежей за короткое время.
* **Число контрагентов и степени в графе** — layering: fan-in/fan-out,
  транзитные счета, циклы.
* **Новизна пары / повторяемость пары** — «накатанная дорожка» между двумя
  счетами отличается от случайного переводa.
"""

from __future__ import annotations

import gc

import gc

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import (
    COL_AMOUNT, COL_PAY_CUR, COL_REC_CUR, COL_SENDER, COL_RECEIVER,
    COL_SENDER_LOC, COL_RECEIVER_LOC, COL_PAY_TYPE, COL_TARGET,
    STRUCTURING_THRESHOLD, FEATURES_DIR,
)

# Сдвиг для упаковки (счёт, время) в один int64 — см. _rolling_window_stats.
_KEY_SHIFT = np.int64(1) << 32


# ---------------------------------------------------------------------------
# 1. ТРАНЗАКЦИОННЫЕ ПРИЗНАКИ (считаются по одной строке, утечки нет по定义)
# ---------------------------------------------------------------------------
def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Признаки, которые видны из самой транзакции, без оглядки на историю счёта.

    amount_log — суммы распределены логнормально, моделям так удобнее.
    amount_near_threshold — 90-100% от порога $10 000 ( structuring ).
    amount_above_threshold — выше порога: попадает под обязательную отчётность,
        поэтому мошенники его избегают, а «честные» крупные платежи там есть.
    is_round_100 — «круглые» суммы: перевод «сам себе» часто делают 50 000,
        а чек в магазине — 1 234.56.
    """
    df = df.copy()
    amt = df[COL_AMOUNT].astype("float64")

    df["amount_log"] = np.log1p(amt).astype("float32")
    df["amount_near_threshold"] = (
        (amt >= STRUCTURING_THRESHOLD * 0.90) & (amt < STRUCTURING_THRESHOLD)
    ).astype("int8")
    df["amount_above_threshold"] = (amt >= STRUCTURING_THRESHOLD).astype("int8")
    df["is_round_100"] = (np.round(amt, 2) % 100 == 0).astype("int8")
    df["is_round_1000"] = (np.round(amt, 2) % 1000 == 0).astype("int8")

    # География и валюты: сравнение по КОДАМ категорий (без выделения памяти
    # под миллионы строк — см. categories_equal в src/eda_utils.py)
    from src.eda_utils import categories_equal
    df["is_cross_border"] = (~categories_equal(df[COL_SENDER_LOC], df[COL_RECEIVER_LOC])).astype("int8")
    df["is_currency_mismatch"] = (~categories_equal(df[COL_PAY_CUR], df[COL_REC_CUR])).astype("int8")

    # Время: ночь и выходные — когда живой контроль слабее всего
    if "hour" in df.columns:
        df["hour"] = df["hour"].astype("float32")
        df["is_night"] = df["hour"].between(0, 5).astype("int8")
    if "day_of_week" in df.columns:
        df["day_of_week"] = df["day_of_week"].astype("int8")
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    # Коды категориальных колонок — модели удобнее работать с числами
    for col, name in [(COL_PAY_TYPE, "payment_type_code"),
                      (COL_PAY_CUR, "pay_currency_code"),
                      (COL_REC_CUR, "rec_currency_code"),
                      (COL_SENDER_LOC, "sender_loc_code"),
                      (COL_RECEIVER_LOC, "receiver_loc_code")]:
        if col in df.columns and isinstance(df[col].dtype, pd.CategoricalDtype):
            df[name] = df[col].cat.codes.astype("int16")

    return df


def fit_amount_percentiles(df_fit: pd.DataFrame, n_bins: int = 1000) -> dict[str, np.ndarray]:
    """
    Строит «линейку» процентилей суммы внутри каждой валюты по train-периоду.

    Зачем: 9 000 долларов и 9 000 рупий — это разные деньги. Процентиль внутри
    валюты приводит всё к одной шкале: «насколько крупная операция для своей
    валюты». Линейка строится на train и применяется к val/test.
    """
    grid = {}
    for cur, sub in df_fit.groupby(COL_PAY_CUR, observed=True):
        q = np.quantile(sub[COL_AMOUNT].to_numpy(dtype="float64"),
                        np.linspace(0, 1, n_bins + 1)[1:-1])
        grid[str(cur)] = np.unique(q)
    return grid


def add_amount_percentile(df: pd.DataFrame, grid: dict[str, np.ndarray]) -> pd.DataFrame:
    """Приклеивает процентиль суммы внутри валюты по заранее построенной линейке."""
    df = df.copy()
    amt = df[COL_AMOUNT].to_numpy(dtype="float64")
    pct = np.full(len(df), np.nan, dtype="float32")

    cur_codes = pd.Categorical(df[COL_PAY_CUR])
    for code, cur in enumerate(cur_codes.categories):
        g = grid.get(str(cur))
        if g is None or len(g) == 0:
            continue
        mask = cur_codes.codes == code
        if not mask.any():
            continue
        pct[mask] = np.searchsorted(g, amt[mask], side="left") / len(g) * 100.0

    df["amount_pct_in_currency"] = pct
    return df


# ---------------------------------------------------------------------------
# 2. ПОВЕДЕНЧЕСКИЕ ПРОФИЛИ СЧЕТОВ (FIT на train -> APPLY везде)
# ---------------------------------------------------------------------------
def fit_account_profiles(df_fit: pd.DataFrame) -> pd.DataFrame:
    """
    Считает «нормальное поведение» каждого счёта по train-периоду:
    сколько операций, на какие суммы, с сколькими контрагентами, как долго живёт.

    Возвращает таблицу, где строка = счёт (индекс из названий счетов).
    """
    by_sender = df_fit.groupby(COL_SENDER, observed=True)
    prof = pd.DataFrame({
        "sender_txn_count": by_sender.size(),
        "sender_median_amount": by_sender[COL_AMOUNT].median(),
        "sender_mean_amount": by_sender[COL_AMOUNT].mean(),
        "sender_std_amount": by_sender[COL_AMOUNT].std(),
        "sender_max_amount": by_sender[COL_AMOUNT].max(),
        "sender_total_amount": by_sender[COL_AMOUNT].sum(),
        "sender_n_receivers": by_sender[COL_RECEIVER].nunique(),
        "sender_active_days": by_sender["date"].nunique(),
    })

    by_receiver = df_fit.groupby(COL_RECEIVER, observed=True)
    prof = prof.join(pd.DataFrame({
        "receiver_txn_count": by_receiver.size(),
        "receiver_median_amount": by_receiver[COL_AMOUNT].median(),
        "receiver_mean_amount": by_receiver[COL_AMOUNT].mean(),
        "receiver_n_senders": by_receiver[COL_SENDER].nunique(),
        "receiver_total_amount": by_receiver[COL_AMOUNT].sum(),
    }), how="outer")

    # Сколько операций в день счёт делает в среднем (в дни, когда он активен)
    prof["sender_txn_per_active_day"] = (
        prof["sender_txn_count"] / prof["sender_active_days"].replace(0, np.nan)
    )
    # Доля «крупных» операций счёта: хабы-прокладки живут на крупных суммах
    prof["sender_share_above_threshold"] = (
        prof["sender_max_amount"] / prof["sender_median_amount"].replace(0, np.nan)
    )
    # Приводим к float64 всё, что можно. Даты (если появятся) не трогаем:
    # они не должны попадать в признаки — модель по ним «запомнит» период.
    num_cols = prof.select_dtypes(include=["number"]).columns
    prof[num_cols] = prof[num_cols].astype("float64")
    return prof


def _merge_profile(df: pd.DataFrame, prof: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """
    Приклеивает профили к транзакциям.
    Делается через merge по названию счёта: так мы не создаём миллионы строковых
    значений (как это было бы при astype(str).map(...)).
    """
    prof = prof.reset_index()
    # имя колонки-индекса после reset_index — это название счёта
    prof = prof.rename(columns={prof.columns[0]: key_col})
    # float64 не нужен: точность профиля (медиана, среднее) важна до копеек,
    # а float32 экономит половину памяти на матрице из 9.5 млн строк.
    for c in prof.columns:
        if pd.api.types.is_float_dtype(prof[c]):
            prof[c] = prof[c].astype("float32")
    n_before = len(df)
    out = df.merge(prof, on=key_col, how="left", validate="many_to_one")
    assert len(out) == n_before, "merge изменил число строк — проверь ключи"
    return out


def add_behavioural_features(df: pd.DataFrame, prof: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет поведенческие признаки: как текущая транзакция выглядит на фоне
    обычного поведения её отправителя и получателя.

    Ключевые (и самые сильные) признаки:
        amount_to_sender_median   — во сколько раз сумма больше обычной для счёта
        amount_zscore_sender      — то же в единицах стандартного отклонения
        sender_txn_per_active_day — «температура» счёта
    """
    # ВАЖНО: делим профиль на «часть отправителя» и «часть получателя».
    # Если приклеить одну и ту же таблицу дважды (по Sender и по Receiver),
    # появятся одинаковые имена колонок и pandas разведёт их в _x / _y —
    # половина признаков «исчезнет», а ошибка всплывёт гораздо позже.
    sender_cols = [c for c in prof.columns if c.startswith("sender_")]
    receiver_cols = [c for c in prof.columns if c.startswith("receiver_")]

    df = _merge_profile(df, prof[sender_cols], COL_SENDER)
    df = _merge_profile(df, prof[receiver_cols], COL_RECEIVER)

    amt = df[COL_AMOUNT].astype("float64")

    # Знаменатели вида 0 заменяем на NaN: деление даёт NaN, бустинг его понимает.
    med_s = df["sender_median_amount"].replace(0, np.nan)
    med_r = df["receiver_median_amount"].replace(0, np.nan)
    std_s = df["sender_std_amount"].replace(0, np.nan)

    df["amount_to_sender_median"] = (amt / med_s).astype("float32")
    df["amount_to_receiver_median"] = (amt / med_r).astype("float32")
    df["amount_zscore_sender"] = ((amt - df["sender_mean_amount"]) / std_s).astype("float32")
    df["amount_to_sender_max"] = (amt / df["sender_max_amount"].replace(0, np.nan)).astype("float32")

    return df


# ---------------------------------------------------------------------------
# 3. VELOCITY: признаки скользящего окна (считаются ТОЛЬКО по прошлому)
# ---------------------------------------------------------------------------
def _rolling_window_stats(codes: np.ndarray, ts: np.ndarray,
                          values: dict[str, np.ndarray],
                          windows_hours: tuple[int, ...]) -> dict[str, np.ndarray]:
    """
    Для каждой строки считает за окно «последние W часов» внутри группы (счёта):
        count  — сколько операций было в окне, ВКЛЮЧАЯ текущую
        sum_*  — сумма значений за окно

    КАК ЭТО РАБОТАЕТ (важно понять, иначе код выглядит магией):
    1. Сортируем строки по (счёт, время).
    2. Упаковываем пару (счёт, время) в ОДНО число int64:
           key = код_счёта * 2^32 + время_в_секундах
       Тогда сортировка по key = сортировка сначала по счёту, потом по времени.
    3. Для окна W ищем позицию элемента с key = key_текущий − W*3600
       обычным np.searchsorted по всему массиву: раз ключи упорядочены,
       позиция гарантированно попадает внутрь блока своего счёта.
       Это даёт векторный расчёт БЕЗ циклов по 120 тыс. счетов.
    4. Суммы берём из заранее посчитанной кумулятивной суммы: sum = cum[i] − cum[left].

    Итог: 9.5 млн строк × 3 окна считаются за секунды вместо минут.
    """
    n = len(codes)

    # ШАГ 1. Сортировка по (счёт, время). np.lexsort по двум int-массивам —
    # самый быстрый способ получить нужный порядок без цикла по 120 тыс. счетов.
    order = np.lexsort((ts, codes))
    codes_s = codes[order]
    ts_s = ts[order]
    # ШАГ 2. Упаковка (счёт, время) в одно int64: сортировка по key =
    # сортировка сначала по счёту, внутри счёта — по времени.
    key_s = codes_s.astype(np.int64) * _KEY_SHIFT + np.maximum(ts_s, 0)
    del codes_s, ts_s
    gc.collect()

    # ШАГ 3. Кумулятивные суммы в том же отсортированном порядке:
    # сумма на отрезке [left, i] = csum[i + 1] - csum[left].
    csums = {}
    for name, arr in values.items():
        csums[name] = np.concatenate(([0.0], np.cumsum(arr[order])))
    # Обратная перестановка: unsort[i] = позиция i-й строки ИСХОДНОГО порядка
    # в отсортированном массиве. Строкой ниже мы возвращаем всё «как было».
    unsort = np.empty(n, dtype=np.int64)
    unsort[order] = np.arange(n, dtype=np.int64)
    del order
    gc.collect()

    out: dict[str, np.ndarray] = {}
    for w in windows_hours:
        secs = np.int64(int(w) * 3600)
        # searchsorted ищет левую границу окна по ВСЕМУ массиву сразу: раз ключи
        # упорядочены по (счёт, время), позиция гарантированно попадает внутрь
        # блока своего счёта. Никаких циклов по счетам.
        # side="right" — левая граница окна СТРОГАЯ: интервал (t - W, t].
        # (Операция ровно W часов назад уже не входит — так же считает и
        # проверка брутфорсом в тетрадке.)
        left = np.searchsorted(key_s, key_s - secs, side="right")
        # Позиция ПОСЛЕДНЕЙ строки с тем же (счёт, секунда). Нужна из-за
        # «двойников»: если у счёта две операции в одну и ту же секунду, обе
        # должны видеть друг друга в окне (иначе счётчик у них differs на 1).
        own = np.searchsorted(key_s, key_s, side="right") - 1
        # +1 — учитываем и САМУ текущую операцию: в правилах мониторинга счётчик
        # «сколько операций за сутки» всегда включает текущую транзакцию.
        out[f"_count_{w}"] = (own - left + 1).astype("float32")
        for name in values:
            out[f"_sum_{name}_{w}"] = (csums[name][own + 1] - csums[name][left]).astype("float32")
        del left
        gc.collect()

    out["_unsort"] = unsort          # чтобы вернуть строки в исходный порядок
    return out


def compute_velocity_features(df: pd.DataFrame,
                              windows_hours: tuple[int, ...] = (1, 6, 24, 168)) -> pd.DataFrame:
    """
    Velocity-признаки ОТПРАВИТЕЛЯ: сколько операций и на какую сумму он сделал
    за последний час / 6 часов / сутки / неделю.

    Отдельно считаем **число операций, прижатых к порогу $10 000**, за сутки и
    неделю: это прямое численное выражение типологии structuring
    («несколько платежей чуть ниже порога за короткое время»).

    Возвращает ТОЛЬКО velocity-колонки отдельным фреймом (float32), в исходном
    порядке строк.

    ПОЧЕМУ ОТДЕЛЬНЫМ ФРЕЙМОМ (а не df.copy() + новые колонки): на 9.5 млн строк
    копия исходного датафрейма — это лишние сотни мегабайт, из-за которых
    сборка падала по памяти. Здесь мы копируем только нужные три колонки.
    """
    # Три «сырых» массива — всё, что нужно для окон. near = флаг «прижато к
    # порогу structuring» (float32: кумулятивная сумма счётчиков точна до 16 млн).
    codes = df[COL_SENDER].cat.codes.to_numpy()
    codes = np.where(codes < 0, codes.max() + 1, codes).astype(np.int64)
    ts = df["txn_ts"].to_numpy().astype("datetime64[s]").astype(np.int64)
    amt = df[COL_AMOUNT].to_numpy(dtype="float64")
    near = ((amt >= STRUCTURING_THRESHOLD * 0.90) & (amt < STRUCTURING_THRESHOLD)).astype("float32")

    stats = _rolling_window_stats(codes, ts, {"amount": amt, "near": near}, windows_hours)
    unsort = stats.pop("_unsort")
    del codes, ts, amt, near
    gc.collect()

    vel = pd.DataFrame(index=pd.RangeIndex(len(df)))
    for w in windows_hours:
        vel[f"vel_count_{w}h"] = stats[f"_count_{w}"][unsort]
        vel[f"vel_amount_sum_{w}h"] = stats[f"_sum_amount_{w}"][unsort]
    for w in (24, 168):
        if f"_sum_near_{w}" in stats:
            vel[f"vel_near_threshold_count_{w}h"] = stats[f"_sum_near_{w}"][unsort]
    del stats, unsort
    gc.collect()
    return vel


def add_velocity_features(df: pd.DataFrame,
                          windows_hours: tuple[int, ...] = (1, 6, 24, 168)) -> pd.DataFrame:
    """Velocity + отношение оборота за сутки к обычной сумме счёта."""
    vel = compute_velocity_features(df, windows_hours)
    df = df.copy()
    for c in vel.columns:
        df[c] = vel[c].to_numpy()
    # Какой долей от обычной суммы счёта является оборот за сутки.
    # Резкий рост = счёт «проснулся»: типичный behavioural change.
    if "sender_median_amount" in df.columns:
        med = df["sender_median_amount"].replace(0, np.nan)
        df["vel_amount_24h_to_median"] = (df["vel_amount_sum_24h"] / med).astype("float32")
    return df


# ---------------------------------------------------------------------------
# 4. ПАРНЫЕ ПРИЗНАКИ (FIT на train -> APPLY везде)
# ---------------------------------------------------------------------------
def fit_pair_stats(df_fit: pd.DataFrame) -> pd.DataFrame:
    """
    Статистика пары «отправитель -> получатель» по train-периоду:
    сколько переводов было, на какую сумму, когда последний раз.
    """
    g = df_fit.groupby([COL_SENDER, COL_RECEIVER], observed=True)
    pair = pd.DataFrame({
        "pair_count": g.size(),
        "pair_amount_sum": g[COL_AMOUNT].sum(),
        "pair_amount_mean": g[COL_AMOUNT].mean(),
        "pair_last_ts": g["txn_ts"].max(),
    })
    return pair.reset_index()


def add_pair_features(df: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    """
    Признаки «накатанной дорожки» между двумя счетами.

    is_new_pair = 1, если эта пара НИ РАЗУ не встречалась в train-периоде.
    В мониторинге новая пара на крупную сумму — отдельный красный флаг.
    """
    out = df.merge(pair, on=[COL_SENDER, COL_RECEIVER], how="left")
    assert len(out) == len(df)
    out["is_new_pair"] = out["pair_count"].isna().astype("int8")
    out["pair_count"] = out["pair_count"].fillna(0).astype("float32")
    out["pair_amount_sum"] = out["pair_amount_sum"].fillna(0).astype("float32")
    out["pair_amount_mean"] = out["pair_amount_mean"].fillna(0).astype("float32")
    out = out.drop(columns=["pair_last_ts"])
    return out


# ---------------------------------------------------------------------------
# 5. СБОРКА МАТРИЦЫ ПРИЗНАКОВ
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    # транзакционные
    "amount_log", "amount_pct_in_currency",
    "amount_near_threshold", "amount_above_threshold",
    "is_round_100", "is_round_1000",
    "is_cross_border", "is_currency_mismatch",
    "hour", "is_night", "day_of_week", "is_weekend",
    "payment_type_code", "pay_currency_code", "rec_currency_code",
    "sender_loc_code", "receiver_loc_code",
    # поведенческие (профиль отправителя и получателя из train)
    "sender_txn_count", "sender_median_amount", "sender_mean_amount",
    "sender_std_amount", "sender_max_amount", "sender_n_receivers",
    "sender_active_days", "sender_txn_per_active_day", "sender_share_above_threshold",
    "receiver_txn_count", "receiver_median_amount", "receiver_n_senders",
    "amount_to_sender_median", "amount_to_receiver_median",
    "amount_zscore_sender", "amount_to_sender_max",
    # velocity
    "vel_count_1h", "vel_count_6h", "vel_count_24h", "vel_count_168h",
    "vel_amount_sum_1h", "vel_amount_sum_6h", "vel_amount_sum_24h", "vel_amount_sum_168h",
    "vel_near_threshold_count_24h", "vel_near_threshold_count_168h",
    "vel_amount_24h_to_median",
    # парные
    "pair_count", "pair_amount_sum", "pair_amount_mean", "is_new_pair",
]

# Сюда graph_features.py дописывает свои колонки
GRAPH_FEATURE_COLUMNS = [
    "g_out_degree", "g_in_degree", "g_total_degree",
    # g_degree_ratio = во сколько раз входящих связей больше исходящих (или наоборот):
    # «транзитный» счёт, который только собирает и тут же рассылает деньги.
    "g_degree_ratio",
    "g_out_amount_sum", "g_in_amount_sum",
    "g_pagerank", "g_in_cycle", "g_core_component_size", "g_component_size",
    "g_nx_scc_size", "g_nx_component_size", "g_in_repeat_cycle",
    "g_is_mutual_pair", "g_receiver_in_cycle",
]


def build_feature_matrix(
    df: pd.DataFrame,
    train_mask: pd.Series,
    with_graph: bool = True,
    graph_max_edges: int = 400_000,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Полный цикл построения признаков:

        FIT на train (профили счетов, пары, процентили, граф)
             -> APPLY ко всем строкам (train + val + test)

    Возвращает (X, y, meta), где meta содержит обученные профили — их нужно
    сохранить, чтобы в продакшене считать признаки по той же линейке.
    """
    # Маску сразу превращаем в numpy-массив: после merge индекс df сбрасывается,
    # и булева Series с «дырявым» индексом перестанет совпадать со строками.
    train_mask_np = np.asarray(train_mask)
    df_train = df[train_mask_np]

    if verbose:
        print(f"[features] FIT на train: {len(df_train):,} строк | "
              f"APPLY на все: {len(df):,} строк")

    # 1. Транзакционные
    df = add_transaction_features(df)
    grid = fit_amount_percentiles(df_train)
    df = add_amount_percentile(df, grid)

    # 2. Поведенческие профили
    prof = fit_account_profiles(df_train)
    df = add_behavioural_features(df, prof)

    # 3. Velocity
    df = add_velocity_features(df)

    # 4. Пары
    pair = fit_pair_stats(df_train)
    df = add_pair_features(df, pair)

    # 5. Граф
    meta = {"amount_percentile_grid": grid, "account_profiles": prof,
            "pair_stats": pair, "graph": None}
    if with_graph:
        from src.graph_features import fit_graph_features
        df, graph_meta = fit_graph_features(df, train_mask_np, max_edges=graph_max_edges,
                                            verbose=verbose)
        meta["graph"] = graph_meta

    cols = [c for c in FEATURE_COLUMNS + GRAPH_FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing and verbose:
        print(f"[features] ВНИМАНИЕ: нет колонок {missing}")

    X = df[cols].copy()
    # Все признаки — float32: экономим память и ускоряем бустинг
    for c in X.columns:
        if X[c].dtype != "float32":
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    y = df[COL_TARGET].astype("int8")

    if verbose:
        print(f"[features] Матрица: {X.shape[0]:,} строк x {X.shape[1]} признаков "
              f"| память {X.memory_usage(deep=True).sum() / 1e6:,.0f} MB")
    return X, y, meta

def build_and_save_features(
    df: pd.DataFrame,
    train_mask: np.ndarray | pd.Series,
    period: pd.Series | None = None,
    out_path: str | Path | None = None,
    chunk_size: int = 1_000_000,
    graph_max_edges: int = 400_000,
    with_graph: bool = True,
    verbose: bool = True,
) -> dict:
    """
    ПОТОКОВАЯ сборка матрицы признаков для больших данных (9.5 млн строк).

    Почему не «всё в памяти»: матрица 9.5 млн x 60 признаков — это ~2.3 ГБ,
    плюс сам датафрейм с промежуточными колонками. На ноутбуке такого может
    просто не быть. Поэтому:

      * velocity считается ОДИН РАЗ на всех данных (ей нужна история счёта,
        и по частям её считать нельзя — на границах чанков окна «сломаются»);
      * профили счетов, пары и граф FITятся на train (это маленькие таблицы);
      * всё остальное собирается чанками, и каждый чанк СРАЗУ дописывается
        в parquet (pyarrow умеет писать row groups последовательно).

    Возвращает meta с обученными профилями — их надо сохранить: в продакшене
    признаки считаются по той же линейке.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    train_np = np.asarray(train_mask)
    if period is None:
        period = pd.Series(np.where(train_np, "train", "other"), index=df.index)
    period_np = np.asarray(period)

    out_path = Path(out_path) if out_path else FEATURES_DIR / "features_full.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_train = df[train_np]

    if verbose:
        print(f"[features] FIT на train: {len(df_train):,} строк | "
              f"пишем в {out_path.name} чанками по {chunk_size:,}")

    def _rss() -> float:
        return int(open("/proc/self/statm").read().split()[1]) * 4096 / 1e6

    # 1. Velocity — один раз на всех данных (окнам нужна полная история)
    vel = compute_velocity_features(df)
    if verbose:
        print(f"[features] RSS после velocity: {_rss():,.0f} MB")

    # 2. FIT на train
    grid = fit_amount_percentiles(df_train)
    prof = fit_account_profiles(df_train)
    pair = fit_pair_stats(df_train)
    accounts = mutual = None
    gmeta = None
    if with_graph:
        from src.graph_features import fit_graph_tables
        accounts, mutual, gmeta = fit_graph_tables(
            df_train, max_edges=graph_max_edges, verbose=verbose)
        if verbose:
            print(f"[features] RSS после FIT графа: {_rss():,.0f} MB")

    # Train-датафрейм больше не нужен: все линейки уже «запомнены» в профилях.
    # Освобождаем ~350 МБ перед тем, как начать гонять чанки.
    del df_train
    gc.collect()

    # 3. Чанки: собрать -> дописать в parquet -> выбросить из памяти
    writer = None
    n = len(df)
    cols = None
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = df.iloc[start:stop].copy()

        chunk = add_transaction_features(chunk)
        chunk = add_amount_percentile(chunk, grid)
        chunk = add_behavioural_features(chunk, prof)
        chunk = add_pair_features(chunk, pair)
        if with_graph:
            from src.graph_features import merge_graph_tables
            chunk = merge_graph_tables(chunk, accounts, mutual)

        # velocity приклеиваем из общего массива (окна считаны по всей истории)
        for c in vel.columns:
            chunk[c] = vel[c].to_numpy()[start:stop]
        med = chunk["sender_median_amount"].replace(0, np.nan)
        chunk["vel_amount_24h_to_median"] = (
            chunk["vel_amount_sum_24h"] / med).astype("float32")

        cols = [c for c in FEATURE_COLUMNS + GRAPH_FEATURE_COLUMNS if c in chunk.columns]
        X = pd.DataFrame({c: pd.to_numeric(chunk[c], errors="coerce").astype("float32")
                          for c in cols})
        X[COL_TARGET] = chunk[COL_TARGET].to_numpy()
        X["period"] = period_np[start:stop]

        table = pa.Table.from_pandas(X, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

        if verbose:
            print(f"  ... обработано {stop:,} / {n:,} строк | RSS {_rss():,.0f} MB", end="\n")
        del chunk, X, table
        gc.collect()

    if writer is not None:
        writer.close()

    if verbose:
        size_mb = out_path.stat().st_size / 1e6
        print(f"\n[features] ГОТОВО: {out_path} ({size_mb:,.0f} MB), "
              f"{len(cols)} признаков, {n:,} строк")

    return {"amount_percentile_grid": grid, "account_profiles": prof,
            "pair_stats": pair, "graph": gmeta, "feature_columns": cols,
            "out_path": out_path}


def save_features(X: pd.DataFrame, y: pd.Series, name: str = "features",
                  extra: pd.DataFrame | None = None, verbose: bool = True) -> Path:
    """Сохраняет матрицу признаков в data/features/ (parquet или pickle)."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.concat([X, y.rename(COL_TARGET)], axis=1)
    if extra is not None:
        out = pd.concat([out, extra], axis=1)
    path = FEATURES_DIR / f"{name}.parquet"
    try:
        out.to_parquet(path, index=False)
    except Exception:
        path = FEATURES_DIR / f"{name}.pkl"
        out.to_pickle(path)
    if verbose:
        print(f"[features] Сохранено: {path} ({path.stat().st_size / 1e6:,.0f} MB)")
    return path

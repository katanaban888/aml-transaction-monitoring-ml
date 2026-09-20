"""
src/data_loader.py
==================
Загрузка, типизация и первичная подготовка датасета SAML-D
(Synthetic Anti-Money Laundering Dataset).

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ, А НЕ КОД ПРЯМО В ТЕТРАДКЕ?
---------------------------------------------------
1. 9.5 млн строк читать с диска долго (1-3 минуты). Если держать всё в ноутбуке,
   каждый рестарт ядра = снова ждать. Здесь мы один раз читаем CSV, оптимизируем
   типы и сохраняем в быстрый бинарный формат (parquet). Дальше всё читается за секунды.
2. Один и тот же код загрузки нужен и в EDA, и в feature engineering, и в ML,
   и в Streamlit. Копипастить между ноутбуками = рассинхрон и баги.
3. На собеседовании это большой плюс: «у меня не ноутбук-портянка, а воспроизводимый
   пайплайн с кэшированием».

Ключевая идея AML-проекта, зашитая в этот модуль:
мы НЕ теряем ни одной строки и НЕ трогаем целевые метки. Всё, что здесь делается, —
это экономия памяти и наведение порядка в датах/времени.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. ПУТИ ПРОЕКТА
# ---------------------------------------------------------------------------
# __file__ = .../aml-transaction-monitoring-ml/src/data_loader.py
# parents[0] = src/, parents[1] = корень проекта.
# Такой подход работает независимо от того, откуда запущен код:
# из ноутбука, из консоли или из Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"           # сырой CSV с Kaggle (в git не идёт)
PROCESSED_DIR = DATA_DIR / "processed"  # очищенные/подготовленные данные (в git не идёт)
FEATURES_DIR = DATA_DIR / "features"    # матрицы признаков (в git не идёт)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"   # сюда сохраняем все графики

# Создаём папки при импорте модуля — чтобы не ловить FileNotFoundError потом.
for _dir in (RAW_DIR, PROCESSED_DIR, FEATURES_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

PREPARED_FILENAME = "saml_d_prepared"   # расширение подставится само (parquet/pickle)


# ---------------------------------------------------------------------------
# 2. КОНТРАКТ ДАННЫХ (схема датасета)
# ---------------------------------------------------------------------------
# Названия колонок SAML-D. Если Kaggle-версия отличается — правим только здесь.
COL_TIME = "Time"
COL_DATE = "Date"
COL_SENDER = "Sender_account"
COL_RECEIVER = "Receiver_account"
COL_AMOUNT = "Amount"
COL_PAY_CUR = "Payment_currency"
COL_REC_CUR = "Received_currency"
COL_SENDER_LOC = "Sender_bank_location"
COL_RECEIVER_LOC = "Receiver_bank_location"
COL_PAY_TYPE = "Payment_type"
COL_TARGET = "Is_laundering"
COL_LAUND_TYPE = "Laundering_type"

RAW_COLUMNS = [
    COL_TIME, COL_DATE, COL_SENDER, COL_RECEIVER, COL_AMOUNT,
    COL_PAY_CUR, COL_REC_CUR, COL_SENDER_LOC, COL_RECEIVER_LOC,
    COL_PAY_TYPE, COL_TARGET, COL_LAUND_TYPE,
]

# Колонки, которые повторяются миллионы раз (id счетов, валюты, страны, типы платежей).
# Тип category хранит каждое уникальное значение ОДИН раз и дальше ссылается на него
# числом. Экономия памяти обычно в 5-20 раз на таких колонках.
CATEGORICAL_COLUMNS = [
    COL_SENDER, COL_RECEIVER, COL_PAY_CUR, COL_REC_CUR,
    COL_SENDER_LOC, COL_RECEIVER_LOC, COL_PAY_TYPE, COL_LAUND_TYPE,
]

# Бизнес-константа AML: порог обязательного отчёта (Currency Transaction Report)
# в США — $10,000. Классическая схема placement'а — «структурирование» (smurfing):
# дробить крупную сумму на платежи ЧУТЬ НИЖЕ порога, чтобы не попасть под отчётность.
# Эту константу мы будем использовать и в EDA, и в правилах (Этап 3), и в признаках.
STRUCTURING_THRESHOLD = 10_000.0

# Насколько близко к порогу считаем «подозрительно близко» (90%..99.99% от порога).
STRUCTURING_BAND_LOW = 0.90
STRUCTURING_BAND_HIGH = 1.00


# ---------------------------------------------------------------------------
# 3. ПОИСК И ЧТЕНИЕ СЫРОГО ФАЙЛА
# ---------------------------------------------------------------------------
def find_raw_csv(verbose: bool = True) -> Path:
    """
    Находит CSV с данными в data/raw/.

    Зачем автоматически: файл с Kaggle может называться по-разному
    (SAML-D.csv, HI-Small_Trans.csv, ...). Переменная окружения SAML_D_CSV
    позволяет явно указать путь, если в папке лежит несколько файлов.
    """
    env_path = os.getenv("SAML_D_CSV")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Файл из переменной окружения SAML_D_CSV не найден: {path}")
        if verbose:
            print(f"[data_loader] Источник задан через SAML_D_CSV: {path}")
        return path

    candidates = sorted(RAW_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"В папке {RAW_DIR} нет ни одного .csv файла.\n"
            f"Положи туда датасет с Kaggle либо укажи путь:\n"
            f"    export SAML_D_CSV=/полный/путь/к/файлу.csv"
        )

    # Если файлов несколько — берём самый большой (простая и надёжная эвристика).
    path = max(candidates, key=lambda p: p.stat().st_size)
    if verbose:
        if len(candidates) > 1:
            print(f"[data_loader] Найдено {len(candidates)} CSV, беру самый большой: {path.name}")
        else:
            print(f"[data_loader] Найден файл: {path.name}")
    return path


def load_raw_csv(nrows: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Читает сырой CSV с правильными типами колонок.

    Почему dtype задаём вручную:
      * Object/str на 9.5 млн строк по id счетов = гигабайты RAM;
      * float64 для Amount не нужен — точность всё равно не важна для AML,
        а float32 экономит половину памяти;
      * если не указать типы, pandas будет «угадывать» и может превратить
        номер счёта в число, а дату — в строку.
    """
    path = find_raw_csv(verbose=verbose)

    dtype_map = {
        # КЛЮЧЕВОЙ МОМЕНТ: читаем повторяющиеся колонки СРАЗУ как category.
        # Если сначала прочитать как строки, а потом делать astype("category"),
        # в памяти одновременно живут и миллионы строк, и их копия в кодах —
        # на 9.5 млн строк это +1-1.5 ГБ и риск нехватки памяти.
        COL_SENDER: "category",
        COL_RECEIVER: "category",
        COL_PAY_CUR: "category",
        COL_REC_CUR: "category",
        COL_SENDER_LOC: "category",
        COL_RECEIVER_LOC: "category",
        COL_PAY_TYPE: "category",
        COL_LAUND_TYPE: "category",
        COL_AMOUNT: "float32",
        COL_TARGET: "int8",
        # Time и Date читаем как строки — разбирать будем отдельной функцией
        # (формат у версий датасета отличается).
        COL_TIME: "string",
        COL_DATE: "string",
    }

    if verbose:
        size_mb = path.stat().st_size / 1e6
        print(f"[data_loader] Читаю {path.name} ({size_mb:,.0f} MB){' (первые %d строк)' % nrows if nrows else ''} ...")

    df = pd.read_csv(path, dtype=dtype_map, nrows=nrows, low_memory=False)

    if verbose:
        print(f"[data_loader] Прочитано строк: {len(df):,}, колонок: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# 4. ОПТИМИЗАЦИЯ ТИПОВ И ПАМЯТИ
# ---------------------------------------------------------------------------
def optimize_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Переводит повторяющиеся строки в category.

    На 9.5 млн строк это разница между «не влезает в 16 ГБ» и «работает спокойно».
    ВАЖНО: category ускоряет groupby и value_counts, но замедляет строковые
    операции (.str.*) — поэтому строковые преобразования делаем ДО этой функции.
    """
    if verbose:
        before = memory_mb(df)

    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")

    if COL_TARGET in df.columns:
        df[COL_TARGET] = df[COL_TARGET].astype("int8")
    if COL_AMOUNT in df.columns:
        df[COL_AMOUNT] = df[COL_AMOUNT].astype("float32")

    if verbose:
        after = memory_mb(df)
        print(f"[data_loader] Память: {before:,.0f} MB -> {after:,.0f} MB "
              f"(-{(1 - after / before) * 100:.0f}%)")
    return df


def memory_mb(df: pd.DataFrame) -> float:
    """Размер датафрейма в мегабайтах (deep=True учитывает строки внутри object-колонок)."""
    return df.memory_usage(deep=True).sum() / 1e6


# ---------------------------------------------------------------------------
# 5. ДАТА И ВРЕМЯ
# ---------------------------------------------------------------------------
_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
    "%d-%m-%Y", "%d.%m.%Y", "%Y.%m.%d", "%d/%m/%y",
]


def _infer_date_format(sample: pd.Series) -> str | None:
    """
    Угадывает формат даты по 500 значениям.

    Зачем: pd.to_datetime без формата на 9.5 млн строк работает медленно
    и может неверно понять, что идёт первым — день или месяц (01/02/2022).
    Ошибка здесь = сдвинутая по времени аналитика velocity, поэтому проверяем явно.
    """
    sample = sample.dropna().astype(str).head(500)
    if sample.empty:
        return None
    best_fmt, best_score = None, -1.0
    for fmt in _DATE_FORMATS:
        parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        score = parsed.notna().mean()
        if score > best_score:
            best_fmt, best_score = fmt, score
        if score == 1.0:
            break
    return best_fmt if best_score > 0.8 else None


def add_time_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Разбирает Date/Time и добавляет календарные признаки:
        hour, minute, day_of_week, is_weekend, month, date, txn_ts

    Бизнес-смысл (Этап 1, блок «velocity»):
      * hour / is_weekend — отмывочные операции часто идут «не в рабочее время»
        и в выходные, когда живой мониторинг слабее;
      * txn_ts (полная метка времени) — нужен для скорости: сколько транзакций
        подряд сделал счёт и с каким интервалом (smurfing = серия платежей за минуты).
    """
    df = df.copy()

    # --- Date -------------------------------------------------------------
    if COL_DATE in df.columns:
        dates = df[COL_DATE]
        fmt = _infer_date_format(dates)
        if fmt:
            parsed = pd.to_datetime(dates, format=fmt, errors="coerce")
        else:  # формат не угадался — даём pandas разобрать самому
            parsed = pd.to_datetime(dates, errors="coerce", format="mixed", dayfirst=False)
            if verbose:
                print("[data_loader] Формат даты не распознан по списку, использован автопарсинг")
        n_bad = int(parsed.isna().sum())
        if n_bad and verbose:
            print(f"[data_loader] ВНИМАНИЕ: не удалось разобрать дату у {n_bad:,} строк")
        df[COL_DATE] = parsed
        df["date"] = parsed.dt.normalize()      # дата без времени
        df["month"] = parsed.dt.month.astype("int8")
        df["day_of_week"] = parsed.dt.dayofweek.astype("int8")  # 0 = понедельник
        df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
        df["day"] = parsed.dt.day.astype("int8")

    # --- Time -------------------------------------------------------------
    # В некоторых версиях SAML-D колонка Time — это строка "HH:MM:SS",
    # в других — уже число (часы). Обрабатываем оба случая.
    if COL_TIME in df.columns:
        if pd.api.types.is_numeric_dtype(df[COL_TIME]):
            hours = df[COL_TIME].astype("float32")
            minutes = np.zeros(len(df), dtype="int16")
        else:
            t = pd.to_datetime(df[COL_TIME].astype("string"), format="%H:%M:%S", errors="coerce")
            if t.isna().mean() > 0.5:  # формат другой — пробуем автопарсинг
                t = pd.to_datetime(df[COL_TIME].astype("string"), errors="coerce", format="mixed")
            hours = t.dt.hour.astype("float32")
            minutes = t.dt.minute.fillna(0).astype("int16")
            n_bad = int(t.isna().sum())
            if n_bad and verbose:
                print(f"[data_loader] ВНИМАНИЕ: не удалось разобрать время у {n_bad:,} строк")

        df["hour"] = hours
        df["minute"] = minutes
        # Дальше колонка Time нам не нужна: время уже разложено на hour/minute/txn_ts.
        # Удаляем её сразу: 9.5 млн Python-строк съедают ~600 МБ и больше ни на что
        # не влияют (все временные расчёты идут по txn_ts / hour / minute).
        df = df.drop(columns=[COL_TIME])
        # Ночное окно 00:00–05:59 — классический «красный флаг» в правилах мониторинга.
        df["is_night"] = df["hour"].between(0, 5).astype("int8")

        # Полная метка времени: дата + время. Нужна для inter-arrival (интервалов
        # между транзакциями одного счёта) — один из сильнейших AML-признаков.
        if "date" in df.columns:
            ts = df["date"] + pd.to_timedelta(df["hour"].fillna(0), unit="h") \
                            + pd.to_timedelta(df["minute"], unit="m")
            df["txn_ts"] = ts

    return df


# ---------------------------------------------------------------------------
# 6. КЭШ: СОХРАНЕНИЕ / ЗАГРУЗКА ПОДГОТОВЛЕННЫХ ДАННЫХ
# ---------------------------------------------------------------------------
def _engine_available() -> str:
    """Возвращает 'parquet', если установлен pyarrow/fastparquet, иначе 'pickle'."""
    try:
        import pyarrow  # noqa: F401
        return "parquet"
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return "parquet"
        except ImportError:
            return "pickle"


def save_table(df: pd.DataFrame, path: Path | str, verbose: bool = True) -> Path:
    """Сохраняет датафрейм. Формат определяется по расширению файла."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)
    if verbose:
        print(f"[data_loader] Сохранено: {path} ({path.stat().st_size / 1e6:,.0f} MB)")
    return path


def load_table(path: Path | str, verbose: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
    if verbose:
        print(f"[data_loader] Загружено из кэша: {path.name} ({len(df):,} строк, {memory_mb(df):,.0f} MB)")
    return df


def cache_path(extension: str | None = None) -> Path:
    ext = extension or ("parquet" if _engine_available() == "parquet" else "pkl")
    return PROCESSED_DIR / f"{PREPARED_FILENAME}.{ext}"


# ---------------------------------------------------------------------------
# 7. ГЛАВНАЯ ФУНКЦИЯ: ПОЛНЫЙ ЦИКЛ ЗАГРУЗКИ
# ---------------------------------------------------------------------------
def load_dataset(
    use_cache: bool = True,
    force_rebuild: bool = False,
    nrows: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Единая точка входа для всех ноутбуков:

        df = load_dataset()          # первый запуск: читает CSV ~1-3 мин и кэширует
        df = load_dataset()          # второй запуск: читает кэш ~5-15 сек

    Параметры
    ---------
    use_cache : bool
        Использовать ли сохранённую копию из data/processed/.
    force_rebuild : bool
        True = перечитать CSV заново и перезаписать кэш
        (нужно после изменения логики подготовки).
    nrows : int | None
        Сколько строк читать из CSV (для быстрых экспериментов).
        ВАЖНО: при nrows не None кэш НЕ пишется, чтобы не сохранить обрезанные данные.
    """
    path = cache_path()

    if use_cache and not force_rebuild and nrows is None and path.exists():
        return load_table(path, verbose=verbose)

    df = load_raw_csv(nrows=nrows, verbose=verbose)
    df = optimize_dtypes(df, verbose=verbose)
    df = add_time_features(df, verbose=verbose)

    if nrows is None and use_cache:
        save_table(df, path, verbose=verbose)
    elif verbose:
        print("[data_loader] Кэш не записан (читалась только часть строк)")
    return df


# ---------------------------------------------------------------------------
# 8. МЕЛКИЕ ПОМОЩНИКИ ДЛЯ ОТЧЁТОВ
# ---------------------------------------------------------------------------
def class_balance(y: pd.Series) -> dict:
    """
    Базовая статистика по целевой переменной.

    Почему это важно проговаривать вслух на собеседовании:
    при 0.1% positives accuracy = 99.9% означает «модель вообще ничего не нашла».
    Поэтому уже здесь фиксируем base rate и будем сравнивать с ним все lift'ы.
    """
    y = np.asarray(y)
    n = len(y)
    pos = int((y == 1).sum())
    return {
        "n": n,
        "positives": pos,
        "negatives": n - pos,
        "base_rate": pos / n if n else 0.0,
        "imbalance_ratio": (n - pos) / pos if pos else np.inf,
    }


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица с долей пропусков по колонкам (в EDA это первое, что смотрим)."""
    miss = df.isna().sum()
    out = pd.DataFrame({"n_missing": miss, "pct_missing": (miss / len(df) * 100).round(4)})
    return out[out["n_missing"] > 0].sort_values("n_missing", ascending=False)


if __name__ == "__main__":
    # Позволяет проверить модуль из консоли:  python src/data_loader.py
    df = load_dataset(nrows=200_000)
    print(df.dtypes)
    print(class_balance(df[COL_TARGET]))

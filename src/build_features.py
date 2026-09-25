"""
Этап 2, финальный шаг: потоковая сборка матрицы признаков на ВСЕХ 9.5 млн строк.

Запускается ОТДЕЛЬНЫМ ПРОЦЕССОМ:

    python -m src.build_features
    python -m src.build_features --chunk-size 1000000 --max-edges 300000
    python -m src.build_features --no-graph      # только «плоские» признаки

Почему отдельным процессом, а не ячейкой ноутбука (это важно понимать):
к моменту сборки ноутбук уже «потрогал» гигабайты данных. Python освобождает
память, но не всегда возвращает её операционной системе: в статистике процесса
выросший «пик» остаётся занятым. Свежий процесс стартует с чистого листа, и
тяжёлая сборка проходит без риска получить «Killed» от системы.

Что делает скрипт:
  1. читает ТОЛЬКО нужные колонки из parquet-кэша (зачем тащить лишние 200 МБ);
  2. считает разбиение по времени train / val / test;
  3. FITит все «линейки» (процентили, профили счетов, пары, граф) на train;
  4. чанками собирает признаки и дописывает их в data/features/features_full.parquet;
  5. сохраняет обученные линейки в data/features/ — они понадобятся в продакшене
     и в Streamlit-приложении (новые транзакции надо считать той же линейкой).
"""
from __future__ import annotations

import argparse
import gc
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import (  # noqa: E402
    COL_AMOUNT, COL_RECEIVER, COL_SENDER, COL_TARGET, COL_PAY_TYPE,
    FEATURES_DIR, cache_path, memory_mb,
)
from src.split import time_split  # noqa: E402
from src.features import build_and_save_features  # noqa: E402

# «Сырые» колонки, которых достаточно для сборки всех признаков.
# Всё остальное (Laundering_type и т.п.) в матрицу не идёт и память не тратит.
RAW_COLUMNS = [
    "txn_ts", "date", "hour", "minute", "day_of_week",
    COL_AMOUNT, COL_SENDER, COL_RECEIVER, COL_PAY_TYPE,
    "Payment_currency", "Received_currency",
    "Sender_bank_location", "Receiver_bank_location",
    COL_TARGET,
]


def cur_mb() -> float:
    """Текущий RSS процесса в МБ (а не накопленный максимум)."""
    return int(open("/proc/self/statm").read().split()[1]) * 4096 / 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description="Потоковая сборка матрицы признаков")
    ap.add_argument("--chunk-size", type=int, default=1_000_000,
                    help="сколько строк обрабатывать за один заход (по умолчанию 1 млн)")
    ap.add_argument("--max-edges", type=int, default=300_000,
                    help="ограничение на число рёбер в графе (PageRank, компоненты)")
    ap.add_argument("--no-graph", action="store_true", help="не считать граф-признаки")
    ap.add_argument("--out", type=str, default=None, help="путь к выходному parquet")
    ap.add_argument("--limit", type=int, default=None,
                    help="взять только первые N строк (для отладки)")
    args = ap.parse_args()

    path = cache_path()
    if not path.exists():
        print(f"[build_features] Нет кэша {path}. Сначала выполни notebooks/01_eda.ipynb")
        return 1

    print(f"[build_features] Читаю {path.name} (только нужные колонки) ...")
    # Читаем по row group'ам, а не весь файл сразу: при разовой конвертации
    # pyarrow -> pandas память на пике подскакивает вдвое, а нам каждый мегабайт
    # на счету (сборка и так требует ~2 ГБ).
    import pyarrow.parquet as pq
    _pf = pq.ParquetFile(path)
    _ng = _pf.metadata.num_row_groups
    df = pd.concat([_pf.read_row_group(i, columns=RAW_COLUMNS).to_pandas()
                    for i in range(_ng)], ignore_index=True)
    del _pf
    if args.limit:
        df = df.head(args.limit).copy()
    gc.collect()
    print(f"[build_features] Прочитано {len(df):,} строк | {memory_mb(df):,.0f} MB "
          f"| RSS {cur_mb():,.0f} MB")

    train_mask, val_mask, test_mask = time_split(df, verbose=True)
    period = pd.Series("train", index=df.index, dtype=object)
    period[val_mask] = "val"
    period[test_mask] = "test"

    out_path = Path(args.out) if args.out else FEATURES_DIR / "features_full.parquet"
    meta = build_and_save_features(
        df,
        train_mask=np.asarray(train_mask),
        period=np.asarray(period),
        out_path=out_path,
        chunk_size=args.chunk_size,
        graph_max_edges=args.max_edges,
        with_graph=not args.no_graph,
        verbose=True,
    )
    del df
    gc.collect()

    # ---- сохраняем обученные «линейки» (пригодятся для прода/дашборда) ----
    # Линейка процентилей — это словарь {валюта: массив}, сохраняем как wide-таблицу
    grid = meta["amount_percentile_grid"]
    pd.DataFrame({cur: pd.Series(vals) for cur, vals in grid.items()}).to_parquet(
        FEATURES_DIR / "fit_amount_percentiles.parquet")
    meta["account_profiles"].to_parquet(FEATURES_DIR / "fit_account_profiles.parquet")
    meta["pair_stats"].to_parquet(FEATURES_DIR / "fit_pair_stats.parquet")
    print(f"[build_features] Линейки сохранены в {FEATURES_DIR}")

    print(f"\n[build_features] ГОТОВО: {out_path} "
          f"({out_path.stat().st_size / 1e6:,.0f} MB)")
    print(f"[build_features] Пик памяти: "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:,.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

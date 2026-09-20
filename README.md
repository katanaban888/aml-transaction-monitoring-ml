# Intelligent AML Transaction Monitoring System using ML and Graph Analytics

End-to-end портфолио-проект по выявлению подозрительных транзакций (AML):
rule-based detection → feature engineering → граф-аналитика → ML (XGBoost/LightGBM)
→ SHAP-объяснения → Streamlit-дашборд для аналитика.

**Датасет:** SAML-D (Synthetic Anti-Money Laundering Dataset, Kaggle) — 9 504 852 транзакций,
12 колонок, доля отмывания `Is_laundering=1` — 0.1039% (9 873 строк), 17 типологий
(`Structuring` 1870, `Cash_Withdrawal` 1334, `Deposit-Send` 945, `Smurfing` 932, ...).

---

## Быстрый старт

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Положить CSV с Kaggle в data/raw/   (папка в git не идёт)
#    либо указать путь явно:  export SAML_D_CSV=/путь/к/SAML-D.csv

# 2. Запустить EDA
jupyter notebook notebooks/01_eda.ipynb

# 3. Проверить, что код жив (без реального датасета, ~7 секунд)
python tests/test_synthetic_pipeline.py
```

Данные читаются один раз (~45 c на 9.5 млн строк / 900 МБ) и кэшируются в
`data/processed/saml_d_prepared.parquet`; дальше всё открывается за секунды.

## Структура

```
data/            raw/ (Kaggle CSV, не в git) · processed/ (кэш) · features/
notebooks/       01_eda · 02_feature_engineering · 03_rule_based_baseline
                 04_ml_modeling · 05_evaluation_shap
src/             data_loader    — чтение, типы, парсинг дат, кэш parquet
                 eda_utils      — lift-таблицы, z-тест двух долей, сохранение графиков
                 split          — разбиение по времени + детектор утечек
                 synthetic_data — генератор SAML-D-подобных данных со всеми 17 типологиями
                 features       — транзакционные, поведенческие, velocity- и парные признаки
                 graph_features — степени, PageRank, циклы, компоненты связности
                 build_features — CLI: потоковая сборка матрицы на всех 9.5 млн строк
                 rules          — справочник правил «красных флагов» и их метрики
                 train · evaluate
streamlit_app/   app.py — дашборд аналитика
models/          обученные модели
reports/         figures/ · eda_findings.md · reference_run_synthetic/
data/features/   features_full.parquet (9.5 млн x 62) + обученные «линейки» FIT
tests/           test_synthetic_pipeline.py — дымовой тест конвейера
```

## Синтетический генератор и эталонный прогон

Реальные данные с Kaggle нельзя положить в git, из-за этого код невозможно
протестировать на другой машине или в CI. Решение: `src/synthetic_data.py`
генерирует датасет **той же схемы и того же масштаба** (9 504 852 строки,
9 873 отмывания, те же 17 типологий с теми же count'ами), причём каждая
типология воспроизводится по своей «физике»:

| Типология | Как моделируется |
|---|---|
| `Structuring` | серия платежей $8 700–9 950 (чуть ниже порога $10 000) |
| `Smurfing` | 8–16 мелких платежей за 1.5 часа, часто ночью |
| `Fan_In` / `Fan_Out` | звезда: 15–35 → 1 и 1 → 12–30 |
| `Layered_Fan_In/Out` | два слоя: через 3 счёта-«мула» |
| `Bipartite` / `Stacked Bipartite` | две группы, связанные «все со всеми» |
| `Cycle` | деньги идут по кругу A→B→C→…→A, сумма тает на ~3% за шаг |
| `Gather-Scatter` / `Scatter-Gather` | много→хаб→много и хаб→мулы→финал |
| `Single_large` / `Over-Invoicing` | одна огромная сумма / завышенный инвойс через границу |
| `Behavioural_Change_1/2` | резкий скачок суммы (×15–40) или смена контрагентов |

Эталонный прогон EDA на этих данных лежит в `reports/reference_run_synthetic/`
(16 графиков, `eda_findings.md`, полный лог `eda_reference_output.txt`).
**Цифры там синтетические** — это ориентир «как должен выглядеть результат»,
а не реальные показатели SAML-D.

## Профиль ресурсов (замерено на 9.5 млн строк)

| Этап | Время | Память |
|---|---|---|
| Чтение CSV (900 МБ) + типы + парсинг дат + запись кэша | ~45 c | пик ~3.4 ГБ, сам фрейм 0.53 ГБ |
| Прогон всех 49 ячеек EDA | ~4 мин | пик ~3.35 ГБ |
| Сборка матрицы признаков (`python -m src.build_features`) | ~90 с | пик ~2.8 ГБ |
| Прогон тетрадки Этапа 2 (включая сборку) | ~4 мин | пик ~2.2 ГБ |

Матрица 9.5 млн x 62 признака занимает 932 МБ в parquet — в память она целиком не
загружается: признаки считаются чанками по 500 тыс. строк и сразу дописываются
в файл (`src/build_features.py`). На машине с 16 ГБ всё проходит свободно; при
8 ГБ — тоже. Если памяти меньше 4 ГБ, запускайте EDA на сэмпле:
`df = load_dataset(nrows=2_000_000)`.

## Как воспроизвести Этап 2

```bash
source venv/bin/activate
python -m src.build_features                 # матрица признаков, ~90 с
#   или с другими параметрами:
python -m src.build_features --chunk-size 250000 --max-edges 100000 --no-graph
jupyter notebook notebooks/02_feature_engineering.ipynb   # разбор каждого признака
python tests/test_synthetic_pipeline.py      # 9 проверок конвейера, ~10 с
```

Вывод эталонных прогонов: `reports/reference_run_synthetic/stage2_findings.md`
и `stage3_findings.md`.

## Прогресс

- [x] Этап 0 — репозиторий, venv (Python 3.11), зависимости, датасет
- [x] Этап 1 — EDA: дисбаланс, типологии, structuring-тест у порога $10 000,
      velocity/inter-arrival, коридоры, fan-in/fan-out, качество данных
- [x] Этап 2 — Feature Engineering: 62 признака (транзакционные, поведенческие,
      velocity, парные, граф), FIT на train, потоковая сборка 9.5 млн строк,
      контроль утечек. Топ lift: vel_amount_sum_1h 4.77, amount_log 4.38,
      g_out_amount_sum 2.40
- [x] Этап 3 — Rule-based baseline: 13 правил, лучший набор — recall 33.7 %
      при 0.44 % трафика (lift 76). Слепые зоны: Smurfing, Bipartite,
      Gather-Scatter, Stacked Bipartite — 0 % (обоснование для ML)
- [ ] Этап 4 — ML-модели (дисбаланс, XGBoost/LightGBM, tuning, калибровка)
- [ ] Этап 5 — Evaluation, SHAP, бизнес-метрики
- [ ] Этап 6 — Streamlit-дашборд
- [ ] Этап 7 — README, презентация, Docker

# Intelligent AML Transaction Monitoring System using ML and Graph Analytics

End-to-end портфолио-проект по выявлению подозрительных транзакций (AML):
rule-based detection → feature engineering → граф-аналитика → ML (XGBoost/LightGBM)
→ SHAP-объяснения → Streamlit-дашборд для аналитика.

**Датасет:** SAML-D (Synthetic Anti-Money Laundering Dataset, Kaggle) — 9 504 852 транзакций,
12 колонок, доля отмывания `Is_laundering=1` — 0.1039% (9 873 строк), 17 типологий.

---

## Быстрый старт

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Положить CSV с Kaggle в data/raw/   (папка в git не идёт)
#    либо указать путь явно:  export SAML_D_CSV=/путь/к/SAML-D.csv

# 2. Запустить EDA
jupyter notebook notebooks/01_eda.ipynb
```

Данные читаются один раз (1–3 мин на 9.5 млн строк) и кэшируются в
`data/processed/saml_d_prepared.parquet`; дальше всё открывается за секунды.

## Структура

```
data/            raw/ (Kaggle CSV, не в git) · processed/ (кэш) · features/
notebooks/       01_eda · 02_feature_engineering · 03_rule_based_baseline
                 04_ml_modeling · 05_evaluation_shap
src/             data_loader · features · graph_features · rules · train · evaluate · eda_utils
streamlit_app/   app.py — дашборд аналитика
models/          обученные модели
reports/         figures/ (графики) · eda_findings.md (авто-сводка EDA)
tests/           make_synthetic_saml_d.py — генератор лёгкой синтетики для тестов
```

## Прогресс

- [x] Этап 0 — репозиторий, venv (Python 3.11), зависимости, датасет
- [ ] Этап 1 — EDA (суммы/structuring, velocity, коридоры, граф-паттерны)
- [ ] Этап 2 — Feature Engineering (транзакционные, поведенческие, velocity, graph)
- [ ] Этап 3 — Rule-based baseline (красные флаги)
- [ ] Этап 4 — ML-модели (дисбаланс, XGBoost/LightGBM, tuning, калибровка)
- [ ] Этап 5 — Evaluation, SHAP, бизнес-метрики
- [ ] Этап 6 — Streamlit-дашборд
- [ ] Этап 7 — README, презентация, Docker

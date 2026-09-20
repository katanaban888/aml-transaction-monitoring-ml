"""
Этап 3. Rule-based baseline («красные флаги»).

ЗАЧЕМ ЭТОТ ЭТАП ВООБЩЕ НУЖЕН, если дальше будет ML:

1. **Точка отсчёта.** Модель обязана обойти правила, иначе она не нужна.
   Если XGBoost даёт recall 40 % при 5 000 алертов в день, а три правила дают
   35 % при 300 алертах — модель не окупается.
2. **Объяснимость.** Для регулятора и для аналитика «сработало правило R2:
   два платежа по 9 500 за сутки» звучит понятнее, чем «SHAP-значение 0.37».
3. **Гибрид.** На практике живая система = правила (жёсткие требования закона)
   + модель (приоритизация очереди алертов).
4. **Тест на утечки.** Правила не обучаются, поэтому их качество — честный
   ориентир «сколько сигнала вообще есть в данных».

КАК УСТРОЕН КОД
---------------
Каждое правило — это функция от матрицы признаков, которая возвращает
булев массив: «транзакция подозрительна / нет». Правила собираются в
`RULE_BOOK` (справочник правил), дальше:

    flags  = apply_rules(X)          # таблица: строка = транзакция, колонка = правило
    score  = risk_score(flags)       # сколько правил сработало = «уровень риска»
    metrics = rule_metrics(flags, y) # precision / recall / lift по каждому правилу

Пороги для числовых правил подбираем на val-периоде и только потом смотрим
test — иначе мы подгоним правила под ответы.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.data_loader import COL_TARGET, STRUCTURING_THRESHOLD


# ---------------------------------------------------------------------------
# 1. ОДНО ПРАВИЛО
# ---------------------------------------------------------------------------
@dataclass
class Rule:
    """Одно правило мониторинга.

    Порог бывает двух видов:
      * **жёсткий** — прописан в законе или в процедуре банка
        (например, "сумма >= 10 000" или ">= 2 платежа чуть ниже порога за сутки");
      * **настраиваемый** — «странно много / странно крупно» по сравнению с
        остальными клиентами. Тут число из головы брать нельзя: на 9.5 млн
        транзакций «20 получателей» — это почти все. Поэтому порог = квантиль
        признака, посчитанный на train-периоде (тот же принцип FIT/APPLY,
        что и у профилей клиентов на Этапе 2).
    """

    code: str                      # короткий код, например "R2"
    name: str                      # человеческое название
    typology: str                  # какую типологию отмывания ловит
    logic: str                     # условий правила «словами» (для отчёта)
    func: Callable[["Rule", pd.DataFrame], np.ndarray]   # сама проверка
    needs: tuple[str, ...] = field(default_factory=tuple)   # нужные признаки
    feature: str | None = None     # какой признак «пороговать» (если порог настраиваемый)
    quantile: float | None = None  # какой квантиль берём порогом (0.95, 0.99 ...)
    threshold: float | None = None # посчитанный порог (заполняется в fit)

    def fit(self, X_fit: pd.DataFrame) -> "Rule":
        """Считает порог по квантили признака на обучающем периоде."""
        if self.feature is not None and self.quantile is not None:
            vals = pd.to_numeric(X_fit[self.feature], errors="coerce").to_numpy(dtype="float64")
            self.threshold = float(np.nanquantile(vals[np.isfinite(vals)], self.quantile))
            self.logic = f"{self.feature} >= {self.threshold:,.2f} ({self.quantile:.0%} перцентиль train)"
        return self

    def __call__(self, X: pd.DataFrame) -> np.ndarray:
        """Возвращает булев массив: сработало / не сработало."""
        missing = [c for c in self.needs if c not in X.columns]
        if missing:
            raise KeyError(f"Правилу {self.code} не хватает признаков: {missing}")
        return np.asarray(self.func(self, X), dtype=bool)


# ---------------------------------------------------------------------------
# 2. СПРАВОЧНИК ПРАВИЛ
# ---------------------------------------------------------------------------
def default_rule_book() -> list[Rule]:
    """
    Набор правил «как в реальном банке»: каждое правило = численное выражение
   typology из обучающих материалов FATF / типичных процедур комплаенса.
    """
    return [
        Rule(
            code="R1",
            name=f"Крупная сумма (>= ${STRUCTURING_THRESHOLD:,.0f})",
            typology="Single_large, Over-Invoicing",
            logic=f"Amount >= {STRUCTURING_THRESHOLD:,.0f} (порог обязательного отчёта)",
            func=lambda r, X: X["amount_above_threshold"].to_numpy() == 1,
            needs=("amount_above_threshold",),
        ),
        Rule(
            code="R2",
            name="Structuring: серия платежей чуть ниже порога",
            typology="Structuring, Smurfing",
            logic="сумма в [0.9 * 10 000, 10 000) И таких же >= 2 за сутки",
            func=lambda r, X: ((X["amount_near_threshold"].to_numpy() == 1)
                             & (X["vel_near_threshold_count_24h"].to_numpy() >= 2)),
            needs=("amount_near_threshold", "vel_near_threshold_count_24h"),
        ),
        Rule(
            code="R3",
            name="Structuring за неделю",
            typology="Structuring",
            logic=">= 3 платежа чуть ниже порога за 7 дней",
            func=lambda r, X: X["vel_near_threshold_count_168h"].to_numpy() >= 3,
            needs=("vel_near_threshold_count_168h",),
        ),
        Rule(
            code="R4",
            name="Всплеск частоты",
            typology="Smurfing",
            logic=">= 3 операций у счёта за час",
            func=lambda r, X: X["vel_count_1h"].to_numpy() >= 3,
            needs=("vel_count_1h",),
        ),
        Rule(
            code="R5",
            name="Всплеск оборота",
            typology="Behavioural_Change_1/2",
            logic="оборот за сутки во много раз больше обычной суммы счёта",
            func=lambda r, X: X["vel_amount_24h_to_median"].fillna(0).to_numpy() >= r.threshold,
            needs=("vel_amount_24h_to_median",),
            feature="vel_amount_24h_to_median", quantile=0.99,
        ),
        Rule(
            code="R6",
            name="Сумма нетипична для счёта",
            typology="Behavioural_Change_1/2, Single_large",
            logic="z-отклонение суммы от истории счёта аномально велико",
            func=lambda r, X: X["amount_zscore_sender"].fillna(0).to_numpy() >= r.threshold,
            needs=("amount_zscore_sender",),
            feature="amount_zscore_sender", quantile=0.99,
        ),
        Rule(
            code="R7",
            name="Fan-out: слишком много получателей",
            typology="Fan_Out, Layered_Fan_Out",
            logic="у счёта аномально много разных получателей",
            func=lambda r, X: X["g_out_degree"].fillna(0).to_numpy() >= r.threshold,
            needs=("g_out_degree",),
            feature="g_out_degree", quantile=0.95,
        ),
        Rule(
            code="R8",
            name="Fan-in: слишком много отправителей",
            typology="Fan_In, Layered_Fan_In, Gather-Scatter",
            logic="у счёта аномально много разных отправителей",
            func=lambda r, X: X["g_in_degree"].fillna(0).to_numpy() >= r.threshold,
            needs=("g_in_degree",),
            feature="g_in_degree", quantile=0.95,
        ),
        Rule(
            code="R9",
            name="Участие в цикле (повторяющиеся связи)",
            typology="Cycle",
            logic="счёт в цикле по повторяющимся связям И связей у него аномально много",
            func=lambda r, X: ((X["g_in_repeat_cycle"].fillna(0).to_numpy() >= 1)
                               & (X["g_total_degree"].fillna(0).to_numpy() >= r.threshold)),
            needs=("g_in_repeat_cycle", "g_total_degree"),
            feature="g_total_degree", quantile=0.95,
        ),
        Rule(
            code="R10",
            name="Взаимные переводы (цикл длины 2)",
            typology="Cycle, Bipartite",
            logic="между парой счетов есть переводы в обе стороны",
            func=lambda r, X: X["g_is_mutual_pair"].fillna(0).to_numpy() >= 1,
            needs=("g_is_mutual_pair",),
        ),
        Rule(
            code="R11",
            name="Новый контрагент + крупная сумма",
            typology="Money mule, Deposit-Send",
            logic="пара видит друг друга впервые И сумма в топ-5 % по валюте",
            func=lambda r, X: ((X["is_new_pair"].fillna(1).to_numpy() == 1)
                             & (X["amount_pct_in_currency"].fillna(0).to_numpy() >= 95)),
            needs=("is_new_pair", "amount_pct_in_currency"),
        ),
        Rule(
            code="R12",
            name="Ночная операция на крупную сумму",
            typology="Layering (сокрытие следа)",
            logic="операция с 00:00 до 06:00 И сумма в топ-10 % по валюте",
            func=lambda r, X: ((X["is_night"].to_numpy() == 1)
                             & (X["amount_pct_in_currency"].fillna(0).to_numpy() >= 90)),
            needs=("is_night", "amount_pct_in_currency"),
        ),
        Rule(
            code="R13",
            name="Транзитный счёт",
            typology="Layering_Fan_In/Fan_Out, Stacked Bipartite",
            logic="перекос между входящими и исходящими связями аномально велик",
            func=lambda r, X: X["g_degree_ratio"].fillna(1).to_numpy() >= r.threshold,
            needs=("g_degree_ratio",),
            feature="g_degree_ratio", quantile=0.99,
        ),
    ]


# ---------------------------------------------------------------------------
# 3. ПРИМЕНЕНИЕ ПРАВИЛ
# ---------------------------------------------------------------------------
def fit_rules(X_fit: pd.DataFrame, rule_book: list[Rule] | None = None) -> list[Rule]:
    """
    Подбирает пороги всех настраиваемых правил на обучающем периоде.

    ПОЧЕМУ НА TRAIN: если подогнать пороги по val или test, мы узнаем ответы
    заранее. В проде пороги пересчитывают раз в месяц на «прошлом» — ровно
    то же самое.
    """
    for rule in (rule_book or default_rule_book()):
        rule.fit(X_fit)
    return rule_book or default_rule_book()


def apply_rules(X: pd.DataFrame, rule_book: list[Rule] | None = None) -> pd.DataFrame:
    """
    Пропускает матрицу признаков через все правила.

    Возвращает таблицу: строка = транзакция, колонка `flag_R1 ... flag_R13`
    (True = правило сработало).
    """
    rule_book = rule_book or default_rule_book()
    out = pd.DataFrame(index=pd.RangeIndex(len(X)))
    for rule in rule_book:
        out[f"flag_{rule.code}"] = rule(X)
    return out


def risk_score(flags: pd.DataFrame) -> pd.Series:
    """Сколько правил сработало на транзакции = простая оценка риска."""
    return flags.sum(axis=1).astype("int16").rename("risk_score")


def any_rule(flags: pd.DataFrame) -> pd.Series:
    """«Хотя бы одно правило» — самая чувствительная (и самая шумная) стратегия."""
    return flags.any(axis=1).rename("any_rule")


# ---------------------------------------------------------------------------
# 4. МЕТРИКИ
# ---------------------------------------------------------------------------
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def confusion(alert: np.ndarray, y: np.ndarray) -> dict:
    """TP / FP / FN / TN + производные метрики одной строки."""
    alert = np.asarray(alert, dtype=bool)
    y = np.asarray(y, dtype=bool)
    tp = int((alert & y).sum())
    fp = int((alert & ~y).sum())
    fn = int((~alert & y).sum())
    tn = int((~alert & ~y).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alerts": tp + fp,
        "alert_rate_%": _safe_div(tp + fp, len(y)) * 100,
        "precision_%": precision * 100,
        "recall_%": recall * 100,
        "lift": _safe_div(precision, (y.mean() if len(y) else np.nan)) if y.mean() else np.nan,
        "f1": _safe_div(2 * precision * recall, precision + recall),
    }


def rule_metrics(flags: pd.DataFrame, y: pd.Series | np.ndarray,
                 rule_book: list[Rule] | None = None) -> pd.DataFrame:
    """Метрики по каждому правилу отдельно."""
    rule_book = rule_book or default_rule_book()
    y = np.asarray(y)
    rows = []
    for rule in rule_book:
        col = f"flag_{rule.code}"
        if col not in flags.columns:
            continue
        m = confusion(flags[col].to_numpy(), y)
        rows.append({"правило": rule.code, "название": rule.name,
                     "типология": rule.typology, "условие": rule.logic, **m})
    return pd.DataFrame(rows).sort_values("recall_%", ascending=False).reset_index(drop=True)


def score_curve(y: pd.Series | np.ndarray, score: pd.Series | np.ndarray,
                max_score: int | None = None) -> pd.DataFrame:
    """
    Метрики при пороге «сработало >= k правил» для каждого k.

    Это и есть главный бизнес-выбор: чем выше порог k, тем меньше алертов
    (дешевле) и тем ниже recall (больше пропусков).
    """
    y = np.asarray(y)
    score = np.asarray(score)
    rows = []
    for k in range(1, (max_score or int(score.max())) + 1):
        m = confusion(score >= k, y)
        rows.append({"порог_k": k, **m})
    return pd.DataFrame(rows)


def recall_by_typology(alert: np.ndarray, y: np.ndarray,
                       typology: pd.Series | np.ndarray,
                       min_support: int = 20) -> pd.DataFrame:
    """
    Какую долю КАЖДОЙ типологии мы ловим — самая полезная таблица этапа.

    По ней видно не «среднюю температуру», а конкретные дыры: например,
    structuring ловится правилами хорошо, а Cycle — почти никак.
    """
    alert = np.asarray(alert, dtype=bool)
    y = np.asarray(y)
    typ = pd.Series(np.asarray(typology)).fillna("— обычная операция —")
    out = typ[y == 1].value_counts().rename("всего_отмываний").to_frame()
    caught = typ[(y == 1) & alert].value_counts().rename("поймано")
    out = out.join(caught, how="left").fillna({"поймано": 0})
    out["recall_%"] = out["поймано"] / out["всего_отмываний"] * 100
    return out[out["всего_отмываний"] >= min_support].sort_values(
        "recall_%", ascending=False).reset_index(names="типология")


def daily_alerts(alert: np.ndarray, dates: pd.Series | np.ndarray) -> pd.Series:
    """
    Сколько алертов в день придёт аналитику — главная операционная метрика.

    Дату приводим к «дню» через datetime64[D], а не через строки: на 9.5 млн
    строк перевод даты в текст съедает почти гигабайт.
    """
    day = np.asarray(pd.to_datetime(pd.Series(np.asarray(dates))).dt.floor("D"))
    return pd.Series(np.asarray(alert, dtype=bool)).groupby(day).sum()


def workload_report(alert: np.ndarray, y: np.ndarray, dates: pd.Series | np.ndarray,
                    analyst_capacity_per_day: int = 30) -> dict:
    """
    Сводка «что это значит для команды»: сколько алертов в день, сколько
    аналитиков нужно, сколько «пустых» проверок на один реальный случай.
    """
    alert = np.asarray(alert, dtype=bool)
    y = np.asarray(y)
    per_day = daily_alerts(alert, dates)
    tp = int((alert & y).sum())
    return {
        "алертов_всего": int(alert.sum()),
        "алертов_в_день_медиана": float(per_day.median()),
        "алертов_в_день_макс": int(per_day.max()),
        "дней_наблюдений": int(len(per_day)),
        "найдено_отмываний": tp,
        "проверок_на_1_реальный_случай": _safe_div(int(alert.sum()), tp),
        "аналитиков_нужно": _safe_div(per_day.median(), analyst_capacity_per_day),
    }

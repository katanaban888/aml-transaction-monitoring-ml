"""
src/graph_features.py
=====================
Граф-признаки: отмывание видно не в одной транзакции, а в СТРУКТУРЕ связей.

                 Fan-Out                    Fan-In                  Cycle
                    A                      -> A                    A -> B
                 /  |  \\                 /   ^                     ^     v
                B   C   D               C    D                     D <- C
           один -> много            много -> один          деньги вернулись к отправителю

Что считаем:
  * степени вершин (in/out) и обороты по ним — ловят fan-in / fan-out / хабы;
  * PageRank — «важность» счёта в сети переводов (считаем сами на разреженных
    матрицах, потому что networkx на 10 млн рёбер не влезает в память);
  * участие в цикле (strongly connected component размера > 1) — прямая детекция
    типологии `Cycle` и «возвратных» схем;
  * размер слабой компоненты связности — размер «компании» вокруг счёта;
  * взаимные пары (A->B и B->A) — цикл длины 2.

ПОЧЕМУ НЕ ПРОСТО networkx НА ВСЕХ ДАННЫХ
-----------------------------------------
В датасете 9.5 млн транзакций и миллионы уникальных рёбер. Объект
networkx.DiGraph хранит каждое ребро в словарях — это примерно 300 байт на
ребро, то есть ~3 ГБ только на структуру графа. Поэтому:

  1) степени, обороты и PageRank считаем на разреженных матрицах / pandas
     (быстро и на ВСЕХ данных);
  2) networkx используем для структурных алгоритмов (SCC, компоненты), но на
     ОТФИЛЬТРОВАННОМ графе — топ-N рёбер по обороту. Именно такой приём
     (сначала фильтрация, потом граф-алгоритмы) применяют в продакшене,
     где счёт идёт на сотни миллионов транзакций.

Всё, что считается здесь, фит ТОЛЬКО на train-периоде.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import COL_AMOUNT, COL_SENDER, COL_RECEIVER


# ---------------------------------------------------------------------------
# 1. РЁБРА ГРАФА
# ---------------------------------------------------------------------------
def build_edge_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сворачивает транзакции в рёбра графа: (отправитель -> получатель)
    с числом переводов и суммарным оборотом.
    """
    g = df.groupby([COL_SENDER, COL_RECEIVER], observed=True)
    edges = pd.DataFrame({
        "n_txn": g.size(),
        "amount_sum": g[COL_AMOUNT].sum(),
        "amount_mean": g[COL_AMOUNT].mean(),
        "amount_max": g[COL_AMOUNT].max(),
        "last_ts": g["txn_ts"].max() if "txn_ts" in df.columns else np.nan,
    }).reset_index()
    return edges.sort_values("amount_sum", ascending=False).reset_index(drop=True)


def compute_degrees(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Степени вершин и обороты по ним.

    g_out_degree  — скольким счётам счёт отправлял деньги (fan-out)
    g_in_degree   — от скольких счетов получал (fan-in)
    g_total_degree — суммарное число контрагентов
    """
    out = edges.groupby(COL_SENDER, observed=True).agg(
        g_out_degree=("n_txn", "size"),
        g_out_txn_sum=("n_txn", "sum"),
        g_out_amount_sum=("amount_sum", "sum"),
    )
    inn = edges.groupby(COL_RECEIVER, observed=True).agg(
        g_in_degree=("n_txn", "size"),
        g_in_txn_sum=("n_txn", "sum"),
        g_in_amount_sum=("amount_sum", "sum"),
    )
    deg = out.join(inn, how="outer").fillna(0)
    deg["g_total_degree"] = deg["g_out_degree"] + deg["g_in_degree"]
    deg["g_degree_ratio"] = (
        deg["g_in_degree"] / deg["g_out_degree"].replace(0, np.nan)
    )
    return deg


# ---------------------------------------------------------------------------
# 2. PAGERANK НА РАЗРЕЖЕННЫХ МАТРИЦАХ (все данные, без networkx)
# ---------------------------------------------------------------------------
def sparse_pagerank(edges: pd.DataFrame, damping: float = 0.85, max_iter: int = 60,
                    tol: float = 1e-8, verbose: bool = False) -> pd.Series:
    """
    PageRank по графу переводов. Вес ребра = число транзакций по нему.

    Смысл в AML: высокий PageRank = счёт, через который по структуре сети
    «протекает» много денег — центральный узел схемы, а не рядовой клиент.

    Реализовано степенным методом на scipy.sparse:
        r <- d * W^T (r / out_weight) + (1 - d) / N
    Это стандартная формулировка, где учитываются и «висячие» вершины
    (у которых нет исходящих рёбер): их вес распределяется равномерно.
    """
    import scipy.sparse as sp

    nodes = pd.Index(
        pd.concat([edges[COL_SENDER], edges[COL_RECEIVER]], ignore_index=True).unique()
    )
    n = len(nodes)
    u = nodes.get_indexer(edges[COL_SENDER]).astype(np.int32)
    v = nodes.get_indexer(edges[COL_RECEIVER]).astype(np.int32)
    w = edges["n_txn"].to_numpy(dtype=np.float64)

    # W[i, j] = вес ребра j -> i (транспонировано для удобства формулы)
    W = sp.csr_matrix((w, (v, u)), shape=(n, n))
    out_weight = np.asarray(W.sum(axis=1)).ravel()      # сколько «уходит» из i
    out_weight_safe = np.where(out_weight > 0, out_weight, 1.0)

    r = np.full(n, 1.0 / n)
    for it in range(max_iter):
        new_r = damping * (W @ (r / out_weight_safe)) + (1 - damping) / n
        # разница по «висячим» вершинам уходит в равномерное распределение
        dangling = r[out_weight == 0].sum()
        new_r += damping * dangling / n
        delta = np.abs(new_r - r).sum()
        r = new_r
        if delta < tol:
            if verbose:
                print(f"[pagerank] сошёлся за {it + 1} итераций")
            break
    return pd.Series(r, index=nodes, name="g_pagerank")


def cyclic_core(edges: pd.DataFrame, max_iter: int = 50, verbose: bool = False):
    """
    Находит все счета, которые лежат хотя бы на одном ЦИКЛЕ, на ПОЛНОМ графе.

    Идея (вместо тяжёлого Тарьяна на 7 млн рёбер):
    в любом ациклическом графе (DAG) обязательно есть вершина без входящих
    рёбер и вершина без исходящих. Значит, если мы будем «шелушить» граф —
    удалять вершины с in_degree = 0 или out_degree = 0 и повторять, — то
    в остатке останутся РОВНО те вершины, у которых в подграфе есть и вход,
    и выход, а это в точности вершины, лежащие на циклах.

    Сложность: несколько проходов groupby по рёбрам (секунды на 7 млн рёбер)
    вместо построения многогигабайтного графа в networkx.

    Возвращает (core_nodes, core_edges).
    """
    # Сохраняем n_txn (если есть): он нужен для визуализации и весов рёбер.
    keep_cols = [COL_SENDER, COL_RECEIVER] + [c for c in ("n_txn",) if c in edges.columns]
    cur = edges[keep_cols].copy()

    for it in range(max_iter):
        out_deg = cur.groupby(COL_SENDER, observed=True).size()
        in_deg = cur.groupby(COL_RECEIVER, observed=True).size()
        # остаются те, у кого есть хотя бы одно входящее И одно исходящее ребро
        keep_out = out_deg.index[(out_deg > 0).to_numpy()]
        keep_in = in_deg.index[(in_deg > 0).to_numpy()]
        nodes = keep_out.intersection(keep_in)
        if len(nodes) == 0:
            cur = cur.iloc[0:0]
            break
        new = cur[cur[COL_SENDER].isin(nodes) & cur[COL_RECEIVER].isin(nodes)]
        if len(new) == len(cur):
            break
        cur = new
        if verbose:
            print(f"  [cyclic_core] итерация {it + 1}: рёбер {len(cur):,}, вершин {len(nodes):,}")

    core_nodes = pd.Index(
        pd.concat([cur[COL_SENDER], cur[COL_RECEIVER]], ignore_index=True).unique()
    )
    return core_nodes, cur


def component_sizes(edges: pd.DataFrame) -> pd.Series:
    """
    Размер слабой (без учёта направления) компоненты связности для каждой вершины.
    Считается на разреженной матрице scipy — быстро и на полном графе.

    Смысл: размер «компании» счетов, связанных переводами.
    У схем типа Bipartite / Gather-Scatter эта цифра резко выше обычной.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    nodes = pd.Index(pd.concat([edges[COL_SENDER], edges[COL_RECEIVER]],
                               ignore_index=True).unique())
    idx = pd.Series(np.arange(len(nodes)), index=nodes)
    ui = idx.reindex(edges[COL_SENDER]).to_numpy()
    vi = idx.reindex(edges[COL_RECEIVER]).to_numpy()
    n = len(nodes)

    adj = sp.coo_matrix((np.ones(len(ui)), (ui, vi)), shape=(n, n)).tocsr()
    n_comp, labels = connected_components(adj, directed=False, connection="weak")
    sizes = pd.Series(labels).value_counts().reindex(range(n_comp)).to_numpy()
    return pd.Series(sizes[labels], index=nodes, name="g_component_size")


# ---------------------------------------------------------------------------
# 3. СТРУКТУРНЫЕ ПРИЗНАКИ ЧЕРЕЗ networkx (на отфильтрованном графе)
# ---------------------------------------------------------------------------
def build_filtered_graph(edges: pd.DataFrame, max_edges: int = 400_000) -> "object":
    """
    Строит networkx.DiGraph из топ-N рёбер по обороту.

    Почему фильтруем: полный граф на 9.5 млн транзакций даёт миллионы рёбер,
    которые не влезают в память ноутбука. При этом схемы отмывания — это
    всегда КРУПНЫЕ и повторяющиеся связи, поэтому фильтрация по обороту
    сохраняет именно ту часть сети, где живёт риск.
    """
    import networkx as nx

    sub = edges.head(max_edges)
    G = nx.DiGraph()
    G.add_nodes_from(pd.concat([sub[COL_SENDER], sub[COL_RECEIVER]]).unique())
    G.add_edges_from(
        (u, v, {"weight": float(wt), "amount": float(amt)})
        for u, v, wt, amt in zip(sub[COL_SENDER].to_numpy(),
                                 sub[COL_RECEIVER].to_numpy(),
                                 sub["n_txn"].to_numpy(),
                                 sub["amount_sum"].to_numpy())
    )
    return G


def cycle_and_component_features(G) -> pd.DataFrame:
    """
    Для каждой вершины: участвует ли в цикле, размер SCC, размер компоненты.

    `g_in_cycle` — главный признак для типологии `Cycle`: вершина лежит в
    сильно связной компоненте размера > 1, значит от неё можно дойти до неё же
    самой по рёбрам переводов (деньги ходят по кругу).
    """
    import networkx as nx

    rows = {}
    for size, comp in enumerate(nx.strongly_connected_components(G)):
        for node in comp:
            rows[node] = {"g_scc_size": len(comp), "g_in_cycle": int(len(comp) > 1)}
    scc = pd.DataFrame.from_dict(rows, orient="index")

    comp_rows = {}
    for comp in nx.weakly_connected_components(G):
        size = len(comp)
        for node in comp:
            comp_rows[node] = {"g_component_size": size}
    comp_df = pd.DataFrame.from_dict(comp_rows, orient="index")

    out = scc.join(comp_df, how="outer").fillna({"g_in_cycle": 0, "g_scc_size": 1,
                                                 "g_component_size": 1})
    return out


def mutual_pairs(edges: pd.DataFrame) -> pd.DataFrame:
    """
    Взаимные пары A->B и B->A (цикл длины 2: деньги ушли и вернулись).
    Возвращает таблицу пар с флагом 1.
    """
    fwd = edges[[COL_SENDER, COL_RECEIVER]].copy()
    rev = edges[[COL_RECEIVER, COL_SENDER]].copy()
    rev.columns = [COL_SENDER, COL_RECEIVER]
    mutual = fwd.merge(rev.drop_duplicates(), on=[COL_SENDER, COL_RECEIVER])
    mutual["g_is_mutual_pair"] = 1
    return mutual.drop_duplicates(subset=[COL_SENDER, COL_RECEIVER])


# ---------------------------------------------------------------------------
# 4. СБОРКА: FIT НА TRAIN -> APPLY НА ВСЕ СТРОКИ
# ---------------------------------------------------------------------------
def fit_graph_tables(df_train: pd.DataFrame, max_edges: int = 400_000,
                     verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Считает граф-статистику по train-периоду.

    Возвращает (accounts, mutual, meta):
      accounts — таблица признаков по каждому счёту (индекс = название счёта);
      mutual   — таблица взаимных пар (A->B и B->A);
      meta     — цифры для отчёта.

    Отделяем FIT от APPLY затем же, зачем и в features.py: статистику графа
    учим на прошлом, а применяем и к будущим периодам.
    """
    edges = build_edge_list(df_train)
    deg = compute_degrees(edges)
    pr = sparse_pagerank(edges, verbose=verbose)

    # 2a. Циклы и компоненты на ПОЛНОМ графе (главные структурные признаки)
    core_nodes, core_edges = cyclic_core(edges, verbose=verbose)
    comp = component_sizes(edges)
    cycle = pd.Series(1, index=core_nodes, name="g_in_cycle")
    if len(core_edges):
        core_comp = component_sizes(core_edges).rename("g_core_component_size")
    else:
        core_comp = pd.Series(dtype="float64", name="g_core_component_size")

    # 2b. networkx — на отфильтрованном графе (SCC как независимая проверка)
    G = build_filtered_graph(edges, max_edges=max_edges)
    struct = cycle_and_component_features(G).rename(
        columns={"g_scc_size": "g_nx_scc_size", "g_component_size": "g_nx_component_size"})
    # Циклы по графу ПОВТОРЯЮЩИХСЯ связей (пара переводила друг другу >= 2 раз).
    # Такой граф на порядки разреженнее и поэтому гораздо избирательнее:
    # случайный перевод «на удачу» не создаёт структуру, а «накатанная дорожка» — да.
    rep = edges[edges["n_txn"] >= 2]
    if len(rep) > 0:
        rep_core, _ = cyclic_core(rep)
        rep_cycle = pd.Series(1, index=rep_core, name="g_in_repeat_cycle")
    else:
        rep_cycle = pd.Series(dtype="int64", name="g_in_repeat_cycle")
    mutual = mutual_pairs(edges)

    accounts = (deg.join(pr, how="outer")
                   .join(comp, how="outer")
                   .join(cycle, how="outer")
                   .join(core_comp, how="outer")
                   .join(rep_cycle, how="outer")
                   .join(struct[["g_nx_scc_size", "g_nx_component_size"]], how="outer"))
    for c, fill in [("g_in_cycle", 0), ("g_component_size", 1),
                    ("g_nx_scc_size", 1), ("g_nx_component_size", 1),
                    ("g_core_component_size", 0), ("g_in_repeat_cycle", 0)]:
        if c in accounts.columns:
            accounts[c] = accounts[c].fillna(fill)
    accounts["g_pagerank"] = accounts["g_pagerank"].fillna(accounts["g_pagerank"].min())

    # Считаем итоговые цифры ДО блока печати: они нужны и в meta, и в логе.
    in_cycle = int(accounts["g_in_cycle"].sum())
    in_rep = int(accounts["g_in_repeat_cycle"].sum())

    if verbose:
        print(f"[graph] рёбер в train: {len(edges):,} | счетов в графе: {len(accounts):,}")
        print(f"[graph] циклическое ядро (полный граф): {len(core_nodes):,} счетов "
              f"({len(core_nodes) / max(len(accounts), 1) * 100:.2f}%)")
        print(f"[graph] граф networkx для SCC: {G.number_of_nodes():,} узлов, "
              f"{G.number_of_edges():,} рёбер (топ-{max_edges:,} по обороту)")
        print(f"[graph] счетов в циклах: {in_cycle:,} | "
              f"в циклах по повторяющимся связям: {in_rep:,} | "
              f"взаимных пар: {len(mutual):,}")

    meta = {
        "n_edges_train": len(edges),
        "n_accounts": len(accounts),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "n_mutual_pairs": len(mutual),
        "n_accounts_in_cycle": in_cycle,
        "n_core_edges": len(core_edges),
        "n_accounts_in_repeat_cycle": int(accounts["g_in_repeat_cycle"].sum()),
    }
    return accounts, mutual, meta


def merge_graph_tables(df: pd.DataFrame, accounts: pd.DataFrame,
                       mutual: pd.DataFrame) -> pd.DataFrame:
    """
    Приклеивает граф-признаки к транзакциям: признаки счёта-отправителя,
    признак «получатель участвует в цикле» и флаг взаимной пары.
    """
    acc = accounts.reset_index()
    acc = acc.rename(columns={acc.columns[0]: COL_SENDER})
    acc[COL_SENDER] = acc[COL_SENDER].astype(df[COL_SENDER].dtype)
    out = df.merge(acc, on=COL_SENDER, how="left")

    # По получателю берём только «участвует в цикле» — остальные его параметры
    # уже есть в поведенческом профиле (receiver_txn_count и т.п.)
    recv = accounts[["g_in_cycle"]].reset_index()
    recv = recv.rename(columns={recv.columns[0]: COL_RECEIVER,
                                "g_in_cycle": "g_receiver_in_cycle"})
    recv[COL_RECEIVER] = recv[COL_RECEIVER].astype(df[COL_RECEIVER].dtype)
    out = out.merge(recv, on=COL_RECEIVER, how="left")

    out = out.merge(mutual, on=[COL_SENDER, COL_RECEIVER], how="left")
    out["g_is_mutual_pair"] = out["g_is_mutual_pair"].fillna(0).astype("float32")
    out["g_receiver_in_cycle"] = out["g_receiver_in_cycle"].fillna(0).astype("float32")
    # Заполняем пропуски только в тех колонках, которые реально есть
    for c, fill in [("g_in_cycle", 0), ("g_in_repeat_cycle", 0),
                    ("g_component_size", 1), ("g_nx_scc_size", 1),
                    ("g_nx_component_size", 1), ("g_core_component_size", 0)]:
        if c in out.columns:
            out[c] = out[c].fillna(fill).astype("float32")
    return out


def fit_graph_features(df: pd.DataFrame, train_mask: pd.Series,
                       max_edges: int = 400_000, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Удобная обёртка: FIT на train -> APPLY на переданный df (для небольших данных)."""
    train_np = np.asarray(train_mask)
    accounts, mutual, meta = fit_graph_tables(df[train_np], max_edges=max_edges, verbose=verbose)
    return merge_graph_tables(df, accounts, mutual), meta

#!/usr/bin/env python3
"""スイープの «1 行 1 (セル × 試行)» の表．

run ディレクトリの読み方そのものは `runvault.read` にある．ここに残るのは CompeteAI
固有の部分だけ — どの列を持つ表なのか (`n_firms` / `final_revenue_gini` …) である．

runvault はこの表をディスクに持たない．sweep 親の子 run (`lineage.parent_run_uid` が
親の `run_uid`) を集め，各子の `config.json` の `parameters`・`run.json` の `rng`・
`metrics.csv` の最終ステップと run スコープ指標から組み直す．列は移行前の
`sweep_summary.csv` と同じにしてある．
"""

from __future__ import annotations

import os

import pandas as pd
from runvault.read import (
    config_parameters,
    load_run_meta,
    metrics_wide,
    run_scope_metrics,
    sweep_children,
)

__all__ = ["sweep_summary_table"]

#: 最終ステップ (= 最終日) の値から作る列 (列名 → metrics.csv の指標名)．
_FINAL_COLUMNS = {
    "final_revenue_gini": "revenue_gini",
    "final_market_share_max": "market_share_max",
    "final_menu_similarity": "menu_similarity",
    "final_alive_firms": "n_alive_firms",
}

#: run スコープ指標から作る列 (列名 → 指標名)．
_SCOPE_COLUMNS = {
    "final_day": "final_day",
    "winner_take_all": "winner_take_all",
    "quality_improved": "quality_improved",
    "cache_hit_rate": "llm_cache_hit_rate",
}


def sweep_summary_table(sweep_dir: str | os.PathLike) -> pd.DataFrame:
    """1 行 1 (セル × 試行) のサマリ表を組み直す．

    どの行も `run_dir` を持つので，呼び出し側は条件からディレクトリ名を組み立てなくて
    よい．`cache_hit_rate` は LLM を 1 度も呼ばなかった run では指標そのものが無い
    (率は «0» ではなく «定義できない») ので `NaN` になる．
    """
    children = sweep_children(sweep_dir)
    if not children:
        raise SystemExit(
            f"エラー: この sweep 親に紐づく子 run が見つかりません: {sweep_dir}\n"
            "  子 run は lineage.parent_run_uid で親を指します．"
            "親と子が同じ results ルートにあるか確認してください．"
        )

    rows: list[dict] = []
    for child in children:
        params = config_parameters(child) or {}
        rng = (load_run_meta(child) or {}).get("rng") or {}
        scoped = run_scope_metrics(child)
        last = metrics_wide(os.path.join(child, "metrics.csv")).iloc[-1]
        row = {
            "n_firms": params.get("n_firms"),
            "n_customers": params.get("n_customers"),
            "customer_mode": params.get("customer_mode"),
            # 同一セルの何本目かは runvault の rng.replicate_index が持つ．
            "run": rng.get("replicate_index"),
            "seed": rng.get("master_seed"),
        }
        row.update({column: scoped.get(name) for column, name in _SCOPE_COLUMNS.items()})
        row.update({column: float(last[name]) for column, name in _FINAL_COLUMNS.items()})
        row["run_dir"] = child
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["n_firms", "n_customers", "customer_mode", "run"])
        .reset_index(drop=True)
    )

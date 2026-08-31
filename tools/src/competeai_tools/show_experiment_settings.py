"""competeai-tools show-experiment-settings — run ディレクトリの設定表示．

runvault の run ディレクトリの `config.json` (封筒．条件は `parameters` の下) を読み，
実行時に使われた全パラメータを整形表示する．`run` か `sweep` か `reproduce` かは
`run.json` の `subcommand` で判別する (`sweep_config.json` はもう書かれない)．
LLM 情報 (モデル・provider・温度) は `run.json` の `llm` ブロック，呼び出し数と
cache-hit 率・勝者総取り・品質改善は `metrics.csv` の run スコープ指標から採る．

run ディレクトリのパスは次で取れる:
    runvault path --experiment competeai --latest --subcommand run --standalone
    runvault path --experiment competeai --latest --subcommand sweep
    runvault path --experiment competeai --latest --subcommand reproduce

Usage:
    uv run competeai-tools show-experiment-settings
    uv run competeai-tools show-experiment-settings --results-dir "$(runvault path --experiment competeai --latest --subcommand sweep)"
    uv run competeai-tools show-experiment-settings --json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from runvault.read import (
    config_parameters,
    load_run_meta,
    run_subcommand,
    runvault_path,
)

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "competeai"

# config キー → 表示ラベル (右コロン位置を揃えるためパディング済み)．
# `run` の条件と，`sweep` / `reproduce` の親が持つ格子の両方を並べる — 親と子で
# キーが重ならないので 1 つの表で足りる．
FIELD_LABELS = {
    # run (掃引の子も同じ形)
    "n_firms": "店舗数 M         ",
    "n_customers": "顧客数 N         ",
    "customer_mode": "顧客構成         ",
    "group_size": "グループ人数     ",
    "days": "日数 days        ",
    "init_funds": "初期資金         ",
    "init_menu_size": "初期メニュー数   ",
    "init_price": "初期価格         ",
    "init_cost_ratio": "初期原価率       ",
    "init_chef_salary": "初期シェフ給与   ",
    "customer_income": "顧客所得         ",
    # sweep 親
    "n_firms_values": "店舗数 M (格子)  ",
    "n_customers_values": "顧客数 N (格子)  ",
    "runs": "試行数 runs      ",
    # reproduce 親
    "individual_runs": "個人客 ラン数    ",
    "group_runs": "グループ客ラン数 ",
    "mock": "mock 駆動        ",
    # 共通
    "seed": "シード (コア)    ",
    "llm_temperature": "LLM 温度         ",
    "llm_seed": "LLM seed         ",
}

# run スコープ指標 → 表示ラベル．
SCOPE_LABELS = {
    "n_units": "観測店舗数       ",
    "final_day": "完了ステップ数   ",
    "winner_take_all": "勝者総取り       ",
    "quality_improved": "品質改善         ",
    "llm_calls": "呼び出し総数     ",
    "llm_cache_hits": "cache-hit        ",
}

# sweep スコープ指標 (reproduce 親) → 表示ラベル．
SWEEP_LABELS = {
    "wta_freq_individual": "WTA 頻度 (個人)  ",
    "wta_freq_group": "WTA 頻度 (群)    ",
    "quality_freq_all": "品質改善 頻度    ",
    "menu_similarity_all": "メニュー類似度   ",
}


def _fmt(value: object) -> str:
    """リストは `, ` 連結，それ以外はそのまま．"""
    if isinstance(value, list):
        return ", ".join(map(str, value))
    return str(value)


def render_config(cfg: dict, source: Path, kind: str) -> str:
    """設定テーブルを整形する．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"実行設定 ({kind})")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    for field, label in FIELD_LABELS.items():
        if field in cfg:
            lines.append(f"{label}: {_fmt(cfg[field])}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm(meta: dict, scoped: dict[str, float]) -> str:
    """LLM ブロックと LLM 関連の run スコープ指標を整形する．

    移行前の `run_metadata.json` は 2 つに分かれた — モデル・provider・温度は
    `run.json` の `llm` ブロック，呼び出し数と cache-hit は指標である．cache-hit 率は
    呼び出しが 1 本も無いと «定義できない» ので，行そのものが無い．
    """
    llm = meta.get("llm")
    lines: list[str] = []
    lines.append("")
    lines.append("LLM (run.json の llm ブロック / run スコープ指標)")
    lines.append("-" * 70)
    if llm:
        lines.append(f"provider         : {llm.get('provider', '-')}")
        lines.append(f"モデル           : {llm.get('model_snapshot', '-')}")
        lines.append(f"温度             : {llm.get('temperature', '-')}")
    else:
        lines.append("llm ブロックなし (LLM を使わない run)")
    for name, label in SCOPE_LABELS.items():
        if name in scoped:
            lines.append(f"{label}: {scoped[name]:g}")
    if "llm_cache_hit_rate" in scoped:
        lines.append(f"cache-hit 率     : {scoped['llm_cache_hit_rate'] * 100:.1f}%")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_scope(scoped: dict[str, float]) -> str:
    """reproduce 親の «条件をまたいだ集約» を整形する．"""
    lines: list[str] = []
    lines.append("")
    lines.append("条件をまたいだ集約 (scope=sweep)")
    lines.append("-" * 70)
    for name, label in SWEEP_LABELS.items():
        if name in scoped:
            lines.append(f"{label}: {scoped[name]:.4f}")
    lines.append("=" * 70)
    return "\n".join(lines)


def scope_metrics(results_dir: Path, scope: str) -> dict[str, float]:
    """`metrics.csv` の «step を持たない» 行を scope で絞って読む．

    `runvault.read.run_scope_metrics` は scope 列を見ないので，掃引の親が持つ
    «条件をまたいだ集約» (scope=sweep) と子の «run 全体の値» (scope=run) が混ざる．
    ここでは分けて表示するので scope で絞る．
    """
    path = results_dir / "metrics.csv"
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["scope"] == scope and row["step"] == "":
                out[row["name"]] = float(row["value"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="competeai-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default=None,
        help="run ディレクトリ (省略時は runvault path が返す直近の run)",
    )
    parser.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="runvault の results ルート (default: results)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = Path(
        args.results_dir
        or runvault_path(
            EXPERIMENT,
            results_root=args.results_root,
            subcommand="run",
            standalone=True,
        )
    )
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    cfg = config_parameters(results_dir, required=False)
    if cfg is None:
        print(
            f"エラー: runvault の run ディレクトリではありません: {results_dir}\n"
            "  config.json (parameters を持つ封筒) と run.json が要ります．",
            file=sys.stderr,
        )
        return 1
    meta = load_run_meta(results_dir) or {}
    kind = run_subcommand(results_dir)
    scoped = scope_metrics(results_dir, "run")
    swept = scope_metrics(results_dir, "sweep")

    if args.json:
        payload = {
            "run_dir": str(results_dir),
            "subcommand": kind,
            "parameters": cfg,
            "llm": meta.get("llm"),
            "run_scope_metrics": scoped,
            "sweep_scope_metrics": swept,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(render_config(cfg, results_dir / "config.json", kind))
    if meta.get("llm") or any(name in scoped for name in SCOPE_LABELS):
        print(render_llm(meta, scoped))
    if swept:
        print(render_sweep_scope(swept))
    return 0


if __name__ == "__main__":
    sys.exit(main())

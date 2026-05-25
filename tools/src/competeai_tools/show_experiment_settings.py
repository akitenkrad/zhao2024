"""competeai-tools show-experiment-settings — 実行結果の設定表示．

results/{timestamp}/config.json (run) または
results/{timestamp}_sweep/sweep_config.json (sweep) を読み，実行時に使われた全
パラメータを整形表示する．存在すれば run_metadata.json の LLM 情報
(モデル・endpoint・温度・seed・cache-hit 率・勝者総取り・品質改善) も併せて表示する．
`results/latest` も解決される．

Usage:
    competeai-tools show-experiment-settings
    competeai-tools show-experiment-settings --results-dir results/20260524_153000
    competeai-tools show-experiment-settings --results-dir results/latest --json

I/O・run 設定テーブルは共有ヘルパ `socsim_tools` に委譲する (出力はバイト等価)．
sweep 設定テーブル・`--json` の `kind` フィールド・run_metadata ブロック
(competeai 固有の勝者総取り/品質改善行を含む) は本モジュールに残す．
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socsim_tools.io import load_run_metadata, resolve_results_dir
from socsim_tools.settings import render_run_config

# config キー → 表示ラベル (右コロン位置を揃えるため空白パディング済み)．
# render_run_config が `f"{label}: {value}"` で整形するため，ラベルは末尾の
# `: ` を含めず，従来の run レンダラと同じ桁揃えになるようパディングする．
FIELD_LABELS = {
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
    "seed": "シード (コア)    ",
    "llm_temperature": "LLM 温度         ",
    "llm_seed": "LLM seed         ",
    "output_dir": "出力先           ",
}


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    """config.json (run) か sweep_config.json (sweep) を探す．"""
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (run) または sweep_config.json (sweep)"
    )


def render_sweep_config(cfg: dict, source: Path) -> str:
    """sweep 設定テーブルを整形する (competeai 固有; リスト項目を `, ` 連結する)．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (sweep)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"店舗数 M         : {', '.join(map(str, cfg.get('n_firms_values', [])))}")
    lines.append(f"顧客数 N         : {', '.join(map(str, cfg.get('n_customers_values', [])))}")
    lines.append(f"顧客構成         : {cfg.get('customer_mode', '-')}")
    lines.append(f"日数 days        : {cfg.get('days', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_run_metadata(meta: dict) -> str:
    """LLM 実行メタデータブロックを整形する．

    competeai は勝者総取り・品質改善の 2 行を含むため共有ヘルパ
    `socsim_tools.settings.render_run_metadata` ではなく本ローカル実装を使う
    (従来出力とバイト等価)．
    """
    lines: list[str] = []
    lines.append("")
    lines.append("LLM 実行メタデータ (run_metadata.json)")
    lines.append("-" * 70)
    lines.append(f"モデル           : {meta.get('llm_model', '-')}")
    lines.append(f"endpoint         : {meta.get('llm_endpoint', '-')}")
    lines.append(f"温度             : {meta.get('llm_temperature', '-')}")
    lines.append(f"seed             : {meta.get('llm_seed', '-')}")
    lines.append(f"呼び出し総数     : {meta.get('total_calls', '-')}")
    lines.append(f"cache-hit        : {meta.get('cache_hits', '-')}")
    rate = meta.get("cache_hit_rate")
    if rate is not None:
        lines.append(f"cache-hit 率     : {rate * 100:.1f}%")
    lines.append(f"勝者総取り       : {meta.get('winner_take_all', '-')}")
    lines.append(f"品質改善         : {meta.get('quality_improved', '-')}")
    note = meta.get("determinism_note")
    if note:
        lines.append("-" * 70)
        lines.append(f"注記: {note}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="competeai-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default="results/latest",
        help="実行結果ディレクトリ (default: results/latest)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path, kind = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    with cfg_path.open() as f:
        cfg = json.load(f)
    meta = load_run_metadata(results_dir)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "run_metadata": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path, FIELD_LABELS))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_run_metadata(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())

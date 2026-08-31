#!/usr/bin/env python3
"""reproduce_paper.py — Zhao et al. (2024) CompeteAI 論文 Table 2 発生頻度の一括再現レポート + 図．

Rust の `competeai reproduce` が書く sweep 親 run を読む．条件をまたいだ観測値は
親の `metrics.csv` の scope=sweep 行に，論文の報告値は `reference.csv` にある．
条件別の試行は `lineage.parent_run_uid` で親を指す子 run で，代表 run の時系列は
そこから採る．3 つの図を出す:

    1. occurrence_frequency.png
       個人客 / グループ客 の «勝者総取り» と «品質改善» の発生頻度を棒グラフで対比．
       論文 Table 2 の «個人 66.7% → グループ 16.7%» (グループ化による勝者総取りの
       緩和) と «品質改善 86.67%» を一目で示す．
    2. matthew_effect.png
       条件別の最終収益 Gini・最終最大市場シェアの棒グラフ．マタイ効果 (市場集中) が
       個人客で強く，グループ客で弱まることを示す．
    3. share_trajectory.png
       代表 run (replicate 0) の最大市場シェア時系列を個人客 vs グループ客で重ね描き．

観測値と論文値の差は出すが，PASS/OFF の «帯» はここでは判定しない — 帯は論文の主張
ではなくこの再現実装が置いたものなので `reference.csv` に載らず，同じ閾値を Python と
Rust の 2 箇所に置くと食い違う余地ができる．帯つきの判定は `competeai reproduce` の
コンソール出力にある．向きの主張 (個人 > グループ) は閾値が要らないのでここでも見る．

`--run` を付けると先に Rust バイナリ (`cargo run --release -- reproduce`) を実行して
最新結果を生成する．サンドボックス・CI では `--mock` も付けてライブ LLM を回避する．

Usage:
    uv run competeai-tools reproduce --run --mock          # mock で一括再現 + 図
    uv run competeai-tools reproduce --run --mock --quick  # 軽量版 (動作確認用)
    uv run competeai-tools reproduce                        # 既存の直近 reproduce を可視化
    uv run competeai-tools reproduce --results-dir "$(runvault path --experiment competeai --latest --subcommand reproduce)"
    uv run competeai-tools reproduce --json

Outputs:
    <results-root>/competeai/figures/<run_slug>/{occurrence_frequency,matthew_effect,share_trajectory}.png
    stdout: 条件別の発生頻度と，論文値との差．
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from runvault.read import (
    config_parameters,
    figures_dir,
    load_run_meta,
    metrics_wide,
    runvault_path,
    sweep_children,
)

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "competeai"

# --------------------------------------------------------------------------- #
# 表示設定 (CJK フォントが利用不能でも落ちないように try)
# --------------------------------------------------------------------------- #
try:
    plt.rcParams["font.family"] = "Hiragino Sans"
except Exception:  # pragma: no cover - フォント未インストール環境用フォールバック
    pass

COLOR_BG = "#FAFAF8"
COLOR_INDIVIDUAL = "#2196F3"
COLOR_GROUP = "#FF9800"
COLOR_WTA = "#F44336"
COLOR_QUALITY = "#4CAF50"

MODES = ("individual", "group")


# --------------------------------------------------------------------------- #
# Rust バイナリ実行
# --------------------------------------------------------------------------- #


def _run_binary(*, mock: bool, quick: bool, seed: int, output_dir: str) -> None:
    """`cargo run --release -- reproduce ...` を実行して最新結果を生成する．"""
    cmd = ["cargo", "run", "--release", "--", "reproduce", "--seed", str(seed),
           "--output-dir", output_dir]
    if mock:
        cmd.append("--mock")
    if quick:
        cmd.append("--quick")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# 親 run の読み取り
# --------------------------------------------------------------------------- #


def sweep_scope_metrics(parent_dir: str) -> dict[str, float]:
    """親の `metrics.csv` の scope=sweep 行 (条件をまたいだ集約)．"""
    path = Path(parent_dir) / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"metrics.csv が見つかりません: {path}\n"
            f"  先に `competeai-tools reproduce --run --mock` を実行してください．"
        )
    out: dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["scope"] == "sweep" and row["step"] == "":
                out[row["name"]] = float(row["value"])
    return out


def paper_values(parent_dir: str) -> list[dict]:
    """`reference.csv` の行 (論文が報告した値だけ)．"""
    path = Path(parent_dir) / "reference.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return [
            {"name": r["name"], "value": float(r["value"]),
             "target_id": r["target_id"], "source": r["source"]}
            for r in csv.DictReader(f)
        ]


def cells(parent_dir: str, scoped: dict[str, float]) -> list[dict]:
    """条件別の集約を «移行前の cells» と同じ形に組み直す．"""
    params = config_parameters(parent_dir) or {}
    runs = {"individual": params.get("individual_runs"), "group": params.get("group_runs")}
    out = []
    for mode in MODES:
        if f"wta_freq_{mode}" not in scoped:
            continue
        out.append({
            "customer_mode": mode,
            "runs": runs.get(mode),
            "wta_freq": scoped[f"wta_freq_{mode}"],
            "quality_freq": scoped[f"quality_freq_{mode}"],
            "mean_menu_similarity": scoped[f"menu_similarity_{mode}"],
            "mean_final_gini": scoped[f"final_gini_{mode}"],
            "mean_final_share_max": scoped[f"final_share_max_{mode}"],
        })
    return out


def representative_children(parent_dir: str) -> dict[str, str]:
    """条件ごとの代表 run (replicate 0) のディレクトリ．"""
    out: dict[str, str] = {}
    for child in sweep_children(parent_dir):
        params = config_parameters(child) or {}
        rng = (load_run_meta(child) or {}).get("rng") or {}
        if rng.get("replicate_index") == 0:
            out[params.get("customer_mode")] = child
    return out


def _cell(cell_rows: list[dict], mode: str) -> dict | None:
    for c in cell_rows:
        if c["customer_mode"] == mode:
            return c
    return None


# --------------------------------------------------------------------------- #
# 描画
# --------------------------------------------------------------------------- #


def _occurrence_frequency(cell_rows: list[dict], out_path: Path) -> None:
    """個人客 / グループ客 の勝者総取り・品質改善 発生頻度の棒グラフ．"""
    indiv = _cell(cell_rows, "individual")
    group = _cell(cell_rows, "group")
    if indiv is None or group is None:
        print("  警告: 条件が不足しているため occurrence_frequency をスキップ")
        return

    metrics = ["勝者総取り (WTA)", "品質改善"]
    x = np.arange(len(metrics))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.bar(x - w / 2, [indiv["wta_freq"] * 100, indiv["quality_freq"] * 100], w,
           color=COLOR_INDIVIDUAL, label="個人客 (individual)")
    ax.bar(x + w / 2, [group["wta_freq"] * 100, group["quality_freq"] * 100], w,
           color=COLOR_GROUP, label="グループ客 (group)")
    # 論文 Table 2 の参照ライン．
    ax.axhline(66.7, color=COLOR_INDIVIDUAL, lw=0.8, ls="--", alpha=0.6)
    ax.axhline(16.7, color=COLOR_GROUP, lw=0.8, ls="--", alpha=0.6)
    ax.axhline(86.67, color=COLOR_QUALITY, lw=0.8, ls=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("発生頻度 (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Zhao et al. (2024) Table 2 — 発生頻度 (破線=論文値 個人66.7%/群16.7%, 点線=品質86.67%)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _matthew_effect(cell_rows: list[dict], out_path: Path) -> None:
    """条件別の最終収益 Gini・最大市場シェアの棒グラフ (マタイ効果の強度)．"""
    if not cell_rows:
        print("  警告: 条件が無いため matthew_effect をスキップ")
        return
    labels = [c["customer_mode"] for c in cell_rows]
    colors = [COLOR_INDIVIDUAL if m == "individual" else COLOR_GROUP for m in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=COLOR_BG)
    fig.suptitle("Zhao et al. (2024) — マタイ効果 (市場集中) の強度", fontsize=13)

    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x, [c["mean_final_gini"] for c in cell_rows], color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("最終収益 Gini")
    ax.set_title("収益不平等 (高いほど集中)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x, [c["mean_final_share_max"] for c in cell_rows], color=colors, alpha=0.9)
    ax.axhline(0.8, color="#888888", lw=0.8, ls="--", label="WTA 閾値 0.8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("最終最大市場シェア")
    ax.set_ylim(0, 1.05)
    ax.set_title("市場シェア集中", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _share_trajectory(children: dict[str, str], out_path: Path) -> None:
    """個人客 vs グループ客 の最大市場シェア時系列 (代表 run)．"""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    styles = {
        "individual": ("個人客 (individual)", COLOR_INDIVIDUAL, "-"),
        "group": ("グループ客 (group)", COLOR_GROUP, "--"),
    }
    plotted = 0
    for mode in MODES:
        child = children.get(mode)
        if child is None:
            continue
        legend, color, ls = styles[mode]
        # 最大市場シェアは全店同値の日次集計なので metrics.csv 側にある．
        df = metrics_wide(os.path.join(child, "metrics.csv"))
        ax.plot(df["step"], df["market_share_max"], color=color, ls=ls, lw=2, label=legend)
        plotted += 1
    if plotted == 0:
        print("  警告: 代表 run が無いため share_trajectory をスキップ")
        plt.close(fig)
        return
    ax.axhline(0.8, color="#888888", lw=0.8, ls=":", label="WTA 閾値 0.8")
    ax.set_xlabel("時刻 (日)")
    ax.set_ylabel("最大市場シェア")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "勝者総取りの時系列 (代表 run)\n個別客は同調で独占へ / グループ客は熟議で市場が割れる",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


# --------------------------------------------------------------------------- #
# レポート出力
# --------------------------------------------------------------------------- #


def _print_report(cell_rows: list[dict], scoped: dict[str, float],
                  papers: list[dict], results_dir: str) -> None:
    print("=" * 78)
    print("Zhao et al. (2024) CompeteAI — 論文 Table 2 発生頻度 一括再現レポート")
    print(f"  source: {results_dir}")
    print("=" * 78)

    print("\n[顧客構成別 発生頻度]")
    print(f"  {'mode':<12}{'runs':>5}{'WTA':>10}{'quality':>12}{'menu_sim':>10}{'Gini':>9}")
    for c in cell_rows:
        print(f"  {c['customer_mode']:<12}{c['runs']:>5}"
              f"{c['wta_freq'] * 100:>9.1f}%{c['quality_freq'] * 100:>11.1f}%"
              f"{c['mean_menu_similarity']:>10.3f}{c['mean_final_gini']:>9.3f}")

    print("\n[観測 vs 論文の報告値 (reference.csv)]")
    for pv in papers:
        obs = scoped.get(pv["name"])
        if obs is None:
            continue
        print(f"  {pv['name']:<28} obs={obs:.4f} paper={pv['value']:.4f} "
              f"diff={obs - pv['value']:+.4f}")
        print(f"  {'':<28} 出典: {pv['source']}")
    gap = scoped.get("wta_freq_gap_individual_minus_group")
    if gap is not None:
        status = "PASS" if gap > 0 else "OFF "
        print(f"\n  [{status}] 個人 > グループ (グループ化が勝者総取りを緩和): "
              f"差 = {gap:+.4f}")
    print("-" * 78)
    print("(中核知見: 個別客は同調で勝者総取り / グループ客は熟議で緩和 / 競争のみで品質改善)")
    print("帯つきの PASS/OFF 判定は `competeai reproduce` のコンソール出力にある — "
          "帯は論文の主張ではないので記録しない．")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="competeai-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None,
                        help="reproduce の親 run (省略時は runvault path が返す直近の reproduce)")
    parser.add_argument("--results-root", "--results_root", default="results",
                        help="runvault の results ルート (default: results)")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (既定: <results-root>/competeai/figures/<run_slug>)")
    parser.add_argument("--run", action="store_true",
                        help="先に Rust バイナリ (reproduce) を実行する．")
    parser.add_argument("--mock", action="store_true",
                        help="--run 時にライブ LLM を使わず mock で駆動する．")
    parser.add_argument("--quick", action="store_true",
                        help="--run 時に軽量モードで実行する (動作確認用)．")
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument("--json", action="store_true", help="JSON 形式で要約を出力する．")
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(mock=args.mock, quick=args.quick, seed=args.seed,
                    output_dir=args.results_root)

    results_dir = args.results_dir or runvault_path(
        EXPERIMENT,
        results_root=args.results_root,
        subcommand="reproduce",
    )
    try:
        scoped = sweep_scope_metrics(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    cell_rows = cells(results_dir, scoped)
    papers = paper_values(results_dir)

    if args.json:
        payload = {
            "run_dir": results_dir,
            "cells": cell_rows,
            "sweep_scope_metrics": scoped,
            "paper_values": papers,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    _print_report(cell_rows, scoped, papers, results_dir)

    out_dir = Path(args.output_dir) if args.output_dir else Path(figures_dir(results_dir))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[図] 出力先: {out_dir}")
    _occurrence_frequency(cell_rows, out_dir / "occurrence_frequency.png")
    _matthew_effect(cell_rows, out_dir / "matthew_effect.png")
    _share_trajectory(representative_children(results_dir), out_dir / "share_trajectory.png")

    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

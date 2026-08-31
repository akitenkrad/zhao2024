#!/usr/bin/env python3
"""
visualize.py — Zhao et al. (2024) CompeteAI 市場競争シミュレーション 可視化スクリプト

runvault の run ディレクトリを読み，
(1) 市場シェア推移 (店舗別の当日客数シェア)，
(2) 収益 Gini 推移 (マタイ効果)，
(3) 店舗別の平均料理スコア推移 (品質改善)，
(4) メニュー類似度推移 (差別化と模倣の動的均衡)
の 4 図 (2×2) を生成する．

店舗ごとの系列 (客数・料理スコア) は `events.jsonl` の `observation` から採る —
«日 × 店舗» のパネルは runvault の `metrics.csv` には載らないからである．全店で同値の
日次集計 (Gini・メニュー類似度) は `metrics.csv` にある．

`--results-dir` を省略すると
`runvault path --experiment competeai --latest --subcommand run --standalone`
が返す run ディレクトリを対象にする (`runvault` が PATH にある必要がある)．
`--standalone` を付けるのは，掃引の子 run も subcommand=run だからである．

図は run の外 (`<results-root>/competeai/figures/<run_slug>/`) に出す．run が終わった
後に作るものは `manifest.csv` に載らないので，run ディレクトリの中には置かない．

Usage:
    uv run competeai-tools visualize
    uv run competeai-tools visualize --results-dir "$(runvault path --experiment competeai --latest --subcommand run --standalone)"
    uv run competeai-tools visualize --output-dir out

Outputs:
    output_dir/
    └── competition_dynamics.png   ← 市場シェア・収益 Gini・料理スコア・メニュー類似度
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from runvault.read import events_table, figures_dir, metrics_wide, runvault_path

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "competeai"

# --------------------------------------------------------------------------- #
# 日本語フォント設定
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

# --------------------------------------------------------------------------- #
# カラー設定
# --------------------------------------------------------------------------- #
COLOR_BG = "#FAFAF8"
COLOR_GINI = "#9C27B0"
COLOR_SIM = "#FF9800"
FIRM_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def load_panel(run_dir: str) -> pd.DataFrame:
    """«日 × 店舗» のパネルを `events.jsonl` の observation から読み込む．

    列名は移行前の long-format `metrics.csv` と同じ (`firm` / `day_customers` /
    `avg_dish_score` …) で，日の列だけ runvault の予約語 `t` になっている．
    """
    df = events_table(run_dir, kind="observation")
    return df.rename(columns={"t": "day"})


def load_daily(run_dir: str) -> pd.DataFrame:
    """全店で同値の日次集計 (Gini・メニュー類似度) を `metrics.csv` から読み込む．"""
    return metrics_wide(os.path.join(run_dir, "metrics.csv")).rename(columns={"step": "day"})


def _firm_color(idx: int) -> str:
    return FIRM_COLORS[idx % len(FIRM_COLORS)]


def save_competition_dynamics(df: pd.DataFrame, daily: pd.DataFrame, out_path: str) -> None:
    """市場シェア・収益 Gini・料理スコア・メニュー類似度の推移図を保存する (2×2)．"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=COLOR_BG)
    fig.suptitle("Zhao et al. (2024) CompeteAI — 競争動態", fontsize=14)

    days = sorted(df["day"].unique())
    firms = sorted(df["firm"].unique())

    # --- (0,0) 市場シェア推移 ---
    ax = axes[0, 0]
    ax.set_facecolor(COLOR_BG)
    # 各日の総客数で正規化したシェア．
    total_by_day = df.groupby("day")["day_customers"].sum()
    for i, firm in enumerate(firms):
        sub = df[df["firm"] == firm].set_index("day").reindex(days)
        share = sub["day_customers"] / total_by_day.reindex(days).replace(0, float("nan"))
        ax.plot(days, share.values * 100.0, color=_firm_color(i), lw=2, marker="o", ms=3,
                label=f"店舗 {firm}")
    ax.axhline(80.0, color="#888888", lw=0.8, linestyle="--")
    ax.set_ylim(0, 100)
    ax.set_xlabel("日 d")
    ax.set_ylabel("市場シェア (%)")
    ax.set_title("市場シェア推移 (80% 線 = 勝者総取り閾値)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- (0,1) 収益 Gini 推移 (日次集計なので metrics.csv 側から取る) ---
    ax = axes[0, 1]
    ax.set_facecolor(COLOR_BG)
    gini_by_day = daily.set_index("day")["revenue_gini"].reindex(days)
    ax.plot(days, gini_by_day.values, color=COLOR_GINI, lw=2, marker="o", ms=3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("日 d")
    ax.set_ylabel("収益 Gini")
    ax.set_title("収益 Gini 推移 (上昇 = マタイ効果)")
    ax.grid(True, alpha=0.3)

    # --- (1,0) 店舗別 平均料理スコア推移 ---
    ax = axes[1, 0]
    ax.set_facecolor(COLOR_BG)
    for i, firm in enumerate(firms):
        sub = df[df["firm"] == firm].set_index("day").reindex(days)
        ax.plot(days, sub["avg_dish_score"].values, color=_firm_color(i), lw=2, marker="o", ms=3,
                label=f"店舗 {firm}")
    ax.set_xlabel("日 d")
    ax.set_ylabel("平均料理スコア s")
    ax.set_title("料理スコア推移 (上昇 = 品質改善)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- (1,1) メニュー類似度推移 ---
    ax = axes[1, 1]
    ax.set_facecolor(COLOR_BG)
    sim_by_day = daily.set_index("day")["menu_similarity"].reindex(days)
    ax.plot(days, sim_by_day.values, color=COLOR_SIM, lw=2, marker="o", ms=3)
    ax.axhline(0.36, color="#888888", lw=0.8, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("日 d")
    ax.set_ylabel("メニュー類似度")
    ax.set_title("メニュー類似度 (論文 約36% 動的均衡を破線で参照)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="competeai-tools visualize",
        description="Zhao et al. (2024) CompeteAI 競争動態 可視化スクリプト",
    )
    p.add_argument(
        "--results-dir",
        "--results_dir",
        default=None,
        help="run ディレクトリ (省略時は runvault path が返す直近の run)",
    )
    p.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="runvault の results ルート (default: results)",
    )
    p.add_argument(
        "--output-dir",
        "--output_dir",
        default=None,
        help="図の保存先ディレクトリ (default: <results-root>/competeai/figures/<run_slug>)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    results_dir = args.results_dir or runvault_path(
        EXPERIMENT,
        results_root=args.results_root,
        subcommand="run",
        standalone=True,
    )
    out_dir = args.output_dir or figures_dir(results_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=== Zhao et al. (2024) CompeteAI 競争動態 可視化 ===")
    print(f"run:    {results_dir}")
    print(f"出力先: {out_dir}")
    print("-----------------------------------------")

    df = load_panel(results_dir)
    daily = load_daily(results_dir)
    n_days = df["day"].nunique()
    n_firms = df["firm"].nunique()
    print(f"      {n_days} 日 × {n_firms} 店舗の日次パネル")
    print("[1/1] 競争動態図 (市場シェア・Gini・スコア・メニュー類似度) を保存中 ...")
    save_competition_dynamics(df, daily, os.path.join(out_dir, "competition_dynamics.png"))

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()

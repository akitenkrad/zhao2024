#!/usr/bin/env python3
"""
visualize_sweep.py — Zhao et al. (2024) CompeteAI スイープ結果 可視化スクリプト

results/latest (または --sweep_dir 指定先) の sweep_summary.csv を読み，
店舗数 M × 顧客数 N の格子について，勝者総取り発生頻度・最終収益 Gini・
品質改善頻度を集計し，棒グラフ/ヒートマップで可視化する (マタイ効果の創発条件)．

Usage:
    uv run competeai-tools visualize-sweep
    uv run competeai-tools visualize-sweep --sweep_dir results/20260524_160000_sweep

Outputs:
    output_dir/
    ├── sweep_wta_by_nfirms.png    ← 店舗数別の勝者総取り頻度 / 最終 Gini
    └── sweep_gini_heatmap.png     ← 最終 Gini (M × N) ヒートマップ
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"


def load_summary(sweep_dir: str) -> pd.DataFrame:
    """sweep_summary.csv を読み込む．"""
    path = os.path.join(sweep_dir, "sweep_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sweep_summary.csv が見つかりません: {path}")
    return pd.read_csv(path)


def save_wta_by_nfirms(df: pd.DataFrame, out_path: str) -> None:
    """店舗数 M 別の勝者総取り発生頻度・最終 Gini を棒グラフで比較する．"""
    n_firms_vals = sorted(df["n_firms"].unique())
    wta_freq = [df[df["n_firms"] == m]["winner_take_all"].mean() * 100.0 for m in n_firms_vals]
    gini_mean = [df[df["n_firms"] == m]["final_revenue_gini"].mean() for m in n_firms_vals]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=COLOR_BG)
    labels = [str(m) for m in n_firms_vals]

    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.bar(labels, wta_freq, color="#F44336", alpha=0.85)
    for i, v in enumerate(wta_freq):
        ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("店舗数 M")
    ax.set_ylabel("勝者総取り発生頻度 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("店舗数↑ → 勝者総取り↓ (シェア分散)")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    ax.bar(labels, gini_mean, color="#9C27B0", alpha=0.85)
    for i, v in enumerate(gini_mean):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("店舗数 M")
    ax.set_ylabel("最終収益 Gini (平均)")
    ax.set_ylim(0, 1)
    ax.set_title("店舗数別の最終収益 Gini (マタイ効果)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_gini_heatmap(df: pd.DataFrame, out_path: str) -> None:
    """最終収益 Gini を (M × N) ヒートマップで可視化する．"""
    agg = df.groupby(["n_firms", "n_customers"])["final_revenue_gini"].mean().reset_index()
    table = agg.pivot(index="n_firms", columns="n_customers", values="final_revenue_gini")
    table = table.sort_index()

    fig, ax = plt.subplots(
        figsize=(1.6 + 1.4 * table.shape[1], 1.4 + 0.9 * table.shape[0]),
        facecolor=COLOR_BG,
    )
    ax.set_facecolor(COLOR_BG)
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([str(m) for m in table.index])
    ax.set_xlabel("顧客数 N")
    ax.set_ylabel("店舗数 M")
    ax.set_title("最終収益 Gini (店舗数 × 顧客数)", fontsize=12)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="competeai-tools visualize-sweep",
        description="Zhao et al. (2024) CompeteAI スイープ結果 可視化スクリプト",
    )
    p.add_argument(
        "--sweep_dir",
        "--sweep-dir",
        default="results/latest",
        help="スイープ出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {sweep_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_dir = args.output_dir if args.output_dir else os.path.join(args.sweep_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Zhao et al. (2024) CompeteAI スイープ可視化 ===")
    print(f"スイープ: {args.sweep_dir}")
    print(f"出力先:   {out_dir}")
    print("-------------------------------------------------")

    print("[1/2] sweep_summary.csv を読み込み中 ...")
    df = load_summary(args.sweep_dir)
    print(
        f"      M {df['n_firms'].nunique()} 種 × N {df['n_customers'].nunique()} 種 "
        f"(計 {len(df)} 実行)"
    )

    print("[2/2] 店舗数別の勝者総取り頻度 / Gini 図を保存中 ...")
    save_wta_by_nfirms(df, os.path.join(out_dir, "sweep_wta_by_nfirms.png"))

    if df["n_customers"].nunique() > 1 and df["n_firms"].nunique() > 1:
        save_gini_heatmap(df, os.path.join(out_dir, "sweep_gini_heatmap.png"))
    else:
        print("      M または N が単一のため Gini ヒートマップはスキップ")

    print("-------------------------------------------------")
    print("店舗数別の勝者総取り発生頻度 (マタイ効果の創発条件):")
    for m in sorted(df["n_firms"].unique()):
        freq = df[df["n_firms"] == m]["winner_take_all"].mean() * 100.0
        gini = df[df["n_firms"] == m]["final_revenue_gini"].mean()
        print(f"  M={m} → WTA = {freq:.1f}% | Ginī = {gini:.3f}")

    print("-------------------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""reproduce_paper.py — Zhao et al. (2024) CompeteAI 論文 Table 2 発生頻度の一括再現レポート + 図．

Rust の `competeai reproduce` が書き出す `reproduce_summary.json` (顧客構成別セル・
論文 Table 2 アンカー) と条件別 `metrics_<mode>.csv` を読み，論文のマクロ的知見を
3 つの図で可視化しつつ PASS/off テーブルを表示する:

    1. occurrence_frequency.png
       個人客 / グループ客 の «勝者総取り» と «品質改善» の発生頻度を棒グラフで対比．
       論文 Table 2 の «個人 66.7% → グループ 16.7%» (グループ化による勝者総取りの
       緩和) と «品質改善 86.67%» を一目で示す．
    2. matthew_effect.png
       条件別の最終収益 Gini・最終最大市場シェアの棒グラフ．マタイ効果 (市場集中) が
       個人客で強く，グループ客で弱まることを示す．
    3. share_trajectory.png
       代表 run の最大市場シェア時系列を個人客 vs グループ客で重ね描き．個人客は
       初期優位が正のフィードバックで増幅し独占へ向かう一方，グループ客は熟議で
       市場が割れて独占に至りにくいことを時系列で対比する．

`--run` を付けると先に Rust バイナリ (`cargo run --release -- reproduce`) を実行して
最新結果を生成する．サンドボックス・CI では `--mock` も付けてライブ LLM を回避する．

Usage:
    uv run competeai-tools reproduce --run --mock          # mock で一括再現 + 図
    uv run competeai-tools reproduce --run --mock --quick  # 軽量版 (動作確認用)
    uv run competeai-tools reproduce                        # 既存 results/latest を可視化
    uv run competeai-tools reproduce --results-dir results/reproduce_20260530_000000
    uv run competeai-tools reproduce --json

Outputs:
    {results_dir}/figures/{occurrence_frequency,matthew_effect,share_trajectory}.png
    stdout: アンカーごとの PASS / OFF．
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from socsim_tools.io import resolve_results_dir

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


def _load_summary(results_dir: Path) -> dict:
    path = results_dir / "reproduce_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"reproduce_summary.json が見つかりません: {path}\n"
            f"  先に `competeai-tools reproduce --run --mock` を実行してください．"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _cell(summary: dict, mode: str) -> dict | None:
    for c in summary.get("cells", []):
        if c["customer_mode"] == mode:
            return c
    return None


# --------------------------------------------------------------------------- #
# 描画
# --------------------------------------------------------------------------- #


def _occurrence_frequency(summary: dict, out_path: Path) -> None:
    """個人客 / グループ客 の勝者総取り・品質改善 発生頻度の棒グラフ．"""
    indiv = _cell(summary, "individual")
    group = _cell(summary, "group")
    if indiv is None or group is None:
        print("  警告: cells が不足しているため occurrence_frequency をスキップ")
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


def _matthew_effect(summary: dict, out_path: Path) -> None:
    """条件別の最終収益 Gini・最大市場シェアの棒グラフ (マタイ効果の強度)．"""
    cells = summary.get("cells", [])
    if not cells:
        print("  警告: cells が無いため matthew_effect をスキップ")
        return
    labels = [c["customer_mode"] for c in cells]
    colors = [COLOR_INDIVIDUAL if m == "individual" else COLOR_GROUP for m in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=COLOR_BG)
    fig.suptitle("Zhao et al. (2024) — マタイ効果 (市場集中) の強度", fontsize=13)

    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x, [c["mean_final_gini"] for c in cells], color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("最終収益 Gini")
    ax.set_title("収益不平等 (高いほど集中)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x, [c["mean_final_share_max"] for c in cells], color=colors, alpha=0.9)
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


def _share_trajectory(results_dir: Path, out_path: Path) -> None:
    """個人客 vs グループ客 の最大市場シェア時系列 (代表 run)．"""
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    pairs = [
        ("metrics_individual.csv", "個人客 (individual)", COLOR_INDIVIDUAL, "-"),
        ("metrics_group.csv", "グループ客 (group)", COLOR_GROUP, "--"),
    ]
    plotted = 0
    for fname, legend, color, ls in pairs:
        path = results_dir / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # 1 日 1 値 (集計量は全店同値なので最初の店の行を採る)．
        per_day = df.groupby("day")["market_share_max"].first()
        ax.plot(per_day.index, per_day.values, color=color, ls=ls, lw=2, label=legend)
        plotted += 1
    if plotted == 0:
        print("  警告: metrics_<mode>.csv が無いため share_trajectory をスキップ")
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


def _print_report(summary: dict, results_dir: Path) -> None:
    print("=" * 78)
    print("Zhao et al. (2024) CompeteAI — 論文 Table 2 発生頻度 一括再現レポート")
    print(f"  source: {results_dir}  (mode={summary.get('mode', '?')})")
    print("=" * 78)

    print("\n[顧客構成別 発生頻度]")
    print(f"  {'mode':<12}{'runs':>5}{'WTA':>10}{'quality':>12}{'menu_sim':>10}{'Gini':>9}")
    for c in summary.get("cells", []):
        print(f"  {c['customer_mode']:<12}{c['runs']:>5}"
              f"{c['wta_freq'] * 100:>9.1f}%{c['quality_freq'] * 100:>11.1f}%"
              f"{c['mean_menu_similarity']:>10.3f}{c['mean_final_gini']:>9.3f}")

    print("\n[論文 Table 2 アンカー (観測 vs 論文)]")
    n_pass = 0
    for a in summary.get("anchors", []):
        hi = a["target_hi"]
        hi_str = "∞" if hi is None or hi > 1e30 else f"{hi:.3f}"
        status = "PASS" if a["pass"] else "OFF "
        if a["pass"]:
            n_pass += 1
        print(f"  [{status}] {a['name']:<42} obs={a['observed']:.4f} "
              f"target=[{a['target_lo']:.3f},{hi_str}] paper={a['paper']}")
    print("-" * 78)
    print(f"{n_pass}/{len(summary.get('anchors', []))} アンカーが in-band")
    print("(中核知見: 個別客は同調で勝者総取り / グループ客は熟議で緩和 / 競争のみで品質改善)")


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
                        help="reproduce_summary.json のあるディレクトリ (既定: results/latest)")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (既定: {results_dir}/figures)")
    parser.add_argument("--run", action="store_true",
                        help="先に Rust バイナリ (reproduce) を実行する．")
    parser.add_argument("--mock", action="store_true",
                        help="--run 時にライブ LLM を使わず mock で駆動する．")
    parser.add_argument("--quick", action="store_true",
                        help="--run 時に軽量モードで実行する (動作確認用)．")
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument("--cargo-output-dir", "--cargo_output_dir", default="results",
                        help="--run 時に cargo の --output-dir へ渡すパス (既定: results)．")
    parser.add_argument("--json", action="store_true", help="JSON 形式で要約を出力する．")
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(mock=args.mock, quick=args.quick, seed=args.seed,
                    output_dir=args.cargo_output_dir)

    results_dir = resolve_results_dir(args.results_dir)
    try:
        summary = _load_summary(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    _print_report(summary, results_dir)

    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[図] 出力先: {out_dir}")
    _occurrence_frequency(summary, out_dir / "occurrence_frequency.png")
    _matthew_effect(summary, out_dir / "matthew_effect.png")
    _share_trajectory(results_dir, out_dir / "share_trajectory.png")

    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

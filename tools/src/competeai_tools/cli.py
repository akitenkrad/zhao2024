"""competeai-tools — Zhao et al. (2024) CompeteAI 市場競争シミュレーション ツール統合 CLI．

Usage:
    competeai-tools visualize [...]
    competeai-tools visualize-sweep [...]
    competeai-tools show-experiment-settings [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

`reproduce` (論文 Table 2 の発生頻度一括再現・グループ客深掘り) は Phase 3 で実装予定 (未提供)．
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="competeai-tools",
        description="Zhao et al. (2024) CompeteAI 市場競争シミュレーション 可視化・分析ツール",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "visualize",
        help="単一実行結果 (市場シェア推移・収益 Gini・料理スコア・メニュー類似度) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "visualize-sweep",
        help="スイープ結果 (店舗数 × 顧客数 の勝者総取り頻度・最終 Gini) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "show-experiment-settings",
        help="実行結果ディレクトリの設定 (config / sweep_config / run_metadata) の表示",
        add_help=False,
    )

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return

    command = argv[0]
    rest = argv[1:]
    if command == "visualize":
        from competeai_tools.visualize import main as run_main

        run_main(rest)
    elif command == "visualize-sweep":
        from competeai_tools.visualize_sweep import main as run_main

        run_main(rest)
    elif command == "show-experiment-settings":
        from competeai_tools.show_experiment_settings import main as run_main

        run_main(rest)
    else:
        # 未知のコマンドは argparse のエラーメッセージに委ねる
        parser.parse_args(argv)


if __name__ == "__main__":
    main()

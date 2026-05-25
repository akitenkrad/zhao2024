"""competeai-tools — Zhao et al. (2024) CompeteAI 市場競争シミュレーション ツール統合 CLI．

Usage:
    competeai-tools visualize [...]
    competeai-tools visualize-sweep [...]
    competeai-tools show-experiment-settings [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

`reproduce` (論文 Table 2 の発生頻度一括再現・グループ客深掘り) は Phase 3 で実装予定 (未提供)．

dispatcher の組み立ては共有ヘルパ `socsim_tools.cli.build_dispatcher` に委譲する
(prog 名・サブコマンド・ヘルプ文・argv ルーティングは従来と同一)．可視化/設定表示の
実体 (visualize / visualize_sweep / show_experiment_settings) は repo 固有のまま．
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="competeai-tools",
    description="Zhao et al. (2024) CompeteAI 市場競争シミュレーション 可視化・分析ツール",
    subcommands={
        "visualize": (
            "単一実行結果 (市場シェア推移・収益 Gini・料理スコア・メニュー類似度) の可視化",
            "competeai_tools.visualize:main",
        ),
        "visualize-sweep": (
            "スイープ結果 (店舗数 × 顧客数 の勝者総取り頻度・最終 Gini) の可視化",
            "competeai_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "実行結果ディレクトリの設定 (config / sweep_config / run_metadata) の表示",
            "competeai_tools.show_experiment_settings:main",
        ),
    },
)


if __name__ == "__main__":
    main()

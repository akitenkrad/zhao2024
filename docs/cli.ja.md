[English](cli.md) | [日本語](cli.ja.md)

# CLI

Rust バイナリ `competeai` は `run`・`sweep`・`reproduce` の 3 サブコマンドを持つ．

## LLM 環境変数

LLM レイヤは **Ollama 第一 → OpenAI フォールバック** で，環境変数で設定する (ハードコードしない):

| 変数 | 既定 | 用途 |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | 第一バックエンド |
| `OLLAMA_MODEL` | `llama3.2:latest` | 第一バックエンド |
| `OPENAI_API_KEY` | (未設定 → フォールバック無効) | フォールバック |
| `OPENAI_MODEL` | `gpt-4o-mini` | フォールバック |

プロンプト→応答キャッシュ (既定 `.llm_cache/cache.json`) が再実行を擬似決定論化する: ウォームキャッシュは同一応答を再生するので再実行は無料である．

## `run`

LLM 駆動の市場競争 ABM を単一設定で実行する．

```bash
cargo run --release -- run \
    --n-firms 2 --n-customers 50 --customer-mode individual \
    --days 15 --runs 9 --seed 42
```

| フラグ | 既定 | 意味 |
|---|---|---|
| `--n-firms` | 2 | 店舗数 M |
| `--n-customers` | 50 | 顧客数 N |
| `--customer-mode` | `individual` | `individual` / `group` |
| `--group-size` | 4 | 1 グループ人数 (group モードのみ) |
| `--days` | 15 | 日数 (ラウンド数) |
| `--runs` | 1 | 独立試行数 (各試行は派生 seed); 最後の試行の詳細を保存 |
| `--seed` | (ランダム) | socsim コアシード |
| `--llm-temperature` | 0.0 | 生成温度 |
| `--llm-seed` | 0 | バックエンド生成シード |
| `--cache-path` | `.llm_cache/cache.json` | プロンプト→応答キャッシュファイル |
| `--mock` | off | 決定論的 scripted mock でオフライン駆動 (ライブ LLM 不要) |
| `--output-dir` | `results` | runvault の results ルート |

[runvault](https://github.com/akitenkrad/rs-runvault) の run 1 本を `<output-dir>/competeai/run_<timestamp>_<config_hash>_<execution_hash>/` へ書く: `config.json` (条件; `parameters` 配下)，`run.json` (`llm` ブロック — モデル/endpoint/温度 — を含む)，`metrics.csv` (日次集計 `revenue_gini`/`market_share_max`/`menu_similarity`/`n_alive_firms` と，run スコープの値 `llm_calls`/`llm_cache_hits`/`llm_cache_hit_rate`/`winner_take_all`/`quality_improved`)，`events.jsonl` (店舗ごとの `observation` を全日ぶん + 各店舗の帰趨を記す `terminal` を 1 行ずつ)，`status.json`，`manifest.csv`．`--runs > 1` のときディスクに記録するのは最後の試行だけ (上の `--runs` の注記を参照) だが，コンソールには全試行横断の勝者総取り頻度・品質改善頻度を表示する．

## `sweep`

店舗数 × 顧客数 を走査しマタイ効果指標を集計する．

```bash
cargo run --release -- sweep \
    --n-firms-values 2,3,4 \
    --n-customers-min 20 --n-customers-max 80 --n-customers-step 20 \
    --days 15 --runs 5 --seed 42
```

| フラグ | 既定 | 意味 |
|---|---|---|
| `--n-firms-values` | `2,3,4` | カンマ区切り店舗数 |
| `--n-customers-min/max/step` | 20 / 80 / 20 | 顧客数レンジ |
| `--customer-mode` | `individual` | `individual` / `group` |
| `--days` | 15 | 日数 |
| `--runs` | 5 | セルあたり試行数 (各派生 seed) |
| `--seed` | 42 | 基点シード (セル/試行が派生) |
| `--cache-path` | `.llm_cache/cache.json` | 共有キャッシュ (セル横断でヒット率向上) |
| `--output-dir` | `results` | runvault の results ルート |

runvault の **親** run を `<output-dir>/competeai/sweep_<timestamp>_<config_hash>_<execution_hash>/` へ書く．`config.json` の `parameters` が掃引の格子 (`n_firms_values`, `n_customers_values`, …) を持ち，親自体はセルごとの指標を持たない．(店舗数 × 顧客数 × 試行) の各セルはそれぞれ独立の **子** run (上記 `run` と同じレイアウト) で，`lineage.parent_run_uid` で親につながる．`sweep_summary.csv` はディスクには無い — `uv run competeai-tools visualize-sweep` が子 run から «1 行 1 (セル×試行)» の表をその都度組み直す ([可視化](visualization.ja.md) を参照)．

## `reproduce`

論文 Table 2 の発生頻度を一括再現する: 個人客条件とグループ客条件をそれぞれ複数回の独立試行で実行し，観測された勝者総取り / 品質改善 / メニュー類似度の発生頻度を論文と突き合わせる．

```bash
# オフライン (scripted mock) — サンドボックス・CI で使う
cargo run --release -- reproduce --mock --seed 42
# ライブ LLM (ビルドして Ollama 起動後)
cargo run --release -- reproduce --seed 42
```

| フラグ | 既定 | 意味 |
|---|---|---|
| `--n-firms` | 2 | 店舗数 M |
| `--n-customers` | 50 | 顧客数 N |
| `--group-size` | 4 | 1 グループ人数 (group 条件) |
| `--days` | 15 | 日数 |
| `--individual-runs` | 9 | 個人客条件の独立試行数 (論文 Table 2 = 9) |
| `--group-runs` | 6 | グループ客条件の独立試行数 (論文 Table 2 = 6) |
| `--seed` | 42 | 基点シード (条件/試行が派生) |
| `--mock` | off | 決定論的 scripted mock でオフライン駆動 (ライブ LLM 不要) |
| `--llm-temperature` | 0.0 | 生成温度 (live 時のみ) |
| `--llm-seed` | 0 | バックエンド生成シード (live 時のみ) |
| `--cache-path` | `.llm_cache/cache.json` | 共有キャッシュ (live 時のみ) |
| `--quick` | off | N / 試行数 / 日数を縮小した高速スモーク (論文値検証には使わない) |
| `--output-dir` | `results` | runvault の results ルート |

runvault の **親** run を `<output-dir>/competeai/reproduce_<timestamp>_<config_hash>_<execution_hash>/` へ書き，試行ごとの **子** run (個人客・グループ客) を `lineage.parent_run_uid` でつなぐ．親の `metrics.csv` は条件をまたいだ集約を `scope=sweep` 指標として持つ — `wta_freq_individual`，`wta_freq_group`，`quality_freq_individual`，`quality_freq_group`，`quality_freq_all`，`menu_similarity_individual`，`menu_similarity_group`，`menu_similarity_all`，`final_gini_individual`，`final_gini_group`，`final_share_max_individual`，`final_share_max_group`，方向性を持つ `wta_freq_gap_individual_minus_group`．親の `reference.csv` には論文 Table 2 / §4 の報告値そのもの (個人客の勝者総取り 66.7%・グループ客の勝者総取り 16.7%・品質改善 86.67%・メニュー類似度 約36%; 各行に `source` 付き) が入る．±15pt / ±10pt の合否バンドは論文の主張ではなく本再現実装が置いたものなので，どのファイルにも **記録しない** — `competeai reproduce` のコンソール出力にのみ現れる．図は親の指標と `reference.csv` から Python `competeai-tools reproduce` が描く．

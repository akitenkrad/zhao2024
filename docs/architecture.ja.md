[English](architecture.md) | [日本語](architecture.ja.md)

# アーキテクチャ

## リポジトリ構成

```
replications/zhao2024/
├── Cargo.toml                  # Rust workspace (members = ["simulation"])
├── pyproject.toml              # uv workspace (members = ["tools"])
├── simulation/                 # Rust crate `competeai-simulation` (bin `competeai`)
│   ├── Cargo.toml              # socsim-core + socsim-engine + socsim-llm (features=["live"])
│   ├── examples/mock_smoke.rs  # オフライン (ライブ LLM 不要) パイプラインスモーク
│   ├── src/
│   │   ├── main.rs             # clap: run / sweep
│   │   ├── lib.rs
│   │   ├── config.rs           # Config, CustomerMode, LLM 設定, seed 派生
│   │   ├── world.rs            # MarketWorld (WorldState), Firm, Customer, Dish, Market
│   │   ├── mechanisms.rs       # 6-phase 上の 5 メカニズム
│   │   ├── llm.rs              # socsim-llm ビルダ (Ollama→OpenAI + cache)
│   │   ├── prompts.rs          # 店舗戦略 / 顧客選択プロンプト + 応答パース
│   │   ├── metrics.rs          # 収益 Gini / 市場シェア / WTA / 料理スコア / メニュー類似度
│   │   └── simulation.rs       # init_world + run ドライバ + 出力ライタ
│   └── tests/integration_test.rs   # mock 駆動 (ScriptedClient); ライブ LLM 不要
├── tools/                      # Python package `competeai-tools` (module `competeai_tools`)
│   └── src/competeai_tools/
│       ├── cli.py
│       ├── visualize.py        # 市場シェア + 収益 Gini + 料理スコア + メニュー類似度
│       ├── visualize_sweep.py  # 店舗数 × 顧客数 の WTA 頻度 / 最終 Gini
│       └── show_experiment_settings.py
└── docs/                       # bilingual (.md + .ja.md)
```

## 二層決定論

socsim コアは決定論的で LLM を含まず，LLM は本質的に非決定的である．設計は LLM を 2 つのメカニズムに閉じ込めて擬似決定論化する:

| 層 | 構成要素 | 再現性 |
|---|---|---|
| 決定論的 socsim コア | 店舗/顧客初期化 (資金・所得・嗜好・メニュー)・活性化順・グループ多数決の同点処理・顧客−店舗の市場マッチング・収益/原価/資金会計・全指標 | seed が決まれば bit 単位で再現 (ChaCha20 `SimRng`) |
| 非決定的 LLM レイヤ | 店舗戦略の反省 + 顧客選択 (2 つの `Decision` メカニズム) | プロンプト→応答キャッシュ + `temperature=0` + 固定 seed で擬似決定論 |

RNG ストリームは単一 root seed から派生する (schelling1971 / axelrod1997 / li2024 と同一規約): `derive_seed(root, &[0])` が世界初期化 (店舗資金・顧客特性) を，`derive_seed(root, &[1])` が engine (活性化順・グループ同点処理) を初期化する．`&[2]`, `&[3]`, … を追加ストリームに予約する．

## 世界状態

`MarketWorld { clock, firms: BTreeMap<AgentId, Firm>, customers: BTreeMap<AgentId, Customer>, market: Market, day }` が `WorldState` を実装する．空間格子はなく，店舗と顧客は市場を介して相互作用する固定エージェントである．店舗は `AgentId` `[0, n_firms)` を，顧客は `[CUSTOMER_ID_BASE, …)` を占めるので，`kind_of(id)` が単一 `AgentId` 空間を決定論的に分割し，`agent_ids()` は店舗 (昇順) → 顧客 (昇順) を返す．`Dish` は論文の品質スコア `s = 0.5·c/p + 0.5·f/5000` を持つ．

## メカニズム (6-phase 上の 5 個)

同期的な日次ステップ (1 エンジン tick = 1 日) は 6 フェーズを順に走り，同一フェーズ内では宣言順に発火する．

| Mechanism | Phase | 役割 |
|---|---|---|
| `MarketResetMechanism` | Environment | 前日 daybook を scratch に退避してから当日市場をリセット |
| `CompetitionMatthewMechanism` | **Decision** | LLM 店舗が daybook + ライバル情報 + 記憶を反省し価格/シェフ給与/広告を改訂．全店オファーを scratch にスナップショット (**LLM**) |
| `CustomerChoiceMechanism` | **Decision** | LLM 顧客が提示オファーから来店店を選択．グループは多数決 (同点は `ctx.rng`) (**LLM**) |
| `PatronageMechanism` | Interaction | 顧客−店舗マッチング・食事体験・他顧客に可視なコメント生成 |
| `RevenueRewardMechanism` | Reward | 収益/原価/資金・評判更新・日次マタイ効果指標 (収益 Gini・市場シェア集中) |
| `ReflectionMechanism` | PostStep | 各店舗が当日を記憶へ要約・撤退判定 (`funds < 0` で `alive=false`)・店舗撤退 or `day == days-1` で `request_stop` |

LLM クライアントと呼び出しメタデータコレクタは 2 つの `Decision` メカニズムと `Rc<RefCell<…>>` で共有し (li2024 パターン)，run ドライバが実行後にキャッシュ保存・cache-hit 率集計に使う．店舗オファーは店舗 `Decision` 完了時にスナップショットして step スコープの `scratch` 経由で `Interaction` に渡すので，日の途中の状態変化が同一日の他エージェント決定に波及しない．

## 指標

`metrics.csv` は **long-format** で，(日, 店舗) ごとに 1 行である．店舗固有列 (`day_customers`, `day_revenue`, `cumulative_revenue`, `avg_dish_score`, `avg_price`, `reputation`, `firm_alive`) は店舗で異なり，日次集計列 (`revenue_gini`, `market_share_max`, `menu_similarity`, `n_alive_firms`) は同一日の各行で同値である．

| 指標 | 定義 | 論文での対応 |
|---|---|---|
| `revenue_gini` | 店舗累積収益の Gini | マタイ効果 (Table 2) |
| `market_share_max` | 当日の `max_r N_r / Σ_r N_r` | 勝者総取り |
| `winner_take_all` | `[Day6, 最終日]` の全日で最大シェア > 0.8 か (bool) | マクロ分析 (66.7% / 16.7%) |
| `avg_dish_score` | 店舗別 平均料理スコア `s` の日次 | 品質改善 (86.67%) |
| `menu_similarity` | メニュー (料理名集合) の Jaccard | 差別化/模倣 (約36%) |
| `quality_improved` | 少なくとも 1 店の平均スコアが Day1→最終日で上昇 (bool) | 品質改善 |

## socsim / socsim-llm

本クレットは `socsim-core` (`WorldState` / `Mechanism` / `Phase` / `SimClock` / `SimRng`) と `socsim-engine` (`SimulationBuilder`, `RandomActivationScheduler`, `run_observed`)，および `features = ["live"]` の `socsim-llm` (Ollama + OpenAI バックエンドを `FallbackClient` で束ねる) のみに依存する．本番クライアント型は `CachingClient<Box<dyn LlmClient>>` で，`FallbackClient<OllamaClient, OpenAiClient>` を `socsim-llm` の `impl LlmClient for Box<T>` (issue #26) で `Box<dyn LlmClient>` に型消去する．専用 newtype は不要で，同じ `CompeteClient` がテストで `mock::ScriptedClient` を受け取れる．git 依存は `Cargo.lock` で具体 commit に固定する．

> 設計書 (§4.2/§7) は当初 `reqwest` + `sha2` を挙げていたが，本スイートは li2024 / chuang2024 と統一して `socsim-llm` に標準化することで上書きした．`socsim-llm` が HTTP と `hash(prompt+model)` キャッシュキーを所有するため，本クレットに `reqwest` / `sha2` は現れない．

## 参考文献

- Zhao, Q., Wang, J., Zhang, Y., Jin, Y., Zhu, K., Chen, H., & Xie, X. (2024). CompeteAI: Understanding the Competition Dynamics of Large Language Model-based Agents. *ICML 2024*, PMLR 235, 61092–61107. arXiv:2310.17512.
- Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*. (仮想タウン設計の基盤)
- Rigney, D. (2010). *The Matthew Effect: How Advantage Begets Further Advantage*. Columbia University Press.
- socsim: [rs-social-simulation-tools](https://github.com/akitenkrad/rs-social-simulation-tools) (`socsim-llm` は issue #21/#26)．

---
*This file was generated by Claude Code.*

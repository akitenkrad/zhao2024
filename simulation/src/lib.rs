//! Zhao et al. (2024) "CompeteAI: Understanding the Competition Dynamics of
//! Large Language Model-based Agents" (ICML 2024) の再現実装ライブラリ．
//!
//! socsim フレームワーク上に構築した LLM 駆動の市場競争 ABM (2 レストラン × 50
//! 顧客 × 15 日の仮想タウン) の公開 API を提供する．設定 (`config`)・世界状態
//! (`world`)・LLM クライアント層 (`llm`)・プロンプト生成と応答パース
//! (`prompts`)・更新メカニズム (`mechanisms`)・実行ドライバ (`simulation`)・
//! 集計メトリクス (`metrics`) をモジュールとして公開し，バイナリ (`competeai`)
//! と統合テストの双方から利用する．
//!
//! # 二層決定論
//!
//! socsim コア層 (店舗/顧客初期化・活性化順・グループ多数決の同点処理・市場
//! マッチング・収益/Gini/シェアの指標) は seed から bit 単位で決定論的である．
//! LLM レイヤ (店舗戦略立案・顧客選択) は socsim の bit 再現性の **外側** にあり，
//! `socsim-llm` のキャッシュ + `temperature=0` + `seed` 固定で擬似決定論化する．
//! 詳細は `crate::llm` を参照．設計書 §4.2/§7 は当初 `reqwest` + `sha2` を挙げて
//! いたが，本スイートは li2024 / chuang2024 と統一して `socsim-llm` (issue
//! #21/#26) に標準化したため `reqwest` / `sha2` は使わない (socsim-llm が HTTP と
//! プロンプトハッシュを所有する)．

pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod prompts;
pub mod simulation;
pub mod world;

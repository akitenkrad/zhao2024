//! LLM クライアント層 (Ollama 第一 → OpenAI フォールバック + キャッシュ)．
//!
//! 本モジュールは `socsim-llm` の合成 API に対する薄いビルダである．二層
//! アーキテクチャの **上層 (非決定的 LLM レイヤ)** をここに閉じ込め，下層の
//! 決定論的 socsim コアからは [`CompeteClient`] 型エイリアス経由でのみ触れる．
//!
//! # 合成 (Ollama 第一 → OpenAI フォールバック → キャッシュ)
//!
//! ```text
//! CachingClient< Box<dyn LlmClient> >
//!   └─ cache: PromptCache (prompt → response; 擬似決定論の本体)
//!      └─ backend (型消去): FallbackClient< OllamaClient, OpenAiClient >
//!         primary:   OllamaClient   (OLLAMA_HOST / OLLAMA_MODEL)
//!         secondary: OpenAiClient   (OPENAI_API_KEY / OPENAI_MODEL)
//! ```
//!
//! `FallbackClient` は socsim-llm が提供する (自前実装しない)．「Ollama を試行
//! → 任意のエラーで OpenAI へフォールバック」を担う．`CachingClient` はその上に
//! プロンプト→応答キャッシュを被せ，`temperature=0` / `seed` 固定と合わせて
//! 再実行を擬似決定論化する．socsim-llm がプロンプトハッシュ (cache key) を
//! 所有するため `sha2` は不要．キャッシュは features=["live"] 経由で有効化される．
//!
//! 設計書 §4.2/§7 は当初 `reqwest` + `sha2` + 手書き `llm.rs` を挙げていたが，
//! 本スイートは li2024 / chuang2024 と統一して `socsim-llm` クレット (issue
//! #21/#26) に標準化したため `reqwest` / `sha2` は使わず，HTTP とハッシュは
//! socsim-llm が所有する．
//!
//! テストでは `socsim-llm::mock::ScriptedClient` を `Box<dyn LlmClient>` として
//! 同じ [`CompeteClient`] に流し込める．`socsim-llm` が `Box<dyn LlmClient>` に
//! 対する [`LlmClient`] の転送実装を提供する (issue #26) ため，専用 newtype は
//! 不要である．

use std::path::Path;

use socsim_llm::{CachingClient, LlmClient, LlmConfig, LlmError, PromptCache};

use crate::config::LlmSettings;

/// 本シミュレーションが用いるキャッシュ付きクライアント型．
///
/// バックエンドは `Box<dyn LlmClient>` に型消去してあり，本番は
/// `FallbackClient<OllamaClient, OpenAiClient>`，テストは `ScriptedClient` を
/// 注入できる．`socsim-llm` の `impl LlmClient for Box<T>` (issue #26) により
/// 専用 newtype なしで `CachingClient` の `C: LlmClient` 境界を満たす．
pub type CompeteClient = CachingClient<Box<dyn LlmClient>>;

/// 本番用の «Ollama 第一 → OpenAI フォールバック + キャッシュ» クライアントを
/// 環境変数から構築する．
///
/// - Ollama: `OLLAMA_HOST` (既定 `http://localhost:11434`) / `OLLAMA_MODEL`
///   (既定 `llama3.2:latest`)．
/// - OpenAI: `OPENAI_API_KEY` / `OPENAI_MODEL` (既定 `gpt-4o-mini`; 原論文は
///   `gpt-4`)．未設定なら空キーのフォールバックを置く (Ollama が成功すれば
///   呼ばれない; 両方失敗時のみ設定エラーになる)．
/// - キャッシュ: `settings.cache_path` があればその JSON ファイル，なければ
///   in-memory．
pub fn build_live_client(settings: &LlmSettings) -> Result<CompeteClient, LlmError> {
    // 既定モデルを llama3.2:latest にする (OLLAMA_MODEL 未設定時)．socsim-llm の
    // OllamaClient::from_env 既定は llama3.1 のため，委譲前に明示的に上書きする
    // (run_metadata に記録される model を従来どおり保つため)．
    if std::env::var("OLLAMA_MODEL").is_err() {
        std::env::set_var("OLLAMA_MODEL", "llama3.2:latest");
    }
    // «Ollama 第一 → OpenAI フォールバック → 型消去 → キャッシュ» の組み立ては
    // socsim-llm の `build_live_client` に委譲する (挙動は従来の手書き実装と等価)．
    // 本ラッパは replication 固有の `LlmSettings` (cache_path) と上記 OLLAMA_MODEL
    // 既定だけを受け持つ薄い層である．
    socsim_llm::build_live_client(settings.cache_path.as_deref().map(Path::new))
}

/// 任意の [`LlmClient`] (例: `mock::ScriptedClient`) をキャッシュで包んだ
/// [`CompeteClient`] を作る (主にテスト用)．
pub fn wrap_client<C: LlmClient + 'static>(backend: C, cache: PromptCache) -> CompeteClient {
    let boxed: Box<dyn LlmClient> = Box::new(backend);
    CachingClient::new(boxed, cache)
}

/// [`LlmSettings`] から socsim-llm の [`LlmConfig`] を組み立てる．
pub fn llm_config(settings: &LlmSettings) -> LlmConfig {
    LlmConfig::deterministic()
        .with_temperature(settings.temperature)
        .with_seed(settings.seed)
}

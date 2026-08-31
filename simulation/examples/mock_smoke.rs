//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (run ディレクトリ・metrics.csv / events.jsonl / config.json) と
//! Python 可視化を検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient` で
//! 決定論的に店舗戦略・顧客選択を駆動し，本番 `run` と同じ runvault の run へ書く．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;

use runvault::{Run, RunOptions};

use competeai_simulation::config::Config;
use competeai_simulation::llm::wrap_client;
use competeai_simulation::record::{self, DOMAIN, EXPERIMENT, REPO_ID};
use competeai_simulation::simulation::run_with_client;
use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, PromptCache};

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());

    let cfg = Config {
        n_firms: 2,
        n_customers: 6,
        days: 3,
        seed: Some(42),
        ..Config::default()
    };

    // 店舗戦略プロンプト (price_factor を含む) には «高めの品質投資» 戦略を，
    // 顧客選択プロンプトには «品質の高い Option 0 を選ぶ» 擬似挙動を返す．これで
    // 市場シェアが偏り，マタイ効果指標 (Gini・最大シェア) に動きが出る．
    let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
        if prompt.contains("price_factor") {
            // 自店が好調なら据え置き，不調なら値下げ + シェフ給与増の擬似戦略．
            if prompt.contains("Yesterday: 0 customers") {
                "{\"price_factor\": 0.95, \"chef_salary\": 2600, \"advertisement\": \"Now cheaper and tastier!\"}"
                    .to_string()
            } else {
                "{\"price_factor\": 1.02, \"chef_salary\": 2400, \"advertisement\": \"Customer favorite!\"}"
                    .to_string()
            }
        } else {
            // 顧客は Option 0 を選好する (初期優位 → 正のフィードバック)．
            "{\"choice\": 0}".to_string()
        }
    });
    let client = wrap_client(backend, PromptCache::in_memory());

    // クライアントは run を開始する前に組む (`llm` ブロックのため)．
    let llm = record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        cfg.llm.temperature,
    );
    let parameters = cfg.to_run_config_json();
    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&base)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(cfg.seed.expect("mock smoke は seed を固定する"))
            .llm(llm)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    let result = run_with_client(&cfg, client).expect("mock run failed");
    record::log_simulation(&mut rv, &result);
    let dir = rv.finish().expect("runvault: run の完了に失敗");

    let last = result.metrics_history.last().unwrap();
    println!("mock smoke wrote: {}", dir.display());
    println!(
        "final day={} revenue_gini={:.3} market_share_max={:.3} menu_similarity={:.3} WTA={} quality_improved={}",
        result.final_day,
        last.revenue_gini,
        last.market_share_max,
        last.menu_similarity,
        result.winner_take_all,
        result.quality_improved,
    );
}

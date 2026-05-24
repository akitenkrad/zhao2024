//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (metrics.csv / run_metadata.json / config.json) と Python
//! 可視化を検証するための補助バイナリ．`socsim-llm::mock::ScriptedClient` で
//! 決定論的に店舗戦略・顧客選択を駆動し，本番 `run` と同じ writer で結果を書き出す．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;
use std::fs;

use chrono::Local;

use competeai_simulation::config::Config;
use competeai_simulation::llm::wrap_client;
use competeai_simulation::simulation::{
    ensure_output_dir, run_with_client, save_metrics, save_run_metadata,
};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = format!("{base}/{timestamp}");

    let cfg = Config {
        n_firms: 2,
        n_customers: 6,
        days: 3,
        seed: Some(42),
        output_dir: output_dir.clone(),
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

    ensure_output_dir(&cfg.output_dir);
    let result = run_with_client(&cfg, client).expect("mock run failed");
    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json
    let cfg_path = format!("{}/config.json", cfg.output_dir);
    let f = fs::File::create(&cfg_path).unwrap();
    serde_json::to_writer_pretty(f, &cfg.to_run_config_json()).unwrap();

    // latest symlink
    let link = format!("{base}/latest");
    let _ = fs::remove_file(&link);
    #[cfg(unix)]
    let _ = std::os::unix::fs::symlink(&timestamp, &link);

    let last = result.metrics_history.last().unwrap();
    println!("mock smoke wrote: {output_dir}");
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

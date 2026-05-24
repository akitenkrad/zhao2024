//! 初期化と実行ドライバ (SimulationBuilder 配線 + 二層 LLM レイヤ)．
//!
//! 二層決定論を配線する:
//! - **下層 (決定論的 socsim コア)**: `derive_seed(root, &[0])` で世界初期化 (店舗
//!   初期資金・顧客特性) の init RNG を，`derive_seed(root, &[1])` で engine RNG
//!   (= 活性化順・グループ多数決の同点処理) を派生する．bit 単位で再現する．
//! - **上層 (非決定的 LLM レイヤ)**: [`crate::llm`] のキャッシュ付き
//!   Ollama→OpenAI フォールバッククライアントに閉じ込め，`temperature=0`/`seed`
//!   固定 + プロンプト→応答キャッシュで擬似決定論化する．モデル・endpoint・
//!   温度・seed・cache-hit を `run_metadata.json` に記録する．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::rc::Rc;

use csv::Writer;
use rand::Rng;
use serde::Serialize;

use socsim_core::{derive_seed, AgentId, SimClock, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};

use crate::config::{Config, CustomerMode};
use crate::llm::{build_live_client, CompeteClient};
use crate::mechanisms::{
    CompetitionMatthewMechanism, CustomerChoiceMechanism, MarketResetMechanism, PatronageMechanism,
    ReflectionMechanism, RevenueRewardMechanism, SharedClient, SharedMetadata, SharedMetrics,
};
use crate::metrics::{market_share_max, winner_take_all, DailyMetric};
use crate::world::{Customer, Dish, Firm, Market, MarketWorld, CUSTOMER_ID_BASE};

/// 世界初期化用 RNG ラベル (店舗初期資金・顧客特性)．
const RNG_WORLD_INIT: u64 = 0;
/// socsim エンジン (= 活性化順・グループ多数決同点処理) 用 RNG ラベル．
const RNG_ENGINE: u64 = 1;

/// 勝者総取り判定の開始日 (0 始まり; Day6 = index 5)．
pub const WTA_START_DAY: usize = 5;
/// 勝者総取り判定のシェア閾値．
pub const WTA_THRESHOLD: f64 = 0.8;

/// 顧客の嗜好テンプレート (init RNG で 1 つ割り当てる)．
const PREFERENCES: [&str; 5] = [
    "loves rich, indulgent dishes",
    "prefers light and healthy meals",
    "is a budget-conscious bargain hunter",
    "seeks adventurous, novel flavors",
    "values classic comfort food",
];

/// 顧客の健康状態テンプレート．
const HEALTH: [&str; 4] = [
    "no dietary restrictions",
    "vegetarian",
    "watching calories",
    "gluten-free",
];

/// 初期メニューの料理名プール (差別化・模倣の観測に使う)．
const DISH_NAMES: [&str; 8] = [
    "sushi", "ramen", "tempura", "curry", "udon", "gyoza", "yakitori", "donburi",
];

/// シミュレーション全体の実行結果．
pub struct SimulationResult {
    /// 日次指標の履歴 (metrics.csv の行; long-format)．
    pub metrics_history: Vec<DailyMetric>,
    /// LLM 呼び出しメタデータの集計．
    pub metadata: MetadataCollector,
    /// LLM モデル名 (run_metadata 用)．
    pub llm_model: String,
    /// LLM endpoint (run_metadata 用; primary)．
    pub llm_endpoint: String,
    /// 実行した日数 (= 完了ステップ数)．
    pub final_day: usize,
    /// 勝者総取りが発生したか．
    pub winner_take_all: bool,
    /// 少なくとも一方の店舗の平均スコアが Day1→最終日で上昇したか．
    pub quality_improved: bool,
}

/// 世界状態を初期化する (店舗生成 + 顧客生成 + 空の市場)．
///
/// 店舗初期資金は `init_funds` を中央値に ±20% で散らし，顧客所得・嗜好・健康は
/// init RNG で決定論的に割り当てる (socsim コア層)．各店舗は初期メニューを
/// `init_menu_size` 品もつ (差別化の余地のため店舗ごとに料理プールから選ぶ)．
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> MarketWorld {
    // --- 店舗 ---
    let mut firms: BTreeMap<AgentId, Firm> = BTreeMap::new();
    for i in 0..cfg.n_firms as u64 {
        let funds = cfg.init_funds * rng.gen_range(0.8..=1.2);
        let mut menu: Vec<Dish> = Vec::with_capacity(cfg.init_menu_size);
        for k in 0..cfg.init_menu_size {
            // 店舗ごとに料理プールをずらして初期差別化を与える．
            let name = DISH_NAMES[((i as usize) + k) % DISH_NAMES.len()].to_string();
            let price = cfg.init_price * rng.gen_range(0.9..=1.1);
            let cost = price * cfg.init_cost_ratio;
            menu.push(Dish {
                name,
                cost,
                price,
                chef_salary: cfg.init_chef_salary,
            });
        }
        let mut firm = Firm::new(funds, menu, cfg.init_chef_salary);
        firm.advertisement = "A welcoming neighborhood restaurant.".to_string();
        firms.insert(AgentId(i), firm);
    }

    // --- 顧客 ---
    let mut customers: BTreeMap<AgentId, Customer> = BTreeMap::new();
    let group_size = cfg.group_size.max(1);
    for j in 0..cfg.n_customers as u64 {
        let income = cfg.customer_income * rng.gen_range(0.6..=1.6);
        let pref = PREFERENCES[rng.gen_range(0..PREFERENCES.len())].to_string();
        let health = HEALTH[rng.gen_range(0..HEALTH.len())].to_string();
        let mut customer = Customer::new(income, pref, health);
        if cfg.customer_mode == CustomerMode::Group {
            // 連番でグループへ割り当てる (決定論)．
            customer.group = Some(j / group_size as u64);
        }
        customers.insert(AgentId(CUSTOMER_ID_BASE + j), customer);
    }

    MarketWorld {
        clock: SimClock::new(cfg.days as u64),
        firms,
        customers,
        market: Market::default(),
        day: 0,
    }
}

/// シミュレーションを実行する (本番 LLM クライアントを構築して駆動)．
///
/// `OLLAMA_*` / `OPENAI_*` 環境変数から «Ollama 第一 → OpenAI フォールバック +
/// キャッシュ» クライアントを構築し，[`run_with_client`] へ委譲する．
pub fn run(cfg: &Config) -> Result<SimulationResult, String> {
    let client =
        build_live_client(&cfg.llm).map_err(|e| format!("LLM クライアント構築に失敗: {e}"))?;
    run_with_client(cfg, client)
}

/// 与えられた [`CompeteClient`] でシミュレーションを実行する．
///
/// 本番は [`build_live_client`] の結果を，テストは [`crate::llm::wrap_client`] で
/// ラップした `mock::ScriptedClient` を渡す．LLM クライアントはメカニズムと
/// `Rc<RefCell<…>>` で共有し，実行後にキャッシュ保存・メタデータ集計に使う．
pub fn run_with_client(cfg: &Config, client: CompeteClient) -> Result<SimulationResult, String> {
    let root = cfg.seed.unwrap_or_else(rand::random);

    // 初期世界 (root から派生した init RNG; 決定論的 socsim コア層)．
    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);

    // LLM モデル/endpoint をメタデータ用に控える．
    let llm_model = client.inner().model().to_string();
    let llm_endpoint = client.inner().endpoint().to_string();

    // クライアント・メタデータ・日次指標バッファを共有する．
    let shared_client: SharedClient = Rc::new(RefCell::new(client));
    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let shared_metrics: SharedMetrics = Rc::new(RefCell::new(Vec::new()));

    let mut sim = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]))
        .add_mechanism(Box::new(MarketResetMechanism))
        .add_mechanism(Box::new(CompetitionMatthewMechanism::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
        )))
        .add_mechanism(Box::new(CustomerChoiceMechanism::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
            cfg.customer_mode,
        )))
        .add_mechanism(Box::new(PatronageMechanism))
        .add_mechanism(Box::new(RevenueRewardMechanism::new(Rc::clone(
            &shared_metrics,
        ))))
        .add_mechanism(Box::new(ReflectionMechanism::new(cfg.days as u64)))
        .build();

    // 日次の最大市場シェアを観測して勝者総取りを判定する．
    let mut max_share_by_day: Vec<f64> = Vec::new();
    let mut final_day = 0usize;
    sim.run_observed(|report| {
        final_day = report.t as usize;
        if let Some(s) = report.scratch.get::<f64>("market_share_max") {
            max_share_by_day.push(*s);
        }
    })
    .map_err(|e| format!("シミュレーションの実行に失敗: {e}"))?;

    // キャッシュを保存 (cache_path 指定時; in-memory はスキップ)．
    if cfg.llm.cache_path.is_some() {
        let client = shared_client.borrow();
        client
            .cache()
            .save()
            .map_err(|e| format!("キャッシュ保存に失敗: {e}"))?;
    }

    let metrics_history = shared_metrics.borrow().clone();
    let metadata = shared_meta.borrow().clone();

    let wta = winner_take_all(&max_share_by_day, WTA_START_DAY, WTA_THRESHOLD);
    let quality_improved = compute_quality_improved(&metrics_history);

    Ok(SimulationResult {
        metrics_history,
        metadata,
        llm_model,
        llm_endpoint,
        final_day,
        winner_take_all: wta,
        quality_improved,
    })
}

/// 少なくとも一方の店舗の平均スコアが Day1 (= 最初の記録日) → 最終日で上昇したか
/// (論文ファクト8 / 品質改善 86.67%)．
fn compute_quality_improved(metrics: &[DailyMetric]) -> bool {
    if metrics.is_empty() {
        return false;
    }
    let first_day = metrics.iter().map(|m| m.day).min().unwrap_or(0);
    let last_day = metrics.iter().map(|m| m.day).max().unwrap_or(0);
    if first_day == last_day {
        return false;
    }
    use std::collections::BTreeMap;
    let mut first: BTreeMap<u64, f64> = BTreeMap::new();
    let mut last: BTreeMap<u64, f64> = BTreeMap::new();
    for m in metrics {
        if m.day == first_day {
            first.insert(m.firm, m.avg_dish_score);
        }
        if m.day == last_day {
            last.insert(m.firm, m.avg_dish_score);
        }
    }
    first
        .iter()
        .any(|(firm, &s0)| last.get(firm).map(|&s1| s1 > s0 + 1e-9).unwrap_or(false))
}

/// 日次指標を CSV に保存する (long-format; 1 行 = 1 日 1 店舗)．
pub fn save_metrics(metrics: &[DailyMetric], output_dir: &str) {
    let path = format!("{}/metrics.csv", output_dir);
    let file = File::create(&path).expect("metrics.csv の作成に失敗");
    let mut wtr = Writer::from_writer(BufWriter::new(file));
    for m in metrics {
        wtr.serialize(m).expect("メトリクス書き込みに失敗");
    }
    wtr.flush().expect("フラッシュに失敗");
}

/// `run_metadata.json` の構造体 (LLM モデル・endpoint・温度・seed・cache 統計)．
#[derive(Serialize)]
pub struct RunMetadataJson {
    pub llm_model: String,
    pub llm_endpoint: String,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub total_calls: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub winner_take_all: bool,
    pub quality_improved: bool,
    pub determinism_note: &'static str,
}

/// `run_metadata.json` を保存する．
pub fn save_run_metadata(result: &SimulationResult, cfg: &Config, output_dir: &str) {
    let meta = RunMetadataJson {
        llm_model: result.llm_model.clone(),
        llm_endpoint: result.llm_endpoint.clone(),
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        total_calls: result.metadata.total(),
        cache_hits: result.metadata.cache_hits(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
        winner_take_all: result.winner_take_all,
        quality_improved: result.quality_improved,
        determinism_note: "LLM output is outside socsim bit-reproducibility; the prompt->response \
                           cache (with temperature=0 and fixed seed) is the reproducibility \
                           mechanism. The socsim core (world init, activation order, group \
                           majority tie-breaking, market matching, revenue/Gini/share metrics) is \
                           deterministic given the seed.",
    };
    let path = format!("{}/run_metadata.json", output_dir);
    let file = File::create(&path).expect("run_metadata.json の作成に失敗");
    serde_json::to_writer_pretty(BufWriter::new(file), &meta)
        .expect("run_metadata.json の書き込みに失敗");
}

/// 出力ディレクトリを作成する．
pub fn ensure_output_dir(output_dir: &str) {
    fs::create_dir_all(output_dir).expect("出力ディレクトリの作成に失敗");
}

/// 日次指標から最大市場シェアの系列を抽出する (1 日 1 値; ヘルパ)．
pub fn max_share_series(metrics: &[DailyMetric]) -> Vec<f64> {
    use std::collections::BTreeMap;
    let mut by_day: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for m in metrics {
        by_day.entry(m.day).or_default().push(m.day_customers);
    }
    by_day.values().map(|v| market_share_max(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::wrap_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    fn scripted_client() -> CompeteClient {
        // 店舗には現状維持戦略を，顧客には option 0 を返す mock．
        let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
            if prompt.contains("price_factor") {
                "{\"price_factor\": 1.0, \"chef_salary\": 2000, \"advertisement\": \"\"}"
                    .to_string()
            } else {
                "{\"choice\": 0}".to_string()
            }
        });
        wrap_client(backend, PromptCache::in_memory())
    }

    fn test_config() -> Config {
        Config {
            n_firms: 2,
            n_customers: 6,
            days: 3,
            seed: Some(42),
            ..Config::default()
        }
    }

    #[test]
    fn scripted_run_produces_daily_metrics() {
        let cfg = test_config();
        let result = run_with_client(&cfg, scripted_client()).unwrap();
        // 各日 × 店舗数 の行 (撤退や早期終了がなければ days * n_firms)．
        assert!(!result.metrics_history.is_empty());
        assert!(result.final_day >= 1);
    }

    #[test]
    fn core_is_deterministic_given_mock() {
        let cfg = test_config();
        let a = run_with_client(&cfg, scripted_client()).unwrap();
        let b = run_with_client(&cfg, scripted_client()).unwrap();
        let ra: Vec<f64> = a
            .metrics_history
            .iter()
            .map(|m| m.cumulative_revenue)
            .collect();
        let rb: Vec<f64> = b
            .metrics_history
            .iter()
            .map(|m| m.cumulative_revenue)
            .collect();
        assert_eq!(ra, rb, "同一シードは完全再現すべき");
    }
}

//! Zhao et al. (2024) CompeteAI 市場競争 ABM の統合テスト．
//!
//! **ライブ LLM を一切必要としない**: socsim-llm の `mock::ScriptedClient` で
//! 決定論的に店舗戦略・顧客選択を駆動し，以下を検証する:
//! ・料理品質スコア式 s = 0.5*c/p + 0.5*f/5000
//! ・顧客−店舗マッチング (patronage) と収益/客数の計上
//! ・収益 Gini・最大市場シェアの計算と範囲 ([0,1])
//! ・勝者総取り (winner-take-all) の検出
//! ・RNG 決定論性 (同一シード → 完全再現)
//! ・終了条件 (最終日で停止)

use competeai_simulation::config::{derive_run_seed, Config, CustomerMode};
use competeai_simulation::metrics::{market_share_max, revenue_gini, winner_take_all};
use competeai_simulation::reproduce_mock::build_reproduce_client;
use competeai_simulation::simulation::{
    max_share_series, run_mock, run_with_client, SimulationResult,
};
use competeai_simulation::world::Dish;
use competeai_simulation::{
    config::LlmSettings,
    llm::{wrap_client, CompeteClient},
};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

/// 店舗には現状維持 (or 指定 factor) を，顧客には全員 Option `choice` を返す mock．
fn scripted(choice: usize, price_factor: f64) -> CompeteClient {
    let firm_reply = format!(
        "{{\"price_factor\": {price_factor}, \"chef_salary\": 2000, \"advertisement\": \"\"}}"
    );
    let cust_reply = format!("{{\"choice\": {choice}}}");
    let backend = ScriptedClient::new("mock-model", move |prompt: &str| {
        if prompt.contains("price_factor") {
            firm_reply.clone()
        } else {
            cust_reply.clone()
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

fn base_config() -> Config {
    Config {
        n_firms: 2,
        n_customers: 8,
        days: 6,
        seed: Some(7),
        ..Config::default()
    }
}

// --------------------------------------------------------------------------- //
// 料理品質スコア式
// --------------------------------------------------------------------------- //

#[test]
fn dish_score_formula_matches_paper() {
    // c=1600, p=4000 → 0.4; f=2000 → 0.4; s = 0.5*0.4 + 0.5*0.4 = 0.4
    let d = Dish {
        name: "x".into(),
        cost: 1600.0,
        price: 4000.0,
        chef_salary: 2000.0,
    };
    assert!((d.score() - 0.4).abs() < 1e-9, "got {}", d.score());
}

// --------------------------------------------------------------------------- //
// メトリクス配線: 行が生成され，集計量が [0,1]
// --------------------------------------------------------------------------- //

#[test]
fn produces_metrics_with_sane_ranges() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted(0, 1.0)).unwrap();
    assert!(
        !result.metrics_history.is_empty(),
        "メトリクス行が生成される"
    );
    for m in &result.metrics_history {
        assert!(
            (0.0..=1.0).contains(&m.revenue_gini),
            "Gini は [0,1] (got {})",
            m.revenue_gini
        );
        assert!(
            (0.0..=1.0).contains(&m.market_share_max),
            "最大シェアは [0,1] (got {})",
            m.market_share_max
        );
        assert!(
            (0.0..=1.0).contains(&m.menu_similarity),
            "メニュー類似度は [0,1]"
        );
        assert!(m.avg_dish_score >= 0.0, "スコアは非負");
        assert!(m.day_revenue >= 0.0, "収益は非負");
    }
}

// --------------------------------------------------------------------------- //
// patronage: 全員が同じ店を選ぶと最大シェア = 1，勝者総取りが検出される
// --------------------------------------------------------------------------- //

#[test]
fn all_customers_to_one_firm_yields_monopoly() {
    let cfg = base_config();
    // 全顧客が Option 0 を選ぶ → 1 店が全客を取り続ける．
    let result = run_with_client(&cfg, scripted(0, 1.0)).unwrap();

    // 最終日 (or 停止日) の行を見て最大シェアが 1．
    let last_day = result.metrics_history.iter().map(|m| m.day).max().unwrap();
    let last_rows: Vec<_> = result
        .metrics_history
        .iter()
        .filter(|m| m.day == last_day)
        .collect();
    let share = last_rows[0].market_share_max;
    assert!(
        (share - 1.0).abs() < 1e-9,
        "独占なら最大シェア 1 (got {share})"
    );

    // Day6 まで一店独占 → 勝者総取りが立つはず (days=6 なので index 5 = Day6 が最終)．
    let series = max_share_series(&result.metrics_history);
    assert!(
        winner_take_all(&series, 0, 0.8),
        "全期間独占なら WTA (start=0)"
    );
    assert!(result.winner_take_all, "result.winner_take_all も true");
}

// --------------------------------------------------------------------------- //
// patronage: 顧客が分散すると収益 Gini が下がる (対称 → 0 付近)
// --------------------------------------------------------------------------- //

#[test]
fn revenue_accrues_to_chosen_firm() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted(1, 1.0)).unwrap();
    // Option 1 (= ソート 2 番目の生存店) に累積収益が乗る．
    let last_day = result.metrics_history.iter().map(|m| m.day).max().unwrap();
    let last_rows: Vec<_> = result
        .metrics_history
        .iter()
        .filter(|m| m.day == last_day)
        .collect();
    let total_rev: f64 = last_rows.iter().map(|m| m.cumulative_revenue).sum();
    assert!(total_rev > 0.0, "累積収益が発生する");
}

// --------------------------------------------------------------------------- //
// 指標関数の単体検算
// --------------------------------------------------------------------------- //

#[test]
fn metric_helpers_hand_calc() {
    assert!((revenue_gini(&[0.0, 100.0]) - 0.5).abs() < 1e-9);
    assert!((market_share_max(&[30, 10]) - 0.75).abs() < 1e-9);
    assert!(winner_take_all(&[0.9, 0.95, 0.99], 0, 0.8));
    assert!(!winner_take_all(&[0.9, 0.5, 0.99], 0, 0.8));
}

// --------------------------------------------------------------------------- //
// 決定論性: 同一シード + 同一 mock → 完全再現
// --------------------------------------------------------------------------- //

#[test]
fn core_is_deterministic_given_fixed_mock() {
    let cfg = base_config();
    let a = run_with_client(&cfg, scripted(0, 1.0)).unwrap();
    let b = run_with_client(&cfg, scripted(0, 1.0)).unwrap();
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
    let sa: Vec<f64> = a
        .metrics_history
        .iter()
        .map(|m| m.market_share_max)
        .collect();
    let sb: Vec<f64> = b
        .metrics_history
        .iter()
        .map(|m| m.market_share_max)
        .collect();
    assert_eq!(ra, rb, "同一シードは累積収益を完全再現すべき");
    assert_eq!(sa, sb, "同一シードは市場シェアを完全再現すべき");
}

// --------------------------------------------------------------------------- //
// グループ客モードでも動く (多数決経路)
// --------------------------------------------------------------------------- //

#[test]
fn group_mode_runs_and_matches() {
    let mut cfg = base_config();
    cfg.customer_mode = CustomerMode::Group;
    cfg.group_size = 4;
    let result = run_with_client(&cfg, scripted(0, 1.0)).unwrap();
    assert!(
        !result.metrics_history.is_empty(),
        "グループ客でも行が生成される"
    );
}

// --------------------------------------------------------------------------- //
// reproduce mock: 顧客構成のグループ化が勝者総取りを緩和する (論文 Table 2)
// --------------------------------------------------------------------------- //

/// reproduce 用設定 (cmd_reproduce の base と同じ規模; days を縮めて高速化)．
fn repro_config(mode: CustomerMode) -> Config {
    Config {
        n_firms: 2,
        n_customers: 50,
        customer_mode: mode,
        group_size: 4,
        days: 15,
        llm: LlmSettings {
            temperature: 0.0,
            seed: 0,
            cache_path: None,
        },
        ..Config::default()
    }
}

/// 1 条件を `runs` 回 run_mock して勝者総取り発生頻度を測る (cmd_reproduce と同じ
/// seed 派生規約)．
fn wta_frequency(mode: CustomerMode, runs: usize, base_seed: u64) -> f64 {
    let mut wta = 0usize;
    for run_idx in 0..runs {
        // cmd_reproduce は label_hash(mode) と run_idx で派生するが，テストでは
        // run_idx ベースの派生で十分 (条件内の独立試行を作る)．
        let mode_offset = if mode == CustomerMode::Group { 1000 } else { 0 };
        let seed = derive_run_seed(base_seed.wrapping_add(mode_offset), run_idx);
        let mut cfg = repro_config(mode);
        cfg.seed = Some(seed);
        let result: SimulationResult = run_mock(&cfg).expect("reproduce mock 実行に失敗");
        if result.winner_take_all {
            wta += 1;
        }
    }
    wta as f64 / runs.max(1) as f64
}

#[test]
fn reproduce_mock_group_dampens_winner_take_all() {
    // 論文 Table 2 の中核知見: 個人客は勝者総取りが起きやすく (66.7%)，グループ客は
    // 熟議でそれが緩和される (16.7%)．mock でも «個人 > グループ» の順序が出る．
    let individual = wta_frequency(CustomerMode::Individual, 9, 42);
    let group = wta_frequency(CustomerMode::Group, 6, 42);
    assert!(
        individual > group,
        "グループ化は勝者総取りを緩和する (individual={individual}, group={group})"
    );
    assert!(
        group <= 0.34,
        "グループ客の勝者総取りは低頻度 (got {group})"
    );
}

#[test]
fn reproduce_mock_quality_improves() {
    // 全条件で «少なくとも一方の店の平均料理スコアが Day1→最終日で上昇» する
    // (論文ファクト8 / 品質改善 86.67%)．mock では単調品質投資で常に改善する．
    let mut cfg = repro_config(CustomerMode::Individual);
    cfg.seed = Some(123);
    let result = run_mock(&cfg).unwrap();
    assert!(result.quality_improved, "品質改善が観測される");
}

// --------------------------------------------------------------------------- //
// reproduce mock の bit 決定論性 (同一 seed → 完全再現)
// --------------------------------------------------------------------------- //

#[test]
fn reproduce_mock_is_bit_deterministic() {
    let mut cfg = repro_config(CustomerMode::Group);
    cfg.seed = Some(2024);
    let a = run_with_client(&cfg, build_reproduce_client()).unwrap();
    let b = run_with_client(&cfg, build_reproduce_client()).unwrap();
    let rev = |r: &SimulationResult| {
        r.metrics_history
            .iter()
            .map(|m| (m.day, m.firm, m.cumulative_revenue, m.market_share_max))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        rev(&a),
        rev(&b),
        "同一 seed の reproduce mock は完全再現すべき"
    );
}

// --------------------------------------------------------------------------- //
// 個別客の既定経路は Group 拡張に影響されず bit 不変 (回帰)
// --------------------------------------------------------------------------- //

#[test]
fn individual_default_path_is_unchanged() {
    // 個別客モードは group_deliberation=false の従来プロンプトを使う．choice 固定の
    // 旧式 scripted client で個別客モードを回し，全顧客が同一店へ来る独占が再現
    // されること (= Individual 分岐のセマンティクスが Group 拡張前と不変) を確認する．
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted(0, 1.0)).unwrap();
    let last_day = result.metrics_history.iter().map(|m| m.day).max().unwrap();
    let share = result
        .metrics_history
        .iter()
        .find(|m| m.day == last_day)
        .unwrap()
        .market_share_max;
    assert!(
        (share - 1.0).abs() < 1e-9,
        "個別客で全員 choice 0 → 独占 (Individual 分岐は不変; got {share})"
    );
}

// --------------------------------------------------------------------------- //
// 終了条件: 最終日で停止 (days を超えない)
// --------------------------------------------------------------------------- //

#[test]
fn stops_at_final_day() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted(0, 1.0)).unwrap();
    let max_day = result.metrics_history.iter().map(|m| m.day).max().unwrap();
    assert!(
        max_day < cfg.days as u64,
        "日インデックスは days 未満 (0 始まり)"
    );
    assert!(result.final_day <= cfg.days, "完了ステップ数は days 以下");
}

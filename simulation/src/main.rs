//! Zhao et al. (2024) "CompeteAI" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で LLM 駆動の市場競争 ABM を実行する (`--mock` でオフライン)．
//! `sweep`     : 店舗数 × 顧客数 (× 顧客構成) を走査し，マタイ効果指標 (収益 Gini・
//!               最大市場シェア・勝者総取り・品質改善) を `sweep_summary.csv` に集計する．
//! `reproduce` : 論文 Table 2 の発生頻度 (個人客/グループ客の勝者総取り・品質改善・
//!               メニュー類似度) を一括再現し，観測 vs 論文の PASS/off を
//!               `reproduce_summary.json` に集計する (`--mock` でオフライン scripted 駆動)．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use competeai_simulation::config::{parse_customer_mode, Config, CustomerMode, LlmSettings};
use competeai_simulation::metrics::mean;
use competeai_simulation::simulation::{
    ensure_output_dir, run, run_mock, save_metrics, save_run_metadata, SimulationResult,
};

// ---------------------------------------------------------------------------
// CLI 定義
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "competeai",
    about = "Zhao et al. (2024) CompeteAI: Competition Dynamics of LLM-based Agents — 再現実験"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一設定で LLM 駆動の市場競争 ABM を実行する．
    Run(RunArgs),
    /// 店舗数 × 顧客数 を走査し，マタイ効果指標を集計する．
    Sweep(SweepArgs),
    /// 論文 Table 2 の発生頻度 (勝者総取り・品質改善・メニュー類似度) を一括再現する．
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// 店舗数 M．
    #[arg(long, default_value_t = 2)]
    n_firms: usize,

    /// 顧客数 N．
    #[arg(long, default_value_t = 50)]
    n_customers: usize,

    /// 顧客構成 (individual / group)．
    #[arg(long, default_value = "individual")]
    customer_mode: String,

    /// グループ客のときの 1 グループ人数．
    #[arg(long, default_value_t = 4)]
    group_size: usize,

    /// シミュレーション日数 (ラウンド数; 論文標準 15)．
    #[arg(long, default_value_t = 15)]
    days: usize,

    /// 独立試行数 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 1)]
    runs: usize,

    /// 乱数シード (省略時はランダム; socsim コア層のみ支配)．
    #[arg(long)]
    seed: Option<u64>,

    /// LLM 生成温度 (既定 0.0)．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード (バックエンドへ渡す)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (既定 .llm_cache/cache.json)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// LLM を呼ばず決定論的 scripted mock で駆動する (オフライン検証用)．
    /// サンドボックス・CI では `--mock` を付ける (ライブ LLM 不要)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// カンマ区切りの店舗数リスト．
    #[arg(long, default_value = "2,3,4")]
    n_firms_values: String,

    /// 顧客数の最小値．
    #[arg(long, default_value_t = 20)]
    n_customers_min: usize,

    /// 顧客数の最大値．
    #[arg(long, default_value_t = 80)]
    n_customers_max: usize,

    /// 顧客数の刻み幅．
    #[arg(long, default_value_t = 20)]
    n_customers_step: usize,

    /// 顧客構成 (individual / group)．
    #[arg(long, default_value = "individual")]
    customer_mode: String,

    /// シミュレーション日数．
    #[arg(long, default_value_t = 15)]
    days: usize,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 5)]
    runs: usize,

    /// 乱数シード基点 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (sweep 全体で共有しヒット率を高める)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// 店舗数 M (論文標準 2)．
    #[arg(long, default_value_t = 2)]
    n_firms: usize,

    /// 顧客数 N (論文標準 50)．
    #[arg(long, default_value_t = 50)]
    n_customers: usize,

    /// グループ客のときの 1 グループ人数．
    #[arg(long, default_value_t = 4)]
    group_size: usize,

    /// シミュレーション日数 (論文標準 15)．
    #[arg(long, default_value_t = 15)]
    days: usize,

    /// 個人客の独立試行数 (論文 Table 2 = 9 ラン)．
    #[arg(long, default_value_t = 9)]
    individual_runs: usize,

    /// グループ客の独立試行数 (論文 Table 2 = 6 ラン)．
    #[arg(long, default_value_t = 6)]
    group_runs: usize,

    /// 乱数シード基点 (各条件・試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM を呼ばず決定論的 scripted mock で駆動する (オフライン検証用)．
    /// サンドボックス・CI では `--mock` を付ける (ライブ LLM 不要)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// LLM 生成温度 (live 時のみ)．
    #[arg(long, default_value_t = 0.0)]
    llm_temperature: f32,

    /// LLM 生成シード (live 時のみ)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (live 時のみ; 全条件で共有)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 軽量モード (N と試行数と日数を縮小; 動作確認用)．
    #[arg(long, default_value_t = false)]
    quick: bool,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

/// `sweep_summary.csv` の 1 行 (条件ごとのマタイ効果指標)．
#[derive(serde::Serialize)]
struct SweepRow {
    n_firms: usize,
    n_customers: usize,
    customer_mode: String,
    run: usize,
    seed: u64,
    final_day: usize,
    /// 最終日の店舗累積収益 Gini．
    final_revenue_gini: f64,
    /// 最終日の最大市場シェア．
    final_market_share_max: f64,
    /// 勝者総取りが発生したか (0/1)．
    winner_take_all: u8,
    /// 品質改善があったか (0/1)．
    quality_improved: u8,
    /// 最終日のメニュー類似度．
    final_menu_similarity: f64,
    /// 最終日の生存店舗数．
    final_alive_firms: u64,
    cache_hit_rate: f64,
}

/// `sweep_config.json` の構造体．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    command: &'static str,
    n_firms_values: Vec<usize>,
    n_customers_values: Vec<usize>,
    customer_mode: String,
    days: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// カンマ区切り文字列を trim 済みの非空リストへ．
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// 顧客数列を [min, max] step 刻みで生成する．
fn n_customers_range(min: usize, max: usize, step: usize) -> Vec<usize> {
    if step == 0 || max < min {
        return vec![min];
    }
    let mut out = Vec::new();
    let mut n = min;
    while n <= max {
        out.push(n);
        n += step;
    }
    out
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let customer_mode =
        parse_customer_mode(&args.customer_mode).unwrap_or_else(|e| panic!("{}", e));

    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    ensure_output_dir(&output_dir);

    println!("=== Zhao et al. (2024) CompeteAI 市場競争 再現実験 ===");
    println!(
        "M (店舗): {} | N (顧客): {} | 構成: {} | 日数: {} | 試行: {}",
        args.n_firms,
        args.n_customers,
        customer_mode.label(),
        args.days,
        args.runs,
    );
    println!(
        "LLM: temp={} llm_seed={} cache={} | seed: {:?}{}",
        args.llm_temperature,
        args.llm_seed,
        args.cache_path,
        args.seed,
        if args.mock { " | MOCK" } else { "" },
    );
    println!("出力先: {}", output_dir);
    println!("-------------------------------------------------");

    let base_seed = args.seed.unwrap_or(42);
    let mut last_result: Option<SimulationResult> = None;
    let mut wta_count = 0usize;
    let mut quality_count = 0usize;

    for run_idx in 0..args.runs.max(1) {
        // 試行ごとに独立シードを派生する．
        let seed = competeai_simulation::config::derive_run_seed(base_seed, run_idx);
        let cfg = Config {
            n_firms: args.n_firms,
            n_customers: args.n_customers,
            customer_mode,
            group_size: args.group_size,
            days: args.days,
            seed: Some(seed),
            llm: LlmSettings {
                temperature: args.llm_temperature,
                seed: args.llm_seed,
                cache_path: Some(args.cache_path.clone()),
            },
            output_dir: output_dir.clone(),
            ..Config::default()
        };

        let result = if args.mock {
            run_mock(&cfg).unwrap_or_else(|e| panic!("mock 実行に失敗: {}", e))
        } else {
            run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e))
        };
        if result.winner_take_all {
            wta_count += 1;
        }
        if result.quality_improved {
            quality_count += 1;
        }

        // 最後の試行の詳細を保存する (代表 run)．
        if run_idx + 1 == args.runs.max(1) {
            save_metrics(&result.metrics_history, &output_dir);
            save_run_metadata(&result, &cfg, &output_dir);
            // config.json (pretty-print JSON; socsim_results::write_json に委譲)．
            let path = format!("{}/config.json", output_dir);
            write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");
            last_result = Some(result);
        }
    }

    // latest シンボリックリンクを再作成する (best-effort; 従来同様エラーは無視)．
    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

    let runs = args.runs.max(1);
    println!(
        "勝者総取り発生: {}/{} ({:.1}%) | 品質改善: {}/{} ({:.1}%)",
        wta_count,
        runs,
        100.0 * wta_count as f64 / runs as f64,
        quality_count,
        runs,
        100.0 * quality_count as f64 / runs as f64,
    );
    if let Some(result) = &last_result {
        if let Some(last) = result.metrics_history.last() {
            println!(
                "最終日 Gini: {:.3} | 最大シェア: {:.3} | メニュー類似度: {:.3} | 生存店: {}",
                last.revenue_gini, last.market_share_max, last.menu_similarity, last.n_alive_firms
            );
        }
        println!(
            "LLM 呼び出し: {} 回 | cache-hit: {} ({:.1}%) | model: {}",
            result.metadata.total(),
            result.metadata.cache_hits(),
            result.metadata.cache_hit_rate() * 100.0,
            result.llm_model,
        );
    }
    println!("メトリクス → {}/metrics.csv", output_dir);
    println!("LLM メタ   → {}/run_metadata.json", output_dir);
    println!("設定       → {}/config.json", output_dir);
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(args: SweepArgs) {
    let customer_mode: CustomerMode =
        parse_customer_mode(&args.customer_mode).unwrap_or_else(|e| panic!("{}", e));
    let n_firms_values: Vec<usize> = split_csv(&args.n_firms_values)
        .iter()
        .map(|s| {
            s.parse::<usize>()
                .unwrap_or_else(|_| panic!("不正な n_firms: {s}"))
        })
        .collect();
    let n_customers_values = n_customers_range(
        args.n_customers_min,
        args.n_customers_max,
        args.n_customers_step,
    );

    let timestamp = timestamp();
    let sweep_dir = format!("{}/{}_sweep", args.output_dir, timestamp);
    fs::create_dir_all(&sweep_dir).expect("sweep ディレクトリの作成に失敗");
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = n_firms_values.len() * n_customers_values.len() * args.runs;

    println!("=== Zhao et al. (2024) CompeteAI パラメータスイープ ===");
    println!(
        "M: {} 種 | N: {} 種 | 構成: {} | 試行: {} | 合計: {} 実行",
        n_firms_values.len(),
        n_customers_values.len(),
        customer_mode.label(),
        args.runs,
        n_total,
    );
    println!("出力先: {}", sweep_dir);
    println!("-----------------------------------------------------------");

    let mut summary_rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    for &n_firms in &n_firms_values {
        for &n_customers in &n_customers_values {
            for run_idx in 0..args.runs {
                let seed = socsim_core_derive(args.seed, n_firms, n_customers, run_idx);
                let cfg = Config {
                    n_firms,
                    n_customers,
                    customer_mode,
                    days: args.days,
                    seed: Some(seed),
                    llm: LlmSettings {
                        temperature: args.llm_temperature,
                        seed: args.llm_seed,
                        cache_path: Some(args.cache_path.clone()),
                    },
                    output_dir: sweep_dir.clone(),
                    ..Config::default()
                };

                let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));
                summary_rows.push(summarize(
                    &result,
                    n_firms,
                    n_customers,
                    customer_mode,
                    run_idx,
                    seed,
                ));
                done += 1;
            }
            println!(
                "[{}/{}] M={} N={} 完了 ({} 試行)",
                done, n_total, n_firms, n_customers, args.runs,
            );
        }
    }

    // sweep_summary.csv (各行を serialize; socsim_results::write_csv に委譲)．
    {
        let path = format!("{}/sweep_summary.csv", sweep_dir);
        write_csv(&summary_rows, &path).expect("sweep_summary.csv の書き込みに失敗");
    }

    // sweep_config.json (pretty-print JSON; socsim_results::write_json に委譲)．
    {
        let config_json = SweepConfigJson {
            command: "sweep",
            n_firms_values: n_firms_values.clone(),
            n_customers_values: n_customers_values.clone(),
            customer_mode: customer_mode.label().to_string(),
            days: args.days,
            runs: args.runs,
            seed: args.seed,
            llm_temperature: args.llm_temperature,
            llm_seed: args.llm_seed,
        };
        let path = format!("{}/sweep_config.json", sweep_dir);
        write_json(&config_json, &path).expect("sweep_config.json の書き込みに失敗");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{}_sweep", timestamp));

    println!("===========================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------");
    println!("店舗数別の勝者総取り発生頻度 / 平均 Gini:");
    for &n_firms in &n_firms_values {
        let rows: Vec<&SweepRow> = summary_rows
            .iter()
            .filter(|r| r.n_firms == n_firms)
            .collect();
        if rows.is_empty() {
            continue;
        }
        let wta_freq =
            rows.iter().filter(|r| r.winner_take_all == 1).count() as f64 / rows.len() as f64;
        let avg_gini = mean(
            &rows
                .iter()
                .map(|r| r.final_revenue_gini)
                .collect::<Vec<_>>(),
        );
        println!(
            "  M={} → WTA = {:.1}% | Ginī = {:.3}",
            n_firms,
            wta_freq * 100.0,
            avg_gini
        );
    }
    println!("-----------------------------------------------------------");
    println!("サマリ → {}/sweep_summary.csv", sweep_dir);
    println!("設定   → {}/sweep_config.json", sweep_dir);
}

/// sweep の試行シードを派生する (店舗数・顧客数・試行 index で独立化)．
fn socsim_core_derive(base: u64, n_firms: usize, n_customers: usize, run_idx: usize) -> u64 {
    socsim_core::derive_seed(base, &[n_firms as u64, n_customers as u64, run_idx as u64])
}

/// 1 実行結果を sweep の 1 行に集約する．
fn summarize(
    result: &SimulationResult,
    n_firms: usize,
    n_customers: usize,
    customer_mode: CustomerMode,
    run_idx: usize,
    seed: u64,
) -> SweepRow {
    let m = &result.metrics_history;
    let last_day = m.iter().map(|r| r.day).max().unwrap_or(0);
    let last_rows: Vec<&competeai_simulation::metrics::DailyMetric> =
        m.iter().filter(|r| r.day == last_day).collect();
    let final_gini = last_rows.first().map(|r| r.revenue_gini).unwrap_or(0.0);
    let final_share = last_rows.first().map(|r| r.market_share_max).unwrap_or(0.0);
    let final_menu_sim = last_rows.first().map(|r| r.menu_similarity).unwrap_or(0.0);
    let final_alive = last_rows.first().map(|r| r.n_alive_firms).unwrap_or(0);

    SweepRow {
        n_firms,
        n_customers,
        customer_mode: customer_mode.label().to_string(),
        run: run_idx,
        seed,
        final_day: result.final_day,
        final_revenue_gini: final_gini,
        final_market_share_max: final_share,
        winner_take_all: if result.winner_take_all { 1 } else { 0 },
        quality_improved: if result.quality_improved { 1 } else { 0 },
        final_menu_similarity: final_menu_sim,
        final_alive_firms: final_alive,
        cache_hit_rate: result.metadata.cache_hit_rate(),
    }
}

// ---------------------------------------------------------------------------
// reproduce
// ---------------------------------------------------------------------------

/// 1 条件 (顧客構成) を `runs` 回回した発生頻度の集計セル．
#[derive(serde::Serialize, Clone)]
struct ReproCell {
    /// 条件ラベル (individual / group)．
    customer_mode: String,
    runs: usize,
    /// 勝者総取り (WTA) が発生した試行数．
    wta_count: usize,
    /// 勝者総取り発生頻度 ∈ [0,1]．
    wta_freq: f64,
    /// 品質改善 (少なくとも一方の店) が発生した試行数．
    quality_count: usize,
    /// 品質改善発生頻度 ∈ [0,1]．
    quality_freq: f64,
    /// 試行平均の最終メニュー類似度 (差別化/模倣の動的均衡)．
    mean_menu_similarity: f64,
    /// 試行平均の最終収益 Gini (マタイ効果の強度)．
    mean_final_gini: f64,
    /// 試行平均の最終最大市場シェア．
    mean_final_share_max: f64,
}

/// 1 条件 (顧客構成) を `runs` 回実行して発生頻度を集計する．
#[allow(clippy::too_many_arguments)]
fn run_repro_cell(
    customer_mode: CustomerMode,
    base: &Config,
    runs: usize,
    root_seed: u64,
    mock: bool,
    out_dir: &str,
) -> ReproCell {
    let mut wta_count = 0usize;
    let mut quality_count = 0usize;
    let mut sum_menu = 0.0;
    let mut sum_gini = 0.0;
    let mut sum_share = 0.0;
    // 代表 (run 0) のメトリクス履歴を CSV に保存し，Python 側で時系列描画に使う．
    let mut representative: Option<Vec<competeai_simulation::metrics::DailyMetric>> = None;

    for run_idx in 0..runs.max(1) {
        let seed = socsim_core::derive_seed(
            root_seed,
            &[label_hash(customer_mode.label()), run_idx as u64],
        );
        let cfg = Config {
            customer_mode,
            seed: Some(seed),
            output_dir: out_dir.to_string(),
            ..base.clone()
        };
        let result = if mock {
            run_mock(&cfg)
                .unwrap_or_else(|e| panic!("mock 実行に失敗 ({}): {e}", customer_mode.label()))
        } else {
            run(&cfg).unwrap_or_else(|e| panic!("実行に失敗 ({}): {e}", customer_mode.label()))
        };

        if result.winner_take_all {
            wta_count += 1;
        }
        if result.quality_improved {
            quality_count += 1;
        }
        // 最終日の集計指標 (全店同値なので先頭行で代表させる)．
        let last_day = result
            .metrics_history
            .iter()
            .map(|m| m.day)
            .max()
            .unwrap_or(0);
        if let Some(last) = result.metrics_history.iter().find(|m| m.day == last_day) {
            sum_menu += last.menu_similarity;
            sum_gini += last.revenue_gini;
            sum_share += last.market_share_max;
        }
        if run_idx == 0 {
            representative = Some(result.metrics_history.clone());
        }
    }

    let n = runs.max(1) as f64;
    if let Some(hist) = representative {
        let path = format!("{out_dir}/metrics_{}.csv", customer_mode.label());
        socsim_results::write_csv(&hist, &path).expect("metrics_<mode>.csv の書き込みに失敗");
    }

    ReproCell {
        customer_mode: customer_mode.label().to_string(),
        runs: runs.max(1),
        wta_count,
        wta_freq: wta_count as f64 / n,
        quality_count,
        quality_freq: quality_count as f64 / n,
        mean_menu_similarity: sum_menu / n,
        mean_final_gini: sum_gini / n,
        mean_final_share_max: sum_share / n,
    }
}

/// ラベルを決定論的な u64 へ畳む (seed 派生用; FNV-1a)．
fn label_hash(label: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in label.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 観測値と論文 Table 2 の発生頻度を突き合わせた 1 アンカー．
#[derive(serde::Serialize)]
struct ReproAnchor {
    name: String,
    /// 論文値の表示文字列．
    paper: String,
    observed: f64,
    target_lo: f64,
    target_hi: f64,
    pass: bool,
}

fn cmd_reproduce(args: ReproduceArgs) {
    // quick モードは軽量化 (動作確認用; 論文値検証には使わない)．
    let n_customers = if args.quick { 12 } else { args.n_customers };
    let days = if args.quick { 6 } else { args.days };
    let individual_runs = if args.quick { 3 } else { args.individual_runs };
    let group_runs = if args.quick { 2 } else { args.group_runs };

    let ts = timestamp();
    let out_dir = format!("{}/reproduce_{}", args.output_dir, ts);
    ensure_output_dir(&out_dir);
    if !args.mock {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    // 基準設定 (全条件で共通; customer_mode/seed のみ条件ごとに差替)．
    let base = Config {
        n_firms: args.n_firms,
        n_customers,
        customer_mode: CustomerMode::Individual,
        group_size: args.group_size,
        days,
        seed: Some(args.seed),
        llm: LlmSettings {
            temperature: args.llm_temperature,
            seed: args.llm_seed,
            cache_path: if args.mock {
                None
            } else {
                Some(args.cache_path.clone())
            },
        },
        output_dir: out_dir.clone(),
        ..Config::default()
    };

    println!("=== Zhao et al. (2024) CompeteAI 論文 Table 2 発生頻度 一括再現 ===");
    println!(
        "M: {} | N: {} | days: {} | individual: {} ラン | group: {} ラン | mode: {}",
        args.n_firms,
        n_customers,
        days,
        individual_runs,
        group_runs,
        if args.mock { "MOCK" } else { "LIVE" },
    );
    println!("出力先: {out_dir}");
    println!("-------------------------------------------------");

    // --- 個人客 / グループ客の発生頻度を集計 ---
    let individual = run_repro_cell(
        CustomerMode::Individual,
        &base,
        individual_runs,
        args.seed,
        args.mock,
        &out_dir,
    );
    let group = run_repro_cell(
        CustomerMode::Group,
        &base,
        group_runs,
        args.seed,
        args.mock,
        &out_dir,
    );

    // --- アンカー評価 (論文 Table 2 / 本文の発生頻度; ±許容幅) ---
    let mut anchors: Vec<ReproAnchor> = Vec::new();
    let mut push = |name: &str, paper: &str, obs: f64, lo: f64, hi: f64| {
        anchors.push(ReproAnchor {
            name: name.to_string(),
            paper: paper.to_string(),
            observed: obs,
            target_lo: lo,
            target_hi: hi,
            pass: obs >= lo && obs <= hi,
        });
    };

    // 全ラン (個人 + グループ) の品質改善頻度 (論文 86.67%, ±10pt)．
    let total_quality = individual.quality_count + group.quality_count;
    let total_runs = (individual.runs + group.runs).max(1);
    let quality_freq_all = total_quality as f64 / total_runs as f64;
    let total_menu = individual.mean_menu_similarity * individual.runs as f64
        + group.mean_menu_similarity * group.runs as f64;
    let menu_all = total_menu / total_runs as f64;

    // 1. 個人客の勝者総取り頻度 (論文 66.7%, ±15pt)．
    push(
        "wta_individual (paper 66.7%)",
        "66.7%",
        individual.wta_freq,
        0.667 - 0.15,
        0.667 + 0.15,
    );
    // 2. グループ客の勝者総取り頻度 (論文 16.7%, ±15pt)．
    push(
        "wta_group (paper 16.7%)",
        "16.7%",
        group.wta_freq,
        0.0,
        0.167 + 0.15,
    );
    // 3. グループ化は勝者総取りを緩和する (個人 > グループ)．
    push(
        "group_dampens_wta (individual > group)",
        "individual > group",
        individual.wta_freq - group.wta_freq,
        0.0,
        f64::INFINITY,
    );
    // 4. 品質改善頻度 (論文 86.67%, ±10pt)．
    push(
        "quality_improved_all (paper 86.67%)",
        "86.67%",
        quality_freq_all,
        0.8667 - 0.10,
        1.0,
    );
    // 5. メニュー類似度 動的均衡 (論文 約36%; 参考値)．
    //    本 Phase 1 モデルは店舗のメニュー品目 (料理名集合) を改訂しないため，
    //    類似度は «初期差別化» の構造値で一定となる (差別化・模倣による «メニュー
    //    変異» は本モデルの範囲外)．論文値 ±10pt を参考バンドとして観測値を記録する
    //    が，モデルの限界として OFF となりうる (発生頻度の中核アンカーではない)．
    push(
        "menu_similarity_all (paper ~36%; structural)",
        "~36% (model holds menus fixed)",
        menu_all,
        0.36 - 0.10,
        0.36 + 0.10,
    );

    // --- コンソール出力 ---
    println!("--- 顧客構成別 発生頻度 ---");
    println!(
        "{:<12} {:>5} {:>10} {:>12} {:>10} {:>8}",
        "mode", "runs", "WTA", "quality", "menu_sim", "Gini"
    );
    for c in [&individual, &group] {
        println!(
            "{:<12} {:>5} {:>9.1}% {:>11.1}% {:>10.3} {:>8.3}",
            c.customer_mode,
            c.runs,
            c.wta_freq * 100.0,
            c.quality_freq * 100.0,
            c.mean_menu_similarity,
            c.mean_final_gini,
        );
    }
    println!("--- 論文 Table 2 アンカー (観測 vs 論文) ---");
    for a in &anchors {
        let hi = if a.target_hi.is_infinite() {
            "∞".to_string()
        } else {
            format!("{:.3}", a.target_hi)
        };
        println!(
            "[{}] {:<42} obs={:.4} target=[{:.3},{}] paper={}",
            if a.pass { "PASS" } else { "OFF " },
            a.name,
            a.observed,
            a.target_lo,
            hi,
            a.paper,
        );
    }
    let n_pass = anchors.iter().filter(|a| a.pass).count();
    println!("-------------------------------------------------");
    println!("{}/{} アンカーが in-band", n_pass, anchors.len());

    // --- reproduce_summary.json ---
    let summary = serde_json::json!({
        "timestamp": ts,
        "mode": if args.mock { "mock" } else { "live" },
        "config": {
            "n_firms": args.n_firms,
            "n_customers": n_customers,
            "days": days,
            "group_size": args.group_size,
            "individual_runs": individual_runs,
            "group_runs": group_runs,
            "seed": args.seed,
        },
        "cells": [individual, group],
        "anchors": anchors,
        "n_pass": n_pass,
        "n_total": anchors.len(),
    });
    let path = format!("{out_dir}/reproduce_summary.json");
    write_json(&summary, &path).expect("reproduce_summary.json の書き込みに失敗");
    let _ = refresh_latest_symlink(&args.output_dir, &format!("reproduce_{ts}"));
    println!("サマリ → {path}");
    println!("条件別メトリクス → {out_dir}/metrics_<mode>.csv");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}

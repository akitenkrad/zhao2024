//! Zhao et al. (2024) "CompeteAI" — 再現実験の CLI エントリポイント．
//!
//! `run`   : 単一設定で LLM 駆動の市場競争 ABM を実行する．
//! `sweep` : 店舗数 × 顧客数 (× 顧客構成) を走査し，マタイ効果指標 (収益 Gini・
//!           最大市場シェア・勝者総取り・品質改善) を `sweep_summary.csv` に集計する．
//!
//! Phase 3 の `reproduce` (論文 Table 2 の発生頻度一括再現・グループ客深掘り) は
//! 未実装 (拡張点)．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use competeai_simulation::config::{parse_customer_mode, Config, CustomerMode, LlmSettings};
use competeai_simulation::metrics::mean;
use competeai_simulation::simulation::{
    ensure_output_dir, run, save_metrics, save_run_metadata, SimulationResult,
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
        "LLM: temp={} llm_seed={} cache={} | seed: {:?}",
        args.llm_temperature, args.llm_seed, args.cache_path, args.seed
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

        let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));
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
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
    }
}

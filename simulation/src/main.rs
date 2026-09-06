//! Zhao et al. (2024) "CompeteAI" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で LLM 駆動の市場競争 ABM を実行する (`--mock` でオフライン)．
//! `sweep`     : 店舗数 × 顧客数 (× 顧客構成) を走査する．親 run 1 本 + セルごとの子 run．
//! `reproduce` : 論文 Table 2 の発生頻度 (個人客/グループ客の勝者総取り・品質改善・
//!               メニュー類似度) を一括再現する．親 run 1 本 + 試行ごとの子 run で，
//!               観測値は親の sweep スコープ指標に，論文の報告値は `reference.csv` に
//!               入る (`--mock` でオフライン scripted 駆動)．
//!
//! サブコマンド 1 回が runvault の run 1 本になる．出力の置き場と同一性 (run ディレ
//! クトリ・`config.json`・`metrics.csv`・`events.jsonl`) は runvault が持つので，
//! ここではタイムスタンプ付きディレクトリも `latest` symlink も作らない．

use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::rc::Rc;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions, Stage};

use competeai_simulation::config::{parse_customer_mode, Config, CustomerMode, LlmSettings};
use competeai_simulation::llm::{build_live_client, CompeteClient};
use competeai_simulation::mechanisms::CallObserver;
use competeai_simulation::metrics::mean;
use competeai_simulation::record::{self, DOMAIN, EXPERIMENT, REPO_ID, SWEEP_SCOPE};
use competeai_simulation::simulation::{run_with_client_observed, SimulationResult};
use socsim_llm::LlmClient;

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

    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
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

/// `sweep` 親 run の `parameters`．掃引の格子そのものを持つ．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    n_firms_values: Vec<usize>,
    n_customers_values: Vec<usize>,
    customer_mode: String,
    days: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// `reproduce` 親 run の `parameters`．条件と試行数を持つ．
#[derive(serde::Serialize)]
struct ReproduceConfigJson {
    n_firms: usize,
    n_customers: usize,
    group_size: usize,
    days: usize,
    individual_runs: usize,
    group_runs: usize,
    seed: u64,
    mock: bool,
    llm_temperature: f32,
    llm_seed: u64,
}

// ---------------------------------------------------------------------------
// LLM クライアント
// ---------------------------------------------------------------------------

/// LLM クライアントを 1 本組む．
///
/// `run.json` の `llm` ブロックに書くモデル名と endpoint は，実際に応答する
/// バックエンドから採らないと意味を持たないので，組み立ては `Run::start` より前に
/// 置く (knoll2013 と同じ理由で `simulation::run` / `run_mock` を消してある —
/// 中でクライアントを組む入口が残っていると，`llm` ブロックを埋めないまま記録
/// できてしまう)．
fn build_client(cfg: &Config, mock: bool) -> CompeteClient {
    if mock {
        competeai_simulation::reproduce_mock::build_reproduce_client()
    } else {
        build_live_client(&cfg.llm).unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"))
    }
}

/// LLM キャッシュの置き場を用意する (ライブ実行のみ; mock は in-memory)．
fn ensure_cache_dir(cfg: &Config) {
    if let Some(parent) = cfg
        .llm
        .cache_path
        .as_deref()
        .and_then(|path| Path::new(path).parent())
    {
        let _ = fs::create_dir_all(parent);
    }
}

/// mock は永続キャッシュを持たないので `cache_path` を落とす．
fn llm_settings(temperature: f32, seed: u64, cache_path: &str, mock: bool) -> LlmSettings {
    LlmSettings {
        temperature,
        seed,
        cache_path: (!mock).then(|| cache_path.to_string()),
    }
}

/// LLM 呼び出しを数える stage を，メカニズムから突ける形にして渡す．
///
/// 呼び出しが起きるのは `CompetitionMatthewMechanism` (店舗の戦略立案) と
/// `CustomerChoiceMechanism` (顧客の来店選択) の中である．メカニズムは
/// `Box<dyn Mechanism<_>>` としてエンジンへ入る = `'static` なので，呼び出し側の
/// `Stage` を借用できない — `Rc` で共有し，走り終えたあとに [`close_shared`] で
/// 取り出して閉じる．
fn share_stage(stage: Stage) -> (Rc<RefCell<Option<Stage>>>, CallObserver) {
    let cell = Rc::new(RefCell::new(Some(stage)));
    let observer: CallObserver = {
        let cell = Rc::clone(&cell);
        Rc::new(RefCell::new(move || {
            if let Some(stage) = cell.borrow_mut().as_mut() {
                stage.tick();
            }
        }))
    };
    (cell, observer)
}

/// 共有していた stage を取り出して閉じる．
///
/// `manifest.csv` は `finish()` で封をされる．そのあとに 1 行足せば，manifest が
/// 食い違うダイジェストを持つことになるので，必ず `finish()` の前に呼ぶ．
fn close_shared(cell: &Rc<RefCell<Option<Stage>>>) {
    if let Some(stage) = cell.borrow_mut().take() {
        stage.close();
    }
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

    let base_seed = args.seed.unwrap_or(42);
    let runs = args.runs.max(1);

    // 記録するのは最後の 1 本．`--runs N` は同じ条件を N 本回して最後の試行の詳細
    // だけを残す既存の動きなので (旧実装も `save_metrics` を最終試行でしか呼んで
    // いない)，`master_seed` には実際に世界を支配した `derive_run_seed(base, N-1)`
    // を書き，`replicate_index` を N-1 にする．CLI で与えた根のシードは
    // `/parameters.seed` にあり，seed_pointers 経由で execution_hash に残る．
    let recorded_seed = competeai_simulation::config::derive_run_seed(base_seed, runs - 1);

    let base_cfg = Config {
        n_firms: args.n_firms,
        n_customers: args.n_customers,
        customer_mode,
        group_size: args.group_size,
        days: args.days,
        // `parameters` に載るのは CLI で与えた根のシード．実際に世界を支配した
        // 派生シードは `master_seed` が持つ．
        seed: Some(base_seed),
        llm: llm_settings(
            args.llm_temperature,
            args.llm_seed,
            &args.cache_path,
            args.mock,
        ),
        ..Config::default()
    };
    ensure_cache_dir(&base_cfg);

    // クライアントは run を開始する前に組む (`llm` ブロックのため)．最初の 1 本で
    // そのまま使い，2 本目以降は旧実装と同じく 1 本ごとに組み直す．
    let mut pending = Some(build_client(&base_cfg, args.mock));
    let llm = pending.as_ref().map(|c| {
        record::llm_block(
            c.inner().model(),
            c.inner().endpoint(),
            base_cfg.llm.temperature,
        )
    });

    let parameters = base_cfg.to_run_config_json();
    let mut options = RunOptions::new(EXPERIMENT, "run")
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(&args.output_dir)
        .parameters(&parameters)
        .expect("runvault: parameters の組み立てに失敗")
        .seed_pointers(["/seed"])
        .master_seed(recorded_seed)
        .replicate_index((runs - 1) as u64)
        .replication(record::replication());
    if let Some(llm) = llm {
        options = options.llm(llm);
    }
    let mut rv = Run::start(options).expect("runvault: run の開始に失敗");

    println!("=== Zhao et al. (2024) CompeteAI 市場競争 再現実験 ===");
    println!(
        "M (店舗): {} | N (顧客): {} | 構成: {} | 日数: {} | 試行: {}",
        args.n_firms,
        args.n_customers,
        customer_mode.label(),
        args.days,
        runs,
    );
    println!(
        "LLM: temp={} llm_seed={} cache={} | seed: {}{}",
        args.llm_temperature,
        args.llm_seed,
        args.cache_path,
        base_seed,
        if args.mock { " | MOCK" } else { "" },
    );
    println!("出力先: {}", rv.dir().display());
    println!("-------------------------------------------------");

    // 進捗の単位は «LLM 呼び出し 1 回»．1 日は生存店 1 軒あたり 1 回 + 顧客 1 人
    // あたり 1 回で，論文標準の M=2 / N=50 なら 52 回になる．ローカルの Ollama
    // (llama3.2) で既定設定を実測すると 780 回で 6 分 18 秒 = 1 回 0.48 秒 —
    // 1 日はおよそ 25 秒で，日を数える counter はその間まったく動かない．
    //
    // 分母を持たないのは，`ReflectionMechanism` が «最終日» だけでなく «資金が
    // 尽きた店舗が出た» ときにも `request_stop` を掛けるからである (mechanisms.rs
    // の `a_firm_exited`)．`days * (M + N) * runs` は到達するとは限らない上限で
    // あって総数ではなく，上限を分母に据えた ETA は自信をもって間違える．
    let (stage, observer) = share_stage(rv.unbounded_stage("decisions"));

    let mut last_result: Option<SimulationResult> = None;
    let mut wta_count = 0usize;
    let mut quality_count = 0usize;

    for run_idx in 0..runs {
        // 試行ごとに独立シードを派生する．
        let seed = competeai_simulation::config::derive_run_seed(base_seed, run_idx);
        let cfg = Config {
            seed: Some(seed),
            ..base_cfg.clone()
        };

        let client = pending
            .take()
            .unwrap_or_else(|| build_client(&cfg, args.mock));
        let result = run_with_client_observed(&cfg, client, Rc::clone(&observer))
            .unwrap_or_else(|e| panic!("実行に失敗: {}", e));
        if result.winner_take_all {
            wta_count += 1;
        }
        if result.quality_improved {
            quality_count += 1;
        }

        // 最後の試行の詳細を記録する (代表 run)．
        if run_idx + 1 == runs {
            record::log_simulation(&mut rv, &result);
            last_result = Some(result);
        }
    }

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

    close_shared(&stage);
    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("日次集計   → {}/metrics.csv", dir.display());
    println!("店舗パネル → {}/events.jsonl", dir.display());
    println!("設定       → {}/config.json", dir.display());
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

    let n_total = n_firms_values.len() * n_customers_values.len() * args.runs;

    // 親 run: 格子の定義そのものを parameters に持つ．個別セルの指標は書かない．
    // 親は単一の master_seed を持たない (セルごとの子が派生シードをそれぞれ持つ)．
    // base seed は /parameters.seed と seed_pointers 経由で execution_hash に残る．
    // sweep_id は runvault が親の run_slug で埋める．
    let sweep_parameters = SweepConfigJson {
        n_firms_values: n_firms_values.clone(),
        n_customers_values: n_customers_values.clone(),
        customer_mode: customer_mode.label().to_string(),
        days: args.days,
        runs: args.runs,
        seed: args.seed,
        llm_temperature: args.llm_temperature,
        llm_seed: args.llm_seed,
    };
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&sweep_parameters)
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: sweep 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

    println!("=== Zhao et al. (2024) CompeteAI パラメータスイープ ===");
    println!(
        "M: {} 種 | N: {} 種 | 構成: {} | 試行: {} | 合計: {} 実行",
        n_firms_values.len(),
        n_customers_values.len(),
        customer_mode.label(),
        args.runs,
        n_total,
    );
    println!("出力先: {}", parent.dir().display());
    println!("-----------------------------------------------------------");

    // コンソールの要約に使うだけの控え (ディスクには書かない; 同じ値は子 run の
    // 指標にある)．
    let mut console: Vec<(usize, bool, f64)> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    // 単位は `run` と同じ «LLM 呼び出し 1 回»．セル 1 つ (= 1 試行) は
    // `days * (M + N)` 回の呼び出しで，既定の格子の一番小さいセルでも
    // 15 * (2 + 20) = 330 回 ≒ 2 分半あり，格子全体では 47,700 回 ≒ 6 時間半に
    // なる (実測した 1 回 0.48 秒から)．セルを数える counter では «いま何をして
    // いるか» が 2 分半見えない．
    //
    // 分母は持たない．試行の本数 (60) は正確だが，1 試行の長さは店舗の撤退で
    // 早く終わりうるので呼び出しの総数は事前に数えられない．なお `sweep` には
    // `--mock` が無く，既定でライブの LLM を必要とする．
    let (stage, observer) = share_stage(parent.unbounded_stage("decisions"));

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
                    llm: llm_settings(args.llm_temperature, args.llm_seed, &args.cache_path, false),
                    ..Config::default()
                };
                ensure_cache_dir(&cfg);
                let client = build_client(&cfg, false);
                let llm = record::llm_block(
                    client.inner().model(),
                    client.inner().endpoint(),
                    cfg.llm.temperature,
                );

                // 子は «そのセルの run» そのもの．master_seed は base から派生した
                // 実際に使われるシードで，同一セルの繰り返しは replicate_index で
                // 分ける．parameters は手で回した `run` と同じ形なので，同じ条件
                // なら config_hash が一致する．
                let parameters = cfg.to_run_config_json();
                let mut child = Run::start(
                    RunOptions::new(EXPERIMENT, "run")
                        .repo_id(REPO_ID)
                        .domain(DOMAIN)
                        .results_root(&args.output_dir)
                        .parameters(&parameters)
                        .expect("runvault: 子 run の parameters の組み立てに失敗")
                        .seed_pointers(["/seed"])
                        .master_seed(seed)
                        .replicate_index(run_idx as u64)
                        .lineage(Lineage {
                            sweep_id: Some(sweep_id.clone()),
                            parent_run_uid: Some(parent_run_uid.clone()),
                            ..Default::default()
                        })
                        .llm(llm)
                        .replication(record::replication()),
                )
                .expect("runvault: 子 run の開始に失敗");

                let result = run_with_client_observed(&cfg, client, Rc::clone(&observer))
                    .unwrap_or_else(|e| panic!("実行に失敗: {}", e));
                record::log_simulation(&mut child, &result);
                child.finish().expect("runvault: 子 run の完了に失敗");

                let last_day = result
                    .metrics_history
                    .iter()
                    .map(|r| r.day)
                    .max()
                    .unwrap_or(0);
                let final_gini = result
                    .metrics_history
                    .iter()
                    .find(|r| r.day == last_day)
                    .map(|r| r.revenue_gini)
                    .unwrap_or(0.0);
                console.push((n_firms, result.winner_take_all, final_gini));
                done += 1;
            }
            println!(
                "[{}/{}] M={} N={} 完了 ({} 試行)",
                done, n_total, n_firms, n_customers, args.runs,
            );
        }
    }

    close_shared(&stage);
    let dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");

    println!("===========================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------");
    println!("店舗数別の勝者総取り発生頻度 / 平均 Gini:");
    for &n_firms in &n_firms_values {
        let rows: Vec<&(usize, bool, f64)> = console.iter().filter(|r| r.0 == n_firms).collect();
        if rows.is_empty() {
            continue;
        }
        let wta_freq = rows.iter().filter(|r| r.1).count() as f64 / rows.len() as f64;
        let avg_gini = mean(&rows.iter().map(|r| r.2).collect::<Vec<_>>());
        println!(
            "  M={} → WTA = {:.1}% | Ginī = {:.3}",
            n_firms,
            wta_freq * 100.0,
            avg_gini
        );
    }
    println!("-----------------------------------------------------------");
    println!("親 run → {}", dir.display());
    println!("子 run は lineage.parent_run_uid で親を指す．");
}

/// sweep の試行シードを派生する (店舗数・顧客数・試行 index で独立化)．
fn socsim_core_derive(base: u64, n_firms: usize, n_customers: usize, run_idx: usize) -> u64 {
    socsim_core::derive_seed(base, &[n_firms as u64, n_customers as u64, run_idx as u64])
}

// ---------------------------------------------------------------------------
// reproduce
// ---------------------------------------------------------------------------

/// 1 条件 (顧客構成) を `runs` 回回した発生頻度の集計．
///
/// ディスクには «この構造体» としては書かない．親 run の sweep スコープ指標に
/// 名前を折り込んで載る (`wta_freq_individual` など)．条件は値ではなく «どの数か»
/// を指す名前なので，`motive_mix_as` (knoll2013) と同じく名前に畳む．
#[derive(Clone)]
struct ReproCell {
    /// 条件ラベル (individual / group)．
    customer_mode: &'static str,
    runs: usize,
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

/// 1 条件 (顧客構成) を `runs` 回実行し，1 試行ごとに子 run を書いて頻度を集計する．
///
/// 旧実装は run 0 の履歴だけを `metrics_<mode>.csv` に残していた．いまは試行が
/// それぞれ模型の別々の実行として子 run になるので，全試行の日次集計と店舗パネルが
/// 残る (旧 CSV はその真部分集合になる)．
#[allow(clippy::too_many_arguments)]
fn run_repro_cell(
    customer_mode: CustomerMode,
    base: &Config,
    runs: usize,
    root_seed: u64,
    mock: bool,
    output_dir: &str,
    sweep_id: &str,
    parent_run_uid: &str,
    observer: CallObserver,
) -> ReproCell {
    let mut wta_count = 0usize;
    let mut quality_count = 0usize;
    let mut sum_menu = 0.0;
    let mut sum_gini = 0.0;
    let mut sum_share = 0.0;

    for run_idx in 0..runs.max(1) {
        let seed = socsim_core::derive_seed(
            root_seed,
            &[label_hash(customer_mode.label()), run_idx as u64],
        );
        let cfg = Config {
            customer_mode,
            seed: Some(seed),
            ..base.clone()
        };
        ensure_cache_dir(&cfg);
        let client = build_client(&cfg, mock);
        let llm = record::llm_block(
            client.inner().model(),
            client.inner().endpoint(),
            cfg.llm.temperature,
        );

        let parameters = cfg.to_run_config_json();
        let mut child = Run::start(
            RunOptions::new(EXPERIMENT, "run")
                .repo_id(REPO_ID)
                .domain(DOMAIN)
                .results_root(output_dir)
                .parameters(&parameters)
                .expect("runvault: 子 run の parameters の組み立てに失敗")
                .seed_pointers(["/seed"])
                .master_seed(seed)
                .replicate_index(run_idx as u64)
                .lineage(Lineage {
                    sweep_id: Some(sweep_id.to_string()),
                    parent_run_uid: Some(parent_run_uid.to_string()),
                    ..Default::default()
                })
                .llm(llm)
                .replication(record::replication()),
        )
        .expect("runvault: 子 run の開始に失敗");

        let result = run_with_client_observed(&cfg, client, Rc::clone(&observer))
            .unwrap_or_else(|e| panic!("実行に失敗 ({}): {e}", customer_mode.label()));
        record::log_simulation(&mut child, &result);
        child.finish().expect("runvault: 子 run の完了に失敗");

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
    }

    let n = runs.max(1) as f64;
    ReproCell {
        customer_mode: customer_mode.label(),
        runs: runs.max(1),
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

/// 論文の報告値 1 つ．`reference.csv` の 1 行になる．
///
/// 入るのは **論文が報告した値だけ**である．この再現実装が選んだ許容幅 (±15pt /
/// ±10pt) は論文の主張ではないので行にしない — 帯と PASS/OFF の判定はコンソールと
/// ドキュメントに残す．
struct PaperValue {
    /// 観測側の指標名と揃える (差分がそのまま取れる)．
    name: &'static str,
    value: f64,
    /// `research.targets[]` の target_id．
    target_id: &'static str,
    source: &'static str,
    /// この再現実装が置いた許容幅 (下限, 上限)．記録はしない．
    band: (f64, f64),
}

/// 論文 Table 2 / 本文が報告した発生頻度と動的均衡値．
const PAPER_VALUES: [PaperValue; 4] = [
    PaperValue {
        name: "wta_freq_individual",
        value: 0.667,
        target_id: "table2",
        source: "Zhao et al. (2024), Table 2 — winner-take-all with individual customers (66.7%)",
        band: (0.667 - 0.15, 0.667 + 0.15),
    },
    PaperValue {
        name: "wta_freq_group",
        value: 0.167,
        target_id: "table2",
        source: "Zhao et al. (2024), Table 2 — winner-take-all with group customers (16.7%)",
        band: (0.0, 0.167 + 0.15),
    },
    PaperValue {
        name: "quality_freq_all",
        value: 0.8667,
        target_id: "table2",
        source: "Zhao et al. (2024), Table 2 — quality improvement across all runs (86.67%)",
        band: (0.8667 - 0.10, 1.0),
    },
    PaperValue {
        name: "menu_similarity_all",
        value: 0.36,
        target_id: "differentiation-imitation-equilibrium",
        source: "Zhao et al. (2024), §4 — menu similarity at the dynamic equilibrium (approx. 36%)",
        band: (0.36 - 0.10, 0.36 + 0.10),
    },
];

fn cmd_reproduce(args: ReproduceArgs) {
    // quick モードは軽量化 (動作確認用; 論文値検証には使わない)．
    let n_customers = if args.quick { 12 } else { args.n_customers };
    let days = if args.quick { 6 } else { args.days };
    let individual_runs = if args.quick { 3 } else { args.individual_runs };
    let group_runs = if args.quick { 2 } else { args.group_runs };

    // 基準設定 (全条件で共通; customer_mode/seed のみ条件ごとに差替)．
    let base = Config {
        n_firms: args.n_firms,
        n_customers,
        customer_mode: CustomerMode::Individual,
        group_size: args.group_size,
        days,
        seed: Some(args.seed),
        llm: llm_settings(
            args.llm_temperature,
            args.llm_seed,
            &args.cache_path,
            args.mock,
        ),
        ..Config::default()
    };

    // 親 run: 条件と試行数を parameters に持ち，条件をまたいだ集約 (発生頻度) を
    // sweep スコープの指標として書く．論文の報告値は reference.csv に入る．
    // 親は単一の master_seed を持たない (条件 × 試行の子が派生シードを持つ)．
    let parent_parameters = ReproduceConfigJson {
        n_firms: args.n_firms,
        n_customers,
        group_size: args.group_size,
        days,
        individual_runs,
        group_runs,
        seed: args.seed,
        mock: args.mock,
        llm_temperature: args.llm_temperature,
        llm_seed: args.llm_seed,
    };
    let mut parent = Run::start(
        RunOptions::new(EXPERIMENT, "reproduce")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parent_parameters)
            .expect("runvault: reproduce の parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: reproduce 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: reproduce 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

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
    println!("出力先: {}", parent.dir().display());
    println!("-------------------------------------------------");

    // --- 個人客 / グループ客の発生頻度を集計 (試行ごとに子 run) ---
    //
    // 単位は `run` / `sweep` と同じ «LLM 呼び出し 1 回»．論文標準では 1 試行が
    // 15 日 × (2 + 50) = 780 回 = 6 分 18 秒 (実測) あるので，試行を数える counter
    // では 6 分動かない．15 試行の全体はおよそ 1 時間半になる．分母は持たない —
    // 店舗の撤退で試行が早く終わりうる以上，呼び出しの総数は事前に数えられない．
    //
    // 顧客構成ごとに別の stage にする．勝者総取りが起きるかどうか (= 資金が尽きて
    // 早く止まるかどうか) は顧客構成で決まる (論文 個人 66.7% / グループ 16.7%)
    // ので，1 試行あたりの呼び出し回数はここで変わる．重みを与えるのではなく分ける
    // — 重みは走らせる前には測れず，速い側が遅い側の見積りを引っぱるだけである．
    let (ind_stage, ind_observer) = share_stage(parent.unbounded_stage("individual"));
    let individual = run_repro_cell(
        CustomerMode::Individual,
        &base,
        individual_runs,
        args.seed,
        args.mock,
        &args.output_dir,
        &sweep_id,
        &parent_run_uid,
        ind_observer,
    );
    close_shared(&ind_stage);

    let (grp_stage, grp_observer) = share_stage(parent.unbounded_stage("group"));
    let group = run_repro_cell(
        CustomerMode::Group,
        &base,
        group_runs,
        args.seed,
        args.mock,
        &args.output_dir,
        &sweep_id,
        &parent_run_uid,
        grp_observer,
    );
    close_shared(&grp_stage);

    // --- 全ラン (個人 + グループ) の集約 ---
    let total_runs = (individual.runs + group.runs).max(1);
    let quality_freq_all =
        (individual.quality_count + group.quality_count) as f64 / total_runs as f64;
    let menu_all = (individual.mean_menu_similarity * individual.runs as f64
        + group.mean_menu_similarity * group.runs as f64)
        / total_runs as f64;

    // --- 親の sweep スコープ指標 (観測値) ---
    let observed: Vec<(&str, f64)> = vec![
        ("wta_freq_individual", individual.wta_freq),
        ("wta_freq_group", group.wta_freq),
        ("quality_freq_individual", individual.quality_freq),
        ("quality_freq_group", group.quality_freq),
        ("quality_freq_all", quality_freq_all),
        (
            "menu_similarity_individual",
            individual.mean_menu_similarity,
        ),
        ("menu_similarity_group", group.mean_menu_similarity),
        ("menu_similarity_all", menu_all),
        ("final_gini_individual", individual.mean_final_gini),
        ("final_gini_group", group.mean_final_gini),
        (
            "final_share_max_individual",
            individual.mean_final_share_max,
        ),
        ("final_share_max_group", group.mean_final_share_max),
        // グループ化が勝者総取りを緩和するか (論文の主張は個人 > グループ)．
        // 差そのものは論文が報告していないので reference は持たない．
        (
            "wta_freq_gap_individual_minus_group",
            individual.wta_freq - group.wta_freq,
        ),
    ];
    parent
        .log_metrics(SWEEP_SCOPE, &observed)
        .expect("reproduce 親の集約指標の記録に失敗");

    // --- 論文の報告値 (reference.csv) ---
    for pv in &PAPER_VALUES {
        parent
            .log_reference(pv.name, pv.value)
            .scope(SWEEP_SCOPE)
            .target(pv.target_id)
            .source(pv.source)
            .send()
            .unwrap_or_else(|e| panic!("論文値 {} の記録に失敗: {e}", pv.name));
    }

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

    // 帯は論文の主張ではなくこの再現実装が置いたものなので，記録せず表示だけする．
    // メニュー類似度は本モデルが料理名集合を改訂しないため «初期差別化» の構造値で
    // 一定になり，OFF となりうる (発生頻度の中核アンカーではない)．
    println!("--- 論文 Table 2 アンカー (観測 vs 論文; 帯は本実装の設定) ---");
    let by_name = |name: &str| -> f64 {
        observed
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
            .expect("観測値が無い")
    };
    let mut n_pass = 0usize;
    for pv in &PAPER_VALUES {
        let obs = by_name(pv.name);
        let pass = obs >= pv.band.0 && obs <= pv.band.1;
        if pass {
            n_pass += 1;
        }
        println!(
            "[{}] {:<28} obs={:.4} paper={:.4} band=[{:.3},{:.3}]",
            if pass { "PASS" } else { "OFF " },
            pv.name,
            obs,
            pv.value,
            pv.band.0,
            pv.band.1,
        );
    }
    println!(
        "[{}] {:<28} obs={:.4} (論文の主張: 個人 > グループ)",
        if individual.wta_freq > group.wta_freq {
            "PASS"
        } else {
            "OFF "
        },
        "wta_freq_gap",
        by_name("wta_freq_gap_individual_minus_group"),
    );
    println!("-------------------------------------------------");
    println!("{}/{} アンカーが in-band", n_pass, PAPER_VALUES.len());

    let dir = parent
        .finish()
        .expect("runvault: reproduce 親 run の完了に失敗");
    println!("集約     → {}/metrics.csv (scope=sweep)", dir.display());
    println!("論文値   → {}/reference.csv", dir.display());
    println!("条件別の試行は lineage.parent_run_uid で親を指す子 run にある．");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}

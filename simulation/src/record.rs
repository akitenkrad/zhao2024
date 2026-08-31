//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` でも `sweep` / `reproduce` の子でも同一なので，
//! ここ 1 箇所で組み立てる．日次の集計指標，run 全体を 1 つの値で表す指標，店舗
//! 1 軒ごとの日次パネル (observation)，店舗の帰趨 (terminal) の書き方もここに集める．
//!
//! # パネルをどこに置くか
//!
//! 旧 `metrics.csv` は `day,firm,…` の long 形式で，1 行が «日 × 店舗» の 2 つの
//! キーで決まっていた．runvault の `metrics.csv` は `run_uid,step,step_unit,scope,
//! name,value` で，系列 (どの店舗か) を入れる列を持たない．店舗ごとの値をここに
//! 並べると全店舗の行が同じ主キー `(name, step, step_unit, scope)` を名乗って衝突
//! する．`scope` は «どの粒度の集約か» であって «どの主体か» ではないので逃げ場に
//! ならない．
//!
//! そこで «日 × 店舗» の数は `observation` イベントに置く．店舗は 1 回の実行の中で
//! 毎日観測される主体であり，資金が尽きれば退出する — つまり到達時間の観測そのもの
//! なので，実験固有の種別ではなくコア語彙の `observation` / `terminal` を使う．
//! 全店舗で同じ値になる日次集計 (Gini・最大シェア・メニュー類似度・生存店舗数) は
//! 系列を持たないので `metrics.csv` に残る．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::metrics::DailyMetric;
use crate::simulation::{FirmOutcome, SimulationResult};

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
pub const EXPERIMENT: &str = "competeai";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "zhao2024";
/// 分野．店舗の初期資金・顧客の所得/嗜好/健康・活性化順・グループ多数決の同点処理が
/// いずれも乱数駆動で `master_seed` が要るので `simulation`．意思決定は LLM が担うが，
/// 測っているのはモデルの安全性ではなく市場競争から創発するマタイ効果なので
/// `llm-safety` ではない．LLM 側の同一性は `run.json` の `llm` ブロックが持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．
///
/// このモデルの 1 刻みは «市場の 1 日» — 店舗が戦略を改訂し，顧客が来店先を選び，
/// 収益が精算されるまでの 1 巡である．語彙に `day` は無く，1 巡を表す語は `round`．
/// 同じく LLM 駆動で 1 日 = 1 巡の gao2023 と揃える．
const T_UNIT: &str = "round";

/// 日次集計と run 全体の指標の粒度．いずれも市場全体の集約なので `run`．
const SCOPE: &str = "run";

/// `reproduce` の親が持つ «条件をまたいだ集約» の粒度．
pub const SWEEP_SCOPE: &str = "sweep";

/// この再現実験が対象としている論文．
///
/// `run` も `sweep` / `reproduce` の子も同じ主張を対象とする — 掃引は店舗数 × 顧客数
/// を変えてマタイ効果の創発条件を見るためのもので，別の対象を持たない．
pub fn replication() -> Replication {
    let mut work = Work::arxiv("2310.17512")
        .title(
            "CompeteAI: Understanding the Competition Dynamics of Large Language Model-based \
             Agents",
        )
        .year(2024)
        .source_version("icml-2024");
    // vault 側の同定にも使えるよう paper-id も残す (work_id は arXiv 側)．
    work.paper_id = Some("P00001797".to_string());
    Replication::new(work)
        .target(Target::table("table2", "Table 2"))
        .target(Target::claim(
            "matthew-effect-winner-take-all",
            "An early advantage compounds through a positive feedback loop into winner-take-all, \
             and grouping the customers dampens it",
        ))
        .target(Target::claim(
            "differentiation-imitation-equilibrium",
            "Differentiation and imitation settle into a dynamic equilibrium of partially \
             overlapping menus",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/zhao2024/設計書.md")
}

// --------------------------------------------------------------------------- //
// LLM ブロック
// --------------------------------------------------------------------------- //

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める．推測しているのは分類だけで，値そのものは記録から採る．
///
/// `model_snapshot` に入るのは `llama3.2:latest` のような動くエイリアスであることが
/// 多い．socsim-llm はスナップショット id を持たないので，持っていない値を作らずに
/// 名乗られた名前を書く．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // 店舗プロンプトと顧客プロンプトは相手ごとに組み立てられ，固定の system
        // prompt を持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// --------------------------------------------------------------------------- //
// シミュレーション 1 本ぶんの記録
// --------------------------------------------------------------------------- //

/// シミュレーション 1 本ぶんを run へ書く (`run` サブコマンドと掃引の子で共通)．
pub fn log_simulation(run: &mut Run, result: &SimulationResult) {
    log_daily_aggregates(run, &result.metrics_history);
    log_observations(run, &result.metrics_history);
    log_run_scope(run, result);
    // 終端の時刻は «最後に観測した日» である．`final_day` は socsim エンジンが数えた
    // 完了ステップ数 (1 始まり) で，日の添字 (0 始まり) より 1 大きい．`verify --deep`
    // は terminal の `t` が同じ `unit_id` の observation の最大 `t` と一致することを
    // 要求するので，観測した側から採る．
    if let Some(last_day) = result.metrics_history.iter().map(|m| m.day).max() {
        log_terminals(run, last_day, &result.firm_outcomes);
    }
}

/// 全店舗で同じ値になる日次集計を `metrics.csv` に書く．
///
/// 旧 `metrics.csv` はこの 4 つを店舗ごとの行すべてに重複して持っていた (long 形式で
/// «日次集計量は全行同値» と注記されていた)．系列を持たない数なので，日ごとに 1 度
/// だけ書く．
fn log_daily_aggregates(run: &mut Run, history: &[DailyMetric]) {
    let mut current: Option<u64> = None;
    for m in history {
        if current == Some(m.day) {
            continue;
        }
        current = Some(m.day);
        run.log_metrics_at(
            m.day,
            T_UNIT,
            SCOPE,
            &[
                ("revenue_gini", m.revenue_gini),
                ("market_share_max", m.market_share_max),
                ("menu_similarity", m.menu_similarity),
                ("n_alive_firms", m.n_alive_firms as f64),
            ],
        )
        .unwrap_or_else(|e| panic!("day {} の日次集計指標の記録に失敗: {e}", m.day));
    }
}

/// `events.jsonl` に書く «日 × 店舗» の観測 1 点．
///
/// 旧 `metrics.csv` の店舗固有の 6 列がそのまま欄になる．`firm_alive` は落とした —
/// 撤退の判定は日次指標を書いた後の `ReflectionMechanism` (PostStep) が行い，撤退が
/// 起きた時点で `request_stop()` が掛かるので，書き出された行の `firm_alive` は
/// 例外なく 1 である．店舗の帰趨は [`TerminalEvent`] の `outcome` / `censored` が
/// 持つ (そちらは «どの店舗が潰れたか» まで言える)．
///
/// 欄の名前は掃引パラメータ (`n_firms` / `n_customers` / `customer_mode` / `days` /
/// `seed` …) と重ならないようにしてある．`runvault.read.sweep_events_table` は
/// 同名のパラメータ列でイベント列を上書きするので，衝突すると黙って消える．
#[derive(Serialize)]
struct ObservationEvent {
    unit_id: String,
    t: u64,
    t_unit: &'static str,
    firm: u64,
    day_customers: u64,
    day_revenue: f64,
    cumulative_revenue: f64,
    avg_dish_score: f64,
    avg_price: f64,
    reputation: f64,
}

/// 店舗 1 軒の 1 日ぶんを 1 行として，日次パネルをすべて書く．
fn log_observations(run: &mut Run, history: &[DailyMetric]) {
    for m in history {
        let event = ObservationEvent {
            unit_id: unit_id(m.firm),
            t: m.day,
            t_unit: T_UNIT,
            firm: m.firm,
            day_customers: m.day_customers,
            day_revenue: m.day_revenue,
            cumulative_revenue: m.cumulative_revenue,
            avg_dish_score: m.avg_dish_score,
            avg_price: m.avg_price,
            reputation: m.reputation,
        };
        run.log_event("observation", &event).unwrap_or_else(|e| {
            panic!(
                "day {} 店舗 {} の observation の記録に失敗: {e}",
                m.day, m.firm
            )
        });
    }
}

/// run 全体を 1 つの値で表す指標．
///
/// `n_units` は予約指標名で «観測主体の数» — このモデルでは `observation` を持つ
/// 店舗の数である (顧客は観測されない)．実行時間は `status.json` の `duration_sec`
/// が正本なので指標にはしない．
///
/// `winner_take_all` と `quality_improved` は 0/1 で書く．カテゴリに番号を振ったの
/// ではなく，«起きたか» を表す指標変数で，複数の run にわたる平均が論文 Table 2 の
/// 発生頻度そのものになる (`reproduce` の親がその平均を書く)．
fn log_run_scope(run: &mut Run, result: &SimulationResult) {
    let calls = result.metadata.total();
    let mut values: Vec<(&str, f64)> = vec![
        ("n_units", result.firm_outcomes.len() as f64),
        ("final_day", result.final_day as f64),
        (
            "winner_take_all",
            if result.winner_take_all { 1.0 } else { 0.0 },
        ),
        (
            "quality_improved",
            if result.quality_improved { 1.0 } else { 0.0 },
        ),
        ("llm_calls", calls as f64),
        ("llm_cache_hits", result.metadata.cache_hits() as f64),
    ];
    // 呼び出しが 1 本も無いときの cache-hit 率は «0» ではなく «定義できない»．
    // 欠測を 0 で埋めず，率の行そのものを書かない．
    if calls > 0 {
        values.push(("llm_cache_hit_rate", result.metadata.cache_hit_rate()));
    }
    run.log_metrics(SCOPE, &values)
        .expect("run スコープの指標の記録に失敗");
}

/// `events.jsonl` に書く店舗 1 軒の終端行．
///
/// 先頭 6 フィールドは runvault の予約語 (`terminal` はこれを全部要求する)．
/// 数は重ねない — 日次の値は最終日の `observation` が持っているので，ここに置くのは
/// «その日にどうなっていたか» を決める資金だけである．
#[derive(Serialize)]
struct TerminalEvent {
    unit_id: String,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    firm: u64,
    funds: f64,
}

/// 店舗 1 軒につき 1 行書く．
///
/// `budget` はこの run で実際に観測できた最後の日である．モデルの終了条件は «いずれ
/// かの店舗の撤退» または «最終日» で，どちらでも観測はそこで打ち切られる．生存した
/// 店舗は «予算を使い切って打ち切られた» ので `censored = true` かつ `t == budget`
/// (runvault が書き込み時に検査する不変条件)．資金が尽きて退出した店舗は事象が起きて
/// いるので `censored = false`．
///
/// 各店舗の `unit_id` は同じ `t` の `observation` にも現れる — 日次パネルを全日ぶん
/// 書いているので，`verify --deep` が要求する «terminal の unit_id が observation の
/// 最大 t と一致する» は自動的に満たされる．
fn log_terminals(run: &mut Run, final_day: u64, outcomes: &[FirmOutcome]) {
    for o in outcomes {
        let event = TerminalEvent {
            unit_id: unit_id(o.firm),
            t: final_day,
            t_unit: T_UNIT,
            outcome: if o.alive { "survive" } else { "exit" },
            censored: o.alive,
            budget: final_day,
            firm: o.firm,
            funds: o.funds,
        };
        run.log_event("terminal", &event)
            .unwrap_or_else(|e| panic!("店舗 {} の terminal の記録に失敗: {e}", o.firm));
    }
}

/// 観測主体の id．店舗の `AgentId` をそのまま使う．
fn unit_id(firm: u64) -> String {
    format!("firm-{firm}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use runvault::meta::TargetKind;

    #[test]
    fn the_work_id_agrees_with_the_arxiv_id() {
        let research: runvault::meta::Research = replication().into();
        let work = research.work.expect("再現実験なので work がある");
        assert_eq!(work.work_id, "arxiv:2310.17512");
        assert_eq!(work.paper_id.as_deref(), Some("P00001797"));
    }

    #[test]
    fn the_targets_are_the_table_and_the_two_headline_claims() {
        let research: runvault::meta::Research = replication().into();
        assert_eq!(research.targets.len(), 3);
        assert!(matches!(research.targets[0].kind, TargetKind::Table));
        assert!(matches!(research.targets[1].kind, TargetKind::Claim));
        assert!(matches!(research.targets[2].kind, TargetKind::Claim));
    }

    #[test]
    fn a_run_that_reproduces_the_paper_passes_the_research_checks() {
        let research: runvault::meta::Research = replication().into();
        runvault::verify::check_research(&research).expect("research の検査に失敗");
    }

    #[test]
    fn the_provider_comes_from_the_endpoint() {
        assert_eq!(llm_block("m", "mock://scripted", 0.0).provider, "mock");
        assert_eq!(
            llm_block("m", "https://api.openai.com/v1", 0.0).provider,
            "openai"
        );
        assert_eq!(
            llm_block("m", "http://localhost:11434", 0.0).provider,
            "ollama"
        );
    }
}

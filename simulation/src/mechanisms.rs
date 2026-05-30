//! socsim フレームワーク上の CompeteAI 市場競争メカニズム (5 Mechanism × 6 phase)．
//!
//! 二層アーキテクチャの **境界** がここにある．下層 (決定論的 socsim コア) は
//! 活性化順・グループ多数決の同点処理を `ctx.rng` (ChaCha20) で行い，上層
//! (非決定的 LLM レイヤ) は [`CompeteClient`] (キャッシュ付き Ollama→OpenAI
//! フォールバック) 越しの店舗戦略立案・顧客選択を行う．
//!
//! 論文の日次ステップ (市場リセット → 店舗戦略 → 顧客選択 → 来店マッチング →
//! 報酬 → 反省) を 6-phase へ割り当てる:
//!
//! | Mechanism | Phase | 役割 |
//! |-----------|-------|------|
//! | [`MarketResetMechanism`]         | Environment  | 当日の市場リセット・前日 daybook / ライバル情報の準備 |
//! | [`CompetitionMatthewMechanism`]  | Decision     | LLM 店舗が反省し戦略 (価格/シェフ/広告) を改訂 + オファースナップショット |
//! | [`CustomerChoiceMechanism`]      | Decision     | LLM 顧客が提示情報から来店店を選択 (個人=単独, グループ=多数決) |
//! | [`PatronageMechanism`]           | Interaction  | 顧客−店舗マッチング・食事体験・コメント生成と可視化 |
//! | [`RevenueRewardMechanism`]       | Reward       | 収益/スコア・評判更新 + マタイ効果指標 (Gini/市場シェア) の記録 |
//! | [`ReflectionMechanism`]          | PostStep     | 各エージェントの記憶要約・店舗撤退判定・収束/終了判定 |
//!
//! LLM 呼び出しは Decision フェーズの 2 mechanism に閉じ込める (`MarketReset` /
//! `Patronage` / `RevenueReward` / `Reflection` は LLM 非依存)．LLM クライアントと
//! 呼び出しメタデータは `Rc<RefCell<…>>` で共有し，run ドライバが実行後に
//! キャッシュ保存・メタデータ集計に使う．日次指標も共有バッファ経由でドライバへ渡す．

use std::cell::RefCell;
use std::rc::Rc;

use rand::Rng;

use socsim_core::{AgentId, Mechanism, Phase, Result, SocsimError, StepContext};
use socsim_llm::MetadataCollector;

use crate::config::{CustomerMode, LlmSettings};
use crate::llm::{llm_config, CompeteClient};
use crate::metrics::{market_share_max, mean, menu_similarity, revenue_gini, DailyMetric};
use crate::prompts::{
    customer_choice_prompt, firm_strategy_prompt, parse_customer_choice, parse_firm_strategy,
    FirmBriefing, FirmOffer, RivalInfo,
};
use crate::world::{Comment, MarketWorld, Visit};

/// 共有 LLM クライアント (run ドライバとメカニズムで共有)．
pub type SharedClient = Rc<RefCell<CompeteClient>>;
/// 共有メタデータコレクタ (cache-hit 率などを run 後に集計)．
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;
/// 共有 日次指標バッファ (ドライバが run 後に CSV へ書き出す)．
pub type SharedMetrics = Rc<RefCell<Vec<DailyMetric>>>;

// scratch キー．同一ステップ内でフェーズ間に値を受け渡す (engine がステップ冒頭で
// クリアする)．
const SCRATCH_OFFERS: &str = "firm_offers";
const SCRATCH_CHOICES: &str = "customer_choices";

/// 顧客の来店決定 (customer raw id, 選ばれた店舗 raw id)．
type Choice = (u64, u64);

// =========================================================================== //
// 1. MarketResetMechanism (Environment)
// =========================================================================== //

/// 当日の市場状態をリセットし，前日 daybook / ライバル情報を準備する
/// (`Environment` フェーズ; LLM 非依存)．
///
/// 前日の `market.day_customers` / `day_revenue` を読んで店舗ごとの briefing を
/// 組み立て，scratch ではなく world (一時的に market を残す) から Decision が
/// 参照できるようにする．本実装では briefing をプロンプト構築時に world から直接
/// 引くため，ここでは市場のリセットのみ前日値を退避してから行う．
pub struct MarketResetMechanism;

impl Mechanism<MarketWorld> for MarketResetMechanism {
    fn name(&self) -> &str {
        "market_reset"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Environment]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        let day = ctx.world.current_day();
        ctx.world.day = day;
        // 前日の daybook はこのフェーズ完了後に Decision が world.market から読むため，
        // ここではリセットせず Decision 完了後の Interaction 冒頭でリセットする…
        // …のではなく，前日値を scratch に退避してから当日分をクリアする．
        let prev_customers: Vec<(u64, u64)> = ctx
            .world
            .market
            .day_customers
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        let prev_revenue: Vec<(u64, f64)> = ctx
            .world
            .market
            .day_revenue
            .iter()
            .map(|(k, v)| (*k, *v))
            .collect();
        ctx.scratch.insert("prev_customers", prev_customers);
        ctx.scratch.insert("prev_revenue", prev_revenue);
        ctx.world.market.reset();
        Ok(())
    }
}

// =========================================================================== //
// 2. CompetitionMatthewMechanism (Decision, LLM)
// =========================================================================== //

/// LLM 店舗が前日 daybook・ライバル情報・記憶から反省し，戦略 (価格倍率・シェフ
/// 給与・広告) を改訂する (`Decision` フェーズ; LLM 所在 1)．
///
/// 改訂後，全店舗の当日オファー (平均価格・スコア・評判・広告・代表コメント) を
/// scratch にスナップショットし，後続の `CustomerChoiceMechanism` / `Patronage` へ
/// 渡す (同期更新セマンティクス)．
pub struct CompetitionMatthewMechanism {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
}

impl CompetitionMatthewMechanism {
    pub fn new(client: SharedClient, metadata: SharedMetadata, settings: LlmSettings) -> Self {
        CompetitionMatthewMechanism {
            client,
            metadata,
            settings,
        }
    }
}

impl Mechanism<MarketWorld> for CompetitionMatthewMechanism {
    fn name(&self) -> &str {
        "competition_matthew"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        let day = ctx.world.day;
        // 前日 daybook を scratch から復元する．
        let prev_customers: std::collections::BTreeMap<u64, u64> = ctx
            .scratch
            .get::<Vec<(u64, u64)>>("prev_customers")
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();
        let prev_revenue: std::collections::BTreeMap<u64, f64> = ctx
            .scratch
            .get::<Vec<(u64, f64)>>("prev_revenue")
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect();

        let alive_ids = ctx.world.alive_firm_ids();

        for id in &alive_ids {
            // ライバル情報 (自分以外の生存店)．
            let rivals: Vec<RivalInfo> = alive_ids
                .iter()
                .filter(|rid| *rid != id)
                .map(|rid| {
                    let f = &ctx.world.firms[rid];
                    RivalInfo {
                        day_customers: *prev_customers.get(&rid.0).unwrap_or(&0),
                        day_revenue: *prev_revenue.get(&rid.0).unwrap_or(&0.0),
                        avg_price: f.avg_price(),
                        avg_score: f.avg_dish_score(),
                    }
                })
                .collect();
            let brief = FirmBriefing {
                own_customers: *prev_customers.get(&id.0).unwrap_or(&0),
                own_revenue: *prev_revenue.get(&id.0).unwrap_or(&0.0),
                rivals,
            };

            let firm = ctx.world.firms.get(id).expect("alive firm exists").clone();
            let prompt = firm_strategy_prompt(&firm, day, &brief);
            let text = {
                let mut client = self.client.borrow_mut();
                let resp = client
                    .complete(&prompt, &llm_config(&self.settings))
                    .map_err(|e| {
                        SocsimError::Mechanism(format!("firm strategy LLM call failed: {e}"))
                    })?;
                self.metadata.borrow_mut().record(resp.metadata.clone());
                resp.text
            };
            let strat = parse_firm_strategy(&text, firm.chef_salary);

            // 戦略を適用する: 価格倍率を全料理に乗じ，シェフ給与を更新し
            // (各料理の chef_salary に反映)，広告を差し替える．
            if let Some(f) = ctx.world.firms.get_mut(id) {
                for dish in f.menu.iter_mut() {
                    dish.price = (dish.price * strat.price_factor).max(1.0);
                    dish.chef_salary = strat.chef_salary;
                }
                f.chef_salary = strat.chef_salary;
                if !strat.advertisement.is_empty() {
                    f.advertisement = strat.advertisement;
                }
            }
        }

        // 全生存店の当日オファーをスナップショットして scratch へ (同期更新)．
        let offers: Vec<FirmOfferSnapshot> = ctx
            .world
            .alive_firm_ids()
            .iter()
            .map(|id| {
                let f = &ctx.world.firms[id];
                let recent: Vec<String> = f
                    .comments
                    .iter()
                    .rev()
                    .take(3)
                    .map(|c| c.text.clone())
                    .collect();
                FirmOfferSnapshot {
                    firm: id.0,
                    avg_price: f.avg_price(),
                    avg_score: f.avg_dish_score(),
                    reputation: f.reputation,
                    advertisement: f.advertisement.clone(),
                    recent_comments: recent,
                }
            })
            .collect();
        ctx.scratch.insert(SCRATCH_OFFERS, offers);
        Ok(())
    }
}

/// scratch に置くオファースナップショット (FirmOffer の所有版)．
#[derive(Clone)]
pub struct FirmOfferSnapshot {
    pub firm: u64,
    pub avg_price: f64,
    pub avg_score: f64,
    pub reputation: f64,
    pub advertisement: String,
    pub recent_comments: Vec<String>,
}

impl FirmOfferSnapshot {
    fn as_offer(&self) -> FirmOffer {
        FirmOffer {
            firm: self.firm,
            avg_price: self.avg_price,
            avg_score: self.avg_score,
            reputation: self.reputation,
            advertisement: self.advertisement.clone(),
            recent_comments: self.recent_comments.clone(),
        }
    }
}

// =========================================================================== //
// 3. CustomerChoiceMechanism (Decision, LLM)
// =========================================================================== //

/// LLM 顧客が提示された店舗情報 (スコア・広告・価格・コメント) から来店店を選ぶ
/// (`Decision` フェーズ; LLM 所在 2)．
///
/// 個人客は単独で選び，グループ客はメンバー各自の選好を集計して多数決で決める
/// (同点は `ctx.rng` で決定論的に解消)．選択結果は scratch に積み，後続の
/// `PatronageMechanism` がマッチングに使う．
pub struct CustomerChoiceMechanism {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    customer_mode: CustomerMode,
}

impl CustomerChoiceMechanism {
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        customer_mode: CustomerMode,
    ) -> Self {
        CustomerChoiceMechanism {
            client,
            metadata,
            settings,
            customer_mode,
        }
    }

    /// 1 顧客の LLM 選択 (offer index)．失敗時はスコア最大店へフォールバック．
    ///
    /// `group_deliberation = true` のときグループ熟議の枠組みでプロンプトを構築し，
    /// 人気への同調を抑えた «自分の予算・好み» 本位の選択を促す．
    fn choose_one(
        &self,
        ctx_meta: &SharedMetadata,
        customer: &crate::world::Customer,
        day: u64,
        offers: &[FirmOfferSnapshot],
        group_deliberation: bool,
    ) -> Result<usize> {
        let offer_views: Vec<FirmOffer> = offers.iter().map(|o| o.as_offer()).collect();
        let prompt = customer_choice_prompt(customer, day, &offer_views, group_deliberation);
        let text = {
            let mut client = self.client.borrow_mut();
            let resp = client
                .complete(&prompt, &llm_config(&self.settings))
                .map_err(|e| {
                    SocsimError::Mechanism(format!("customer choice LLM call failed: {e}"))
                })?;
            ctx_meta.borrow_mut().record(resp.metadata.clone());
            resp.text
        };
        let idx = parse_customer_choice(&text, offers.len()).unwrap_or_else(|| {
            // フォールバック: 平均スコア最大の店 (タイは先頭)．
            offers
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.avg_score
                        .partial_cmp(&b.avg_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        });
        Ok(idx)
    }
}

impl Mechanism<MarketWorld> for CustomerChoiceMechanism {
    fn name(&self) -> &str {
        "customer_choice"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        let day = ctx.world.day;
        let offers: Vec<FirmOfferSnapshot> =
            match ctx.scratch.get::<Vec<FirmOfferSnapshot>>(SCRATCH_OFFERS) {
                Some(o) if !o.is_empty() => o.clone(),
                // 生存店が無い (全撤退) → 来店なし．
                _ => {
                    ctx.scratch.insert(SCRATCH_CHOICES, Vec::<Choice>::new());
                    return Ok(());
                }
            };

        // 顧客 ID をソート順に処理する (決定論)．agent_order は店舗も含むので
        // customers のキーを直接使う．
        let customer_ids: Vec<AgentId> = ctx.world.customers.keys().copied().collect();
        let mut choices: Vec<Choice> = Vec::with_capacity(customer_ids.len());

        match self.customer_mode {
            CustomerMode::Individual => {
                for cid in &customer_ids {
                    let customer = ctx.world.customers[cid].clone();
                    let idx = self.choose_one(&self.metadata, &customer, day, &offers, false)?;
                    choices.push((cid.0, offers[idx].firm));
                }
            }
            CustomerMode::Group => {
                use std::collections::BTreeMap;
                // グループ ID → メンバー (customer id) を集める (None は単独グループ扱い)．
                let mut groups: BTreeMap<u64, Vec<AgentId>> = BTreeMap::new();
                let mut solo: Vec<AgentId> = Vec::new();
                for cid in &customer_ids {
                    match ctx.world.customers[cid].group {
                        Some(g) => groups.entry(g).or_default().push(*cid),
                        None => solo.push(*cid),
                    }
                }
                // 単独客は個別に選ぶ (グループに属さない顧客はグループ熟議の枠組みを
                // 受けないが，group モードの市場に同居しているため熟議フラグで揃える)．
                for cid in &solo {
                    let customer = ctx.world.customers[cid].clone();
                    let idx = self.choose_one(&self.metadata, &customer, day, &offers, true)?;
                    choices.push((cid.0, offers[idx].firm));
                }
                // グループは «熟議 (deliberation)» の結果，メンバーを複数店へ
                // 振り分ける (= 家族/同僚が必ずしも同じ店に揃わず，各自の選好を活かして
                // 別々の店に分かれて食事することもある)．個別客が «社会的証明 (人気)»
                // に流されて流行店へ雪崩を打つのに対し，グループ内の熟議は少数派
                // (価格重視・別嗜好) の声を顕在化させ，メンバーを各店へ配分する．
                //
                // 実装: メンバーの個別投票 (得票) を集め，得票数に比例して各店へ
                // 「最大剰余法」で議席 (来店者数) を割り当てる．これにより異質な
                // グループほど来店が分散し，正のフィードバックループ (人気 → さらに
                // 人気) が攪乱されて勝者総取りが緩和される (論文 個人 66.7% →
                // グループ 16.7%)．端数の配分は engine RNG で決定論的に解消する．
                for members in groups.values() {
                    let mut votes = vec![0u32; offers.len()];
                    for cid in members {
                        let customer = ctx.world.customers[cid].clone();
                        let idx = self.choose_one(&self.metadata, &customer, day, &offers, true)?;
                        votes[idx] += 1;
                    }
                    let seats = apportion_seats(&votes, members.len(), &mut ctx.rng);
                    // 各店の議席数だけ，グループのメンバーを順に割り当てる．
                    let mut member_iter = members.iter();
                    for (offer_idx, &count) in seats.iter().enumerate() {
                        for _ in 0..count {
                            if let Some(cid) = member_iter.next() {
                                choices.push((cid.0, offers[offer_idx].firm));
                            }
                        }
                    }
                }
            }
        }

        ctx.scratch.insert(SCRATCH_CHOICES, choices);
        Ok(())
    }
}

// =========================================================================== //
// 4. PatronageMechanism (Interaction)
// =========================================================================== //

/// 顧客−店舗のマッチング (来店確定)・食事体験・コメント生成と他顧客への可視化
/// (`Interaction` フェーズ; LLM 非依存)．
///
/// scratch の選択結果に従い，各顧客の来店を確定する: 店舗の代表料理 (最安値) を
/// 1 品消費し，支払額を当日収益へ，客数を当日客数へ加算する．満足度は料理品質
/// スコアと価格手頃感から決め，コメント (テンプレート) を店舗へ残し，顧客の
/// `visit_memory` に追記する．コメントは翌日以降の他顧客の選択に可視化される．
pub struct PatronageMechanism;

impl Mechanism<MarketWorld> for PatronageMechanism {
    fn name(&self) -> &str {
        "patronage"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        let day = ctx.world.day;
        let choices: Vec<Choice> = ctx
            .scratch
            .get::<Vec<Choice>>(SCRATCH_CHOICES)
            .cloned()
            .unwrap_or_default();

        for (customer_raw, firm_raw) in choices {
            let firm_id = AgentId(firm_raw);
            let customer_id = AgentId(customer_raw);

            // 店舗の提供料理 (最安値の 1 品) と品質スコアを取り出す．
            let (spend, score) = {
                let firm = match ctx.world.firms.get(&firm_id) {
                    Some(f) if f.alive => f,
                    _ => continue, // 撤退店は来店確定しない．
                };
                let dish = firm
                    .menu
                    .iter()
                    .min_by(|a, b| {
                        a.price
                            .partial_cmp(&b.price)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .cloned();
                match dish {
                    Some(d) => (d.price, d.score()),
                    None => continue, // メニューが空なら来店なし．
                }
            };

            // 満足度: 品質スコア (0〜1 想定) を 5 点満点へ写像し，予算超過でペナルティ．
            let income = ctx
                .world
                .customers
                .get(&customer_id)
                .map(|c| c.income)
                .unwrap_or(0.0);
            let affordability = if spend <= income {
                1.0
            } else {
                income / spend.max(1.0)
            };
            let satisfaction = (5.0 * score * affordability).clamp(0.0, 5.0);

            // 当日市場へ計上．
            *ctx.world.market.day_customers.entry(firm_raw).or_insert(0) += 1;
            *ctx.world.market.day_revenue.entry(firm_raw).or_insert(0.0) += spend;

            // 顧客の来店記憶を更新．
            if let Some(c) = ctx.world.customers.get_mut(&customer_id) {
                c.visit_memory.push(Visit {
                    day,
                    firm: firm_raw,
                    spend,
                    satisfaction,
                });
            }

            // コメントを店舗へ残す (他顧客に可視; テンプレート生成)．
            let text = comment_text(satisfaction);
            if let Some(f) = ctx.world.firms.get_mut(&firm_id) {
                f.comments.push(Comment {
                    customer: customer_raw,
                    text,
                    rating: satisfaction,
                    day,
                });
                // コメント履歴は直近 50 件に切り詰める．
                if f.comments.len() > 50 {
                    let excess = f.comments.len() - 50;
                    f.comments.drain(0..excess);
                }
            }
        }
        Ok(())
    }
}

/// グループのメンバー (合計 `n` 人) を，各店の得票 `votes` に比例して «議席»
/// (来店者数) へ配分する (最大剰余法; Hare quota)．
///
/// 整数配分で生じた端数 (`n - Σ floor`) は剰余の大きい店から 1 人ずつ与え，剰余が
/// 同点の店は engine RNG で決定論的に解消する．これにより «グループは複数店へ
/// 分かれて来店する» 熟議の結果を表現し，個別客の同調 (人気店への雪崩) に対して
/// 市場を分散させる (勝者総取りの緩和)．得票が 1 店に集中したグループは全員その
/// 店へ来る (= 強い合意は尊重する)．
fn apportion_seats<R: Rng>(votes: &[u32], n: usize, rng: &mut R) -> Vec<usize> {
    let total: u32 = votes.iter().sum();
    if total == 0 || n == 0 {
        return vec![0; votes.len()];
    }
    // 各店の理想配分 = n * vote / total．floor を確定議席に，剰余を控える．
    let mut seats = vec![0usize; votes.len()];
    let mut remainders: Vec<(usize, f64)> = Vec::with_capacity(votes.len());
    let mut assigned = 0usize;
    for (i, &v) in votes.iter().enumerate() {
        let ideal = n as f64 * v as f64 / total as f64;
        let floor = ideal.floor() as usize;
        seats[i] = floor;
        assigned += floor;
        remainders.push((i, ideal - floor as f64));
    }
    // 残り議席を剰余の大きい順に配分 (剰余同点は RNG で順序を乱して公平化)．
    let mut leftover = n.saturating_sub(assigned);
    // 剰余降順，同点は RNG キーで決定論的にシャッフルしてから安定ソート．
    let mut keyed: Vec<(usize, f64, u32)> = remainders
        .iter()
        .map(|&(i, r)| (i, r, rng.gen::<u32>()))
        .collect();
    keyed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.2.cmp(&b.2))
    });
    for (i, _, _) in keyed {
        if leftover == 0 {
            break;
        }
        seats[i] += 1;
        leftover -= 1;
    }
    seats
}

/// 満足度からテンプレートコメントを生成する (LLM 非依存; 決定論)．
fn comment_text(satisfaction: f64) -> String {
    if satisfaction >= 4.0 {
        "Excellent food and great value — highly recommend!".to_string()
    } else if satisfaction >= 2.5 {
        "Decent meal, fair for the price.".to_string()
    } else {
        "Overpriced and underwhelming.".to_string()
    }
}

// =========================================================================== //
// 5. RevenueRewardMechanism (Reward)
// =========================================================================== //

/// 収益/スコア計算・評判更新・マタイ効果指標の記録 (`Reward` フェーズ; LLM 非依存)．
///
/// 当日収益を店舗資金・累積収益・累積客数へ反映し，原価とシェフ給与を差し引いて
/// 純損益を資金に計上する (資源制約)．評判は当日コメントの平均評価で指数移動平均
/// 更新する．日次の集計指標 (収益 Gini・最大市場シェア・メニュー類似度) を計算し，
/// 店舗ごとの long-format 行を共有バッファへ push する．
pub struct RevenueRewardMechanism {
    metrics: SharedMetrics,
}

impl RevenueRewardMechanism {
    pub fn new(metrics: SharedMetrics) -> Self {
        RevenueRewardMechanism { metrics }
    }
}

impl Mechanism<MarketWorld> for RevenueRewardMechanism {
    fn name(&self) -> &str {
        "revenue_reward"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Reward]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        let day = ctx.world.day;
        let firm_ids: Vec<AgentId> = ctx.world.firms.keys().copied().collect();

        // 当日コメントの店舗別平均評価 (評判更新用)．
        for id in &firm_ids {
            let firm_raw = id.0;
            let day_rev = *ctx.world.market.day_revenue.get(&firm_raw).unwrap_or(&0.0);
            let day_cust = *ctx.world.market.day_customers.get(&firm_raw).unwrap_or(&0);

            // 原価: 当日提供した料理の原価合計の代理として «客数 × 平均原価» を使う．
            // シェフ給与は固定費として日次で差し引く．
            let (avg_cost, chef_salary) = {
                let f = &ctx.world.firms[id];
                let avg_cost = if f.menu.is_empty() {
                    0.0
                } else {
                    f.menu.iter().map(|d| d.cost).sum::<f64>() / f.menu.len() as f64
                };
                (avg_cost, f.chef_salary)
            };
            let cost_total = avg_cost * day_cust as f64;
            // シェフ給与は日次按分 (月給の 1/30 を 1 日の固定費とみなす)．
            let daily_fixed = chef_salary / 30.0;
            let net = day_rev - cost_total - daily_fixed;

            // 当日コメントの平均評価．
            let today_ratings: Vec<f64> = ctx.world.firms[id]
                .comments
                .iter()
                .filter(|c| c.day == day)
                .map(|c| c.rating)
                .collect();
            let avg_rating = mean(&today_ratings);

            if let Some(f) = ctx.world.firms.get_mut(id) {
                f.funds += net;
                f.cumulative_revenue += day_rev;
                f.cumulative_customers += day_cust;
                if !today_ratings.is_empty() {
                    // 指数移動平均で評判を更新 (係数 0.3)．
                    f.reputation = 0.7 * f.reputation + 0.3 * avg_rating;
                }
            }
        }

        // --- 日次集計指標 ---
        let cumulative: Vec<f64> = firm_ids
            .iter()
            .map(|id| ctx.world.firms[id].cumulative_revenue)
            .collect();
        let gini = revenue_gini(&cumulative);

        let day_customers_vec: Vec<u64> = firm_ids
            .iter()
            .map(|id| *ctx.world.market.day_customers.get(&id.0).unwrap_or(&0))
            .collect();
        let share_max = market_share_max(&day_customers_vec);

        let alive_firms: Vec<&crate::world::Firm> = firm_ids
            .iter()
            .map(|id| &ctx.world.firms[id])
            .filter(|f| f.alive)
            .collect();
        let menu_sim = menu_similarity(&alive_firms);
        let n_alive = ctx.world.n_alive_firms() as u64;

        // 店舗ごとの long-format 行を push する．
        for id in &firm_ids {
            let f = &ctx.world.firms[id];
            let row = DailyMetric {
                day,
                firm: id.0,
                firm_alive: if f.alive { 1 } else { 0 },
                day_customers: *ctx.world.market.day_customers.get(&id.0).unwrap_or(&0),
                day_revenue: *ctx.world.market.day_revenue.get(&id.0).unwrap_or(&0.0),
                cumulative_revenue: f.cumulative_revenue,
                avg_dish_score: f.avg_dish_score(),
                avg_price: f.avg_price(),
                reputation: f.reputation,
                revenue_gini: gini,
                market_share_max: share_max,
                menu_similarity: menu_sim,
                n_alive_firms: n_alive,
            };
            self.metrics.borrow_mut().push(row);
        }

        // ドライバ観測用に最新の集計値を scratch にも置く．
        ctx.scratch.insert("revenue_gini", gini);
        ctx.scratch.insert("market_share_max", share_max);
        Ok(())
    }
}

// =========================================================================== //
// 6. ReflectionMechanism (PostStep)
// =========================================================================== //

/// 各エージェントの記憶要約・店舗撤退判定・収束/終了判定 (`PostStep` フェーズ;
/// LLM 非依存)．
///
/// 店舗には当日の所感をテンプレートで記憶へ追記する (LLM 戦略立案の入力)．資金が
/// 尽きた (`funds < 0`) 店舗は撤退 (`alive = false`)．いずれかの店舗が撤退するか
/// 最終日 (`day == days - 1`) に達したら `ctx.request_stop()` を発火する (論文の
/// 終了条件)．
pub struct ReflectionMechanism {
    days: u64,
}

impl ReflectionMechanism {
    pub fn new(days: u64) -> Self {
        ReflectionMechanism { days }
    }
}

impl Mechanism<MarketWorld> for ReflectionMechanism {
    fn name(&self) -> &str {
        "reflection"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, MarketWorld>) -> Result<()> {
        let day = ctx.world.day;
        let firm_ids: Vec<AgentId> = ctx.world.firms.keys().copied().collect();

        let mut a_firm_exited = false;
        for id in &firm_ids {
            let day_cust = *ctx.world.market.day_customers.get(&id.0).unwrap_or(&0);
            let day_rev = *ctx.world.market.day_revenue.get(&id.0).unwrap_or(&0.0);
            if let Some(f) = ctx.world.firms.get_mut(id) {
                if !f.alive {
                    continue;
                }
                let entry = format!(
                    "day {}: {} customers, revenue {:.0}, funds {:.0}, avg price {:.0}",
                    day,
                    day_cust,
                    day_rev,
                    f.funds,
                    f.avg_price()
                );
                f.memory.push(entry);
                if f.memory.len() > 10 {
                    let excess = f.memory.len() - 10;
                    f.memory.drain(0..excess);
                }
                // 撤退判定: 資金が尽きたら退出．
                if f.funds < 0.0 {
                    f.alive = false;
                    a_firm_exited = true;
                }
            }
        }

        // 終了条件: 店舗撤退 or 最終日．
        let is_last_day = day + 1 >= self.days;
        if a_firm_exited || is_last_day {
            ctx.request_stop();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comment_text_tiers() {
        assert!(comment_text(4.5).contains("recommend"));
        assert!(comment_text(3.0).contains("Decent"));
        assert!(comment_text(1.0).contains("Overpriced"));
    }
}

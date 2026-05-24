//! socsim フレームワーク上の CompeteAI 市場競争シミュレーションの世界状態．
//!
//! エージェント = 移動する空間主体ではなく，市場 (顧客−店舗の二部マッチング) を
//! 通じて相互作用する固定のレストラン (競争者) と顧客 (審判) である．したがって
//! `socsim-grid` (`GridIndex` / `CellGrid`) は採用せず，店舗属性を
//! `BTreeMap<AgentId, Firm>` に，顧客属性を `BTreeMap<AgentId, Customer>` に，
//! 当日の市場ブロックを単一の [`Market`] に保持する．店舗と顧客を 1 つの
//! `AgentId` 空間に同居させ，種別 ([`AgentKind`]) で区別する．`agent_ids()` は
//! 店舗 ID → 顧客 ID の順 (各 `BTreeMap` は昇順キー) を返し決定論を担保する
//! (socsim コア層)．
//!
//! `#[derive(Clone, Serialize, Deserialize)]` でスナップショット (save/resume) と
//! 感度分析の比較実験に対応する．

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use socsim_core::{AgentId, SimClock, WorldState};

/// 1 つの料理 (メニュー項目)．論文 §2.5 の品質スコアを持つ．
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Dish {
    /// 料理名 (差別化・模倣の観測用)．
    pub name: String,
    /// 原価 c．
    pub cost: f64,
    /// 販売価格 p．
    pub price: f64,
    /// この料理を提供するシェフの給与 f (品質スコアの第 2 項を駆動)．
    pub chef_salary: f64,
}

impl Dish {
    /// 料理品質スコア `s = 0.5 * c/p + 0.5 * f/5000` (論文 §2.5)．
    ///
    /// 第 1 項は原価率 (顧客が受け取る価値の代理)，第 2 項はシェフ技能 (給与連動)．
    /// 価格 0 などの退化入力では原価率項を 0 に倒す．
    pub fn score(&self) -> f64 {
        let cost_ratio = if self.price > 1e-9 {
            self.cost / self.price
        } else {
            0.0
        };
        0.5 * cost_ratio + 0.5 * (self.chef_salary / 5000.0)
    }
}

/// 顧客が残した来店コメント (他顧客の意思決定にも可視)．
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Comment {
    /// コメントを残した顧客の `AgentId` (生の `u64`)．
    pub customer: u64,
    /// 本文 (LLM 生成 or テンプレート)．
    pub text: String,
    /// 体験の主観評価 (0〜5; 評判更新に使う)．
    pub rating: f64,
    /// 何日目のコメントか．
    pub day: u64,
}

/// レストラン (競争者)．戦略状態・評判・累積実績・記憶を持つ．
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Firm {
    /// 資金 (資源制約; 負になり日次赤字が続けば撤退)．
    pub funds: f64,
    /// メニュー (各 `Dish { cost, price, chef_salary }`)．
    pub menu: Vec<Dish>,
    /// シェフ給与 f (メニュー全体の品質を底上げするレバー)．
    pub chef_salary: f64,
    /// 広告テキスト (顧客選択プロンプトに提示)．
    pub advertisement: String,
    /// 評判スコア (顧客コメントの主観評価の集約)．
    pub reputation: f64,
    /// 累積客数 (マタイ効果の駆動量)．
    pub cumulative_customers: u64,
    /// 累積収益．
    pub cumulative_revenue: f64,
    /// 顧客コメント履歴 (他顧客に可視)．
    pub comments: Vec<Comment>,
    /// 反省・要約の記憶 (LLM 戦略立案の入力)．
    pub memory: Vec<String>,
    /// レースから降りたら false (撤退)．
    pub alive: bool,
}

impl Firm {
    /// 初期資金・初期メニュー・シェフ給与から店舗を作る．
    pub fn new(funds: f64, menu: Vec<Dish>, chef_salary: f64) -> Self {
        Firm {
            funds,
            menu,
            chef_salary,
            advertisement: String::new(),
            reputation: 0.0,
            cumulative_customers: 0,
            cumulative_revenue: 0.0,
            comments: Vec::new(),
            memory: Vec::new(),
            alive: true,
        }
    }

    /// メニュー平均品質スコア (空メニューは 0)．
    pub fn avg_dish_score(&self) -> f64 {
        if self.menu.is_empty() {
            return 0.0;
        }
        self.menu.iter().map(Dish::score).sum::<f64>() / self.menu.len() as f64
    }

    /// メニュー平均価格 (空メニューは 0)．
    pub fn avg_price(&self) -> f64 {
        if self.menu.is_empty() {
            return 0.0;
        }
        self.menu.iter().map(|d| d.price).sum::<f64>() / self.menu.len() as f64
    }
}

/// グループ識別子 (家族・同僚・カップル・友人など顧客のまとまり)．
pub type GroupId = u64;

/// 1 回の来店記録 (顧客の記憶)．
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Visit {
    /// 何日目か．
    pub day: u64,
    /// 訪れた店舗の `AgentId` (生の `u64`)．
    pub firm: u64,
    /// 支払った金額．
    pub spend: f64,
    /// 主観満足度 (0〜5)．
    pub satisfaction: f64,
}

/// 顧客 (審判)．特性・記憶を持つ．個人客とグループ客を表現する．
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Customer {
    /// 所得 (1 日に使える予算の代理)．
    pub income: f64,
    /// 嗜好 (LLM 選択プロンプトに提示するテキスト特性)．
    pub preference: String,
    /// 健康状態・食事制限 (同上)．
    pub health: String,
    /// 所属グループ (None = 個人客)．グループは多数決で来店店を決める．
    pub group: Option<GroupId>,
    /// 過去の来店・体験の記憶 (LLM 選択の入力)．
    pub visit_memory: Vec<Visit>,
}

impl Customer {
    /// 個人客 (グループ無所属) を作る．
    pub fn new(income: f64, preference: String, health: String) -> Self {
        Customer {
            income,
            preference,
            health,
            group: None,
            visit_memory: Vec::new(),
        }
    }
}

/// エージェント種別 (店舗 / 顧客)．`AgentId` 空間を分割するためのレイアウト規約．
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentKind {
    /// レストラン (競争者)．
    Firm,
    /// 顧客 (審判)．
    Customer,
}

/// 店舗 `AgentId` は `[0, FIRM_ID_BASE + n_firms)` を，顧客 `AgentId` は
/// `[FIRM_ID_BASE_CUSTOMER, …)` を占める．種別判定を `AgentId` の範囲で決定論的に
/// 行うため，顧客 ID は十分大きなオフセットから始める．
pub const CUSTOMER_ID_BASE: u64 = 1_000_000;

/// 当日の市場ブロック (オファースナップショット・マッチング・指標スクラッチ)．
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Market {
    /// 当日の各店舗の来店客数 (店舗 `AgentId` の生 `u64` → 客数)．
    pub day_customers: BTreeMap<u64, u64>,
    /// 当日の各店舗の収益 (店舗 `AgentId` の生 `u64` → 収益)．
    pub day_revenue: BTreeMap<u64, f64>,
}

impl Market {
    /// 当日の市場状態をリセットする (新しい日の冒頭で呼ぶ)．
    pub fn reset(&mut self) {
        self.day_customers.clear();
        self.day_revenue.clear();
    }
}

/// CompeteAI 市場競争シミュレーションの世界状態．
#[derive(Clone, Serialize, Deserialize)]
pub struct MarketWorld {
    /// シミュレーションクロック (1 tick = 1 日)．
    pub clock: SimClock,
    /// 各レストラン (競争者) の状態 (ソート済みキー)．
    pub firms: BTreeMap<AgentId, Firm>,
    /// 各顧客 (審判) の状態 (ソート済みキー)．
    pub customers: BTreeMap<AgentId, Customer>,
    /// 当日の市場ブロック．
    pub market: Market,
    /// ラウンド (= 1 日; 0 始まり)．
    pub day: u64,
}

impl MarketWorld {
    /// 店舗数 M．
    pub fn n_firms(&self) -> usize {
        self.firms.len()
    }

    /// 顧客数 N．
    pub fn n_customers(&self) -> usize {
        self.customers.len()
    }

    /// 現在の日 (0 始まり)．
    ///
    /// socsim エンジンはステップ先頭で `tick()` するため，クロックは 1..=days を
    /// 走る．本モデルは日を 0 始まり (0..days) で扱うので `t() - 1` を返す．
    pub fn current_day(&self) -> u64 {
        self.clock.t().saturating_sub(1)
    }

    /// 与えられた `AgentId` の種別を範囲で判定する (決定論)．
    pub fn kind_of(id: AgentId) -> AgentKind {
        if id.0 >= CUSTOMER_ID_BASE {
            AgentKind::Customer
        } else {
            AgentKind::Firm
        }
    }

    /// 生存している店舗 ID をソート順で返す．
    pub fn alive_firm_ids(&self) -> Vec<AgentId> {
        self.firms
            .iter()
            .filter(|(_, f)| f.alive)
            .map(|(id, _)| *id)
            .collect()
    }

    /// 生存している店舗数．
    pub fn n_alive_firms(&self) -> usize {
        self.firms.values().filter(|f| f.alive).count()
    }
}

impl WorldState for MarketWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        // 店舗 ID (昇順) のあとに顧客 ID (昇順)．両 BTreeMap のキーはソート済みで，
        // 顧客 ID は CUSTOMER_ID_BASE 以上なので連結後も全体としてソート済み → 決定論．
        let mut ids: Vec<AgentId> = self.firms.keys().copied().collect();
        ids.extend(self.customers.keys().copied());
        ids
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dish_score_formula() {
        // c=2000, p=4000, f=2500 → 0.5*0.5 + 0.5*0.5 = 0.5
        let d = Dish {
            name: "x".into(),
            cost: 2000.0,
            price: 4000.0,
            chef_salary: 2500.0,
        };
        assert!((d.score() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn dish_score_zero_price_safe() {
        let d = Dish {
            name: "x".into(),
            cost: 100.0,
            price: 0.0,
            chef_salary: 0.0,
        };
        assert!(d.score().abs() < 1e-9);
    }

    #[test]
    fn agent_kind_partitioned_by_id() {
        assert_eq!(MarketWorld::kind_of(AgentId(0)), AgentKind::Firm);
        assert_eq!(MarketWorld::kind_of(AgentId(5)), AgentKind::Firm);
        assert_eq!(
            MarketWorld::kind_of(AgentId(CUSTOMER_ID_BASE)),
            AgentKind::Customer
        );
        assert_eq!(
            MarketWorld::kind_of(AgentId(CUSTOMER_ID_BASE + 9)),
            AgentKind::Customer
        );
    }
}

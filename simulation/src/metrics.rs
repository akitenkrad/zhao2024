//! 評価指標 (論文 §2.5 / マクロ分析)．
//!
//! 日次の市場指標 — 収益 Gini・最大市場シェア・平均料理スコア・メニュー類似度 —
//! を計算する．勝者総取り (Day6–15 で最大シェアが常時 >0.8) と品質改善
//! (Day1→DayN でスコア上昇) は時系列全体から判定するため，run ドライバ側で
//! 日次レコードを集計する．

use serde::Serialize;

use crate::world::Firm;

/// Gini 係数 G = Σ_i Σ_j |x_i − x_j| / (2 N Σ_i x_i) ∈ [0,1]．
///
/// 店舗の累積収益に適用するとマタイ効果 (収益不平等) の進行を測れる．負値や
/// 総和 0 のときは 0 を返す (定義不可)．
pub fn revenue_gini(values: &[f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let total: f64 = values.iter().sum();
    if total.abs() < 1e-12 {
        return 0.0;
    }
    let mut sum_abs_diff = 0.0;
    for &xi in values {
        for &xj in values {
            sum_abs_diff += (xi - xj).abs();
        }
    }
    let g = sum_abs_diff / (2.0 * n as f64 * total);
    g.clamp(0.0, 1.0)
}

/// 当日の最大市場シェア `max_r N_r / Σ_r N_r` ∈ [0,1]．
///
/// 客が誰も来なかった日 (総客数 0) は 0 を返す (シェア定義不可)．
pub fn market_share_max(day_customers: &[u64]) -> f64 {
    let total: u64 = day_customers.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let max = day_customers.iter().copied().max().unwrap_or(0);
    max as f64 / total as f64
}

/// 勝者総取り (winner-take-all) の判定 (論文マクロ分析)．
///
/// 競争開始 5 日後 (Day 6, 0 始まりで day index 5) 以降，一方の店舗が客の 80%
/// 超を最終日まで保持し続けたか．`max_share_by_day[d]` は日 `d` (0 始まり) の
/// 最大市場シェア．`start_day` (0 始まり; 既定 5 = Day6) 以降の全日で >0.8 なら
/// true．判定対象日が無い (シミュレーションが短すぎる) 場合は false．
pub fn winner_take_all(max_share_by_day: &[f64], start_day: usize, threshold: f64) -> bool {
    let window: Vec<f64> = max_share_by_day
        .iter()
        .enumerate()
        .filter(|(d, _)| *d >= start_day)
        .map(|(_, s)| *s)
        .collect();
    if window.is_empty() {
        return false;
    }
    window.iter().all(|&s| s > threshold)
}

/// 2 店舗のメニュー類似度 ∈ [0,1] (差別化と模倣の動的均衡，論文 約36%)．
///
/// 料理名の集合 Jaccard 類似度 `|A ∩ B| / |A ∪ B|` で定義する．多店舗 (M>2) では
/// 全ペアの平均を返す．いずれかが空メニューのペアは 0 として扱う．
pub fn menu_similarity(firms: &[&Firm]) -> f64 {
    if firms.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut pairs = 0u32;
    for i in 0..firms.len() {
        for j in (i + 1)..firms.len() {
            total += jaccard_menu(firms[i], firms[j]);
            pairs += 1;
        }
    }
    if pairs == 0 {
        0.0
    } else {
        total / pairs as f64
    }
}

/// 2 店舗のメニュー (料理名集合) の Jaccard 類似度．
fn jaccard_menu(a: &Firm, b: &Firm) -> f64 {
    use std::collections::BTreeSet;
    let set_a: BTreeSet<&str> = a.menu.iter().map(|d| d.name.as_str()).collect();
    let set_b: BTreeSet<&str> = b.menu.iter().map(|d| d.name.as_str()).collect();
    if set_a.is_empty() && set_b.is_empty() {
        return 0.0;
    }
    let inter = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

/// 平均値 (空なら 0)．
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// 1 日分の市場指標 (metrics.csv の 1 行; long-format)．
///
/// 各日について «店舗ごとの行» を縦に並べた long 形式で出力する．日次集計量
/// (revenue_gini, market_share_max, menu_similarity, n_alive_firms) は全行で
/// 同一値を持ち，店舗固有量 (dish_score, day_customers, day_revenue,
/// cumulative_revenue, reputation, avg_price) は店舗ごとに異なる．
#[derive(Debug, Clone, Serialize)]
pub struct DailyMetric {
    /// 日 d (0 始まり)．
    pub day: u64,
    /// 店舗 `AgentId` の生 `u64`．
    pub firm: u64,
    /// この店舗が生存しているか (1 / 0)．
    pub firm_alive: u8,
    /// この店舗の当日来店客数．
    pub day_customers: u64,
    /// この店舗の当日収益．
    pub day_revenue: f64,
    /// この店舗の累積収益．
    pub cumulative_revenue: f64,
    /// この店舗の平均料理スコア s．
    pub avg_dish_score: f64,
    /// この店舗の平均価格．
    pub avg_price: f64,
    /// この店舗の評判スコア．
    pub reputation: f64,
    // --- 日次集計量 (全行同値) ---
    /// 店舗累積収益の Gini 係数 ∈ [0,1]．
    pub revenue_gini: f64,
    /// 当日の最大市場シェア ∈ [0,1]．
    pub market_share_max: f64,
    /// 2 店舗メニューの類似度 ∈ [0,1]．
    pub menu_similarity: f64,
    /// 生存店舗数．
    pub n_alive_firms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::Dish;

    fn firm_with_menu(names: &[&str]) -> Firm {
        let menu: Vec<Dish> = names
            .iter()
            .map(|n| Dish {
                name: (*n).into(),
                cost: 1600.0,
                price: 4000.0,
                chef_salary: 2000.0,
            })
            .collect();
        Firm::new(100_000.0, menu, 2000.0)
    }

    #[test]
    fn gini_of_equal_is_zero() {
        assert!(revenue_gini(&[100.0, 100.0]) < 1e-9);
    }

    #[test]
    fn gini_of_monopoly_positive() {
        // 一方が全収益 → 2 店舗の Gini = 0.5．
        let g = revenue_gini(&[0.0, 100.0]);
        assert!((g - 0.5).abs() < 1e-9, "got {g}");
    }

    #[test]
    fn market_share_range_and_monopoly() {
        assert!((market_share_max(&[0, 50]) - 1.0).abs() < 1e-9);
        assert!((market_share_max(&[25, 25]) - 0.5).abs() < 1e-9);
        assert_eq!(market_share_max(&[0, 0]), 0.0);
    }

    #[test]
    fn winner_take_all_detection() {
        // Day0..=5 で最大シェアが (Day6=index5 以降) 常時 >0.8 → true
        let shares = vec![0.5, 0.6, 0.7, 0.85, 0.9, 0.95];
        assert!(winner_take_all(&shares, 5, 0.8));
        // index5 が 0.8 ちょうど → not > → false
        let shares2 = vec![0.5, 0.6, 0.7, 0.85, 0.9, 0.80];
        assert!(!winner_take_all(&shares2, 5, 0.8));
        // 短すぎ (start_day を超える日がない) → false
        assert!(!winner_take_all(&[0.9, 0.9], 5, 0.8));
    }

    #[test]
    fn menu_similarity_jaccard() {
        let a = firm_with_menu(&["sushi", "ramen", "tempura"]);
        let b = firm_with_menu(&["sushi", "ramen", "curry"]);
        // 共通 {sushi, ramen} = 2, 和集合 {sushi, ramen, tempura, curry} = 4 → 0.5
        let sim = menu_similarity(&[&a, &b]);
        assert!((sim - 0.5).abs() < 1e-9, "got {sim}");
    }

    #[test]
    fn menu_similarity_identical_is_one() {
        let a = firm_with_menu(&["sushi", "ramen"]);
        let b = firm_with_menu(&["sushi", "ramen"]);
        assert!((menu_similarity(&[&a, &b]) - 1.0).abs() < 1e-9);
    }
}

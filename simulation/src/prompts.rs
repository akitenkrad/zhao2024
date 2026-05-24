//! LLM プロンプト生成と応答パース．
//!
//! CompeteAI の 2 つの Decision プロンプトを構築する:
//! 1. **店舗戦略立案** (`CompetitionMatthewMechanism`): 直近 daybook・ライバル
//!    情報・記憶から，価格係数・シェフ給与・広告を改訂する JSON を求める．
//! 2. **顧客選択** (`CustomerChoiceMechanism`): 提示された店舗情報 (品質スコア・
//!    広告・価格・コメント) から来店店を 1 つ選ぶ．
//!
//! 応答パースは「まず JSON として読む → 失敗時はフォールバック」の二段で頑健化
//! する (ローカルモデルは厳密 JSON を返さないことがある)．

use crate::world::{Customer, Firm};

/// 店舗に提示する «前日 daybook + ライバル情報» のサマリ．
pub struct FirmBriefing {
    /// この店舗の前日来店客数．
    pub own_customers: u64,
    /// この店舗の前日収益．
    pub own_revenue: f64,
    /// ライバル各店の (前日来店客数, 前日収益, 平均価格, 平均スコア)．
    pub rivals: Vec<RivalInfo>,
}

/// ライバル店 1 店の公開情報．
pub struct RivalInfo {
    pub day_customers: u64,
    pub day_revenue: f64,
    pub avg_price: f64,
    pub avg_score: f64,
}

/// 店舗戦略立案プロンプトを構築する．
///
/// LLM には «価格倍率 price_factor (0.5〜1.5)・シェフ給与 chef_salary・広告文
/// advertisement» を JSON で答えさせる．プロンプト末尾の固定文 `Answer with JSON`
/// で応答形式を安定させ，キャッシュキー (= プロンプト全文 + モデル名) を
/// 決定論化する．
pub fn firm_strategy_prompt(firm: &Firm, day: u64, brief: &FirmBriefing) -> String {
    let mut s = String::new();
    s.push_str(
        "You are the manager of a restaurant competing for customers in a virtual town. \
         Each day you may adjust your prices, the salary you pay your chef (which raises dish \
         quality), and your advertisement. Your goal is to maximize cumulative revenue while \
         staying solvent.\n\n",
    );
    s.push_str(&format!("## Day {}\n", day));
    s.push_str("## Your restaurant\n");
    s.push_str(&format!("- Funds: {:.0}\n", firm.funds));
    s.push_str(&format!("- Average price: {:.0}\n", firm.avg_price()));
    s.push_str(&format!("- Chef salary: {:.0}\n", firm.chef_salary));
    s.push_str(&format!(
        "- Average dish quality score: {:.3}\n",
        firm.avg_dish_score()
    ));
    s.push_str(&format!("- Reputation: {:.2}\n", firm.reputation));
    s.push_str(&format!(
        "- Yesterday: {} customers, revenue {:.0}\n",
        brief.own_customers, brief.own_revenue
    ));

    if !brief.rivals.is_empty() {
        s.push_str("\n## Rivals (yesterday)\n");
        for (i, r) in brief.rivals.iter().enumerate() {
            s.push_str(&format!(
                "- Rival {}: {} customers, revenue {:.0}, avg price {:.0}, avg quality {:.3}\n",
                i + 1,
                r.day_customers,
                r.day_revenue,
                r.avg_price,
                r.avg_score
            ));
        }
    }

    if !firm.memory.is_empty() {
        s.push_str("\n## Your reflections\n");
        for m in &firm.memory {
            s.push_str("- ");
            s.push_str(m);
            s.push('\n');
        }
    }

    s.push_str(
        "\n## Decision\n\
         Choose a price_factor (multiply all dish prices, 0.5 = halve, 1.5 = raise by 50%), a \
         chef_salary (>= 0; higher raises quality), and a short advertisement.\n\
         Answer with JSON only, e.g. \
         {\"price_factor\": 1.05, \"chef_salary\": 2200, \"advertisement\": \"Fresh sushi daily!\"}\n",
    );
    s
}

/// 店舗 1 店の戦略決定 (パース結果)．
pub struct FirmStrategy {
    /// 価格倍率 (0.5〜1.5 にクランプ)．
    pub price_factor: f64,
    /// 新しいシェフ給与 (>= 0)．
    pub chef_salary: f64,
    /// 広告文．
    pub advertisement: String,
}

/// 店舗戦略応答をパースする (JSON → フォールバックは現状維持)．
pub fn parse_firm_strategy(text: &str, current_chef_salary: f64) -> FirmStrategy {
    if let Some(json) = extract_json_object(text) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
            if let Some(obj) = v.as_object() {
                let price_factor = number_field(obj, &["price_factor", "price", "factor"])
                    .unwrap_or(1.0)
                    .clamp(0.5, 1.5);
                let chef_salary = number_field(obj, &["chef_salary", "salary", "chef"])
                    .unwrap_or(current_chef_salary)
                    .max(0.0);
                let advertisement = obj
                    .get("advertisement")
                    .or_else(|| obj.get("ad"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                return FirmStrategy {
                    price_factor,
                    chef_salary,
                    advertisement,
                };
            }
        }
    }
    // フォールバック: 現状維持 (price_factor=1, 給与据え置き, 広告変更なし)．
    FirmStrategy {
        price_factor: 1.0,
        chef_salary: current_chef_salary,
        advertisement: String::new(),
    }
}

/// 顧客に提示する店舗 1 店のオファースナップショット．
pub struct FirmOffer {
    /// 店舗 `AgentId` の生 `u64` (選択結果のキー)．
    pub firm: u64,
    /// 平均価格．
    pub avg_price: f64,
    /// 平均品質スコア．
    pub avg_score: f64,
    /// 評判．
    pub reputation: f64,
    /// 広告文．
    pub advertisement: String,
    /// 直近の代表的コメント (最大 3 件)．
    pub recent_comments: Vec<String>,
}

/// 顧客選択プロンプトを構築する．
///
/// LLM には «来店する店舗の index (0 始まり)» を JSON で答えさせる．グループ客では
/// 各メンバーが個別に答え，呼び出し側が多数決を取る．
pub fn customer_choice_prompt(customer: &Customer, day: u64, offers: &[FirmOffer]) -> String {
    let mut s = String::new();
    s.push_str(
        "You are a customer in a virtual town deciding which restaurant to visit for dinner \
         today. Pick the single restaurant that best fits your taste, budget and the available \
         information.\n\n",
    );
    s.push_str(&format!("## Day {}\n", day));
    s.push_str("## You\n");
    s.push_str(&format!("- Daily budget: {:.0}\n", customer.income));
    s.push_str(&format!("- Preference: {}\n", customer.preference));
    s.push_str(&format!("- Health / diet: {}\n", customer.health));
    if let Some(last) = customer.visit_memory.last() {
        s.push_str(&format!(
            "- Last visit: restaurant {} (spent {:.0}, satisfaction {:.1}/5)\n",
            last.firm, last.spend, last.satisfaction
        ));
    }

    s.push_str("\n## Restaurants\n");
    for (i, o) in offers.iter().enumerate() {
        s.push_str(&format!(
            "- Option {}: avg price {:.0}, quality {:.3}, reputation {:.2}",
            i, o.avg_price, o.avg_score, o.reputation
        ));
        if !o.advertisement.is_empty() {
            s.push_str(&format!(", ad: \"{}\"", o.advertisement));
        }
        s.push('\n');
        for c in &o.recent_comments {
            s.push_str(&format!("    comment: \"{}\"\n", c));
        }
    }

    s.push_str(&format!(
        "\n## Decision\n\
         Reply with the option index of the restaurant you will visit (0 to {}).\n\
         Answer with JSON only, e.g. {{\"choice\": 0}}\n",
        offers.len().saturating_sub(1)
    ));
    s
}

/// 顧客選択応答から «選んだ option index» を抽出する．
///
/// 1. JSON `{"choice": k}` を試す．
/// 2. 失敗時は本文中の最初の整数を拾う．
/// 3. それも失敗なら `None` (呼び出し側がスコア最大店などにフォールバック)．
///
/// 範囲外の index は `None` 扱い．
pub fn parse_customer_choice(text: &str, n_offers: usize) -> Option<usize> {
    if n_offers == 0 {
        return None;
    }
    if let Some(json) = extract_json_object(text) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
            if let Some(obj) = v.as_object() {
                if let Some(k) = number_field(obj, &["choice", "option", "index", "restaurant"]) {
                    let idx = k.round() as i64;
                    if idx >= 0 && (idx as usize) < n_offers {
                        return Some(idx as usize);
                    }
                }
            }
        }
    }
    // 本文中の最初の非負整数を拾う．
    for tok in text.split(|c: char| !c.is_ascii_digit()) {
        if tok.is_empty() {
            continue;
        }
        if let Ok(k) = tok.parse::<usize>() {
            if k < n_offers {
                return Some(k);
            }
        }
    }
    None
}

/// 文字列から最初の `{ … }` ブロックを切り出す ({ から対応する最後の } まで)．
fn extract_json_object(text: &str) -> Option<String> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    Some(text[start..=end].to_string())
}

/// JSON オブジェクトから候補キー名のいずれかで数値を引く (文字列数値も許容)．
fn number_field(obj: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> Option<f64> {
    for k in keys {
        if let Some(v) = obj.get(*k) {
            if let Some(f) = v.as_f64() {
                return Some(f);
            }
            if let Some(s) = v.as_str() {
                if let Ok(f) = s.trim().parse::<f64>() {
                    return Some(f);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_firm_strategy_json() {
        let st = parse_firm_strategy(
            "{\"price_factor\": 1.1, \"chef_salary\": 2500, \"advertisement\": \"Best ramen!\"}",
            2000.0,
        );
        assert!((st.price_factor - 1.1).abs() < 1e-9);
        assert!((st.chef_salary - 2500.0).abs() < 1e-9);
        assert_eq!(st.advertisement, "Best ramen!");
    }

    #[test]
    fn firm_strategy_clamps_price_factor() {
        let st = parse_firm_strategy("{\"price_factor\": 9.0, \"chef_salary\": -5}", 2000.0);
        assert!((st.price_factor - 1.5).abs() < 1e-9);
        assert!(st.chef_salary.abs() < 1e-9);
    }

    #[test]
    fn firm_strategy_fallback_holds() {
        let st = parse_firm_strategy("I am not sure.", 1800.0);
        assert!((st.price_factor - 1.0).abs() < 1e-9);
        assert!((st.chef_salary - 1800.0).abs() < 1e-9);
    }

    #[test]
    fn parses_customer_choice_json() {
        assert_eq!(parse_customer_choice("{\"choice\": 1}", 2), Some(1));
        assert_eq!(
            parse_customer_choice("I will go to {\"choice\": 0} today", 2),
            Some(0)
        );
    }

    #[test]
    fn customer_choice_out_of_range_is_none() {
        assert_eq!(parse_customer_choice("{\"choice\": 5}", 2), None);
    }

    #[test]
    fn customer_choice_prose_fallback() {
        assert_eq!(parse_customer_choice("option 1 please", 2), Some(1));
        assert_eq!(parse_customer_choice("no idea", 2), None);
    }
}

//! オフライン (LLM 不要) 再現用のスクリプト化クライアント．
//!
//! 論文 (Zhao et al. 2024, CompeteAI) の **マクロ的知見** を，ライブ LLM 無しで
//! 構造的に再現するための決定論的 mock を提供する．`reproduce` サブコマンドと
//! `run --mock`，および各種テストがこの mock を共用する．
//!
//! 再現する定性的挙動 (論文 §マクロ分析 / Table 2):
//! - **品質改善 (86.67%)**: 店舗は «振り返り» のたびにシェフ給与を引き上げ料理品質を
//!   高める誘因を持つ (特に劣勢店は巻き返しのため強く投資する)．時間とともに平均
//!   料理スコアが上昇する．
//! - **マタイ効果 / 勝者総取り (個人客 66.7%)**: 顧客は観測可能なシグナル (品質 ×
//!   評判) が最大の店へ «同調 (herding)» する．初期のわずかな優位が «客 → 評判 →
//!   さらに客» の正のフィードバックループで自己強化され，一方の店が市場を独占する．
//! - **グループ化によるマタイ効果の緩和 (グループ客 16.7%)**: グループは多数決で
//!   来店店を決める．メンバーの嗜好が割れると票が分散し，同点は engine RNG で
//!   解消されるため，個人客のような «全員が勝者へ流れる» 同調が攪乱される (正の
//!   フィードバックが弱まり勝者総取りが起きにくい)．
//!
//! この mock は ground-truth LLM ではなく，論文の定性的結論を再現するための
//! «市場ヒューリスティクスの戯画» である．プロンプト文字列から «各店のオファー
//! (品質・評判)» と «自店/ライバルの直近実績» を読み取って次手を決める．ライブ
//! llama3.2 ではこの戯画ではなく実モデルの応答を用いる (cache 経由)．

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

use crate::llm::{wrap_client, CompeteClient};

/// 店舗戦略プロンプトを判別するためのマーカ (prompts.rs と一致させる)．
const FIRM_MARK: &str = "price_factor";
/// 顧客選択プロンプトを判別するためのマーカ (prompts.rs と一致させる)．
const CHOICE_MARK: &str = "option index";
/// グループ熟議プロンプトを判別するためのマーカ (prompts.rs と一致させる)．
const DELIBERATION_MARK: &str = "group deliberates together";
/// 価格に敏感とみなす予算しきい値 (init RNG の所得分布の下側; これ未満は最安店へ)．
///
/// 所得は `8000 * U(0.6, 1.6)` (init RNG)．しきい値 6000 で価格敏感層は平均 約15%，
/// seed ごとに揺らぐ．薄い seed では勝ち店が 80% を超え勝者総取りが成立し，厚い
/// seed では劣勢店が踏みとどまり成立しない → 発生頻度が試行間でばらつく．
const BUDGET_THRESHOLD: f64 = 6000.0;
/// 同調 (herding) が点火するシグナル差のしきい値．
///
/// 2 店の «評判 + 品質» シグナル差がこれ未満のあいだは顧客は予算パリティで分かれ，
/// 初期分割が温存される (市場が割れたまま)．集客に比例した品質投資 (rich-get-richer)
/// で勝ち店の品質が先行し，シグナル差がこれを超えると顧客が一斉に勝ち店へ同調する
/// (正のフィードバック点火 → 勝者総取り)．グループ客は多数決 + engine RNG 同点処理が
/// 点火前の分割を均し，点火を遅らせる/起こさせないため勝者総取りが緩和される．
const HERD_GAP: f64 = 0.04;
/// 劣勢店とみなす前日集客数 (これ未満なら価格競争で対抗する)．
const FIRM_LOSING_CUSTOMERS: u64 = 15;

/// 店舗戦略応答 (JSON) を組み立てる (mock の店舗ロジック)．
///
/// 劣勢店 (前日客数が少ない) ほど巻き返しのためシェフ給与を強く上げ価格を下げる．
/// 好調店は品質を維持しつつ微増する．いずれの場合もシェフ給与は単調に増やすため
/// 平均料理スコアは時間とともに上昇する (論文ファクト8 / 品質改善)．
pub fn reproduce_firm_reply(prompt: &str) -> String {
    let own = parse_own_customers(prompt);
    let cur_salary = parse_chef_salary(prompt);
    // いずれの店も «シェフ給与を引き上げ料理品質を高める» 誘因を持つ (論文ファクト8)．
    // 価格は据え置きに留め，原価率項 (c/p) を保ったまま品質スコア (f/5000 項) を
    // 単調に上げる → 平均料理スコアが時間とともに上昇する (品質改善)．
    //
    // 投資量は «前日の集客 (= 資力)» に比例させる: 多く集めた店ほど潤沢な収益を
    // 品質に再投資できる (rich-get-richer)．集客に比例した増分でシェフ給与を引き上げ，
    // 好調店と劣勢店の品質差を時間とともに開かせる → 評判が分岐し，正の
    // フィードバック (マタイ効果) を生む駆動力となる．
    let invest = 80.0 + 12.0 * own as f64; // 集客 0 で +80, 集客 30 で +440．
    let chef_salary = (cur_salary + invest).min(5000.0);
    // 価格はいずれの店も据え置く (原価率項 c/p を保ち品質スコアを単調に上げる)．
    // 集客に比例した品質投資 (rich-get-richer) で勝ち店の品質が劣勢店を引き離し，
    // 一度開いた品質差は «勝ち店ほど多く再投資できる» ため縮まらない (正の
    // フィードバックが固定化し勝者総取りへ向かう)．広告のみ状況で出し分ける．
    let ad = if own < FIRM_LOSING_CUSTOMERS {
        "Come discover us — fresh quality every day!"
    } else {
        "A customer favorite — quality you can taste."
    };
    format!(
        "{{\"price_factor\": 1.0, \"chef_salary\": {chef_salary}, \"advertisement\": \"{ad}\"}}"
    )
}

/// 顧客選択応答 (JSON) を組み立てる (mock の顧客ロジック)．
///
/// 顧客は «評判 (reputation) が最も高い店へ同調 (herding)» する．評判は来店客の
/// コメントで自己強化されるため，初期にわずかでも客を集めた店が «客 → 評判 →
/// さらに客» の正のフィードバックループで独走しうる (マタイ効果 / 勝者総取り)．
///
/// **初期の対称性破れ**: 競争初日は両店とも評判 0 でシグナルが拮抗する．このとき
/// 顧客は «自分の予算» のパリティ (seed 由来) で 2 店に振り分かれる ([`split_seed`])．
/// この初期分割が seed ごとに揺らぐため，一方に偏った seed では勝者総取りが起き，
/// 拮抗した seed では市場が割れる．価格敏感型 (低予算 or «bargain hunter») は同調を
/// 弱め最安店に寄る «反同調» 勢力で，分割を温存する方向に働く．
///
/// グループ客では多数決が異質なメンバーの振り分けを混ぜ，engine RNG の同点処理も
/// 加わって初期分割が均されるため，正のフィードバックが弱まり勝者総取りが
/// 起きにくい (個人 → グループで緩和)．解析不能時は Option 0．
pub fn reproduce_customer_reply(prompt: &str) -> String {
    let idx = choose_option(prompt).unwrap_or(0);
    format!("{{\"choice\": {idx}}}")
}

/// 顧客の来店店 index を決める (評判同調 + 初期対称性破れ + 価格敏感)．
fn choose_option(prompt: &str) -> Option<usize> {
    let opts: Vec<(usize, f64, f64, f64)> = option_lines(prompt)
        .map(|(idx, rest)| {
            let price = field_after(rest, "avg price ").unwrap_or(f64::INFINITY);
            let quality = field_after(rest, "quality ").unwrap_or(0.0);
            let reputation = field_after(rest, "reputation ").unwrap_or(0.0);
            (idx, price, quality, reputation)
        })
        .collect();
    if opts.is_empty() {
        return None;
    }

    // グループ熟議: 人気 (評判) に流されず «自分の予算・好み» 本位で選ぶ (反同調)．
    // 価格敏感型は最安店，それ以外は «品質 / 価格» の費用対効果が最良の店を選ぶ．
    // いずれも評判 (社会的証明) を無視するため，個別客のような正のフィードバック
    // 同調が生じず，メンバーの予算分布に応じて来店が複数店へ分散する → 配分
    // (apportion_seats) と相まって市場が割れ，勝者総取りが緩和される．
    if prompt.contains(DELIBERATION_MARK) {
        if is_price_sensitive(prompt) {
            return cheapest_split(&opts, prompt);
        }
        // 自分の嗜好に合う店を選ぶ — 嗜好は顧客ごとに異なる (init RNG) ため，熟議客は
        // 評判 (人気) ではなく «自分の好み» で各店へ散らばる (反同調)．嗜好ハッシュで
        // 店を割り当てることで，個別客のような流行店への雪崩が起きない．
        let k = preference_seed(prompt) as usize % opts.len();
        return Some(opts[k].0);
    }

    // 個別客 — 価格敏感型は最安店へ (反同調)．同価格 (拮抗) のときは予算パリティで
    // 分かれ，特定店に偏らない «反同調の床» を両店に分散させる．
    if is_price_sensitive(prompt) {
        return cheapest_split(&opts, prompt);
    }

    // 同調シグナル «評判 + 品質» の最大店と次点を求める．
    let signal = |o: &(usize, f64, f64, f64)| o.3 + o.2;
    let mut sorted = opts.clone();
    sorted.sort_by(|a, b| {
        signal(b)
            .partial_cmp(&signal(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top = sorted[0];
    let second = sorted[1];

    // シグナル差が小さい (初日など拮抗) → 同調の手がかりが弱い．このとき顧客は
    // «自分の予算パリティ» (seed 由来) で 2 店に分かれる (対称性破れ)．常連の慣性も
    // ここに含め，初期分割を温存する «粘り» を与える．差が大きく開いたら同調へ
    // 転じる (= 評判が分岐して初めて正のフィードバックが点火する)．
    if signal(&top) - signal(&second) < HERD_GAP {
        let k = split_seed(prompt) % opts.len() as u64;
        return Some(opts[k as usize].0);
    }

    // シグナルが明瞭に開いた → 最良店へ同調 (herding; マタイ効果を点火)．
    Some(top.0)
}

/// 最安店を返す (同価格は予算パリティで分散; 反同調の床を両店へ散らす)．
fn cheapest_split(opts: &[(usize, f64, f64, f64)], prompt: &str) -> Option<usize> {
    let min_price = opts.iter().map(|o| o.1).fold(f64::INFINITY, f64::min);
    let cheapest: Vec<usize> = opts
        .iter()
        .filter(|o| (o.1 - min_price).abs() < 1e-6)
        .map(|o| o.0)
        .collect();
    let k = split_seed(prompt) as usize % cheapest.len().max(1);
    cheapest.get(k).copied()
}

/// 顧客が価格敏感型か (低予算)．
fn is_price_sensitive(prompt: &str) -> bool {
    if let Some(idx) = prompt.find("Daily budget: ") {
        let rest = &prompt[idx + "Daily budget: ".len()..];
        let token: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(v) = token.parse::<f64>() {
            return v < BUDGET_THRESHOLD;
        }
    }
    false
}

/// 熟議客の «嗜好に合う店» 割り当てに使う顧客固有のシード．
///
/// «Preference: …» 行の本文を FNV-1a で畳む．嗜好は init RNG (実験 seed) が 5 種から
/// 割り当てるため，熟議客は嗜好ごとに異なる店へ分かれる (反同調)．予算 (所得) も
/// 嗜好と相関なく散るので，グループ内のメンバーは複数店へ配分される．
fn preference_seed(prompt: &str) -> u64 {
    let pref = if let Some(idx) = prompt.find("Preference: ") {
        let rest = &prompt[idx + "Preference: ".len()..];
        rest.lines().next().unwrap_or("")
    } else {
        ""
    };
    let mut h: u64 = 0xcbf29ce484222325;
    for b in pref.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 初期対称性破れに使う顧客固有のシード (予算の整数値を FNV-1a で畳む)．
///
/// 予算は init RNG (実験 seed) が散らすため，初期分割の «どちらに寄るか» が seed
/// ごとに揺らぐ → 勝者総取りの発生頻度が試行間でばらつく．
fn split_seed(prompt: &str) -> u64 {
    let budget = field_after(prompt, "Daily budget: ").unwrap_or(0.0);
    let mut h: u64 = 0xcbf29ce484222325;
    for b in (budget as u64).to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// 店舗戦略プロンプトから «自店の前日来店客数» を読み取る．
///
/// prompts.rs は «- Yesterday: {n} customers, revenue …» を埋め込む．読めなければ
/// 0 (= 劣勢扱いで巻き返し戦略) を返す．
fn parse_own_customers(prompt: &str) -> u64 {
    if let Some(idx) = prompt.find("Yesterday: ") {
        let rest = &prompt[idx + "Yesterday: ".len()..];
        let token: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(v) = token.parse::<u64>() {
            return v;
        }
    }
    0
}

/// 店舗戦略プロンプトから «現在のシェフ給与» を読み取る (単調引き上げの基点)．
///
/// prompts.rs は «- Chef salary: {f}» を埋め込む．読めなければ初期値 2000 を返す．
fn parse_chef_salary(prompt: &str) -> f64 {
    field_after(prompt, "Chef salary: ").unwrap_or(2000.0)
}

/// プロンプトの «- Option {i}: …» 行を `(index, 行の残り)` で列挙する．
fn option_lines(prompt: &str) -> impl Iterator<Item = (usize, &str)> {
    prompt.lines().filter_map(|line| {
        let rest = line.trim_start().strip_prefix("- Option ")?;
        let idx_tok: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        let idx = idx_tok.parse::<usize>().ok()?;
        Some((idx, rest))
    })
}

/// 行 `s` 中の `marker` 直後の浮動小数を読み取る (符号・小数点を許容)．
fn field_after(s: &str, marker: &str) -> Option<f64> {
    let idx = s.find(marker)?;
    let rest = &s[idx + marker.len()..];
    let token: String = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    token.parse::<f64>().ok()
}

/// 再現用の決定論的スクリプトクライアントを構築する (in-memory cache)．
///
/// 店舗戦略プロンプトには [`reproduce_firm_reply`] を，顧客選択プロンプトには
/// [`reproduce_customer_reply`] (同調) を返す．グループ多数決・engine RNG 同点処理は
/// メカニズム側が担い，個人客では同調が勝者総取りを，グループ客では票割れが
/// それを緩和する．
pub fn build_reproduce_client() -> CompeteClient {
    let backend = ScriptedClient::new("mock-reproduce", |prompt: &str| {
        if prompt.contains(FIRM_MARK) {
            reproduce_firm_reply(prompt)
        } else if prompt.contains(CHOICE_MARK) {
            reproduce_customer_reply(prompt)
        } else {
            // 想定外プロンプトは無害な現状維持を返す．
            "{\"choice\": 0}".to_string()
        }
    });
    wrap_client(backend, PromptCache::in_memory())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn firm_invests_proportional_to_traffic() {
        // 集客に比例して品質投資 (rich-get-richer): 0 客で +80, 30 客で +440．
        let losing = reproduce_firm_reply(
            "- Chef salary: 2000\n- Yesterday: 0 customers, revenue 0\nprice_factor",
        );
        assert!(losing.contains("2080"), "got {losing}");
        let winning = reproduce_firm_reply(
            "- Chef salary: 2000\n- Yesterday: 30 customers, revenue 90000\nprice_factor",
        );
        assert!(winning.contains("2440"), "got {winning}");
        // 価格は両店とも据え置き (原価率項を保つ)．
        assert!(losing.contains("\"price_factor\": 1.0"), "got {losing}");
        assert!(winning.contains("\"price_factor\": 1.0"), "got {winning}");
    }

    #[test]
    fn customer_herds_to_higher_reputation() {
        // 評判に差がある (拮抗していない) → 高評判の Option 1 へ同調．予算は十分高く
        // 価格敏感型でないこと，bargain でないことを満たす．
        let prompt = "Daily budget: 12000\n\
                      Preference: values classic comfort food\n\
                      option index\n\
                      - Option 0: avg price 4000, quality 0.500, reputation 1.00\n\
                      - Option 1: avg price 4000, quality 0.500, reputation 4.00\n";
        assert_eq!(reproduce_customer_reply(prompt), "{\"choice\": 1}");
    }

    #[test]
    fn price_sensitive_picks_cheapest() {
        // 低予算 (< しきい値) の顧客は最安の Option 1 へ (反同調)．
        let prompt = "Daily budget: 5000\n\
                      Preference: values classic comfort food\n\
                      option index\n\
                      - Option 0: avg price 5000, quality 0.500, reputation 4.00\n\
                      - Option 1: avg price 3000, quality 0.500, reputation 1.00\n";
        assert_eq!(reproduce_customer_reply(prompt), "{\"choice\": 1}");
    }

    #[test]
    fn tied_reputation_breaks_by_budget_parity() {
        // 評判拮抗 (初日) → 予算パリティで対称性を破る．決定論なので同一予算は同一
        // 選択になる (2 店なので 0/1 のどちらか一方に確定する)．
        let mk = |budget: u64| {
            format!(
                "Daily budget: {budget}\nPreference: values classic comfort food\noption index\n\
                 - Option 0: avg price 4000, quality 0.500, reputation 0.00\n\
                 - Option 1: avg price 4000, quality 0.500, reputation 0.00\n"
            )
        };
        let a = reproduce_customer_reply(&mk(12000));
        let b = reproduce_customer_reply(&mk(12000));
        assert_eq!(a, b, "同一予算は同一選択 (決定論)");
        assert!(a == "{\"choice\": 0}" || a == "{\"choice\": 1}");
    }

    #[test]
    fn parse_own_customers_reads_value() {
        assert_eq!(
            parse_own_customers("- Yesterday: 17 customers, revenue 51000"),
            17
        );
        assert_eq!(parse_own_customers("no yesterday line"), 0);
    }
}

//! シミュレーション設定．
//!
//! Zhao et al. (2024) CompeteAI のコアモデル (LLM 駆動の市場競争 ABM) と感度分析
//! パラメータを保持する [`Config`] と，その JSON シリアライズ表現を定義する．
//! 店舗数・顧客数・顧客構成 (個人/グループ)・日数・初期資金/価格・LLM 設定を
//! ここに集約する．

use serde::Serialize;

// --------------------------------------------------------------------------- //
// 顧客構成モード
// --------------------------------------------------------------------------- //

/// 顧客構成モード (審判の意思決定単位)．
///
/// 論文は個人客 (個別判断) とグループ客 (家族/同僚/カップル/友人; 多数決) を比較し，
/// グループ化が正のフィードバックループを攪乱して勝者総取りを緩和することを示す
/// (個人 66.7% → グループ 16.7%)．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CustomerMode {
    /// 個人客 (各顧客が単独で来店店を選ぶ; 論文の基本ケース)．
    Individual,
    /// グループ客 (グループ単位で多数決により来店店を選ぶ)．
    Group,
}

impl CustomerMode {
    pub fn label(&self) -> &'static str {
        match self {
            CustomerMode::Individual => "individual",
            CustomerMode::Group => "group",
        }
    }
}

/// 文字列から [`CustomerMode`] をパースする．
pub fn parse_customer_mode(s: &str) -> Result<CustomerMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "individual" | "indiv" | "single" => Ok(CustomerMode::Individual),
        "group" | "grouped" => Ok(CustomerMode::Group),
        _ => Err(format!(
            "不正な顧客構成モード: \"{}\" (individual / group)",
            s
        )),
    }
}

// --------------------------------------------------------------------------- //
// LLM 設定
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (provider / model / temperature / seed / cache)．
///
/// プロバイダ優先順位は «Ollama 第一 → OpenAI フォールバック» 固定．モデル・
/// ホスト・API キーは環境変数で渡す (`OLLAMA_HOST` / `OLLAMA_MODEL` /
/// `OPENAI_API_KEY` / `OPENAI_MODEL`)．`temperature`/`seed` で擬似決定論化する．
#[derive(Debug, Clone)]
pub struct LlmSettings {
    /// 生成温度 (既定 0.0; 再現性のため．論文は GPT-4 既定温度近傍)．
    pub temperature: f32,
    /// 生成シード (バックエンドへ渡す; Ollama は honour，OpenAI は best-effort)．
    pub seed: u64,
    /// プロンプト→応答キャッシュの保存先 (None なら in-memory)．
    pub cache_path: Option<String>,
}

impl Default for LlmSettings {
    fn default() -> Self {
        LlmSettings {
            temperature: 0.0,
            seed: 0,
            cache_path: None,
        }
    }
}

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// 単一実行の設定．
///
/// 既定値は論文 §2 の標準設定 (2 店 × 50 客 × 15 日，個人客) に近い．
#[derive(Debug, Clone)]
pub struct Config {
    /// 店舗数 M (競争者)．
    pub n_firms: usize,
    /// 顧客数 N (審判)．
    pub n_customers: usize,
    /// 顧客構成 (individual / group)．
    pub customer_mode: CustomerMode,
    /// グループ客のときの 1 グループあたり人数 (個人客では無視)．
    pub group_size: usize,
    /// シミュレーション日数 (= ラウンド数; 論文標準 15)．
    pub days: usize,

    // --- 初期化パラメータ ---
    /// 店舗の初期資金 (中央値; ±レンジを init RNG で散らす)．
    pub init_funds: f64,
    /// 各店舗の初期メニュー品目数．
    pub init_menu_size: usize,
    /// 料理の初期価格 (中央値)．
    pub init_price: f64,
    /// 料理の初期原価率 (cost / price)．
    pub init_cost_ratio: f64,
    /// 店舗の初期シェフ給与 (品質スコアの第 2 項を駆動)．
    pub init_chef_salary: f64,
    /// 顧客の所得 (中央値; init RNG で散らす)．
    pub customer_income: f64,

    /// 乱数シード (None の場合はランダム; socsim コア層のみ支配)．
    pub seed: Option<u64>,
    /// LLM レイヤ設定．
    pub llm: LlmSettings,
    /// 結果出力ディレクトリ．
    pub output_dir: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_firms: 2,
            n_customers: 50,
            customer_mode: CustomerMode::Individual,
            group_size: 4,
            days: 15,
            init_funds: 100_000.0,
            init_menu_size: 4,
            init_price: 4_000.0,
            init_cost_ratio: 0.4,
            init_chef_salary: 2_000.0,
            customer_income: 8_000.0,
            seed: Some(42),
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

/// `run` の試行シードを派生する (試行 index で独立化する)．
///
/// `run --runs K` のとき各試行は同じ base seed から `derive_seed(base, &[idx])` で
/// 独立な socsim コアシードを得る (LLM レイヤとは独立)．
pub fn derive_run_seed(base: u64, run_idx: usize) -> u64 {
    socsim_core::derive_seed(base, &[run_idx as u64])
}

/// `config.json` (run 用) のシリアライズ表現．
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub n_firms: usize,
    pub n_customers: usize,
    pub customer_mode: String,
    pub group_size: usize,
    pub days: usize,
    pub init_funds: f64,
    pub init_menu_size: usize,
    pub init_price: f64,
    pub init_cost_ratio: f64,
    pub init_chef_salary: f64,
    pub customer_income: f64,
    pub seed: Option<u64>,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub output_dir: String,
}

impl Config {
    /// `config.json` 用の表現を組み立てる．
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            n_firms: self.n_firms,
            n_customers: self.n_customers,
            customer_mode: self.customer_mode.label().to_string(),
            group_size: self.group_size,
            days: self.days,
            init_funds: self.init_funds,
            init_menu_size: self.init_menu_size,
            init_price: self.init_price,
            init_cost_ratio: self.init_cost_ratio,
            init_chef_salary: self.init_chef_salary,
            customer_income: self.customer_income,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            output_dir: self.output_dir.clone(),
        }
    }
}

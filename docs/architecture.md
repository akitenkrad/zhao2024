[English](architecture.md) | [日本語](architecture.ja.md)

# Architecture

## Repository layout

```
replications/zhao2024/
├── Cargo.toml                  # Rust workspace (members = ["simulation"])
├── pyproject.toml              # uv workspace (members = ["tools"])
├── simulation/                 # Rust crate `competeai-simulation` (bin `competeai`)
│   ├── Cargo.toml              # socsim-core + socsim-engine + socsim-llm (features=["live"]) + runvault
│   ├── examples/mock_smoke.rs  # offline (no live LLM) pipeline smoke
│   ├── src/
│   │   ├── main.rs             # clap: run / sweep / reproduce
│   │   ├── lib.rs
│   │   ├── config.rs           # Config, CustomerMode, LLM settings, seed derivation
│   │   ├── world.rs            # MarketWorld (WorldState), Firm, Customer, Dish, Market
│   │   ├── mechanisms.rs       # the five mechanisms over the six phases
│   │   ├── llm.rs              # socsim-llm builder (Ollama→OpenAI + cache)
│   │   ├── prompts.rs          # firm-strategy / customer-choice prompts + response parsing
│   │   ├── metrics.rs          # revenue Gini / market share / WTA / dish score / menu similarity
│   │   ├── record.rs           # runvault recording: daily aggregates, observation/terminal events, LLM block
│   │   ├── reproduce_mock.rs   # scripted client wiring for `reproduce --mock`
│   │   └── simulation.rs       # init_world + run drivers (returns SimulationResult; writes no files)
│   └── tests/integration_test.rs   # mock-driven (ScriptedClient); no live LLM
├── tools/                      # Python package `competeai-tools` (module `competeai_tools`)
│   └── src/competeai_tools/
│       ├── cli.py
│       ├── visualize.py        # market share + revenue Gini + dish score + menu similarity (from a run's events.jsonl/metrics.csv)
│       ├── visualize_sweep.py  # store-count × customer-count WTA frequency / final Gini (rebuilds the sweep table from child runs)
│       ├── sweep_summary.py    # rebuilds the "one row per cell×trial" sweep table from a sweep parent's child runs
│       ├── reproduce_paper.py  # reads a `reproduce` parent's scope=sweep metrics + reference.csv, prints the diff, renders figures
│       └── show_experiment_settings.py
└── docs/                       # bilingual (.md + .ja.md)
```

## Two-layer determinism

socsim's core is deterministic and LLM-free; an LLM is inherently not. The design confines the LLM to two mechanisms and pseudo-determinises it:

| Layer | Components | Reproducibility |
|---|---|---|
| Deterministic socsim core | restaurant/customer init (funds, incomes, preferences, menus), activation order, group deliberation/apportionment tie-breaking, customer–firm market matching, revenue/cost/funds accounting, all metrics | bit-for-bit given the seed (ChaCha20 `SimRng`) |
| Non-deterministic LLM layer | firm strategy reflection + customer choice (the two `Decision` mechanisms) | pseudo-deterministic via prompt→response cache + `temperature=0` + fixed seed |

The RNG streams are derived from one root seed (matching schelling1971 / axelrod1997 / li2024): `derive_seed(root, &[0])` initialises the world (firm funds, customer traits) and `derive_seed(root, &[1])` seeds the engine (activation order, group deliberation/apportionment tie-breaking). `&[2]`, `&[3]`, … are reserved for additional streams.

## The world

`MarketWorld { clock, firms: BTreeMap<AgentId, Firm>, customers: BTreeMap<AgentId, Customer>, market: Market, day }` implements `WorldState`. There is no spatial grid: restaurants and customers are fixed agents that interact through the market. Firms occupy `AgentId` `[0, n_firms)` and customers occupy `[CUSTOMER_ID_BASE, …)`, so `kind_of(id)` partitions the single `AgentId` space deterministically and `agent_ids()` returns firms (ascending) then customers (ascending). `Dish` carries the paper's quality score `s = 0.5·c/p + 0.5·f/5000`.

## Mechanisms (five over six phases)

The synchronous daily step (1 engine tick = 1 day) runs the six phases in order; declaration order is the firing order within a phase.

| Mechanism | Phase | Role |
|---|---|---|
| `MarketResetMechanism` | Environment | stash the previous day's daybook into scratch, then reset the day's market |
| `CompetitionMatthewMechanism` | **Decision** | each LLM firm reflects on the daybook + rival info + memory and revises price / chef salary / advertisement; then snapshots all firm offers into scratch (**LLM**) |
| `CustomerChoiceMechanism` | **Decision** | each LLM customer picks a restaurant from the presented offers. Individual customers choose independently; group customers deliberate (a group-framed prompt that resists social proof) and the group then apportions its members across restaurants by the vote distribution (largest-remainder, RNG-broken), which dampens the herd. (**LLM**) |
| `PatronageMechanism` | Interaction | customer–firm matching (patronage), the dining experience, comment generation visible to other customers |
| `RevenueRewardMechanism` | Reward | revenue/cost/funds, reputation update, and the daily Matthew metrics (revenue Gini, market-share concentration) |
| `ReflectionMechanism` | PostStep | each firm summarises the day into memory; firm exit (`alive = false` when funds < 0); `request_stop` when a firm exits or `day == days - 1` |

The LLM client and the call-metadata collector are shared with the two `Decision` mechanisms via `Rc<RefCell<…>>` (the li2024 pattern); the run driver uses them afterwards to persist the cache and aggregate the cache-hit rate. Firm offers are snapshotted at the end of the firm `Decision` and passed to `Interaction` through the step-scoped `scratch`, so within-day state changes do not leak into other agents' same-day decisions.

## Output layout (runvault)

One subcommand invocation is one [runvault](https://github.com/akitenkrad/rs-runvault) run: a directory `<results-root>/competeai/<subcommand>_<timestamp>_<config_hash>_<execution_hash>/` holding `run.json`, `config.json` (an envelope whose `parameters` block holds the conditions), `metrics.csv`, `events.jsonl`, `status.json`, `manifest.csv`, and — for `reproduce` — `reference.csv`. runvault owns the naming and identity of the output directory, so this crate creates no timestamped directories or `latest` symlink of its own; `--output-dir` is the runvault results root (default `results`). `sweep` and `reproduce` are a parent run plus one child run per trial, linked by `lineage.parent_run_uid` (see [CLI](cli.md)).

## Metrics

runvault's `metrics.csv` is `run_uid,step,step_unit,scope,name,value` and has no column for a series id, so the (day, firm) panel cannot live there — every firm's row for a day would collide on the primary key `(name, step, step_unit, scope)`. `crate::record` therefore splits the numbers by grain:

| Where | Grain | Fields |
|---|---|---|
| `events.jsonl`, kind `observation` | one row per (day, firm) | `unit_id=firm-<id>`, `t`=day, `t_unit=round`, `firm`, `day_customers`, `day_revenue`, `cumulative_revenue`, `avg_dish_score`, `avg_price`, `reputation` |
| `metrics.csv`, `scope=run`, per day | identical across firms | `revenue_gini`, `market_share_max`, `menu_similarity`, `n_alive_firms` |
| `metrics.csv`, `scope=run`, no step | one value per run | `n_units`, `final_day`, `winner_take_all` (0/1), `quality_improved` (0/1), `llm_calls`, `llm_cache_hits`, `llm_cache_hit_rate` (omitted when there were no calls — a rate over zero calls is undefined, not zero) |
| `events.jsonl`, kind `terminal` | one row per firm | `outcome` (`survive`/`exit`), `censored` (`true` for survivors), `budget` = last observed day |

`firm_alive` no longer exists as a column: a firm's exit is decided by `ReflectionMechanism` *after* the day's metrics row is written, and stops the run — so on the old long-format `metrics.csv` it was `1` on every row ever written. A firm's fate now lives on its `terminal` event instead.

| Metric | Definition | Paper correspondence |
|---|---|---|
| `revenue_gini` | Gini of cumulative firm revenue | Matthew effect (Table 2) |
| `market_share_max` | `max_r N_r / Σ_r N_r` for the day | winner-take-all |
| `winner_take_all` | max share > 0.8 for every day in `[Day6, last]` (bool) | macro analysis (66.7% / 16.7%) |
| `avg_dish_score` | per-firm mean dish score `s`, daily | quality improvement (86.67%) |
| `menu_similarity` | Jaccard overlap of menus (dish-name sets) | differentiation/imitation (≈ 36%) |
| `quality_improved` | at least one firm's mean score rose Day1→last (bool) | quality improvement |

`sweep`'s and `reproduce`'s cross-condition tables (the old `sweep_summary.csv`) are not written to disk either — `competeai_tools.sweep_summary.sweep_summary_table()` rebuilds the "one row per cell×trial" table on demand from a sweep parent's child runs. `reproduce`'s parent carries the cross-condition aggregates as `scope=sweep` metrics (`wta_freq_individual`, `wta_freq_group`, `quality_freq_all`, `menu_similarity_all`, …) and the paper's own reported values in `reference.csv` (each with a `source`). The ±15pt / ±10pt pass/off band is this replication's own choice, not the paper's, so it is **not** recorded — it stays in the `competeai reproduce` console output.

## socsim / socsim-llm / runvault

The crate depends only on `socsim-core` (the `WorldState` / `Mechanism` / `Phase` / `SimClock` / `SimRng` primitives) and `socsim-engine` (the `SimulationBuilder`, `RandomActivationScheduler`, `run_observed`), plus `socsim-llm` with `features = ["live"]` for the Ollama + OpenAI backends behind a `FallbackClient`. The production client type is `CachingClient<Box<dyn LlmClient>>`: the `FallbackClient<OllamaClient, OpenAiClient>` is type-erased into `Box<dyn LlmClient>` using `socsim-llm`'s `impl LlmClient for Box<T>` (issue #26), so no local newtype is needed and the same `CompeteClient` accepts a `mock::ScriptedClient` in tests. Output recording is a separate concern from the socsim/`socsim-llm` dependencies: it is owned by [runvault](https://github.com/akitenkrad/rs-runvault) (see [Output layout](#output-layout-runvault) above), which the crate depends on as a fourth git dependency alongside `socsim-core` / `socsim-engine` / `socsim-llm`. The git dependencies are pinned to a concrete commit in `Cargo.lock`.

> The design doc (§4.2/§7) originally listed `reqwest` + `sha2`; this suite supersedes that by standardising on `socsim-llm` (matching li2024 / chuang2024). `socsim-llm` owns the HTTP transport and the `hash(prompt+model)` cache key, so neither `reqwest` nor `sha2` appears in this crate.

## References

- Zhao, Q., Wang, J., Zhang, Y., Jin, Y., Zhu, K., Chen, H., & Xie, X. (2024). CompeteAI: Understanding the Competition Dynamics of Large Language Model-based Agents. *ICML 2024*, PMLR 235, 61092–61107. arXiv:2310.17512.
- Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*. (the virtual-town design basis)
- Rigney, D. (2010). *The Matthew Effect: How Advantage Begets Further Advantage*. Columbia University Press.
- socsim: [rs-social-simulation-tools](https://github.com/akitenkrad/rs-social-simulation-tools) (`socsim-llm` is issue #21/#26).

---
*This file was generated by Claude Code.*

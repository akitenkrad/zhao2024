[English](architecture.md) | [日本語](architecture.ja.md)

# Architecture

## Repository layout

```
replications/zhao2024/
├── Cargo.toml                  # Rust workspace (members = ["simulation"])
├── pyproject.toml              # uv workspace (members = ["tools"])
├── simulation/                 # Rust crate `competeai-simulation` (bin `competeai`)
│   ├── Cargo.toml              # socsim-core + socsim-engine + socsim-llm (features=["live"])
│   ├── examples/mock_smoke.rs  # offline (no live LLM) pipeline smoke
│   ├── src/
│   │   ├── main.rs             # clap: run / sweep
│   │   ├── lib.rs
│   │   ├── config.rs           # Config, CustomerMode, LLM settings, seed derivation
│   │   ├── world.rs            # MarketWorld (WorldState), Firm, Customer, Dish, Market
│   │   ├── mechanisms.rs       # the five mechanisms over the six phases
│   │   ├── llm.rs              # socsim-llm builder (Ollama→OpenAI + cache)
│   │   ├── prompts.rs          # firm-strategy / customer-choice prompts + response parsing
│   │   ├── metrics.rs          # revenue Gini / market share / WTA / dish score / menu similarity
│   │   └── simulation.rs       # init_world + run drivers + output writers
│   └── tests/integration_test.rs   # mock-driven (ScriptedClient); no live LLM
├── tools/                      # Python package `competeai-tools` (module `competeai_tools`)
│   └── src/competeai_tools/
│       ├── cli.py
│       ├── visualize.py        # market share + revenue Gini + dish score + menu similarity
│       ├── visualize_sweep.py  # store-count × customer-count WTA frequency / final Gini
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

## Metrics

`metrics.csv` is **long-format**: one row per (day, firm). Per-firm columns (`day_customers`, `day_revenue`, `cumulative_revenue`, `avg_dish_score`, `avg_price`, `reputation`, `firm_alive`) vary by firm; daily aggregates (`revenue_gini`, `market_share_max`, `menu_similarity`, `n_alive_firms`) repeat across the rows of one day.

| Metric | Definition | Paper correspondence |
|---|---|---|
| `revenue_gini` | Gini of cumulative firm revenue | Matthew effect (Table 2) |
| `market_share_max` | `max_r N_r / Σ_r N_r` for the day | winner-take-all |
| `winner_take_all` | max share > 0.8 for every day in `[Day6, last]` (bool) | macro analysis (66.7% / 16.7%) |
| `avg_dish_score` | per-firm mean dish score `s`, daily | quality improvement (86.67%) |
| `menu_similarity` | Jaccard overlap of menus (dish-name sets) | differentiation/imitation (≈ 36%) |
| `quality_improved` | at least one firm's mean score rose Day1→last (bool) | quality improvement |

## socsim / socsim-llm

The crate depends only on `socsim-core` (the `WorldState` / `Mechanism` / `Phase` / `SimClock` / `SimRng` primitives) and `socsim-engine` (the `SimulationBuilder`, `RandomActivationScheduler`, `run_observed`), plus `socsim-llm` with `features = ["live"]` for the Ollama + OpenAI backends behind a `FallbackClient`. The production client type is `CachingClient<Box<dyn LlmClient>>`: the `FallbackClient<OllamaClient, OpenAiClient>` is type-erased into `Box<dyn LlmClient>` using `socsim-llm`'s `impl LlmClient for Box<T>` (issue #26), so no local newtype is needed and the same `CompeteClient` accepts a `mock::ScriptedClient` in tests. The git dependencies are pinned to a concrete commit in `Cargo.lock`.

> The design doc (§4.2/§7) originally listed `reqwest` + `sha2`; this suite supersedes that by standardising on `socsim-llm` (matching li2024 / chuang2024). `socsim-llm` owns the HTTP transport and the `hash(prompt+model)` cache key, so neither `reqwest` nor `sha2` appears in this crate.

## References

- Zhao, Q., Wang, J., Zhang, Y., Jin, Y., Zhu, K., Chen, H., & Xie, X. (2024). CompeteAI: Understanding the Competition Dynamics of Large Language Model-based Agents. *ICML 2024*, PMLR 235, 61092–61107. arXiv:2310.17512.
- Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*. (the virtual-town design basis)
- Rigney, D. (2010). *The Matthew Effect: How Advantage Begets Further Advantage*. Columbia University Press.
- socsim: [rs-social-simulation-tools](https://github.com/akitenkrad/rs-social-simulation-tools) (`socsim-llm` is issue #21/#26).

---
*This file was generated by Claude Code.*

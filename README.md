**English** | [日本語](README.ja.md)

# CompeteAI: Understanding the Competition Dynamics of LLM-based Agents — Zhao et al. (2024)

A reimplementation of the LLM-driven market-competition agent-based model of Zhao et al. (2024), "CompeteAI: Understanding the Competition Dynamics of Large Language Model-based Agents" (*ICML 2024*, PMLR 235, 61092–61107; arXiv:2310.17512). A virtual town hosts two restaurant agents (the *competitors*) and fifty customer agents (the *judges*) over fifteen days. Each day every restaurant reflects on the previous day's daybook, its rivals and its own memory and revises its strategy — prices, chef salary (which raises dish quality), and advertisement — while every customer chooses where to dine from the presented quality scores, prices, adverts and visible comments. The paper's headline finding is that competition for revenue and reputation *alone* makes classical market and sociological regularities emerge from the bottom up — a dynamic equilibrium of **differentiation and imitation** (menu similarity ≈ 36%), and a self-reinforcing **Matthew effect** where an early advantage compounds into **winner-take-all**. The deterministic [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) core handles world initialisation, activation order, group majority tie-breaking, the market matching and all metrics, while the non-deterministic LLM layer is confined to the two decision mechanisms and pseudo-determinised via the `socsim-llm` crate (prompt→response cache + `temperature=0` + fixed seed).

## Two-layer determinism (read this first)

LLM output is **outside** socsim's bit-reproducibility. The design therefore splits into two layers:

- **Deterministic socsim core** — restaurant/customer initialisation (funds, incomes, preferences), activation order, group majority tie-breaking, the customer–firm market matching, the fiscal accounting (revenue, costs, funds) and all metrics (revenue Gini, market-share concentration, winner-take-all, dish scores, menu similarity). Given a seed this reproduces bit-for-bit (`ctx.rng`, ChaCha20 `SimRng`).
- **Non-deterministic LLM layer** — the two `Decision` mechanisms: the firm strategy reflection (`CompetitionMatthewMechanism`) and the customer choice (`CustomerChoiceMechanism`). Pseudo-determinised by `socsim-llm`'s `CachingClient` (a `hash(prompt+model)` → response cache), `temperature=0` and a fixed seed. The provider order is **Ollama first → OpenAI fallback** via `socsim-llm`'s `FallbackClient`.

The cache — not the model — is the reproducibility mechanism: a warm cache replays identical responses, so a rerun is free and stable. Each run writes `run_metadata.json` recording the model, endpoint, temperature, seed and cache-hit rate. Because the local default model (`llama3.2`) differs from the paper's `gpt-4`, reproduction targets are **qualitative**: the occurrence of the Matthew effect / market concentration, winner-take-all occurrence, and quality improvement — not the exact paper frequencies (winner-take-all 66.7% individuals / 16.7% groups, quality improvement 86.67%, menu similarity ≈ 36%). The `reproduce` subcommand batches the individual-vs-group runs and scores the observed frequencies against the paper's Table 2 with a directional pass/off band (see [CLI](docs/cli.md)).

> This project standardises on the `socsim-llm` crate for the LLM layer; it does **not** use `reqwest` or `sha2` (socsim-llm owns the HTTP transport and the prompt-cache hashing). It needs no spatial grid or network (`socsim-grid` / `socsim-net`): the interaction is market-mediated, so it depends only on `socsim-core` + `socsim-engine` + `socsim-llm`.

## Install & Quick start

```bash
# Build the Rust simulation (fetches socsim incl. socsim-llm with the Ollama+OpenAI backends)
cargo build --release

# Make sure a local Ollama is running and a model is pulled, e.g.:
#   ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
# Optional OpenAI fallback:
#   export OPENAI_API_KEY=sk-...   OPENAI_MODEL=gpt-4o-mini

# Run a small simulation (a quick smoke; the paper uses M=2, N=50, days=15)
cargo run --release -- run --n-firms 2 --n-customers 6 --days 3 --runs 1 --seed 42

# The full paper-scale base experiment (individual customers):
#   cargo run --release -- run --n-firms 2 --n-customers 50 --customer-mode individual --days 15 --runs 9 --seed 42

# Install the Python visualization tools (at the workspace root)
uv sync

# Visualize the most recent run (market share, revenue Gini, dish score, menu similarity)
uv run competeai-tools visualize

# Inspect the run's settings and LLM metadata
uv run competeai-tools show-experiment-settings --results-dir results/latest
```

### Offline (no-LLM) smoke

The full day loop, output writers and Python visualization can be exercised without any live LLM via a scripted mock client. `run`, `reproduce` and the `mock_smoke` example all accept an offline path (`--mock`, or the dedicated example), which a sandbox/CI uses to drive the whole pipeline deterministically:

```bash
cargo run --release --example mock_smoke -- results
uv run competeai-tools visualize
```

### Sensitivity sweep

```bash
cargo run --release -- sweep \
    --n-firms-values 2,3,4 \
    --n-customers-min 20 --n-customers-max 80 --n-customers-step 20 \
    --days 15 --runs 5 --seed 42
uv run competeai-tools visualize-sweep
```

### Reproduce the paper's Table 2 occurrence frequencies

`reproduce` batches the individual-customer runs and the group-customer runs, scores the observed frequencies (winner-take-all, quality improvement, menu similarity) against the paper's Table 2 with a pass/off band, and writes `reproduce_summary.json` plus per-condition metrics. The companion Python tool renders the occurrence-frequency, Matthew-effect and share-trajectory figures.

```bash
# Offline (scripted mock) batch reproduction + figures
uv run competeai-tools reproduce --run --mock
# Or with a live LLM, after building and starting Ollama:
#   cargo run --release -- reproduce --seed 42 && uv run competeai-tools reproduce
```

## What this project does

The project implements the full CompeteAI virtual-town model and its analyses end to end:

- **`run`** — one configuration of the LLM-driven market-competition ABM (`MarketWorld` + five mechanisms over the six-phase loop, the LLM decision layer with Ollama→OpenAI fallback + caching, and the Matthew-effect metrics). `--customer-mode {individual,group}` selects whether the judges act as independent individuals or as deliberating groups, and `--mock` drives it offline.
- **`sweep`** — a parameter sweep over store count × customer count, summarising the Matthew-effect metrics per condition.
- **`reproduce`** — a one-shot batch of the paper's Table 2 occurrence frequencies, comparing the individual-customer and group-customer conditions and scoring the observed winner-take-all / quality-improvement / menu-similarity frequencies against the paper.
- **Python `competeai-tools`** — `visualize`, `visualize-sweep`, `show-experiment-settings` and `reproduce` for plotting and inspecting the results.

### Customer-mode comparison (individual vs group)

The judges can act either as independent individuals or as **deliberating groups**. Individual customers are susceptible to social proof and herd toward the popular restaurant, so an early lead compounds into winner-take-all (the Matthew effect). A group instead deliberates: members voice their own budget and taste rather than following the crowd, and the group apportions its members across restaurants by that internal diversity. This disrupts the positive-feedback loop and dampens winner-take-all — the model reproduces the paper's individual → group attenuation (Table 2: 66.7% → 16.7%). Select the mode with `--customer-mode {individual,group}` on `run`, `sweep` and `reproduce`.

## Documentation

- [Use cases](docs/usecases.md) — what you can do with this project, with pointers to the rest of the docs.
- [CLI](docs/cli.md) — the Rust CLI: the `run`, `sweep` and `reproduce` subcommands and their flags, plus the LLM environment variables.
- [Visualization](docs/visualization.md) — the Python `competeai-tools` and how to interpret the outputs.
- [Architecture](docs/architecture.md) — repository structure, the two-layer determinism, the socsim/`socsim-llm` framework, the mechanisms, the metrics, and references.

## License

MIT

---
*This file was generated by Claude Code.*

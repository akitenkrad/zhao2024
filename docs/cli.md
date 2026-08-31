[English](cli.md) | [日本語](cli.ja.md)

# CLI

The Rust binary `competeai` has three subcommands: `run`, `sweep` and `reproduce`.

## LLM environment variables

The LLM layer is **Ollama-first → OpenAI-fallback**, configured by environment (never hard-coded):

| Variable | Default | Used by |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | primary backend |
| `OLLAMA_MODEL` | `llama3.2:latest` | primary backend |
| `OPENAI_API_KEY` | (unset → fallback disabled) | fallback backend |
| `OPENAI_MODEL` | `gpt-4o-mini` | fallback backend |

A prompt→response cache (default `.llm_cache/cache.json`) pseudo-determinises reruns: a warm cache replays identical responses, so a rerun is free.

## `run`

Run a single configuration of the LLM-driven market-competition ABM.

```bash
cargo run --release -- run \
    --n-firms 2 --n-customers 50 --customer-mode individual \
    --days 15 --runs 9 --seed 42
```

| Flag | Default | Meaning |
|---|---|---|
| `--n-firms` | 2 | number of restaurants M |
| `--n-customers` | 50 | number of customers N |
| `--customer-mode` | `individual` | `individual` or `group` |
| `--group-size` | 4 | members per group (group mode only) |
| `--days` | 15 | number of days (rounds) |
| `--runs` | 1 | independent trials (each gets a derived seed); the last trial's details are saved |
| `--seed` | (random) | socsim core seed |
| `--llm-temperature` | 0.0 | generation temperature |
| `--llm-seed` | 0 | backend generation seed |
| `--cache-path` | `.llm_cache/cache.json` | prompt→response cache file |
| `--mock` | off | drive offline with a deterministic scripted mock (no live LLM) |
| `--output-dir` | `results` | runvault results root |

Writes one [runvault](https://github.com/akitenkrad/rs-runvault) run under `<output-dir>/competeai/run_<timestamp>_<config_hash>_<execution_hash>/`: `config.json` (the conditions, under `parameters`), `run.json` (including the `llm` block — model/endpoint/temperature), `metrics.csv` (per-day aggregates — `revenue_gini`/`market_share_max`/`menu_similarity`/`n_alive_firms` — plus run-scope values — `llm_calls`/`llm_cache_hits`/`llm_cache_hit_rate`/`winner_take_all`/`quality_improved`), `events.jsonl` (a per-firm `observation` row for every day, plus one `terminal` row per firm recording its outcome), `status.json` and `manifest.csv`. When `--runs > 1`, only the last trial is recorded to disk (see the `--runs` note above); the console still prints the winner-take-all and quality-improvement frequencies across all trials.

## `sweep`

Sweep store count × customer count and aggregate the Matthew-effect metrics.

```bash
cargo run --release -- sweep \
    --n-firms-values 2,3,4 \
    --n-customers-min 20 --n-customers-max 80 --n-customers-step 20 \
    --days 15 --runs 5 --seed 42
```

| Flag | Default | Meaning |
|---|---|---|
| `--n-firms-values` | `2,3,4` | comma-separated store counts |
| `--n-customers-min/max/step` | 20 / 80 / 20 | customer-count range |
| `--customer-mode` | `individual` | `individual` or `group` |
| `--days` | 15 | number of days |
| `--runs` | 5 | trials per cell (each derived seed) |
| `--seed` | 42 | base seed (cells/trials derive from it) |
| `--cache-path` | `.llm_cache/cache.json` | shared cache (raises hit rate across cells) |
| `--output-dir` | `results` | runvault results root |

Writes a runvault **parent** run under `<output-dir>/competeai/sweep_<timestamp>_<config_hash>_<execution_hash>/` whose `config.json` `parameters` hold the sweep grid (`n_firms_values`, `n_customers_values`, …); the parent has no per-cell metrics. Each (store count × customer count × trial) cell is its own **child** run — same layout as `run` above — linked to the parent via `lineage.parent_run_uid`. There is no `sweep_summary.csv` on disk: `uv run competeai-tools visualize-sweep` rebuilds the one-row-per-cell×trial table from the children on demand (see [Visualization](visualization.md)).

## `reproduce`

Batch the paper's Table 2 occurrence frequencies: run the individual-customer condition and the group-customer condition for several independent trials each, then score the observed winner-take-all / quality-improvement / menu-similarity frequencies against the paper.

```bash
# Offline (scripted mock) — used in sandbox/CI
cargo run --release -- reproduce --mock --seed 42
# Live LLM (after building and starting Ollama)
cargo run --release -- reproduce --seed 42
```

| Flag | Default | Meaning |
|---|---|---|
| `--n-firms` | 2 | number of restaurants M |
| `--n-customers` | 50 | number of customers N |
| `--group-size` | 4 | members per group (group condition) |
| `--days` | 15 | number of days |
| `--individual-runs` | 9 | independent trials for the individual condition (paper Table 2 = 9) |
| `--group-runs` | 6 | independent trials for the group condition (paper Table 2 = 6) |
| `--seed` | 42 | base seed (conditions/trials derive from it) |
| `--mock` | off | drive offline with a deterministic scripted mock (no live LLM) |
| `--llm-temperature` | 0.0 | generation temperature (live only) |
| `--llm-seed` | 0 | backend generation seed (live only) |
| `--cache-path` | `.llm_cache/cache.json` | shared cache (live only) |
| `--quick` | off | shrink N / trials / days for a fast smoke (not for validating paper values) |
| `--output-dir` | `results` | runvault results root |

Writes a runvault **parent** run under `<output-dir>/competeai/reproduce_<timestamp>_<config_hash>_<execution_hash>/` with one **child** run per trial (individual and group), linked via `lineage.parent_run_uid`. The parent's `metrics.csv` carries the cross-condition aggregates as `scope=sweep` metrics — `wta_freq_individual`, `wta_freq_group`, `quality_freq_individual`, `quality_freq_group`, `quality_freq_all`, `menu_similarity_individual`, `menu_similarity_group`, `menu_similarity_all`, `final_gini_individual`, `final_gini_group`, `final_share_max_individual`, `final_share_max_group`, and the directional `wta_freq_gap_individual_minus_group` — and the parent's `reference.csv` carries the paper's own Table 2 / §4 values (individual winner-take-all 66.7%, group winner-take-all 16.7%, quality improvement 86.67%, menu similarity ≈ 36%; each row has a `source`). The ±15pt / ±10pt pass/off band is this replication's own choice, not the paper's, so it is **not** recorded in any file — it appears only in the `competeai reproduce` console output. The Python `competeai-tools reproduce` renders the figures from the parent's metrics and `reference.csv`.

---
*This file was generated by Claude Code.*

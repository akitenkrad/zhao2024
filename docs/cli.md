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
| `--output-dir` | `results` | output base directory |

Outputs into `results/{timestamp}/`: `config.json`, `metrics.csv` (long-format, one row per day×firm), `run_metadata.json` (LLM model/endpoint/temperature/seed/cache-hit + winner_take_all + quality_improved). A `results/latest` symlink points at the newest run. When `--runs > 1`, the console prints the winner-take-all and quality-improvement frequencies across trials.

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
| `--output-dir` | `results` | output base directory |

Outputs into `results/{timestamp}_sweep/`: `sweep_config.json` and `sweep_summary.csv` (one row per cell×trial: final revenue Gini, final market share, winner_take_all, quality_improved, final menu similarity, surviving firms, cache-hit rate). `results/latest` points at the sweep directory.

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
| `--output-dir` | `results` | output base directory |

Outputs into `results/reproduce_{timestamp}/`: `reproduce_summary.json` (per-condition cells + Table 2 anchors with observed-vs-paper and pass/off), and `metrics_individual.csv` / `metrics_group.csv` (the representative trial of each condition). `results/latest` points at the reproduce directory. The anchors are: the individual winner-take-all frequency (paper 66.7%), the group winner-take-all frequency (paper 16.7%), the directional individual > group attenuation, the quality-improvement frequency (paper 86.67%), and the menu similarity (paper ≈ 36%; reported as a structural reference, since this Phase holds menu items fixed). The Python `competeai-tools reproduce` renders the figures.

---
*This file was generated by Claude Code.*

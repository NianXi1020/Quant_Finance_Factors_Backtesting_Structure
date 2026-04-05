# Quant Finance Factors Backtesting Structure (Stage 1)

This repository is Stage-1 of an A-share single-factor research pipeline, currently focused on **momentum** factors with strict anti-look-ahead conventions.

## What is implemented now

- CN raw data loading and schema standardization (daily HFQ, stock list, delisting)
- Reusable universe construction (`is_alive`, listing-age filter, `in_universe`)
- Reusable forward return label (`fwd_1d_return`)
- Momentum factor generation (e.g., `rs_20`) with winsorization/z-score/industry-neutral variants
- IC / Rank-IC analysis outputs
- Quantile backtest outputs (including long-short summary)
- Process-based parallelism for heavy stages (daily file cleaning and per-stock factor computation)
- Staged caching to avoid recomputing expensive upstream data

---

## Storage layout (refactored)

### 1) `data/interim/` (reusable shared artifacts)

```text
data/interim/
  panels/
    daily_panel_clean.parquet
  metadata/
    stock_list_clean.parquet
    delist_clean.parquet
  universe/
    universe_basic.parquet
  labels/
    forward_returns_1d.parquet
```

### 2) `data/processed/` (factor-specific processed artifacts)

```text
data/processed/
  factors/
    momentum/
      rs_20.parquet
      rs_40.parquet
      ...
  manifest/
    factor_registry.csv
```

Each factor file stores only factor-relevant columns (not full OHLCV table), e.g.:
`date, ts_code, stock_code, factor_raw, factor_win, factor_z, factor_indneu`.

### 3) `outputs/` (research/evaluation results)

```text
outputs/
  single_factor/
    momentum/
      rs_20/
        ic_analysis/
        quantile_backtest/
        stability/
        summaries/
        logs/
```

The pipeline writes IC summaries, Rank-IC summaries, quantile returns/NAV, long-short metrics, yearly stability tables, and factor run snapshots.

---

## Pipeline stages

- **Stage A**: clean raw daily/metadata/delisting and cache to `data/interim/`
- **Stage B**: build reusable universe + labels and cache to `data/interim/`
- **Stage C**: compute selected factor and store to `data/processed/factors/...`
- **Stage D**: run IC + quantile evaluation and store artifacts under `outputs/single_factor/...`

The previous single giant `pipeline_output.parquet` behavior is deprecated. A debug combined panel export exists but is **off by default**.

---

## Caching behavior

Configured in `src/config.py`:

- `use_cache`
- `force_rebuild_cleaning`
- `force_rebuild_universe`
- `force_rebuild_labels`
- `force_rebuild_factors`

Typical behavior:
- if cached artifact exists and force flag is `False`, pipeline loads cache directly;
- if raw data changes, set corresponding `force_rebuild_* = True`.

---

## Parallelism controls

Configured in `src/config.py` (`RuntimeConfig`):

- `use_parallel` (default `True`)
- `n_jobs` (default `max(1, os.cpu_count()-1)`)
- `verbose_timing`

Parallelized bottlenecks:
- loading/cleaning many daily HFQ per-stock files
- per-stock momentum computation

Memory note: process-based parallelism can increase peak memory usage.

---

## Structured progress logging

Pipeline emits lightweight step logs such as:

- `[STEP] Stage A: Loading/Cleaning daily panel...`
- `[OK] Daily panel ready (time=..., rows=...)`
- `[PIPELINE COMPLETE] total time = ...s`

Controls in `LogConfig`:
- `enabled`
- `verbose`

---

## Data location assumptions

Daily HFQ files:

```text
data/raw/A_share_data/daily_hfq/
  000001_daily_hfq.csv
  000002_daily_hfq.csv
  ...
```

Delisting file:

```text
data/raw/A_share_data/Delisting/delisting.csv
```

Default paths are defined in `src/config.py`.

---

## Run

```bash
python -m src.main
```

---

## Notebook

Demo notebook:
- `notebooks/stage1_momentum_factor_showcase.ipynb`

(Notebook is for presentation/inspection only; modular pipeline under `src/` is the primary workflow.)

---

## Roadmap

Current stage focuses on momentum single-factor research infrastructure.
Planned next stages include multi-factor modeling, return prediction, portfolio construction/optimization, and expanded diagnostics.

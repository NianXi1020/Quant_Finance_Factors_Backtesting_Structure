# Quant Finance Factors Backtesting Structure (Stage 1)

This project is the **first version** of an A-share single-factor research pipeline, currently focused on a **momentum factor** workflow.  
It is designed with modular Python code under `src/` to keep data handling, factor construction, and research steps maintainable and extensible.

## Current scope (implemented in Stage 1)

- **Data loading and cleaning**
  - Explicit Chinese-to-English schema mapping
  - Robust date parsing (`YYYYMMDD` / `YYYY-MM-DD`)
  - Consistent `stock_code` / `ts_code` normalization
  - Daily HFQ data loaded from a **directory of per-stock files** (`*_daily_hfq.csv`)
- **Universe construction**
  - Delisting-aware alive mask (`is_alive`)
  - IPO listing-age filter (`is_old_enough`)
  - Combined tradable universe flag (`in_universe`)
- **Forward return label**
  - 1-day forward return (`fwd_1d_return`) for evaluation
- **Momentum factor construction**
  - Simple cross-sectional momentum from rolling price change
- **Factor preprocessing**
  - Cross-sectional z-score
  - Industry-aware neutralized version (`mom_ind_neutral`)
  - 1-day signal lag (`signal`) to avoid look-ahead bias
- **IC analysis**
  - Demonstrated in showcase notebook (Spearman IC / ICIR)
- **Quantile backtest**
  - Demonstrated in showcase notebook (quantile group returns / long-short spread)
- **Output saving**
  - Pipeline output is saved to `output/pipeline_output.parquet`

---

## Project structure

```text
.
├── README.md
├── src/
│   ├── config.py                 # path and pipeline configuration
│   ├── main.py                   # lightweight executable entrypoint
│   ├── pipeline.py               # end-to-end pipeline orchestration
│   ├── data/
│   │   └── loaders.py            # raw CSV loading + cleaning + schema mapping
│   ├── factors/
│   │   └── basic.py              # momentum factor + z-score preprocessing
│   ├── universe.py               # universe eligibility logic
│   └── utils/
│       └── identifiers.py        # stock_code / ts_code normalization helpers
└── notebooks/
    └── stage1_momentum_factor_showcase.ipynb
```

---

## How to run the main pipeline

1. Place your raw files under `data/raw/A_share_data/` and make sure daily data is organized as:

```text
data/raw/A_share_data/daily_hfq/
├── 000001_daily_hfq.csv
├── 000002_daily_hfq.csv
├── ...
```

And delisting data is expected at:

```text
data/raw/A_share_data/Delisting/delisting.csv
```

The default paths are configured in `src/config.py`.
2. Run:

```bash
python -m src.main
```

(Equivalent: `python src/main.py`)

---

## Output location

After successful execution, the main result table is saved to:

- `output/pipeline_output.parquet`

This table includes daily panel fields such as universe flags, momentum factor values, lagged signal, and forward return label.

---

## Notebook usage (showcase / inspection)

The demo notebook is:

- `notebooks/stage1_momentum_factor_showcase.ipynb`

Recommended usage:

1. Start Jupyter from project root:

```bash
jupyter lab
```

2. Open `notebooks/stage1_momentum_factor_showcase.ipynb`.

The first code cell includes a small `sys.path` setup so `from src...` imports work even when the notebook kernel runs with `notebooks/` as working directory.

---

## Parallel performance settings

To improve CPU utilization on large universes, Stage-1 now supports optional process-based parallelism in the main bottlenecks:

- **Daily HFQ loading/cleaning** across many `*_daily_hfq.csv` files
- **Per-stock momentum computation** (independent by stock)

Controls are in `src/config.py` (`RuntimeConfig`):

- `use_parallel` (default `True`)
- `n_jobs` (default `max(1, os.cpu_count()-1)`)
- `verbose_timing` (print simple stage timing logs)

Notes/caveats:

- A **serial fallback path** is always available by setting `use_parallel=False` or `n_jobs=1` (useful for debugging/repro checks).
- Process-based parallelism may increase memory usage because workers hold intermediate DataFrames.
- Final concatenation/sorting still enforces deterministic date/stock ordering and preserves signal/return alignment logic.

---

## Roadmap

This repository is currently **Stage 1**, focused on momentum single-factor research and clean pipeline foundations.  
Later stages are planned to cover:

- multi-factor modeling
- return prediction workflows
- portfolio construction and optimization
- expanded backtesting and risk diagnostics

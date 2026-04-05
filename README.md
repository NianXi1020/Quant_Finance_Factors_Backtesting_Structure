# Quant Finance Factors Backtesting Structure (Stage 1)

This project is the **first version** of an A-share single-factor research pipeline, currently focused on a **momentum factor** workflow.  
It is designed with modular Python code under `src/` to keep data handling, factor construction, and research steps maintainable and extensible.

## Current scope (implemented in Stage 1)

- **Data loading and cleaning**
  - Explicit Chinese-to-English schema mapping
  - Robust date parsing (`YYYYMMDD` / `YYYY-MM-DD`)
  - Consistent `stock_code` / `ts_code` normalization
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

1. Place your raw files under `data/` (the pipeline expects configured file names/paths in `src/config.py`).
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

## Roadmap

This repository is currently **Stage 1**, focused on momentum single-factor research and clean pipeline foundations.  
Later stages are planned to cover:

- multi-factor modeling
- return prediction workflows
- portfolio construction and optimization
- expanded backtesting and risk diagnostics

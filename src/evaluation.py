"""Backward-compatible re-export module.

Core evaluation computation has moved to `src.evaluation_core`.
"""

from src.evaluation_core import (  # noqa: F401
    EvaluationParams,
    params_to_dict,
    run_ic_analysis,
    run_quantile_backtest,
    save_run_metadata,
)

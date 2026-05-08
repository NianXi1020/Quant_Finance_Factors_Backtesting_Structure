from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_feature_importance(output_path: Path, train_end_date, feature_cols: list[str], importances) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"train_end_date": train_end_date, "feature": feature_cols, "importance": importances}).to_csv(output_path, mode="a", header=not output_path.exists(), index=False)

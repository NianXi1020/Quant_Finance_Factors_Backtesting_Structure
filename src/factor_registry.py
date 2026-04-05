from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorSpec:
    key: str
    family: str
    name: str
    lookback: int | None
    eval_signal_col: str
    description: str = ""


def build_factor_registry() -> list[FactorSpec]:
    specs: list[FactorSpec] = []
    for lb in [30, 60, 90, 180]:
        specs.append(FactorSpec(key=f"rs_{lb}", family="momentum", name="rs", lookback=lb, eval_signal_col="factor_indneu", description="Relative strength"))
        specs.append(FactorSpec(key=f"hl_{lb}", family="momentum", name="hl", lookback=lb, eval_signal_col="factor_indneu", description="High-low ratio"))
        specs.append(FactorSpec(key=f"vol_{lb}", family="momentum", name="vol", lookback=lb, eval_signal_col="factor_indneu", description="Rolling volatility"))
        specs.append(FactorSpec(key=f"turnover_{lb}", family="momentum", name="turnover", lookback=lb, eval_signal_col="factor_indneu", description="Rolling turnover mean"))
        specs.append(FactorSpec(key=f"improved_mom_{lb}", family="momentum", name="improved_mom", lookback=lb, eval_signal_col="factor_indneu", description="Rolling sum(return*turnover)"))

    specs.append(
        FactorSpec(
            key="macd",
            family="momentum",
            name="macd",
            lookback=None,
            eval_signal_col="macd_bar_indneu",
            description="MACD family (DIF/DEA/MACD_bar)",
        )
    )
    return specs


def registry_by_key() -> dict[str, FactorSpec]:
    return {s.key: s for s in build_factor_registry()}

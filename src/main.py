from __future__ import annotations

import argparse
from dataclasses import replace

from src.config import PipelineConfig
from src.orchestrator import run_factor_orchestration, run_visualization_only


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Factor orchestration runner")
    p.add_argument("--factors", nargs="+", default=None, help="Specific factor keys, e.g. rs_30 rs_90 macd")
    p.add_argument("--group", default=None, help="Run all factors in a group, e.g. momentum")
    p.add_argument("--only-missing", action="store_true", default=True, help="Run only missing/incomplete factors (default)")
    p.add_argument("--force", action="store_true", help="Force rerun selected factors")
    p.add_argument("--use-cached-interim", action="store_true", default=True, help="Reuse cached shared artifacts")
    p.add_argument("--skip-evaluation", action="store_true", help="Compute factor files but skip evaluation")
    p.add_argument("--run-evaluation-only", action="store_true", help="Reuse factor files and run evaluation only")
    p.add_argument("--run-visualization-only", action="store_true", help="Generate plots only from existing evaluation artifacts")
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = PipelineConfig()
    stage = cfg.stage

    if args.skip_evaluation:
        stage = replace(stage, run_evaluation=False)
    if args.run_evaluation_only:
        stage = replace(stage, run_factor=False, run_evaluation=True)

    stage = replace(stage, use_cached_interim=args.use_cached_interim)
    cfg = replace(cfg, stage=stage)

    if args.run_visualization_only:
        result = run_visualization_only(
            cfg,
            factors=args.factors,
            group=args.group,
        )
    else:
        result = run_factor_orchestration(
            cfg,
            factors=args.factors,
            group=args.group,
            only_missing=args.only_missing,
            force=args.force,
        )

    print("\nRun summary:")
    print(f"  selected: {result['selected']}")
    if "run" in result:
        print(f"  run: {result['run']}")
    if "generated" in result:
        print(f"  generated: {result['generated']}")
    print(f"  skipped: {result['skipped']}")


if __name__ == "__main__":
    main()

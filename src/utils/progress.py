from __future__ import annotations

import time


class StepLogger:
    """Simple structured step logger for pipeline stages."""

    def __init__(self, enabled: bool = True, verbose: bool = False) -> None:
        self.enabled = enabled
        self.verbose = verbose

    def step(self, message: str) -> float:
        if self.enabled:
            print(f"[STEP] {message}...")
        return time.perf_counter()

    def done(self, message: str, started_at: float, extra: str | None = None) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - started_at
        suffix = f", {extra}" if extra else ""
        print(f"[OK] {message} (time={elapsed:.2f}s{suffix})")

    def info(self, message: str) -> None:
        if self.enabled and self.verbose:
            print(f"[INFO] {message}")

    def complete(self, started_at: float) -> None:
        if self.enabled:
            print(f"[PIPELINE COMPLETE] total time = {time.perf_counter() - started_at:.2f}s")

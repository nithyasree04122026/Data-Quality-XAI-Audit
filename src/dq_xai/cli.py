"""Command-line interface for experiment matrices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .experiment import Config, run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit data-quality effects on ML and XAI")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("results/matrix.csv"))
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    seeds = config["seeds"]
    missing_rates = config["missing_rates"]
    label_flip_rates = config["label_flip_rates"]
    repeats = config.get("permutation_repeats", 3)
    rows = [
        run(Config(seed=seed, missing_rate=missing, label_flip_rate=flip, repeats=repeats))
        for seed in seeds
        for missing in missing_rates
        for flip in label_flip_rates
    ]
    if not rows:
        parser.error("experiment matrix is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path


def combine_split(base: Path, split: str) -> Path:
    files = sorted((base / split).glob(f"CDPK_*_{split}.csv"))
    rows = []
    fields = None
    for fp in files:
        with fp.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            fields = r.fieldnames
            rows.extend(list(r))
    if fields is None:
        raise RuntimeError(f"No files found for split={split}")
    outdir = base / "combined"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"CDPK_all_{split}.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{split}: files={len(files)} rows={len(rows)} out={out}")
    return out


def main() -> None:
    base = Path("/home/ubuntu/pedagogy-benchmark/data/Chile/CDPK_per_category")
    combine_split(base, "dev")
    combine_split(base, "test")


if __name__ == "__main__":
    main()

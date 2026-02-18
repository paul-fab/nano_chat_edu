#!/usr/bin/env python
"""
Compute full-subset training iterations from local parquet shards.

This expects prebuilt subset shards already downloaded to data_dir.
If a token-count column exists, the token total is exact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import duckdb


def main() -> None:
    p = argparse.ArgumentParser(description="Calculate iterations to train on all subset tokens.")
    p.add_argument("--data-dir", default=os.path.expanduser("~/.cache/nanochat/base_data"))
    p.add_argument("--glob", default="shard_*.parquet")
    p.add_argument("--token-col", default="token_count")
    p.add_argument("--total-batch-size", type=int, default=524288, help="Tokens per optimization step.")
    args = p.parse_args()

    if args.total_batch_size <= 0:
        raise ValueError("--total-batch-size must be > 0")

    pattern = os.path.join(args.data_dir, args.glob).replace("\\", "/")
    if not os.path.isdir(args.data_dir):
        print(f"ERROR: data dir not found: {args.data_dir}")
        sys.exit(1)

    conn = duckdb.connect()
    escaped_pattern = pattern.replace("'", "''")
    nfiles = conn.execute(
        f"SELECT COUNT(*) FROM glob('{escaped_pattern}')"
    ).fetchone()[0]
    if nfiles == 0:
        print(f"ERROR: no parquet files matched: {pattern}")
        sys.exit(1)

    first_file = conn.execute(
        f"SELECT file FROM glob('{escaped_pattern}') LIMIT 1"
    ).fetchone()[0]
    escaped_first_file = str(first_file).replace("'", "''")
    schema_rows = conn.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{escaped_first_file}')"
    ).fetchall()
    columns = {row[0] for row in schema_rows}

    if args.token_col not in columns:
        print(
            json.dumps(
                {
                    "status": "missing_token_col",
                    "message": f"Column '{args.token_col}' not found. Available columns in first shard: {sorted(columns)}",
                    "files": nfiles,
                    "pattern": pattern,
                },
                indent=2,
            )
        )
        sys.exit(2)

    total_rows = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{escaped_pattern}', union_by_name=true)"
    ).fetchone()[0]
    total_tokens = conn.execute(
        f"SELECT SUM(COALESCE(CAST({args.token_col} AS BIGINT), 0)) FROM read_parquet('{escaped_pattern}', union_by_name=true)"
    ).fetchone()[0]
    total_tokens = int(total_tokens or 0)
    num_iterations = int(math.ceil(total_tokens / args.total_batch_size)) if total_tokens > 0 else 0

    payload = {
        "status": "ok",
        "files": int(nfiles),
        "rows": int(total_rows),
        "token_col": args.token_col,
        "tokens": total_tokens,
        "total_batch_size": int(args.total_batch_size),
        "num_iterations": num_iterations,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

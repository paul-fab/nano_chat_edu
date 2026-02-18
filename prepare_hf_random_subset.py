#!/usr/bin/env python
"""
Stream a Hugging Face dataset split and materialize a deterministic random subset
into nanochat-style parquet shards (shard_00000.parquet, ...).
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import get_dataset_config_names, load_dataset


def keep_row(key: str, ratio: float) -> bool:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return (value / 2**64) < ratio


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare deterministic random subset from HF dataset.")
    p.add_argument("--dataset", required=True, help="HF dataset id, e.g. airtrain-ai/fineweb-edu-fortified")
    p.add_argument("--config", default=None, help="Optional HF dataset config.")
    p.add_argument("--split", default="train")
    p.add_argument("--ratio", type=float, default=0.165, help="Subset fraction in (0,1], e.g. 0.165")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default=os.path.expanduser("~/.cache/nanochat/base_data"))
    p.add_argument("--rows-per-shard", type=int, default=14000)
    p.add_argument(
        "--text-column-candidates",
        nargs="+",
        default=["text", "content", "raw_text"],
        help="Columns to try (in order) for training text.",
    )
    args = p.parse_args()

    if args.ratio <= 0 or args.ratio > 1:
        raise ValueError("--ratio must be in (0, 1].")

    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.config:
        configs = [args.config]
    else:
        configs = get_dataset_config_names(args.dataset)
        if not configs:
            raise RuntimeError(f"No configs found for dataset: {args.dataset}")
        print(f"No --config provided. Streaming all configs ({len(configs)}): {configs[:5]}{' ...' if len(configs) > 5 else ''}")

    selected_text_col = None
    has_token_count = False

    shard_idx = 0
    kept = 0
    seen = 0
    batch_text: list[str] = []
    batch_tokens: list[int] = []

    def flush() -> None:
        nonlocal shard_idx, batch_text, batch_tokens
        if not batch_text:
            return
        data = {"text": pa.array(batch_text, type=pa.string())}
        if has_token_count:
            data["token_count"] = pa.array(batch_tokens, type=pa.int64())
        table = pa.table(data)
        out_path = out_dir / f"shard_{shard_idx:05d}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        shard_idx += 1
        batch_text = []
        batch_tokens = []

    for cfg in configs:
        print(f"Streaming config: {cfg}")
        ds = load_dataset(
            args.dataset,
            cfg,
            split=args.split,
            streaming=True,
        )
        for row_idx, row in enumerate(ds):
            if selected_text_col is None:
                for cand in args.text_column_candidates:
                    if cand in row and row[cand] is not None:
                        selected_text_col = cand
                        break
                if selected_text_col is None:
                    raise RuntimeError(
                        f"Could not find text column. Tried: {args.text_column_candidates}. "
                        f"Available columns: {list(row.keys())}"
                    )
                has_token_count = "token_count" in row
                print(
                    f"Using text column '{selected_text_col}'"
                    + (" with token_count" if has_token_count else " (token_count not found)")
                )

            seen += 1
            text = row.get(selected_text_col)
            if text is None:
                continue
            text = str(text)
            if not text.strip():
                continue

            row_key = str(row.get("id", f"{cfg}:{row_idx}:{text[:128]}"))
            hash_key = f"{args.seed}|{row_key}"
            if not keep_row(hash_key, args.ratio):
                continue

            kept += 1
            batch_text.append(text)
            if has_token_count:
                tok = row.get("token_count")
                batch_tokens.append(int(tok) if tok is not None else 0)

            if len(batch_text) >= args.rows_per_shard:
                flush()

            if seen % 1_000_000 == 0:
                pct = (kept / seen) * 100 if seen else 0.0
                print(f"seen={seen:,} kept={kept:,} keep_rate={pct:.2f}% shards={shard_idx}")

    flush()
    print(f"Done. seen={seen:,} kept={kept:,} shards={shard_idx}")


if __name__ == "__main__":
    main()

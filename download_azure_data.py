"""
download_azure_data.py -- Stream parquet files from Azure Blob Storage,
drop only embeddings (keep text + quality metadata), and save as nanochat-compatible shards.

After downloading, sorts all rows globally by quality score (descending) and re-shards,
so the highest-quality documents are in the first shards that nanochat reads.

Usage:
    python download_azure_data.py                    # download all shards, 8 workers
    python download_azure_data.py -w 16              # 16 parallel workers
    python download_azure_data.py -n 100             # download first 100 shards only
    python download_azure_data.py --data-dir /data   # custom output directory
    python download_azure_data.py --sort-key pedagogical_structure_average  # sort by specific metric

Requires .env with AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_KEY.
"""

import math
import os
import shutil
import subprocess
import sys
import time
import argparse
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
import pyarrow as pa
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobSasPermissions, generate_container_sas
from dotenv import load_dotenv

load_dotenv()

ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT", "quratingscoressa")
KEY = os.environ.get("AZURE_STORAGE_KEY")
CONTAINER = os.environ.get("AZURE_CONTAINER", "quratingfiltered")
DATA_DIR = os.path.expanduser("~/.cache/nanochat/base_data")

# Columns to drop (large embeddings we don't need for training)
DROP_COLUMNS = {"embedding"}

# Columns to keep for quality sorting (plus text)
QUALITY_COLUMNS = [
    "education_level_average",
    "education_level_primary_average",
    "education_level_secondary_average",
    "factual_accuracy_average",
    "lesson_engagement_average",
    "pedagogical_structure_average",
]

if not KEY:
    print("ERROR: AZURE_STORAGE_KEY not set. Create a .env file or export it.")
    sys.exit(1)


def get_container_client(container_name: str) -> ContainerClient:
    conn_str = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={ACCOUNT};"
        f"AccountKey={KEY};"
        f"EndpointSuffix=core.windows.net"
    )
    return BlobServiceClient.from_connection_string(conn_str).get_container_client(container_name)


def list_blobs(container_client: ContainerClient, prefix: str | None = None) -> list[str]:
    """List all parquet blob names in the container, sorted."""
    blobs = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        if blob.name.endswith(".parquet"):
            blobs.append(blob.name)
    blobs.sort()
    return blobs


def download_shard(
    container_client: ContainerClient,
    blob_name: str,
    shard_index: int,
    data_dir: str,
    max_retries: int = 3,
) -> tuple[int, str, int]:
    """Download a blob, drop embeddings, keep text + quality metadata.

    Returns (shard_index, status, bytes_written).
    """
    out_path = os.path.join(data_dir, f"shard_{shard_index:05d}.parquet")

    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        return shard_index, "skipped", size

    for attempt in range(max_retries):
        try:
            blob_client = container_client.get_blob_client(blob_name)
            blob_data = blob_client.download_blob().readall()

            buf = pa.BufferReader(blob_data)
            table = pq.read_table(buf)

            # Drop embedding column (and any other large columns)
            cols_to_drop = [c for c in table.column_names if c in DROP_COLUMNS]
            if cols_to_drop:
                table = table.drop(cols_to_drop)

            if table.num_rows == 0:
                return shard_index, f"warning: empty table in {blob_name}", 0

            tmp_path = out_path + ".tmp"
            pq.write_table(table, tmp_path, compression="zstd")
            os.rename(tmp_path, out_path)

            size = os.path.getsize(out_path)
            return shard_index, "done", size

        except Exception as e:
            tmp_path = out_path + ".tmp"
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                return shard_index, f"error after {max_retries} retries: {e}", 0


def transform_local_parquet_to_shard(
    in_path: str,
    out_path: str,
) -> tuple[str, int]:
    """Load local parquet, drop embedding-like columns, and write shard parquet."""
    table = pq.read_table(in_path)
    cols_to_drop = [c for c in table.column_names if c in DROP_COLUMNS]
    if cols_to_drop:
        table = table.drop(cols_to_drop)
    if table.num_rows == 0:
        return "warning: empty table", 0
    tmp_path = out_path + ".tmp"
    pq.write_table(table, tmp_path, compression="zstd")
    os.rename(tmp_path, out_path)
    return "done", os.path.getsize(out_path)


def _run_azcopy_bulk_download(
    *,
    account: str,
    key: str,
    container: str,
    prefix: str,
    raw_dir: str,
) -> bool:
    """Use azcopy for high-throughput parquet transfer into raw_dir."""
    if shutil.which("azcopy") is None:
        print("azcopy not found in PATH; falling back to python download.")
        return False

    expiry = datetime.now(timezone.utc) + timedelta(hours=12)
    sas = generate_container_sas(
        account_name=account,
        container_name=container,
        account_key=key,
        permission=BlobSasPermissions(read=True, list=True),
        expiry=expiry,
    )
    src = f"https://{account}.blob.core.windows.net/{container}"
    if prefix:
        src = f"{src}/{prefix}"
    src = f"{src}?{sas}"

    os.makedirs(raw_dir, exist_ok=True)
    cmd = [
        "azcopy",
        "copy",
        src,
        raw_dir,
        "--recursive=true",
        "--overwrite=false",
        "--include-pattern=*.parquet",
        "--output-level=essential",
    ]
    print(f"Starting azcopy bulk download to {raw_dir} ...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"azcopy failed ({e}); falling back to python download.")
        return False


def bulk_download_and_transform(
    *,
    container_name: str,
    blobs: list[str],
    blob_prefix: str,
    data_dir: str,
    workers: int,
) -> tuple[int, int, int]:
    """Bulk transfer via azcopy, then local transform into shard_XXXXX.parquet files."""
    raw_dir = os.path.join(data_dir, "_azcopy_raw")
    ok = _run_azcopy_bulk_download(
        account=ACCOUNT,
        key=KEY,
        container=container_name,
        prefix=blob_prefix,
        raw_dir=raw_dir,
    )
    if not ok:
        return -1, -1, -1

    done = 0
    errors = 0
    total_bytes = 0
    t0 = time.time()

    def resolve_raw_path(blob_name: str) -> str:
        rel_name = blob_name
        if blob_prefix and rel_name.startswith(blob_prefix):
            rel_name = rel_name[len(blob_prefix):].lstrip("/")
        path = os.path.join(raw_dir, rel_name)
        if os.path.exists(path):
            return path
        base = os.path.basename(blob_name)
        alt = os.path.join(raw_dir, base)
        if os.path.exists(alt):
            return alt
        return path

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {}
        for i, blob_name in enumerate(blobs):
            out_path = os.path.join(data_dir, f"shard_{i:05d}.parquet")
            if os.path.exists(out_path):
                done += 1
                total_bytes += os.path.getsize(out_path)
                continue
            in_path = resolve_raw_path(blob_name)
            futures[executor.submit(transform_local_parquet_to_shard, in_path, out_path)] = (i, in_path)

        for future in as_completed(futures):
            idx, in_path = futures[future]
            try:
                status, nbytes = future.result()
                if status == "done":
                    done += 1
                    total_bytes += nbytes
                else:
                    errors += 1
                    print(f"  [{idx:05d}] {status} ({in_path})")
            except Exception as e:
                errors += 1
                print(f"  [{idx:05d}] error: {e} ({in_path})")

            if done % 50 == 0 and done > 0:
                elapsed = max(time.time() - t0, 1e-6)
                rate = done / elapsed
                print(f"  Transform progress: {done}/{len(blobs)} shards | {rate:.2f} shards/s")

    shutil.rmtree(raw_dir, ignore_errors=True)
    return done, errors, total_bytes


def sort_and_reshard(data_dir: str, sort_keys: list[str], rows_per_shard: int = 14000):
    """Read all shards, sort globally by composite quality score, re-write as sorted shards.

    Computes a composite score as the average of the specified quality columns.
    Documents that are strong across ALL metrics rank highest.
    After sorting, shard_00000 contains the highest-quality documents.
    """
    print(f"\n=== Sorting by composite score: {' + '.join(sort_keys)} ===")

    # Read all shards
    shard_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir)
         if f.endswith(".parquet") and not f.endswith(".tmp")]
    )
    print(f"Reading {len(shard_files)} shards...")

    tables = []
    for i, f in enumerate(shard_files):
        tables.append(pq.read_table(f))
        if (i + 1) % 500 == 0:
            print(f"  Read {i + 1}/{len(shard_files)}")

    combined = pa.concat_tables(tables, promote_options="default")
    total_rows = combined.num_rows
    print(f"Total rows: {total_rows:,}")

    # Cast string columns to large_string to avoid 32-bit offset overflow on sort
    new_columns = []
    for i, field in enumerate(combined.schema):
        col = combined.column(i)
        if field.type == pa.string():
            col = col.cast(pa.large_string())
        new_columns.append(col)
    combined = pa.table({field.name: col for field, col in zip(combined.schema, new_columns)})
    print("  Cast string columns to large_string")

    # Verify all sort keys exist
    for key in sort_keys:
        if key not in combined.column_names:
            print(f"ERROR: sort key '{key}' not in columns: {combined.column_names}")
            return False

    # Compute composite score = average of the quality columns
    # Null values treated as 0
    arrays = []
    for key in sort_keys:
        col = combined.column(key)
        filled = pa.compute.if_else(pa.compute.is_null(col), 0.0, col)
        arrays.append(filled)
        print(f"  {key}: min={pa.compute.min(col).as_py():.2f}, max={pa.compute.max(col).as_py():.2f}, mean={pa.compute.mean(col).as_py():.2f}")

    # Average the columns
    composite = arrays[0]
    for arr in arrays[1:]:
        composite = pa.compute.add(composite, arr)
    composite = pa.compute.divide(composite, len(sort_keys))

    # Sort descending by composite score
    # Add composite as a column so we can use table-based sorting
    combined = combined.append_column("_composite", composite)
    combined = combined.sort_by([("_composite", "descending")])
    composite_sorted = combined.column("_composite")
    combined = combined.drop("_composite")

    # Show composite distribution (use compute functions, not to_pylist on 52M rows)
    print(f"\n  Composite score range: {pa.compute.min(composite_sorted).as_py():.2f} to {pa.compute.max(composite_sorted).as_py():.2f}")
    print(f"  Mean: {pa.compute.mean(composite_sorted).as_py():.2f}")
    # Sample quantiles
    n = len(composite_sorted)
    print(f"  Top-10% threshold: {composite_sorted[n // 10].as_py():.2f}")
    print(f"  Top-25% threshold: {composite_sorted[n // 4].as_py():.2f}")
    print(f"  Median: {composite_sorted[n // 2].as_py():.2f}")

    # Strip to text-only for nanochat compatibility
    text_only = combined.select(["text"])

    # Clear old shards
    for f in shard_files:
        os.remove(f)

    # Write new sorted shards
    num_shards = (total_rows + rows_per_shard - 1) // rows_per_shard
    print(f"Writing {num_shards} sorted shards...")

    for i in range(num_shards):
        start = i * rows_per_shard
        end = min(start + rows_per_shard, total_rows)
        shard = text_only.slice(start, end - start)
        out_path = os.path.join(data_dir, f"shard_{i:05d}.parquet")
        pq.write_table(shard, out_path, compression="zstd")

    print(f"Done! {num_shards} sorted shards written (text-only, highest quality first)")
    return True


def sort_top_percent_external(
    data_dir: str,
    sort_keys: list[str],
    top_percent: float,
    rows_per_shard: int = 14000,
    memory_limit: str = "4GB",
):
    """Memory-safe top-percent selection using DuckDB external sorting.

    Produces shards with only:
      - score (composite average across sort_keys)
      - text
    """
    print(f"\n=== Selecting top {top_percent:.2f}% by composite score ===")

    shard_files = sorted(
        [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )
    if not shard_files:
        print("ERROR: No parquet shards found.")
        return False

    # Verify required columns exist before running a long query.
    schema = pq.read_schema(shard_files[0])
    missing = [k for k in (["text"] + sort_keys) if k not in schema.names]
    if missing:
        print(f"ERROR: Missing required columns in shard schema: {missing}")
        print(f"Available columns: {schema.names}")
        return False

    try:
        import duckdb
    except ImportError:
        print("ERROR: 'duckdb' is required for --top-percent mode.")
        print("Install with: pip install duckdb")
        return False

    temp_dir = os.path.join(data_dir, "_duckdb_tmp")
    out_tmp_dir = os.path.join(data_dir, "_top_shards_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    if os.path.isdir(out_tmp_dir):
        shutil.rmtree(out_tmp_dir)
    os.makedirs(out_tmp_dir, exist_ok=True)

    quoted_files = ", ".join(f"'{p.replace(chr(39), chr(39) + chr(39))}'" for p in shard_files)
    key_exprs = [f"coalesce(CAST({k} AS DOUBLE), 0.0)" for k in sort_keys]
    composite_expr = f"({' + '.join(key_exprs)}) / {len(sort_keys)}"

    conn = duckdb.connect()
    conn.execute(f"PRAGMA temp_directory='{temp_dir.replace(chr(39), chr(39) + chr(39))}'")
    conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
    conn.execute("PRAGMA threads=4")

    base_query = (
        f"SELECT text, {composite_expr} AS score "
        f"FROM parquet_scan([{quoted_files}], union_by_name=true)"
    )

    total_rows = conn.execute(f"SELECT COUNT(*) FROM ({base_query}) AS t").fetchone()[0]
    if total_rows == 0:
        print("ERROR: No rows found after scan.")
        return False

    k = max(1, math.ceil(total_rows * (top_percent / 100.0)))
    print(f"Total rows: {total_rows:,}")
    print(f"Rows kept: {k:,}")

    stream_query = (
        f"SELECT score, text FROM ({base_query}) AS t "
        f"ORDER BY score DESC "
        f"LIMIT {k}"
    )

    reader = conn.execute(stream_query).fetch_record_batch(rows_per_batch=rows_per_shard)

    shard_idx = 0
    rows_written = 0
    for batch in reader:
        table = pa.Table.from_batches([batch])
        out_path = os.path.join(out_tmp_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, out_path, compression="zstd")
        rows_written += table.num_rows
        shard_idx += 1
        if shard_idx % 250 == 0:
            print(f"  Wrote {shard_idx} shards ({rows_written:,} rows)")

    conn.close()

    # Replace old shard files atomically-ish: only after successful write.
    for f in shard_files:
        os.remove(f)

    new_shards = sorted(
        [
            f
            for f in os.listdir(out_tmp_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )
    for f in new_shards:
        shutil.move(os.path.join(out_tmp_dir, f), os.path.join(data_dir, f))
    shutil.rmtree(out_tmp_dir, ignore_errors=True)

    print(f"Done! Wrote {len(new_shards)} shards with columns: ['score', 'text']")
    print(f"Total rows written: {rows_written:,}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download Azure education data for nanochat training"
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=8,
        help="Parallel download workers (default: 8)"
    )
    parser.add_argument(
        "-n", "--num-files", type=int, default=-1,
        help="Max files to download, -1 = all (default: -1)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=DATA_DIR,
        help=f"Output directory (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--container", type=str, default=CONTAINER,
        help=f"Azure Blob container name (default: {CONTAINER})"
    )
    parser.add_argument(
        "--blob-prefix", type=str, default="",
        help="Optional blob path prefix to limit download to a prebuilt subset."
    )
    parser.add_argument(
        "--download-mode",
        choices=["python", "bulk"],
        default="python",
        help="Download mode: 'python' (Azure SDK per-blob) or 'bulk' (azcopy + local transform).",
    )
    parser.add_argument(
        "--sort-keys", type=str, nargs="+",
        default=[
            "pedagogical_structure_average",
            "factual_accuracy_average",
            "lesson_engagement_average",
        ],
        help="Quality columns to combine into composite score (default: pedagogical + factual + engagement)"
    )
    parser.add_argument(
        "--skip-sort", action="store_true",
        help="Skip the sorting step (download only)"
    )
    parser.add_argument(
        "--top-percent", type=float, default=100.0,
        help="Keep only the top X%% rows by composite score (memory-safe external sort). "
             "Default: 100 (keep all rows)"
    )
    parser.add_argument(
        "--rows-per-shard", type=int, default=14000,
        help="Rows per output shard after sorting (default: 14000)"
    )
    parser.add_argument(
        "--duckdb-memory-limit", type=str, default="4GB",
        help="DuckDB memory limit for --top-percent mode (default: 4GB)"
    )
    args = parser.parse_args()

    if args.top_percent <= 0 or args.top_percent > 100:
        print("ERROR: --top-percent must be in the range (0, 100].")
        sys.exit(1)

    os.makedirs(args.data_dir, exist_ok=True)

    # Connect to Azure
    print(f"Connecting to Azure: {ACCOUNT}/{args.container}")
    container_client = get_container_client(args.container)

    # List all blobs
    print("Listing blobs (this may take a moment)...")
    prefix = args.blob_prefix if args.blob_prefix else None
    blobs = list_blobs(container_client, prefix=prefix)
    total = len(blobs)
    if prefix:
        print(f"Found {total} parquet files in container with prefix '{prefix}'")
    else:
        print(f"Found {total} parquet files in container")

    if total == 0:
        print("ERROR: No parquet files found. Check credentials and container name.")
        sys.exit(1)

    if args.num_files > 0:
        blobs = blobs[: args.num_files]
        total = len(blobs)
        print(f"Limiting to first {total} files")

    # Count already-downloaded shards
    existing = 0
    existing_bytes = 0
    for i in range(total):
        path = os.path.join(args.data_dir, f"shard_{i:05d}.parquet")
        if os.path.exists(path):
            existing += 1
            existing_bytes += os.path.getsize(path)

    print(f"Already downloaded: {existing}/{total} ({existing_bytes / 1e9:.1f} GB)")

    if existing < total:
        # Download with thread pool
        done = existing
        errors = 0
        total_bytes = existing_bytes
        t0 = time.time()

        used_bulk = False
        if args.download_mode == "bulk":
            bulk_done, bulk_errors, bulk_bytes = bulk_download_and_transform(
                container_name=args.container,
                blobs=blobs,
                blob_prefix=(args.blob_prefix or ""),
                data_dir=args.data_dir,
                workers=args.workers,
            )
            if bulk_done >= 0:
                done = bulk_done
                errors = bulk_errors
                total_bytes = bulk_bytes
                used_bulk = True

        if not used_bulk:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                for i, blob_name in enumerate(blobs):
                    out_path = os.path.join(args.data_dir, f"shard_{i:05d}.parquet")
                    if os.path.exists(out_path):
                        continue
                    future = executor.submit(
                        download_shard, container_client, blob_name, i, args.data_dir
                    )
                    futures[future] = (i, blob_name)

                for future in as_completed(futures):
                    idx, status, nbytes = future.result()

                    if status == "done":
                        done += 1
                        total_bytes += nbytes
                    elif status.startswith("error") or status.startswith("warning"):
                        errors += 1
                        print(f"  [{idx:05d}] {status}")
                    else:
                        done += 1

                    elapsed = time.time() - t0
                    new_done = done - existing
                    rate = new_done / elapsed if elapsed > 0 else 0
                    remaining = (total - done) / rate if rate > 0 else float("inf")

                    if new_done % 25 == 0 and new_done > 0:
                        print(
                            f"  Progress: {done}/{total} shards | "
                            f"{total_bytes / 1e9:.1f} GB | "
                            f"{rate:.1f} shards/s | "
                            f"ETA: {remaining / 60:.0f} min"
                        )

        elapsed = time.time() - t0
        print(f"\nDownload complete: {done}/{total} shards in {elapsed / 60:.1f} min")
        print(f"Total data: {total_bytes / 1e9:.1f} GB (with metadata, before sort)")
        print(f"Errors: {errors}")
    else:
        print("All shards already downloaded.")

    # Sort by quality
    if not args.skip_sort:
        if args.top_percent < 100.0:
            sort_top_percent_external(
                args.data_dir,
                args.sort_keys,
                top_percent=args.top_percent,
                rows_per_shard=args.rows_per_shard,
                memory_limit=args.duckdb_memory_limit,
            )
        else:
            sort_and_reshard(args.data_dir, args.sort_keys, rows_per_shard=args.rows_per_shard)
    else:
        print("\nSkipping sort step (--skip-sort). Shards retain metadata.")

    # Final verification
    final_shards = sorted(
        [f for f in os.listdir(args.data_dir)
         if f.endswith(".parquet") and not f.endswith(".tmp")]
    )
    print(f"\nFinal shard count: {len(final_shards)}")
    if final_shards:
        schema = pq.read_schema(os.path.join(args.data_dir, final_shards[0]))
        print(f"First shard columns: {schema.names}")
        total_size = sum(
            os.path.getsize(os.path.join(args.data_dir, f)) for f in final_shards
        )
        print(f"Total size on disk: {total_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()

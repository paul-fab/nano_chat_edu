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

import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
import pyarrow as pa
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import load_dotenv

load_dotenv()

ACCOUNT = os.environ.get("AZURE_STORAGE_ACCOUNT", "quratingscoressa")
KEY = os.environ.get("AZURE_STORAGE_KEY")
CONTAINER = "quratingfiltered"
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


def get_container_client() -> ContainerClient:
    conn_str = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={ACCOUNT};"
        f"AccountKey={KEY};"
        f"EndpointSuffix=core.windows.net"
    )
    return BlobServiceClient.from_connection_string(conn_str).get_container_client(CONTAINER)


def list_blobs(container_client: ContainerClient) -> list[str]:
    """List all parquet blob names in the container, sorted."""
    blobs = []
    for blob in container_client.list_blobs():
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
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Connect to Azure
    print(f"Connecting to Azure: {ACCOUNT}/{CONTAINER}")
    container_client = get_container_client()

    # List all blobs
    print("Listing blobs (this may take a moment)...")
    blobs = list_blobs(container_client)
    total = len(blobs)
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
        sort_and_reshard(args.data_dir, args.sort_keys)
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

"""
patch_nanochat.py -- Apply minimal modifications to nanochat for Azure education data.

Changes:
  1. dataset.py: MAX_SHARD 1822 → actual shard count (auto-detected from data dir)
  2. dataset.py: Disables automatic HuggingFace download in the CLI entry point

This is idempotent -- safe to run multiple times.

Usage:
    python patch_nanochat.py                          # auto-detect nanochat location
    python patch_nanochat.py --nanochat-dir ~/nanochat # explicit path
"""

import os
import re
import sys
import argparse
from pathlib import Path


def find_nanochat_dir() -> Path:
    """Find nanochat installation directory."""
    candidates = [
        Path.home() / "nanochat",
        Path.cwd(),
        Path.cwd().parent / "nanochat",
    ]
    for candidate in candidates:
        dataset_path = candidate / "nanochat" / "dataset.py"
        if dataset_path.exists():
            return candidate
    return None


def count_shards(data_dir: str | None = None) -> int:
    """Count existing parquet shards in the data directory."""
    if data_dir is None:
        data_dir = os.path.expanduser("~/.cache/nanochat/base_data")
    if not os.path.isdir(data_dir):
        return -1
    shards = [f for f in os.listdir(data_dir) if f.endswith(".parquet") and not f.endswith(".tmp")]
    return len(shards)


def patch_dataset(nanochat_dir: Path, shard_count: int) -> bool:
    """Patch nanochat/dataset.py with our shard count and disable HF downloads."""
    dataset_path = nanochat_dir / "nanochat" / "dataset.py"

    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found")
        return False

    content = dataset_path.read_text(encoding="utf-8")
    original = content

    # 1. Update MAX_SHARD to our shard count - 1
    max_shard = shard_count - 1
    content, n1 = re.subn(
        r"^(MAX_SHARD\s*=\s*)\d+",
        rf"\g<1>{max_shard}",
        content,
        count=1,
        flags=re.MULTILINE,
    )

    # 2. Replace the HuggingFace BASE_URL with a comment indicating local data
    content, n2 = re.subn(
        r'^(BASE_URL\s*=\s*)("https://huggingface\.co[^"]*")',
        r'\1"local"  # Data pre-downloaded via download_azure_data.py (was: \2)',
        content,
        count=1,
        flags=re.MULTILINE,
    )

    # 3. Patch download_single_file to be a no-op (line-by-line approach for robustness)
    n3 = 0
    if "def download_single_file" in content and "# PATCHED: downloads disabled" not in content:
        lines = content.split("\n")
        new_lines = []
        in_func = False
        func_indent = 0
        patched_body = False

        for line in lines:
            if not in_func:
                new_lines.append(line)
                if re.match(r"^def download_single_file\(", line):
                    in_func = True
                    func_indent = 4  # standard indent for function body
                    # Insert replacement body
                    new_lines.append('    # PATCHED: downloads disabled -- data is pre-downloaded via download_azure_data.py')
                    new_lines.append('    print(f"WARNING: download_single_file called for index {index}, but downloads are disabled.")')
                    new_lines.append('    return')
                    new_lines.append('')
                    patched_body = True
            else:
                # Skip original function body lines (indented lines or blank lines within)
                stripped = line.lstrip()
                current_indent = len(line) - len(stripped)
                if stripped == "" or current_indent >= func_indent:
                    continue  # skip original body
                else:
                    # Hit next top-level definition -- function is over
                    in_func = False
                    new_lines.append(line)

        if patched_body:
            content = "\n".join(new_lines)
            n3 = 1

    if content == original:
        print("No changes needed -- already patched or unexpected format.")
        return True

    # Write patched file
    dataset_path.write_text(content, encoding="utf-8")

    print(f"Patched {dataset_path}:")
    if n1: print(f"  - MAX_SHARD = {max_shard} (was 1822)")
    if n2: print(f"  - BASE_URL set to 'local'")
    if n3: print(f"  - download_single_file disabled")

    return True


def patch_common_tf32(nanochat_dir: Path) -> bool:
    """Patch nanochat/common.py to avoid mixed TF32 API usage."""
    common_path = nanochat_dir / "nanochat" / "common.py"

    if not common_path.exists():
        print(f"ERROR: {common_path} not found")
        return False

    content = common_path.read_text(encoding="utf-8")
    original = content

    old_line = '        torch.backends.fp32_precision = "tf32" # uses tf32 instead of fp32 for matmuls'
    replacement = (
        "        # Use legacy TF32 toggles to stay compatible with torch.compile/inductor checks\n"
        "        torch.backends.cuda.matmul.allow_tf32 = True\n"
        "        torch.backends.cudnn.allow_tf32 = True"
    )

    if old_line in content:
        content = content.replace(old_line, replacement, 1)

    if content == original:
        print("No common.py TF32 changes needed -- already patched or unexpected format.")
        return True

    common_path.write_text(content, encoding="utf-8")
    print(f"Patched {common_path}:")
    print("  - Replaced torch.backends.fp32_precision with legacy TF32 flags")
    return True


def patch_dataloader_ddp_sharding(nanochat_dir: Path) -> bool:
    """Patch dataloader sharding so DDP works with 1-row-group parquet files."""
    dataloader_path = nanochat_dir / "nanochat" / "dataloader.py"

    if not dataloader_path.exists():
        print(f"ERROR: {dataloader_path} not found")
        return False

    content = dataloader_path.read_text(encoding="utf-8")
    original = content

    old = """            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1"""
    new = """            else:
                # If row groups are fewer than world size (common with 1-row-group shards),
                # shard by file index so every rank still receives data.
                if pf.num_row_groups < ddp_world_size:
                    if (pq_idx % ddp_world_size) != ddp_rank:
                        pq_idx += 1
                        continue
                    rg_idx = 0
                    rg_step = 1
                else:
                    rg_idx = ddp_rank
                    rg_step = ddp_world_size
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += rg_step
            pq_idx += 1"""

    if old in content:
        content = content.replace(old, new, 1)

    if content == original:
        print("No dataloader sharding changes needed -- already patched or unexpected format.")
        return True

    dataloader_path.write_text(content, encoding="utf-8")
    print(f"Patched {dataloader_path}:")
    print("  - Added file-level DDP sharding fallback for low row-group parquet files")
    return True


def verify_patch(nanochat_dir: Path) -> bool:
    """Verify the patch was applied correctly."""
    dataset_path = nanochat_dir / "nanochat" / "dataset.py"
    content = dataset_path.read_text(encoding="utf-8")

    # Check MAX_SHARD was updated
    match = re.search(r"^MAX_SHARD\s*=\s*(\d+)", content, re.MULTILINE)
    if not match:
        print("VERIFY FAIL: MAX_SHARD not found")
        return False

    max_shard = int(match.group(1))
    if max_shard == 1822:
        print("VERIFY FAIL: MAX_SHARD still at default 1822")
        return False

    print(f"  Verified: MAX_SHARD = {max_shard}")

    # Check list_parquet_files still works
    sys.path.insert(0, str(nanochat_dir))
    try:
        from nanochat.dataset import list_parquet_files
        files = list_parquet_files()
        print(f"  Verified: list_parquet_files() returns {len(files)} files")
        return True
    except Exception as e:
        print(f"  WARNING: Could not import nanochat.dataset: {e}")
        print("  (This is OK if nanochat deps aren't installed in this Python env)")
        return True


def main():
    parser = argparse.ArgumentParser(description="Patch nanochat for Azure education data")
    parser.add_argument(
        "--nanochat-dir", type=str, default=None,
        help="Path to nanochat repo (auto-detected if not specified)"
    )
    parser.add_argument(
        "--shard-count", type=int, default=None,
        help="Override shard count (auto-detected from data dir if not specified)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing shard_*.parquet files (default: ~/.cache/nanochat/base_data)"
    )
    args = parser.parse_args()

    # Find nanochat
    if args.nanochat_dir:
        nanochat_dir = Path(args.nanochat_dir)
    else:
        nanochat_dir = find_nanochat_dir()

    if nanochat_dir is None:
        print("ERROR: Could not find nanochat directory.")
        print("Clone it first: git clone https://github.com/karpathy/nanochat.git ~/nanochat")
        sys.exit(1)

    print(f"nanochat directory: {nanochat_dir}")

    # Count shards
    if args.shard_count:
        shard_count = args.shard_count
    else:
        shard_count = count_shards(args.data_dir)
        if shard_count <= 0:
            default_dir = args.data_dir or "~/.cache/nanochat/base_data"
            print(f"WARNING: No shards found in {default_dir}")
            print("Using --shard-count 3250 (known dataset size).")
            print("Alternatively, run download_azure_data.py first.")
            shard_count = 3250

    print(f"Shard count: {shard_count}")

    # Apply patch
    if not patch_dataset(nanochat_dir, shard_count):
        sys.exit(1)
    if not patch_common_tf32(nanochat_dir):
        sys.exit(1)
    if not patch_dataloader_ddp_sharding(nanochat_dir):
        sys.exit(1)

    # Verify
    print("\nVerifying patch...")
    verify_patch(nanochat_dir)

    print("\nPatch complete.")


if __name__ == "__main__":
    main()

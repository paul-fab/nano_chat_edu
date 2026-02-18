#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import urllib.request
from urllib.error import HTTPError
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument(
        "--remote-base",
        default="https://fabdvcbenchmarksa.blob.core.windows.net/llm-benchmarks/dvc/files/md5",
    )
    args = p.parse_args()

    root = Path(args.repo_root)
    count = 0
    for dvc in root.glob("data/**/*.dvc"):
        txt = dvc.read_text(encoding="utf-8")
        md5m = re.search(r"md5:\s*([0-9a-f]{32})", txt)
        pathm = re.search(r"path:\s*(.+)", txt)
        if not md5m or not pathm:
            continue
        md5 = md5m.group(1)
        rel = pathm.group(1).strip()
        out = dvc.parent / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"{args.remote_base}/{md5[:2]}/{md5[2:]}"
        try:
            urllib.request.urlretrieve(url, out)
            count += 1
            print(f"downloaded {out}")
        except HTTPError as exc:
            print(f"skipped {dvc} ({exc.code})")
    print(f"downloaded_files={count}")


if __name__ == "__main__":
    main()

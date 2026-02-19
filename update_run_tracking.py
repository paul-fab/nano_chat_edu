#!/usr/bin/env python3
"""Refresh RUN_TRACKING.md from live run logs on remote servers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RunSpec:
    label: str
    wandb_name: str
    run_ids: List[str]
    host: str
    run_dirs: List[str]
    data_variant: str


RUN_SPECS: List[RunSpec] = [
    RunSpec(
        label="top50",
        wandb_name="edu-d26-top50",
        run_ids=["zoebqeam"],
        host="192.222.50.180",
        run_dirs=["run-20260217_204650-zoebqeam"],
        data_variant="Full filtered + extra top-50% quality layer",
    ),
    RunSpec(
        label="top20",
        wandb_name="edu-d26-top20",
        run_ids=["pb0ezqrx"],
        host="192.222.58.132",
        run_dirs=["run-20260217_160901-pb0ezqrx"],
        data_variant="Full filtered + extra top-20% quality layer",
    ),
    RunSpec(
        label="full-filtered (split)",
        wandb_name="exp-top16-d26-8gpu",
        run_ids=["dbwtodle", "n9hithr0", "bk8zeag4"],
        host="192.222.53.38",
        run_dirs=[
            "run-20260218_132009-dbwtodle",
            "run-20260218_142627-n9hithr0",
            "run-20260219_053929-bk8zeag4",
        ],
        data_variant="Full filtered (no extra top20/top50 layer)",
    ),
    RunSpec(
        label="random-8gpu (split)",
        wandb_name="exp-rand16-d26-8gpu",
        run_ids=["aw30lam9", "4ul335c6"],
        host="192.222.53.38",
        run_dirs=["run-20260218_200005-aw30lam9", "run-20260218_210528-4ul335c6"],
        data_variant="Random/staging subset",
    ),
]


RE_STEP = re.compile(r"^step\s+(\d+)/(\d+)", re.MULTILINE)
RE_VAL = re.compile(r"Step\s+(\d+)\s+\|\s+Validation bpb:\s+([0-9.]+)")
RE_CORE = re.compile(r"Step\s+(\d+)\s+\|\s+CORE metric:\s+([0-9.]+)")
RE_CDPK = re.compile(r"Step\s+(\d+)\s+\|\s+CDPK accuracy:\s+([0-9.]+)\s+\((\d+)/(\d+)\)")
RE_ITER_AUTO = re.compile(r"Calculated number of iterations from target data:param ratio:\s*([0-9,]+)")
RE_ITER_FIXED = re.compile(r"Using user-provided number of iterations:\s*([0-9,]+)")
RE_TOKENS = re.compile(r"Total number of training tokens:\s*([0-9,]+)")
RE_BATCH_AUTO = re.compile(r"Auto-computed optimal batch size:\s*([0-9,]+)\s+tokens")
RE_BATCH_TOTAL = re.compile(r"Total batch size\s*([0-9,]+)\s*=>")


def run_ssh(host: str, remote_cmd: str) -> str:
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"ubuntu@{host}",
        remote_cmd,
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"SSH command failed on {host}: {remote_cmd}\n"
            f"exit={proc.returncode}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def cat_remote_file(host: str, path: str) -> str:
    return run_ssh(host, f"cat {path}")


def parse_int(s: str) -> int:
    return int(s.replace(",", ""))


def parse_log_metrics(log_text: str) -> Dict[str, object]:
    step_matches = RE_STEP.findall(log_text)
    latest_step = max((int(s) for s, _ in step_matches), default=0)

    vals = [(int(s), float(v)) for s, v in RE_VAL.findall(log_text)]
    cores = [(int(s), float(v)) for s, v in RE_CORE.findall(log_text)]
    cdpks = [(int(s), float(v), int(n), int(d)) for s, v, n, d in RE_CDPK.findall(log_text)]

    milestones: Dict[int, Dict[str, object]] = {}
    for s, v in vals:
        milestones.setdefault(s, {})["val"] = v
    for s, v in cores:
        milestones.setdefault(s, {})["core"] = v
    for s, v, n, d in cdpks:
        milestones.setdefault(s, {})["cdpk"] = {"acc": v, "n": n, "d": d}

    best_val = min(vals, key=lambda x: x[1]) if vals else None
    best_core = max(cores, key=lambda x: x[1]) if cores else None
    best_cdpk = max(cdpks, key=lambda x: x[1]) if cdpks else None

    iter_match = RE_ITER_AUTO.search(log_text) or RE_ITER_FIXED.search(log_text)
    token_match = RE_TOKENS.search(log_text)
    batch_match = RE_BATCH_AUTO.search(log_text) or RE_BATCH_TOTAL.search(log_text)

    return {
        "latest_step": latest_step,
        "vals": vals,
        "cores": cores,
        "cdpks": cdpks,
        "milestones": milestones,
        "best_val": best_val,
        "best_core": best_core,
        "best_cdpk": best_cdpk,
        "planned_iterations": parse_int(iter_match.group(1)) if iter_match else None,
        "planned_tokens": parse_int(token_match.group(1)) if token_match else None,
        "batch_size_tokens": parse_int(batch_match.group(1)) if batch_match else None,
    }


def fmt_float(x: Optional[float], ndigits: int = 6) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{ndigits}f}"


def fmt_int(x: Optional[int]) -> str:
    if x is None:
        return "n/a"
    return f"{x:,}"


def utc_date() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")


def build_markdown(rows: List[Dict[str, object]]) -> str:
    top50 = next(r for r in rows if r["label"] == "top50")
    top20 = next(r for r in rows if r["label"] == "top20")
    split = next(r for r in rows if r["label"] == "full-filtered (split)")
    random8 = next(r for r in rows if r["label"] == "random-8gpu (split)")

    def wb_urls(run_ids: List[str]) -> str:
        return " + ".join(f"https://wandb.ai/fabdata/nanochat/runs/{rid}" for rid in run_ids)

    def ids_fmt(run_ids: List[str]) -> str:
        return " + ".join(f"`{rid}`" for rid in run_ids)

    def started_fmt(phases: List[Dict[str, object]]) -> str:
        return " + ".join(m["startedAt"].replace("T", " ").replace("Z", "") for m in phases)

    def row_meta_latest(r: Dict[str, object]) -> Dict[str, object]:
        return r["metadata"] if "metadata" in r else r["metadata_phases"][-1]

    def is_split(r: Dict[str, object]) -> bool:
        return "metadata_phases" in r

    def headline_best_val(r: Dict[str, object]) -> str:
        best = r["metrics"]["best_val"]
        if not best:
            return "n/a"
        step, val = best
        return f"{val:.6f} (step {step})"

    def headline_best_core(r: Dict[str, object]) -> str:
        best = r["metrics"]["best_core"]
        if not best:
            return "n/a"
        step, val = best
        return f"{val:.4f} (step {step})"

    def headline_best_cdpk(r: Dict[str, object]) -> str:
        best = r["metrics"]["best_cdpk"]
        if not best:
            return "n/a"
        step, acc, n, d = best
        return f"{acc:.4f} ({n}/{d}, step {step})"

    def milestone_rows_for_single(r: Dict[str, object]) -> List[str]:
        lines: List[str] = []
        milestones: Dict[int, Dict[str, object]] = r["metrics"]["milestones"]
        steps = sorted(s for s in milestones if s > 0)
        for s in steps:
            m = milestones[s]
            val = fmt_float(m.get("val"), 6) if "val" in m else "n/a"
            core = fmt_float(m.get("core"), 4) if "core" in m else "n/a"
            cdpk_obj = m.get("cdpk")
            cdpk = fmt_float(cdpk_obj["acc"], 4) if cdpk_obj else "n/a"
            lines.append(f"| {r['label']} | {s} | {val} | {core} | {cdpk} | |")
        return lines

    def milestone_rows_for_split(r: Dict[str, object]) -> List[str]:
        lines: List[str] = []
        milestones: Dict[int, Dict[str, object]] = r["metrics"]["milestones"]
        phase_last_steps: List[int] = r["phase_last_steps"]
        phase_names: List[str] = [f"phase {chr(ord('A') + i)}" for i in range(len(phase_last_steps))]
        steps = sorted(s for s in milestones if s > 0)
        for s in steps:
            m = milestones[s]
            val = fmt_float(m.get("val"), 6) if "val" in m else "n/a"
            core = fmt_float(m.get("core"), 4) if "core" in m else "n/a"
            cdpk_obj = m.get("cdpk")
            cdpk = fmt_float(cdpk_obj["acc"], 4) if cdpk_obj else "n/a"
            comment = phase_names[-1]
            for i, cutoff in enumerate(phase_last_steps):
                if s <= cutoff:
                    comment = phase_names[i]
                    break
            lines.append(f"| {r['label']} | {s} | {val} | {core} | {cdpk} | {comment} |")
        return lines

    top50_meta = row_meta_latest(top50)
    top20_meta = row_meta_latest(top20)
    split_phases = split["metadata_phases"] if is_split(split) else [split["metadata"]]
    split_meta_latest = split_phases[-1]
    random8_phases = random8["metadata_phases"] if is_split(random8) else [random8["metadata"]]
    random8_meta_latest = random8_phases[-1]

    md = []
    md.append("# Run Tracking Sheet")
    md.append("")
    md.append(f"Last updated: {utc_date()}")
    md.append("")
    md.append("## Scope")
    md.append("")
    md.append("This sheet tracks the main experiment lines:")
    md.append("")
    md.append("1. `edu-d26-top50` (extra quality filter on full filtered dataset)")
    md.append("2. `edu-d26-top20` (extra quality filter on full filtered dataset)")
    md.append("3. `exp-top16-d26-8gpu` (full filtered dataset run, split across multiple W&B runs)")
    md.append("4. `exp-rand16-d26-8gpu` (random/staging data arm on 8x H100)")
    md.append("")
    md.append("## Run Registry")
    md.append("")
    md.append("| Run label | W&B run name | W&B run ID(s) | Host | GPU setup | Data variant | Start (UTC) | W&B URL |")
    md.append("|---|---|---|---|---|---|---|---|")
    md.append(
        f"| top50 | `{top50['wandb_name']}` | `{top50['run_ids'][0]}` | `{top50['host']}` | "
        f"{top50_meta['gpu_count']}x {top50_meta['gpu']} | {top50['data_variant']} | "
        f"{top50_meta['startedAt'].replace('T', ' ').replace('Z', '')} | {wb_urls(top50['run_ids'])} |"
    )
    md.append(
        f"| top20 | `{top20['wandb_name']}` | `{top20['run_ids'][0]}` | `{top20['host']}` | "
        f"{top20_meta['gpu_count']}x {top20_meta['gpu']} | {top20['data_variant']} | "
        f"{top20_meta['startedAt'].replace('T', ' ').replace('Z', '')} | {wb_urls(top20['run_ids'])} |"
    )
    md.append(
        f"| full-filtered (split) | `{split['wandb_name']}` | {ids_fmt(split['run_ids'])} | "
        f"`{split['host']}` | {split_meta_latest['gpu_count']}x {split_meta_latest['gpu']} | {split['data_variant']} | "
        f"{started_fmt(split_phases)} | {wb_urls(split['run_ids'])} |"
    )
    md.append(
        f"| random-8gpu (split) | `{random8['wandb_name']}` | {ids_fmt(random8['run_ids'])} | `{random8['host']}` | "
        f"{random8_meta_latest['gpu_count']}x {random8_meta_latest['gpu']} | {random8['data_variant']} | "
        f"{started_fmt(random8_phases)} | {wb_urls(random8['run_ids'])} |"
    )
    md.append("")
    md.append("## Comparability Notes")
    md.append("")
    md.append("- Same model depth across these runs: `d26`.")
    md.append(
        "- `top50` and `top20` use single-GPU GH200, auto batch size "
        f"`{fmt_int(top50['metrics']['batch_size_tokens'])}` tokens, and computed horizon "
        f"`{fmt_int(top50['metrics']['planned_iterations'])}` iterations "
        f"(`{fmt_int(top50['metrics']['planned_tokens'])}` tokens)."
    )
    md.append(
        "- `exp-top16-d26-8gpu` uses fixed `--total-batch-size "
        f"{fmt_int(split['metrics']['batch_size_tokens'])}` and `--num-iterations "
        f"{fmt_int(split['metrics']['planned_iterations'])}` "
        f"(`{fmt_int(split['metrics']['planned_tokens'])}` tokens planned), so it is a much longer-horizon run."
    )
    md.append("- The 8x run is split:")
    md.append("  - Phase A: `dbwtodle` (no CDPK integrated)")
    md.append("  - Phase B: `n9hithr0` (`--resume-from-step 2000`, CDPK integrated every 1000 steps)")
    md.append("  - Phase C: `bk8zeag4` (`--resume-from-step 12000`, CDPK integrated every 1000 steps)")
    md.append(
        "- `exp-rand16-d26-8gpu` had an early failed startup run (`45vxrhi8`, CUDA OOM), "
        "then successful split phases (`aw30lam9` and `4ul335c6`)."
    )
    md.append("")
    md.append("## Current Headline Snapshot")
    md.append("")
    md.append("| Run label | Latest training step seen | Best validation bpb seen | Best CORE seen | Best CDPK seen | Notes |")
    md.append("|---|---:|---:|---:|---:|---|")
    md.append(
        f"| top50 (`{top50['run_ids'][0]}`) | {top50['metrics']['latest_step']} | "
        f"{headline_best_val(top50)} | {headline_best_core(top50)} | {headline_best_cdpk(top50)} | "
        "CDPK/CORE every 1000 steps |"
    )
    md.append(
        f"| top20 (`{top20['run_ids'][0]}`) | {top20['metrics']['latest_step']} | "
        f"{headline_best_val(top20)} | {headline_best_core(top20)} | {headline_best_cdpk(top20)} | "
        "CDPK/CORE every 1000 steps |"
    )
    md.append(
        f"| full-filtered split ({ids_fmt(split['run_ids'])}) | "
        f"{split['metrics']['latest_step']} (latest phase) | {headline_best_val(split)} | "
        f"{headline_best_core(split)} | {headline_best_cdpk(split)} | "
        f"Phase A reached step {split['phase_last_steps'][0]}; phase B resumed at 2000; phase C resumed at 12000 |"
    )
    md.append(
        f"| random-8gpu split ({ids_fmt(random8['run_ids'])}) | {random8['metrics']['latest_step']} | "
        f"{headline_best_val(random8)} | {headline_best_core(random8)} | {headline_best_cdpk(random8)} | "
        "Phase A reached step 1740, then resumed from checkpoint 1000 in phase B |"
    )
    md.append("")
    md.append("## Milestone Table (fill as runs progress)")
    md.append("")
    md.append("Use this for side-by-side comparisons at common checkpoints.")
    md.append("")
    md.append("| Run label | Step | Validation bpb | CORE | CDPK accuracy | Comment |")
    md.append("|---|---:|---:|---:|---:|---|")
    md.extend(milestone_rows_for_single(top50))
    md.extend(milestone_rows_for_single(top20))
    md.extend(milestone_rows_for_split(split))
    md.extend(milestone_rows_for_split(random8))
    md.append("")
    md.append("## Update Procedure")
    md.append("")
    md.append("1. Run `python update_run_tracking.py` from this repository root.")
    md.append("2. Review the diff in `RUN_TRACKING.md`.")
    md.append("3. Commit when the refresh looks correct.")
    md.append("")
    return "\n".join(md) + "\n"


def collect_run(spec: RunSpec) -> Dict[str, object]:
    metadata_phases = []
    logs = []
    metrics_phases = []
    for run_dir in spec.run_dirs:
        metadata = json.loads(cat_remote_file(spec.host, f"~/nanochat/wandb/{run_dir}/files/wandb-metadata.json"))
        log_text = cat_remote_file(spec.host, f"~/nanochat/wandb/{run_dir}/files/output.log")
        metadata_phases.append(metadata)
        logs.append(log_text)
        metrics_phases.append(parse_log_metrics(log_text))

    if len(spec.run_dirs) == 1:
        return {
            "label": spec.label,
            "wandb_name": spec.wandb_name,
            "run_ids": spec.run_ids,
            "host": spec.host,
            "data_variant": spec.data_variant,
            "metadata": metadata_phases[0],
            "metrics": metrics_phases[0],
        }

    combined_metrics = parse_log_metrics("\n".join(logs))
    return {
        "label": spec.label,
        "wandb_name": spec.wandb_name,
        "run_ids": spec.run_ids,
        "host": spec.host,
        "data_variant": spec.data_variant,
        "metadata_phases": metadata_phases,
        "phase_last_steps": [m["latest_step"] for m in metrics_phases],
        "metrics": combined_metrics,
        "metrics_phases": metrics_phases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Update RUN_TRACKING.md from remote logs.")
    parser.add_argument(
        "--output",
        default="RUN_TRACKING.md",
        help="Output markdown path (default: RUN_TRACKING.md)",
    )
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for spec in RUN_SPECS:
        rows.append(collect_run(spec))

    md = build_markdown(rows)
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

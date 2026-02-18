#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from nanochat.checkpoint_manager import load_model

from logprob_mcq_eval import (
    DEFAULT_PROMPT_HEADER,
    MCQExample,
    build_prompt,
    choose_fewshot_examples,
    load_mcq_file,
)
from logprob_mcq_eval_nanochat import average_logprob_completion_nanochat


def parse_checkpoint(raw: str) -> Tuple[str | None, int | None, str]:
    text = raw.strip()
    if not text:
        raise ValueError("Empty checkpoint value")

    if ":" not in text:
        return text, None, f"{text}:last"

    tag, step_raw = text.split(":", 1)
    tag = tag.strip()
    step_raw = step_raw.strip().lower()
    if not tag:
        raise ValueError(f"Invalid checkpoint value: {raw}")

    if step_raw in ("", "last", "latest"):
        return tag, None, f"{tag}:last"

    try:
        step = int(step_raw)
    except ValueError as exc:
        raise ValueError(f"Invalid step in checkpoint value: {raw}") from exc

    return tag, step, f"{tag}:{step}"


def checkpoint_slug(tag: str | None, step: int | None) -> str:
    safe_tag = (tag or "auto").replace("/", "_")
    safe_step = str(step) if step is not None else "last"
    return f"{safe_tag}_step{safe_step}"


def evaluate_seed(
    *,
    model,
    tokenizer,
    device_t: torch.device,
    max_input_tokens: int | None,
    eval_examples: Sequence[MCQExample],
    fewshot_pool: Sequence[MCQExample],
    fewshot_k: int,
    header: str,
    max_examples: int | None,
    seed: int,
) -> Dict[str, object]:
    total = 0
    correct = 0
    skipped = 0
    rows = []

    examples = list(eval_examples)
    if max_examples is not None:
        examples = examples[:max_examples]

    for idx, ex in enumerate(examples):
        if ex.answer_index is None:
            skipped += 1
            continue

        fewshot = choose_fewshot_examples(
            fewshot_pool,
            k=fewshot_k,
            seed=seed + idx,
            target_qid=ex.qid,
        )
        prompt = build_prompt(ex, fewshot=fewshot, header=header)

        scores: List[float] = []
        for opt in ex.options:
            completion = f" {opt}"
            score = average_logprob_completion_nanochat(
                model,
                tokenizer,
                prompt,
                completion,
                device=device_t,
                max_input_tokens=max_input_tokens,
            )
            scores.append(score)

        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        is_correct = pred == ex.answer_index
        total += 1
        correct += int(is_correct)
        rows.append(
            {
                "id": ex.qid,
                "gold_index": ex.answer_index,
                "pred_index": pred,
                "gold_letter": chr(ord("A") + ex.answer_index),
                "pred_letter": chr(ord("A") + pred),
                "correct": is_correct,
                "scores": scores,
            }
        )

    accuracy = (correct / total) if total else 0.0
    return {
        "total_scored": total,
        "total_skipped": skipped,
        "accuracy": accuracy,
        "correct": correct,
        "rows": rows,
    }


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(summary_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[float]] = {}
    for row in summary_rows:
        label = str(row["checkpoint"])
        grouped.setdefault(label, []).append(float(row["accuracy"]))

    out = []
    for label in sorted(grouped.keys()):
        vals = grouped[label]
        n = len(vals)
        mean = sum(vals) / n
        std = statistics.stdev(vals) if n > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
        out.append(
            {
                "checkpoint": label,
                "runs": n,
                "mean_accuracy": f"{mean:.6f}",
                "std_accuracy": f"{std:.6f}",
                "ci95_half_width": f"{ci95:.6f}",
                "min_accuracy": f"{min(vals):.6f}",
                "max_accuracy": f"{max(vals):.6f}",
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Repeated-seed CDPK log-prob eval for nanochat checkpoints.")
    p.add_argument("--eval-file", required=True)
    p.add_argument("--dev-file", default=None)
    p.add_argument("--fewshot-k", type=int, default=3)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-end", type=int, default=29, help="Inclusive")
    p.add_argument("--prompt-header", default=DEFAULT_PROMPT_HEADER)
    p.add_argument("--source", default="base", choices=["base", "sft", "rl"])
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint in TAG:STEP format (e.g. d20:3000) or TAG:last. Repeatable.",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", default="eval/results/repeats")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start")

    eval_examples = load_mcq_file(Path(args.eval_file))
    fewshot_pool = load_mcq_file(Path(args.dev_file)) if args.dev_file else []
    if not fewshot_pool:
        fewshot_pool = eval_examples

    if args.device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_checkpoints = [parse_checkpoint(x) for x in args.checkpoint]
    seeds = list(range(args.seed_start, args.seed_end + 1))

    summary_rows: List[Dict[str, object]] = []

    for model_tag, step, label in parsed_checkpoints:
        slug = checkpoint_slug(model_tag, step)
        print(f"[load] {label}")
        model, tokenizer, _ = load_model(args.source, device_t, phase="eval", model_tag=model_tag, step=step)
        max_input_tokens = getattr(model.config, "sequence_len", None)

        for seed in seeds:
            out_json = output_dir / f"{slug}_seed{seed}.json"
            if out_json.exists() and not args.overwrite:
                print(f"[skip] {out_json} already exists")
                continue

            t0 = time.time()
            result = evaluate_seed(
                model=model,
                tokenizer=tokenizer,
                device_t=device_t,
                max_input_tokens=max_input_tokens,
                eval_examples=eval_examples,
                fewshot_pool=fewshot_pool,
                fewshot_k=args.fewshot_k,
                header=args.prompt_header,
                max_examples=args.max_examples,
                seed=seed,
            )
            elapsed = time.time() - t0

            payload = {
                "model": f"nanochat:{args.source}:{model_tag or 'auto'}:{step or 'last'}",
                "checkpoint": label,
                "seed": seed,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total_scored": result["total_scored"],
                "total_skipped": result["total_skipped"],
                "rows": result["rows"],
            }
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            print(
                f"[ok ] {label} seed={seed} acc={result['accuracy']:.4f} "
                f"({result['correct']}/{result['total_scored']}) elapsed={elapsed:.1f}s"
            )

            summary_rows.append(
                {
                    "checkpoint": label,
                    "model_tag": model_tag or "",
                    "step": step if step is not None else "last",
                    "seed": seed,
                    "accuracy": f"{result['accuracy']:.6f}",
                    "correct": result["correct"],
                    "total_scored": result["total_scored"],
                    "total_skipped": result["total_skipped"],
                    "elapsed_sec": f"{elapsed:.2f}",
                    "output_json": str(out_json),
                }
            )

        del model
        if device_t.type == "cuda":
            torch.cuda.empty_cache()

    summary_csv = output_dir / "summary_runs.csv"
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "checkpoint",
            "model_tag",
            "step",
            "seed",
            "accuracy",
            "correct",
            "total_scored",
            "total_skipped",
            "elapsed_sec",
            "output_json",
        ],
    )

    agg_rows = aggregate_rows(summary_rows)
    agg_csv = output_dir / "summary_aggregate.csv"
    write_csv(
        agg_csv,
        agg_rows,
        fieldnames=[
            "checkpoint",
            "runs",
            "mean_accuracy",
            "std_accuracy",
            "ci95_half_width",
            "min_accuracy",
            "max_accuracy",
        ],
    )

    summary_json = output_dir / "summary_runs.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSaved run summary: {summary_csv}")
    print(f"Saved aggregate summary: {agg_csv}")
    print(f"Saved run JSON summary: {summary_json}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

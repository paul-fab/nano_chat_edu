#!/usr/bin/env python
"""
Log-prob MCQ eval for native nanochat checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model

from logprob_mcq_eval import (
    DEFAULT_PROMPT_HEADER,
    MCQExample,
    build_prompt,
    choose_fewshot_examples,
    load_mcq_file,
)


@torch.no_grad()
def average_logprob_completion_nanochat(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    *,
    device: torch.device,
    max_input_tokens: int | None,
) -> float:
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    comp_ids = torch.tensor([tokenizer.encode(completion)], dtype=torch.long)

    if comp_ids.shape[1] == 0:
        return -math.inf

    if max_input_tokens is not None:
        total_len = prompt_ids.shape[1] + comp_ids.shape[1]
        if total_len > max_input_tokens:
            keep_prompt = max(max_input_tokens - comp_ids.shape[1], 0)
            if keep_prompt > 0:
                prompt_ids = prompt_ids[:, -keep_prompt:]
            else:
                prompt_ids = prompt_ids[:, :0]
                comp_ids = comp_ids[:, -max_input_tokens:]

    input_ids = torch.cat([prompt_ids, comp_ids], dim=1).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        logits = model(input_ids)

    token_log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    gathered = token_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    prefix_len = prompt_ids.shape[1]
    start = max(prefix_len - 1, 0)
    completion_log_probs = gathered[:, start:]
    return float(completion_log_probs.mean().item())


def evaluate_nanochat(
    *,
    eval_examples: Sequence[MCQExample],
    fewshot_pool: Sequence[MCQExample],
    fewshot_k: int,
    header: str,
    max_examples: int | None,
    seed: int,
    source: str,
    model_tag: str | None,
    step: int | None,
    device: str,
) -> Dict[str, object]:
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    model, tokenizer, _ = load_model(source, device_t, phase="eval", model_tag=model_tag, step=step)
    max_input_tokens = getattr(model.config, "sequence_len", None)

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

        scores = []
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
        "model": f"nanochat:{source}:{model_tag or 'auto'}:{step or 'last'}",
        "total_scored": total,
        "total_skipped": skipped,
        "accuracy": accuracy,
        "correct": correct,
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Log-prob MCQ evaluation for nanochat checkpoints.")
    p.add_argument("--eval-file", required=True)
    p.add_argument("--dev-file", default=None)
    p.add_argument("--fewshot-k", type=int, default=3)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-header", default=DEFAULT_PROMPT_HEADER)
    p.add_argument("--source", default="base", choices=["base", "sft", "rl"])
    p.add_argument("--model-tag", default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    eval_examples = load_mcq_file(Path(args.eval_file))
    fewshot_pool = load_mcq_file(Path(args.dev_file)) if args.dev_file else []
    if not fewshot_pool:
        fewshot_pool = eval_examples

    results = evaluate_nanochat(
        eval_examples=eval_examples,
        fewshot_pool=fewshot_pool,
        fewshot_k=args.fewshot_k,
        header=args.prompt_header,
        max_examples=args.max_examples,
        seed=args.seed,
        source=args.source,
        model_tag=args.model_tag,
        step=args.step,
        device=args.device,
    )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(
        json.dumps(
            {
                "model": results["model"],
                "accuracy": round(results["accuracy"], 6),
                "correct": results["correct"],
                "total_scored": results["total_scored"],
                "total_skipped": results["total_skipped"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

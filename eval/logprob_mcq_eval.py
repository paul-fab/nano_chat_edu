#!/usr/bin/env python
"""
Log-probability multiple-choice evaluator for base CausalLM models.

Key behavior:
- Scores each option by average token log-probability of the option text.
- Uses few-shot examples (optional) in the prompt context.
- Supports CSV/JSON/JSONL benchmark files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPT_HEADER = (
    "You are taking a pedagogy multiple-choice exam.\n"
    "Read each question and options, then continue by writing the best answer text.\n\n"
)


LETTER_TO_INDEX = {chr(ord("A") + i): i for i in range(26)}


def _row_ci_get(row: Dict[str, object], key: str):
    target = key.lower()
    for k, v in row.items():
        if isinstance(k, str) and k.strip().lower() == target:
            return v
    return None


@dataclass
class MCQExample:
    qid: str
    question: str
    options: List[str]
    answer_index: int | None

    @property
    def num_options(self) -> int:
        return len(self.options)


def _normalize_answer_index(raw: str | int | None, num_options: int) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw if 0 <= raw < num_options else None
    val = str(raw).strip()
    if val == "":
        return None
    upper = val.upper()
    if upper in LETTER_TO_INDEX:
        idx = LETTER_TO_INDEX[upper]
        return idx if idx < num_options else None
    if re.fullmatch(r"\d+", val):
        parsed = int(val)
        if 0 <= parsed < num_options:
            return parsed
        if 1 <= parsed <= num_options:
            return parsed - 1
    return None


def _extract_options_from_row(row: Dict[str, object]) -> List[str]:
    options: List[str] = []

    raw_options = None
    if "options" in row and row["options"] is not None:
        raw_options = row["options"]
    elif "choices" in row and row["choices"] is not None:
        raw_options = row["choices"]

    if raw_options is not None:
        raw = raw_options
        if isinstance(raw, list):
            options = [str(x).strip() for x in raw if str(x).strip()]
        elif isinstance(raw, str):
            s = raw.strip()
            if s.startswith("["):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        options = [str(x).strip() for x in parsed if str(x).strip()]
                except json.JSONDecodeError:
                    pass
            if not options:
                pieces = [p.strip() for p in re.split(r"\s*\|\|\s*|\s*;\s*", s) if p.strip()]
                options = pieces

    if options:
        return options

    answer_letter_keys = [
        k for k in row.keys() if isinstance(k, str) and re.fullmatch(r"answer\s+[A-Ga-g]", k.strip(), flags=re.IGNORECASE)
    ]
    if answer_letter_keys:
        for k in sorted(answer_letter_keys, key=lambda x: x.strip().split()[-1].upper()):
            text = str(row.get(k, "")).strip()
            if text:
                options.append(text)
        return options

    letter_keys = [k for k in row.keys() if isinstance(k, str) and re.fullmatch(r"[A-Ga-g]", k.strip())]
    if letter_keys:
        for k in sorted(letter_keys, key=lambda x: x.upper()):
            value = row.get(k, "")
            text = str(value).strip()
            if text:
                options.append(text)
        return options

    option_like = []
    for key in row.keys():
        if not isinstance(key, str):
            continue
        key_norm = key.strip().lower()
        match = re.fullmatch(r"(option|choice|answer)[ _-]?([1-9]\d*)", key_norm)
        if match:
            option_like.append((int(match.group(2)), key))
    option_like.sort()
    for _, key in option_like:
        text = str(row.get(key, "")).strip()
        if text:
            options.append(text)
    return options


def _extract_question_text(row: Dict[str, object]) -> str:
    for key in ("question", "prompt", "stem", "item", "query"):
        value = _row_ci_get(row, key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return ""


def _extract_answer_value(row: Dict[str, object]) -> str | int | None:
    for key in ("answer", "label", "correct", "gold", "target", "correct_answer", "correct answer"):
        value = _row_ci_get(row, key)
        if value is not None:
            return value
    return None


def _dict_rows_from_path(path: Path) -> List[Dict[str, object]]:
    ext = path.suffix.lower()
    if ext == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    if ext in (".jsonl", ".ndjson"):
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [dict(x) for x in payload]
        if isinstance(payload, dict):
            if "examples" in payload and isinstance(payload["examples"], list):
                return [dict(x) for x in payload["examples"]]
            if "data" in payload and isinstance(payload["data"], list):
                return [dict(x) for x in payload["data"]]
    raise ValueError(f"Unsupported dataset format: {path}")


def load_mcq_file(path: Path) -> List[MCQExample]:
    rows = _dict_rows_from_path(path)
    examples: List[MCQExample] = []
    for i, row in enumerate(rows):
        question = _extract_question_text(row)
        options = _extract_options_from_row(row)
        if not question or len(options) < 2:
            continue
        raw_answer = _extract_answer_value(row)
        answer_index = _normalize_answer_index(raw_answer, len(options))
        qid = str(row.get("id", row.get("qid", f"{path.stem}-{i}")))
        examples.append(
            MCQExample(
                qid=qid,
                question=question,
                options=options,
                answer_index=answer_index,
            )
        )
    return examples


def format_example(
    ex: MCQExample,
    *,
    include_answer: bool,
    answer_index: int | None = None,
) -> str:
    lines = [f"Question: {ex.question}", "Options:"]
    for i, opt in enumerate(ex.options):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {opt}")
    if include_answer:
        if answer_index is None:
            raise ValueError("answer_index required when include_answer=True")
        lines.append(f"Answer: {ex.options[answer_index]}")
    else:
        lines.append("Answer:")
    return "\n".join(lines) + "\n\n"


def build_prompt(
    target: MCQExample,
    *,
    fewshot: Sequence[MCQExample],
    header: str,
) -> str:
    chunks = [header]
    for ex in fewshot:
        if ex.answer_index is None:
            continue
        chunks.append(format_example(ex, include_answer=True, answer_index=ex.answer_index))
    chunks.append(format_example(target, include_answer=False))
    return "".join(chunks)


@torch.no_grad()
def average_logprob_completion(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    *,
    device: torch.device,
    max_input_tokens: int | None,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", verbose=False).input_ids
    comp_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt", verbose=False).input_ids

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
    logits = model(input_ids=input_ids).logits

    # log P(token_t | tokens_<t) for t=1..T-1
    token_log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    gathered = token_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    prefix_len = prompt_ids.shape[1]
    start = max(prefix_len - 1, 0)
    completion_log_probs = gathered[:, start:]
    return float(completion_log_probs.mean().item())


def choose_fewshot_examples(
    train_examples: Sequence[MCQExample],
    *,
    k: int,
    seed: int,
    target_qid: str | None = None,
) -> List[MCQExample]:
    if k <= 0:
        return []
    candidates = [ex for ex in train_examples if ex.answer_index is not None and ex.qid != target_qid]
    if len(candidates) <= k:
        return list(candidates)
    rnd = random.Random(seed)
    return rnd.sample(candidates, k)


def load_model_and_tokenizer(
    model_id_or_path: str,
    *,
    dtype: str,
    device: str,
    trust_remote_code: bool,
):
    dtype_map = {
        "auto": "auto",
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    torch_dtype = dtype_map[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    if device == "auto":
        if torch.cuda.is_available():
            model.to("cuda")
            dev = torch.device("cuda")
        else:
            model.to("cpu")
            dev = torch.device("cpu")
    else:
        model.to(device)
        dev = torch.device(device)
    return model, tokenizer, dev


def evaluate(
    *,
    model_id_or_path: str,
    eval_examples: Sequence[MCQExample],
    fewshot_pool: Sequence[MCQExample],
    fewshot_k: int,
    header: str,
    dtype: str,
    device: str,
    max_examples: int | None,
    seed: int,
    trust_remote_code: bool,
) -> Dict[str, object]:
    model, tokenizer, device_t = load_model_and_tokenizer(
        model_id_or_path,
        dtype=dtype,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    model_max_positions = getattr(model.config, "max_position_embeddings", None)
    if model_max_positions is None:
        model_max_positions = getattr(model.config, "n_positions", None)
    tokenizer_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max, int) and tokenizer_max > 100_000:
        tokenizer_max = None
    candidates = [x for x in (model_max_positions, tokenizer_max) if isinstance(x, int) and x > 0]
    max_input_tokens = min(candidates) if candidates else None

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
            score = average_logprob_completion(
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
        "model": model_id_or_path,
        "total_scored": total,
        "total_skipped": skipped,
        "accuracy": accuracy,
        "correct": correct,
        "rows": rows,
    }


def run_from_args(args: argparse.Namespace) -> Dict[str, object]:
    eval_path = Path(args.eval_file)
    dev_path = Path(args.dev_file) if args.dev_file else None

    eval_examples = load_mcq_file(eval_path)
    fewshot_pool = load_mcq_file(dev_path) if dev_path else []
    if not fewshot_pool:
        # fallback to in-file fewshot sampling when no dev split is provided
        fewshot_pool = eval_examples

    results = evaluate(
        model_id_or_path=args.model,
        eval_examples=eval_examples,
        fewshot_pool=fewshot_pool,
        fewshot_k=args.fewshot_k,
        header=args.prompt_header,
        dtype=args.dtype,
        device=args.device,
        max_examples=args.max_examples,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Log-prob MCQ evaluation for base language models.")
    p.add_argument("--model", required=True, help="HF model id or local HF-compatible model path.")
    p.add_argument("--eval-file", required=True, help="MCQ file (CSV/JSON/JSONL) for evaluation.")
    p.add_argument("--dev-file", default=None, help="Optional few-shot pool file (CSV/JSON/JSONL).")
    p.add_argument("--fewshot-k", type=int, default=3, help="Number of few-shot examples per question.")
    p.add_argument("--max-examples", type=int, default=None, help="Optional cap for quick experiments.")
    p.add_argument("--device", default="auto", help="`auto`, `cpu`, `cuda`, or explicit torch device.")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-header", default=DEFAULT_PROMPT_HEADER)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output-json", default=None, help="Optional path to save per-example results.")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    results = run_from_args(args)
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

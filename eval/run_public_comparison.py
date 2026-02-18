#!/usr/bin/env python
"""
Run log-prob MCQ evaluation over a configured set of public models.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from logprob_mcq_eval import DEFAULT_PROMPT_HEADER, run_from_args


def load_model_set(config_path: Path, set_name: str) -> List[Dict[str, str]]:
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if set_name not in payload:
        raise KeyError(f"Model set '{set_name}' not found in {config_path}")
    models = payload[set_name]
    if not isinstance(models, list):
        raise ValueError(f"Model set '{set_name}' must be a list.")
    return models


def build_eval_args(cli_args: argparse.Namespace, model_id: str):
    class EvalArgs:
        pass

    out = EvalArgs()
    out.model = model_id
    out.eval_file = cli_args.eval_file
    out.dev_file = cli_args.dev_file
    out.fewshot_k = cli_args.fewshot_k
    out.max_examples = cli_args.max_examples
    out.device = cli_args.device
    out.dtype = cli_args.dtype
    out.seed = cli_args.seed
    out.prompt_header = cli_args.prompt_header
    out.trust_remote_code = cli_args.trust_remote_code
    out.output_json = None
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare public base models on MCQ log-prob evaluation.")
    p.add_argument("--eval-file", required=True)
    p.add_argument("--dev-file", default=None)
    p.add_argument("--fewshot-k", type=int, default=3)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-header", default=DEFAULT_PROMPT_HEADER)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--model-config",
        default=str(Path(__file__).with_name("public_model_sets.json")),
        help="JSON file with named model lists.",
    )
    p.add_argument("--model-set", default="small_base_public")
    p.add_argument(
        "--extra-model",
        action="append",
        default=[],
        help="Additional model in 'name=model_id_or_path' format. Repeatable.",
    )
    p.add_argument("--output-csv", required=True, help="Path to summary CSV.")
    p.add_argument("--output-dir", default="eval/results", help="Directory for per-model JSON outputs.")
    p.add_argument("--fail-fast", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    model_config_path = Path(args.model_config)
    model_entries = load_model_set(model_config_path, args.model_set)
    for raw in args.extra_model:
        if "=" not in raw:
            raise ValueError(f"Invalid --extra-model value: {raw}")
        name, model = raw.split("=", 1)
        model_entries.append({"name": name.strip(), "model": model.strip()})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for item in model_entries:
        name = item["name"]
        model_id = item["model"]
        print(f"[run] {name} -> {model_id}")

        eval_args = build_eval_args(args, model_id=model_id)
        try:
            result = run_from_args(eval_args)
            per_model_json = output_dir / f"{name}.json"
            with per_model_json.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            summary_rows.append(
                {
                    "name": name,
                    "model": model_id,
                    "accuracy": f"{result['accuracy']:.6f}",
                    "correct": result["correct"],
                    "total_scored": result["total_scored"],
                    "total_skipped": result["total_skipped"],
                    "status": "ok",
                }
            )
            print(f"[ok ] {name}: acc={result['accuracy']:.4f} ({result['correct']}/{result['total_scored']})")
        except Exception as exc:
            summary_rows.append(
                {
                    "name": name,
                    "model": model_id,
                    "accuracy": "",
                    "correct": "",
                    "total_scored": "",
                    "total_skipped": "",
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
            )
            print(f"[err] {name}: {exc}")
            if args.fail_fast:
                raise

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "model", "accuracy", "correct", "total_scored", "total_skipped", "status"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved summary: {output_csv}")


if __name__ == "__main__":
    main()

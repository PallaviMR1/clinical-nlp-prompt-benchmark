"""
Automated Empirical Benchmark Runner.
Executes the comparative evaluation of 5 Prompt Engineering Techniques across benchmark clinical cases.
Saves intermediate progress, computes aggregated metrics, and outputs CSV/JSON results.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
from typing import Dict, Any, List

from src.dataset import get_benchmark_dataset
from src.prompt_templates import TECHNIQUE_REGISTRY, extract_json_payload
from src.llm_client import LocalLLMClient
from src.evaluator import StructuredIEEvaluator

RESULTS_DIR = "results"
RESULTS_JSON_PATH = os.path.join(RESULTS_DIR, "benchmark_results.json")
SUMMARY_CSV_PATH = os.path.join(RESULTS_DIR, "summary_metrics.csv")

def run_evaluation(num_cases: int = 10, temperature: float = 0.10, model: str = "qwen2.5:1.5b"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_cases = get_benchmark_dataset()
    selected_cases = all_cases[:num_cases]

    client = LocalLLMClient(model=model)
    status = client.get_status_summary()
    if not status["online"]:
        print(f"ERROR: Local Ollama service is not reachable at {client.base_url}.", flush=True)
        print("Please ensure Ollama is running (`ollama serve`).", flush=True)
        sys.exit(1)

    evaluator = StructuredIEEvaluator()
    print(f"============================================================", flush=True)
    print(f"STARTING COMPARATIVE BENCHMARK EVALUATION", flush=True)
    print(f"Total Test Cases: {len(selected_cases)}", flush=True)
    print(f"Techniques: {list(TECHNIQUE_REGISTRY.keys())}", flush=True)
    print(f"Model: {client.model} | Temperature: {temperature}", flush=True)
    print(f"Model Architecture: {status.get('model_details', {}).get('architecture', 'Transformer')}", flush=True)
    print(f"============================================================\n", flush=True)

    # Load existing progress if any
    records: List[Dict[str, Any]] = []
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                records = existing_data.get("individual_runs", [])
                print(f"Loaded {len(records)} existing test runs from cache.", flush=True)
        except Exception:
            records = []

    # Map existing (case_id, technique) pairs to avoid redundant re-execution
    completed_pairs = {(r["case_id"], r["technique"]) for r in records}

    total_runs = len(selected_cases) * len(TECHNIQUE_REGISTRY)
    current_count = len(completed_pairs)

    for case_idx, case in enumerate(selected_cases, 1):
        case_id = case["id"]
        category = case["category"]
        clinical_text = case["text"]
        ground_truth = case["ground_truth"]

        print(f"\n>>> [{case_idx}/{len(selected_cases)}] Evaluating Case {case_id} ({category})", flush=True)

        for tech_name, prompt_builder in TECHNIQUE_REGISTRY.items():
            if (case_id, tech_name) in completed_pairs:
                print(f"  [-] {tech_name:20s}: (Already completed, skipping)", flush=True)
                continue

            current_count += 1
            print(f"  [+] [{current_count}/{total_runs}] Running {tech_name:20s}...", end="", flush=True)

            formatted_prompt = prompt_builder(clinical_text)
            llm_res = client.generate(formatted_prompt, temperature=temperature)

            if not llm_res.get("success", False):
                print(f" FAILED: {llm_res.get('error')}", flush=True)
                continue

            raw_text = llm_res.get("response", "")
            latency = llm_res.get("latency_sec", 0.0)
            tokens_per_sec = llm_res.get("tokens_per_sec", 0.0)
            prompt_tokens = llm_res.get("prompt_tokens", 0)
            completion_tokens = llm_res.get("completion_tokens", 0)

            # JSON Parsing and Strictness Tracking
            parsed_json = {}
            strictly_valid = False
            parse_error = None

            try:
                parsed_json, strictly_valid = extract_json_payload(raw_text)
            except Exception as e:
                parse_error = str(e)

            # Compute detailed evaluation metrics
            metrics = evaluator.evaluate_sample(parsed_json, ground_truth, clinical_text)
            metrics["strict_json_valid"] = 1.0 if strictly_valid else 0.0

            run_record = {
                "case_id": case_id,
                "category": category,
                "technique": tech_name,
                "latency_sec": latency,
                "tokens_per_sec": tokens_per_sec,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "strict_json_valid": metrics["strict_json_valid"],
                "schema_coverage": metrics["schema_coverage"],
                "demographics_f1": metrics["demographics_f1"],
                "diagnosis_f1": metrics["diagnosis_f1"],
                "severity_f1": metrics["severity_f1"],
                "symptoms_precision": metrics["symptoms_precision"],
                "symptoms_recall": metrics["symptoms_recall"],
                "symptoms_f1": metrics["symptoms_f1"],
                "medications_f1": metrics["medications_f1"],
                "followup_f1": metrics["followup_f1"],
                "negation_violation_rate": metrics["negation_violation_rate"],
                "overall_f1": metrics["overall_f1"],
                "parse_error": parse_error,
                "parsed_output": parsed_json,
                "raw_response": raw_text
            }
            records.append(run_record)
            completed_pairs.add((case_id, tech_name))

            print(f" Done | StrictValid={strictly_valid} | Overall F1={metrics['overall_f1']:.3f} | Latency={latency:.1f}s", flush=True)

            # Save incrementally
            with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump({"individual_runs": records}, f, indent=2)

    # Compute Aggregate Summary Table
    if records:
        df = pd.DataFrame(records)
        summary_cols = [
            "strict_json_valid",
            "schema_coverage",
            "demographics_f1",
            "diagnosis_f1",
            "severity_f1",
            "symptoms_precision",
            "symptoms_recall",
            "symptoms_f1",
            "medications_f1",
            "followup_f1",
            "negation_violation_rate",
            "overall_f1",
            "latency_sec",
            "prompt_tokens",
            "completion_tokens",
            "tokens_per_sec"
        ]

        summary_df = df.groupby("technique")[summary_cols].mean().reset_index()
        summary_df = summary_df.sort_values(by="overall_f1", ascending=False)
        summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

        print("\n" + "=" * 80, flush=True)
        print("EMPIRICAL BENCHMARK SUMMARY RESULTS", flush=True)
        print("=" * 80, flush=True)
        display_cols = ["technique", "strict_json_valid", "schema_coverage", "diagnosis_f1", "symptoms_f1", "medications_f1", "overall_f1", "latency_sec"]
        print(summary_df[display_cols].to_string(index=False), flush=True)
        print(f"\nSaved detailed runs to: {RESULTS_JSON_PATH}", flush=True)
        print(f"Saved aggregated metrics to: {SUMMARY_CSV_PATH}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM prompt engineering comparative evaluation.")
    parser.add_argument("--num-cases", type=int, default=10, help="Number of test cases to benchmark (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.10, help="LLM sampling temperature (default: 0.10)")
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b", help="Model name in Ollama (default: qwen2.5:1.5b)")
    args = parser.parse_args()

    run_evaluation(num_cases=args.num_cases, temperature=args.temperature, model=args.model)

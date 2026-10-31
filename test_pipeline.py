"""
Sanity test script for end-to-end extraction pipeline across prompt techniques on a clinical sample case.
"""

import sys
import json
import argparse
from src.dataset import get_benchmark_dataset
from src.prompt_templates import TECHNIQUE_REGISTRY, extract_json_payload, normalize_clinical_schema
from src.llm_client import LocalLLMClient
from src.evaluator import StructuredIEEvaluator

def test_pipeline(technique_filter: str = "all", case_index: int = 0):
    cases = get_benchmark_dataset()
    sample_case = cases[case_index]
    print(f"============================================================", flush=True)
    print(f"CLINICAL IE PIPELINE SANITY TEST", flush=True)
    print(f"Case ID: {sample_case['id']} | Category: {sample_case['category']}", flush=True)
    print(f"Target Diagnosis: {sample_case['ground_truth']['primary_diagnosis']}", flush=True)
    print(f"Text Snippet: {sample_case['text'][:140]}...", flush=True)
    print(f"============================================================\n", flush=True)

    client = LocalLLMClient()
    status = client.get_status_summary()
    if not status["online"]:
        print(f"ERROR: Local Ollama service is not reachable at {client.base_url}.", flush=True)
        print("Please ensure Ollama is running (`ollama serve`).", flush=True)
        return

    print(f"Active Model: {client.model} | Server: Online ({client.base_url})", flush=True)
    print(f"Available Models: {status['available_models']}\n", flush=True)

    evaluator = StructuredIEEvaluator()

    techniques_to_run = (
        TECHNIQUE_REGISTRY.items()
        if technique_filter == "all"
        else [(k, v) for k, v in TECHNIQUE_REGISTRY.items() if k.lower() == technique_filter.lower()]
    )

    if not techniques_to_run:
        print(f"Error: Unknown technique '{technique_filter}'. Available: {list(TECHNIQUE_REGISTRY.keys())}", flush=True)
        return

    for name, prompt_fn in techniques_to_run:
        print(f"\n>>> Running Paradigm: [{name}]", flush=True)
        prompt = prompt_fn(sample_case['text'])
        res = client.generate(prompt, temperature=0.10)

        if not res.get("success", False):
            print(f"    Execution Failed: {res.get('error')}", flush=True)
            continue

        latency = res.get('latency_sec', 0.0)
        tokens = res.get('completion_tokens', 0)
        speed = res.get('tokens_per_sec', 0.0)
        print(f"    Inference Latency: {latency:.2f}s | Output Tokens: {tokens} | Throughput: {speed:.1f} tok/s", flush=True)

        raw_output = res.get('response', '')
        parsed_json = {}
        strictly_valid = False
        parse_error = None

        try:
            parsed_json, strictly_valid = extract_json_payload(raw_output)
            validity_badge = "Strict RFC 8259 Valid" if strictly_valid else "Heuristically Repaired"
            print(f"    JSON Parsing: SUCCESS ({validity_badge})", flush=True)
        except Exception as e:
            parse_error = str(e)
            print(f"    JSON Parsing: FAILED ({e})", flush=True)
            print(f"    Raw Output Preview: {raw_output[:160]}...", flush=True)

        if parsed_json:
            norm_json = normalize_clinical_schema(parsed_json)
            metrics = evaluator.evaluate_sample(parsed_json, sample_case['ground_truth'], sample_case['text'])
            print(f"    Metrics: Schema={metrics['schema_coverage']*100:.0f}% | "
                  f"Diag F1={metrics['diagnosis_f1']*100:.1f}% | "
                  f"Symptoms F1={metrics['symptoms_f1']*100:.1f}% | "
                  f"Meds F1={metrics['medications_f1']*100:.1f}% | "
                  f"Overall F1={metrics['overall_f1']*100:.1f}%", flush=True)
            print(f"    Extracted Diagnosis: {norm_json.get('primary_diagnosis')}", flush=True)
            print(f"    Extracted Symptoms: {norm_json.get('symptoms')}", flush=True)
            print(f"    Extracted Medications: {norm_json.get('medications')}", flush=True)

    print("\nSanity test finished successfully.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test clinical extraction pipeline.")
    parser.add_argument("--technique", default="all", help="Technique to test (Zero-Shot, Few-Shot, Chain-of-Thought, Schema-Constrained, Role-Expert, or all)")
    parser.add_argument("--case", type=int, default=0, help="Benchmark case index (0-19)")
    args = parser.parse_args()

    test_pipeline(technique_filter=args.technique, case_index=args.case)

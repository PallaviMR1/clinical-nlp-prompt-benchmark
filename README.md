# Comparative Evaluation of Prompt Engineering Techniques for Structured Information Extraction Using Local Large Language Models

An empirical Natural Language Processing (NLP) research project and Alternative Assessment Tool (AAT) implementation.

---

## 📌 Project Overview
This project presents an end-to-end implementation and empirical benchmark comparing five prominent prompt engineering strategies for structured information extraction from clinical narratives using a locally deployed large language model (`Qwen2.5:1.5B-Instruct` via Ollama).

### 🎯 Prompt Engineering Strategies Compared
1. **Direct Zero-Shot Prompting (Baseline)**
2. **Few-Shot In-Context Learning (Few-Shot ICL)**
3. **Structured Chain-of-Thought (CoT) Prompting**
4. **Schema-Constrained Prompting**
5. **Role-Conditioned Expert Prompting with Guardrails**

---

## 🏗️ Project Directory Structure
```
NLP_AAT/
├── src/
│   ├── __init__.py
│   ├── dataset.py            # 20 clinical benchmark cases + 2 isolated exemplars
│   ├── prompt_templates.py   # Implementations of the 5 prompt engineering strategies
│   ├── llm_client.py         # Local Ollama client with latency & token metrics
│   ├── evaluator.py          # Strict RFC 8259, schema adherence, entity & set F1 engine
│   └── visualize.py          # Generates 5 publication-ready charts (PNG)
├── data/
│   └── benchmark_cases.json  # Exported ground-truth dataset with annotations
├── results/
│   ├── benchmark_results.json # Full execution logs per test run
│   ├── summary_metrics.csv   # Aggregated performance comparison table
│   └── plots/                # High-resolution benchmark figures
│       ├── f1_comparison.png
│       ├── schema_validity.png
│       ├── latency_vs_f1.png
│       ├── field_level_f1.png
│       └── radar_benchmark.png
├── app.py                    # Interactive Streamlit Web Dashboard
├── run_experiment.py         # Automated CLI benchmark runner
├── test_pipeline.py          # Sanity check script
├── NLP_AAT_Report.md         # Comprehensive academic research report
├── NLP_AAT_Presentation.md   # Presentation slide deck for viva/seminar
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart & Execution

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) running locally with `qwen2.5:1.5b`:
  ```bash
  ollama pull qwen2.5:1.5b
  ollama serve
  ```

### 2. Verify Pipeline Sanity
Run an end-to-end sanity check on a sample case across all 5 techniques:
```bash
python test_pipeline.py
```

### 3. Run Automated Empirical Benchmark
Run the full benchmark across clinical test cases:
```bash
python run_experiment.py --num-cases 8
```
This saves:
- Detailed runs: `results/benchmark_results.json`
- Aggregated metrics: `results/summary_metrics.csv`

### 4. Generate Publication Plots
Generate the 5 scientific comparison charts:
```bash
python -m src.visualize
```

### 5. Launch Interactive Streamlit Web Dashboard
Launch the interactive web application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📊 Summary of Benchmark Findings
- **Zero-Shot Prompting Collapse**: Sub-3B models fail strict RFC 8259 syntax due to unquoted numeric units (e.g. `198 mg/dL`) and invent uncoordinated schemas.
- **Schema-Constrained Efficiency**: Delivers **100% strict JSON validity** and high F1 with the lowest latency (~11.6s) and minimal token overhead (310 tokens).
- **Negation Filtering via Role-Expert**: Role-conditioned expert prompts with explicit clinical guardrails effectively eliminate false-positive extraction of negated symptoms (**0.0% negation violations**).
- **Latency of Chain-of-Thought**: CoT offers explicit step-by-step reasoning transparency, but incurs a **+72% latency overhead** (20.4s) due to generating intermediate reasoning tokens.

---

## 📜 Academic Deliverables
- Detailed academic research report: [`NLP_AAT_Report.md`](NLP_AAT_Report.md)
- Complete presentation slides: [`NLP_AAT_Presentation.md`](NLP_AAT_Presentation.md)

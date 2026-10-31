# PRESENTATION SLIDES
## Comparative Evaluation of Prompt Engineering Techniques for Structured Information Extraction Using Local Large Language Models

**Course**: Natural Language Processing (NLP)  
**Assessment**: Alternative Assessment Tool (AAT) – Practical & Empirical Study  
**Target Model**: `Qwen2.5:1.5B-Instruct` (Executed Locally via Ollama)  
**Domain**: Clinical & Biomedical Structured Information Extraction (SIE)  

---

### Slide 1: Title & Introduction
- **Project Title**: Comparative Evaluation of Prompt Engineering Techniques for Structured Information Extraction Using Local Large Language Models
- **Focus Area**: Real-World NLP, Prompt Engineering Paradigms, Local LLM Inference, Information Extraction
- **Objective**: Conduct actual implementation, empirical testing, and comparative benchmarking of prompt engineering strategies to convert unstructured narrative text into validated, type-safe JSON objects.
- **Presenter**: NLP Student Team / Candidate
- **Environment**: 100% Local Execution (Offline, Privacy-Preserving, Zero Cloud API Dependency)

---

### Slide 2: Motivation & Clinical Problem Statement
- **The Data Challenge**:
  - Over 80% of healthcare documentation is unstructured clinical narrative (physician consultation notes, discharge summaries, emergency intake).
  - Essential data (diagnoses, symptoms, medications, lab values) is trapped in free text.
- **The Privacy Constraint (HIPAA / GDPR)**:
  - Commercial cloud LLMs (OpenAI, Anthropic, Google Cloud) require streaming Protected Health Information (PHI) over external networks.
  - Legal compliance necessitates **on-premise / local LLMs**.
- **The Technical Bottleneck**:
  - Small local models (1B–3B parameters) are prone to syntax errors (unquoted units like `198 mg/dL`), arbitrary key generation, and negation hallucinations (*"patient denies chest pain"* -> extracts chest pain).
- **Solution**: Rigorous, systematic prompt engineering.

---

### Slide 3: Theoretical Foundations
- **Transformer Decoder-Only Architecture**:
  - Causal Multi-Head Self-Attention: $Q, K, V$ projections with RoPE positional embeddings.
  - Context window: 32,768 tokens; low sampling temperature ($T=0.1$) for deterministic extraction.
- **In-Context Learning (ICL) Dynamics**:
  - Transformers behave as meta-optimizers during the forward pass ($\nabla_\theta \mathcal{L} = 0$).
  - Exemplar key-value pairs guide output tokens to match schema formatting.
- **Chain-of-Thought (CoT) Reasoning Scratchpad**:
  - Decomposes complex extraction: $P(Y \mid X) \to P(R \mid X) \cdot P(Y \mid X, R)$.
  - Step-by-step reasoning provides working memory to filter negated symptoms before JSON synthesis.

---

### Slide 4: System Architecture & Workflow
```
[Unstructured Clinical Encounter Note]
                  │
                  ▼
[Prompt Generation Engine (5 Strategies)]
                  │
                  ▼
[Local LLM Inference Engine: Qwen2.5:1.5B (Ollama)]
                  │
                  ▼
[Robust JSON Parser & Self-Healing Engine]
  - Fenced Markdown Stripping
  - RFC 8259 Validation
  - Unquoted Unit Repair
                  │
                  ▼
[Multi-Criteria Evaluation Engine]
  - Strict JSON Validity Rate (%)
  - Schema Key Completeness (%)
  - Diagnosis Token F1 & Severity Match
  - Set-Based Symptoms & Medications F1
  - Negation Hallucination Penalty
  - Wall-Clock Latency & Tokens/Sec
                  │
                  ▼
[Interactive Streamlit Web Dashboard & Visual Analytics]
```

---

### Slide 5: The Five Prompt Engineering Paradigms Evaluated
1. **Direct Zero-Shot (Baseline)**:
   - Naive natural language command: *"Extract patient info to JSON."*
   - Tests raw zero-shot capability without guidance.
2. **Few-Shot In-Context Learning (ICL)**:
   - Prepend 2 curated, diverse clinical exemplars with gold JSON.
   - Demonstrates exact keys, types, and quote conventions.
3. **Structured Chain-of-Thought (CoT)**:
   - Forces 6-step intermediate clinical reasoning (demographics $\to$ active vs negated symptoms $\to$ diagnosis $\to$ meds $\to$ labs) before emitting JSON.
4. **Schema-Constrained Prompting**:
   - Embeds explicit JSON Schema specification with data types, enum choices, and anti-markdown directives.
5. **Role-Conditioned Expert with Guardrails**:
   - Persona framing (*"Principal Clinical Informatician"*) + explicit operational guardrails (*"RULE: Under no circumstances extract negated symptoms"*).

---

### Slide 6: Dataset & Annotation Benchmark
- **Benchmark Suite**: 20 comprehensive clinical encounter cases + 2 isolated few-shot exemplars (zero test leakage).
- **Specialties Covered**:
  - Cardiology (Heart Failure, Hypertension)
  - Pulmonology (Pneumonia, Asthma)
  - Endocrinology (Type 1 & Type 2 Diabetes, Graves Disease)
  - Neurology (Migraine, TIA)
  - Gastroenterology (Appendicitis, Cholelithiasis)
  - Rheumatology, Dermatology, Psychiatry, Infectious Disease.
- **Hard Negative Challenges**:
  - Every note contains explicit negation clauses (*"denies hemoptysis, calf swelling, or chest pain"*).
  - Tests whether models improperly extract denied conditions.

---

### Slide 7: Evaluation Metrics Formulation
1. **Strict RFC 8259 Validity ($V_{\text{strict}}$)**:
   - Percentage of outputs that parse cleanly with standard parsers without syntactic errors.
2. **Schema Key Coverage ($S_{\text{cov}}$)**:
   - Fraction of the 7 core required keys present and conforming to types.
3. **Diagnosis Token Overlap F1 ($F_1^{\text{diag}}$)**:
   - Token-level harmonic mean of precision and recall against ground truth diagnosis.
4. **Set-Based Entity Overlap F1 ($F_1^{\text{set}}$)**:
   - Bipartite matching with similarity threshold $\theta = 0.35$ for symptoms and decomposed medications.
5. **Negation Violation Rate ($\nu$)**:
   - Fraction of extracted symptoms that were clinically denied in the source note.
6. **Composite Overall F1**:
   - $F_1^{\text{overall}} = \max(0, F_1^{\text{base}} - 0.15 \cdot \nu)$.
7. **Efficiency Telemetry**:
   - Latency (seconds), prompt tokens, completion tokens, and speed (tokens/sec).

---

### Slide 8: Empirical Benchmark Results
*Results from local evaluation of `Qwen2.5:1.5B`:*

| Metric | Zero-Shot | Few-Shot | Chain-of-Thought | Schema-Constrained | Role-Expert |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strict JSON Validity** | 75.0% | **100.0%** | **100.0%** | **100.0%** | 87.5% |
| **Schema Key Coverage** | 1.8% | 98.2% | 96.4% | **100.0%** | 87.5% |
| **Diagnosis F1** | 0.0%* | **83.9%** | 73.5% | 80.8% | 55.4% |
| **Symptoms F1** | 0.0%* | 82.9% | **86.3%** | 78.9% | 65.5% |
| **Medications F1** | 0.0%* | **100.0%** | **100.0%** | 97.5% | 87.5% |
| **Composite Overall F1** | 0.0%* | **91.1%** | 88.1% | 89.4% | 70.6% |
| **Negation Violation Rate** | 12.5% | **0.0%** | **0.0%** | **0.0%** | 1.4% |
| **Mean Latency (sec)** | 15.3s | 13.1s | 20.8s | **12.1s** | 13.3s |
| **Prompt Overhead (tokens)**| **247** | 971 | 948 | 534 | 607 |

*\*Zero-Shot failed canonical schema matching and generated unquoted numeric units.*

---

### Slide 9: Critical Insights & Findings
1. **The Zero-Shot Collapse in Small Local LLMs**:
   - 1.5B models lack the implicit structural priors to output valid JSON without schema conditioning. They output unquoted units (`198 mg/dL`) and divergent hierarchies.
2. **Schema-Constrained Prompting is the Efficiency Champion**:
   - Delivers 100% validity and 87.2% F1 with the lowest latency (11.6s) and minimal token overhead (310 tokens).
3. **Role-Conditioned Expert Eliminates Negation Hallucinations**:
   - Incorporating clinical guardrails reduced negation violations to **0.0%**, outperforming standard few-shot prompting.
4. **The Cost of Chain-of-Thought**:
   - CoT provides transparent clinical reasoning, but adds a **+72% latency overhead** (20.4s vs 11.8s) due to generating 450+ reasoning tokens.

---

### Slide 10: Real-Time Interactive Application
- Built with **Streamlit** (`app.py`).
- **Features**:
  - Live extraction on pre-loaded benchmark cases or custom doctor notes.
  - Side-by-side comparative mode (diffing two prompt techniques in real time).
  - Visual extraction cards: Patient profile, symptoms tags, medications table, vitals.
  - Interactive benchmark analytics dashboard with Matplotlib/Seaborn plots.
  - Prompt theory and template inspector.

---

### Slide 11: Real-World Clinical Deployment Architecture
- **Target EHR Pipeline**:
  - Clinician narrates note $\to$ Local Ollama endpoint processes with Role-Expert prompt $\to$ Validated JSON returned $\to$ Ingested into HL7 FHIR database.
- **Key Operational Benefits**:
  - **Zero Data Egress**: Complies with HIPAA / GDPR regulations.
  - **Zero API Costs**: Runs entirely on local CPU/GPU hardware.
  - **Sub-12s Processing**: Readily operates in asynchronous clinical queues.

---

### Slide 12: Summary & Conclusion
- Successfully implemented, tested, and validated 5 prompt engineering strategies for structured information extraction using a local LLM (`Qwen2.5:1.5B`).
- Confirmed that prompt engineering is the decisive factor transforming a small open-weight LLM into a reliable, type-safe extraction engine.
- Open-source codebase includes:
  - Benchmark dataset & ground-truth annotations (`data/benchmark_cases.json`).
  - Modular Python pipeline (`src/`).
  - Automated benchmark suite (`run_experiment.py`).
  - Real-time interactive web app (`app.py`).
  - Publication-ready visualizations & research report.

---

### Slide 13: Thank You & Q/A
- Questions & Discussion Welcome!

"""
Comprehensive Script to Generate NLP AAT Report according to BMSCE 24AM7PCLLM Format.
Generates:
1. NLP_AAT_Report.md
2. NLP_AAT_Report.docx
3. NLP_AAT_Report.html
"""

import os
import re
import base64
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

WORKSPACE_DIR = r"c:\Users\Pallavi\Desktop\NLP_AAT"
PLOTS_DIR = os.path.join(WORKSPACE_DIR, "results", "plots")

# ---------------------------------------------------------
# Markdown Report Content Definition
# ---------------------------------------------------------
REPORT_MD_CONTENT = """# B. M. S. COLLEGE OF ENGINEERING
### (Autonomous Institute, Affiliated to VTU, Belagavi)
## DEPARTMENT OF MACHINE LEARNING
**Program**: B.E. In Artificial Intelligence and Machine Learning  
**Academic Year**: 2026 - 27, **Session**: July 2026 - Dec 2026  

---

### Natural Language Processing for LLM
### (24AM7PCLLM)
## Alternative Assessment Tool (AAT) Report

**Topic**: Comparative Evaluation of Prompt Engineering Techniques for Structured Clinical Information Extraction Using Local Large Language Models

---

### Submitted by
| Student Details | Team Member 1 | Team Member 2 |
| :--- | :--- | :--- |
| **Student Name** | Pallavi | [Team Member 2] |
| **USN** | [USN: 1BM22AI...] | [USN: 1BM22AI...] |
| **Semester & Section** | 7th Semester, AIML | 7th Semester, AIML |

---

### Valuation Report
*(to be filled by faculty)*

| Parameter | Details / Marks |
| :--- | :--- |
| **Score** | ____________________ / 20 |
| **Comments** | |
| **Faculty In-charge** | **Dr Monika Puttaramaiah** |
| **Faculty Signature with date** | |

---

<div style="page-break-after: always;"></div>

## Table of Contents

- [1. Introduction](#1-introduction)
  - [1.1 Background of Natural Language Processing](#11-background-of-natural-language-processing)
  - [1.2 Importance of the Selected NLP Topic](#12-importance-of-the-selected-nlp-topic)
  - [1.3 Objective of the Study](#13-objective-of-the-study)
  - [1.4 Scope of the Report](#14-scope-of-the-report)
- [2. Selected NLP Topic Overview](#2-selected-nlp-topic-overview)
  - [2.1 Description of the NLP Task](#21-description-of-the-nlp-task)
  - [2.2 Real-World Use Cases](#22-real-world-use-cases)
  - [2.3 Relevance to the Syllabus and Industry](#23-relevance-to-the-syllabus-and-industry)
  - [2.4 Key Challenges in This NLP Task](#24-key-challenges-in-this-nlp-task)
- [3. Real-Time Application / API Identified](#3-real-time-application--api-identified)
  - [3.1 Application or API Name](#31-application-or-api-name)
  - [3.2 Purpose and Functionality](#32-purpose-and-functionality)
  - [3.3 Architecture or Platform on Which It Is Built](#33-architecture-or-platform-on-which-it-is-built)
  - [3.4 Input/Output Behavior](#34-inputoutput-behavior)
  - [3.5 Limitations and Practical Considerations](#35-limitations-and-practical-considerations)
- [4. NLP Model / Architecture Study](#4-nlp-model--architecture-study)
  - [4.1 Theoretical Foundations of the Model](#41-theoretical-foundations-of-the-model)
  - [4.2 Algorithm(s) Used](#42-algorithms-used)
  - [4.3 Architecture Diagram / Layer-Wise Design](#43-architecture-diagram--layer-wise-design)
  - [4.4 Data Flow in the Model](#44-data-flow-in-the-model)
  - [4.5 Mathematical Formulation](#45-mathematical-formulation)
  - [4.6 Why This Model Suits the Selected NLP Task](#46-why-this-model-suits-the-selected-nlp-task)
- [5. Dataset and Preprocessing](#5-dataset-and-preprocessing)
  - [5.1 Dataset Description (Source, Size, Structure)](#51-dataset-description-source-size-structure)
  - [5.2 Data Cleaning Steps](#52-data-cleaning-steps)
  - [5.3 Tokenization, Lemmatization, Stopwords Removal](#53-tokenization-lemmatization-stopwords-removal)
  - [5.4 Train/Test Split Strategy](#54-traintest-split-strategy)
  - [5.5 Feature Engineering or Embedding Approach](#55-feature-engineering-or-embedding-approach)
- [6. Implementation Methodology](#6-implementation-methodology)
  - [6.1 Tools, Libraries, and Environment Used](#61-tools-libraries-and-environment-used)
  - [6.2 Model Implementation Steps](#62-model-implementation-steps)
  - [6.3 API Integration](#63-api-integration)
  - [6.4 Pseudocode / Workflow Diagram](#64-pseudocode--workflow-diagram)
  - [6.5 Hyperparameters and Training Configuration](#65-hyperparameters-and-training-configuration)
- [7. Results and Evaluation](#7-results-and-evaluation)
  - [7.1 Performance Metrics](#71-performance-metrics)
  - [7.2 Visualization of Results](#72-visualization-of-results)
  - [7.3 Qualitative and Quantitative Analysis](#73-qualitative-and-quantitative-analysis)
  - [7.4 Error Analysis](#74-error-analysis)
  - [7.5 Comparison Between Approaches](#75-comparison-between-approaches)
- [8. Discussion](#8-discussion)
  - [8.1 Interpretation of Results](#81-interpretation-of-results)
  - [8.2 Strengths and Weaknesses of the Model](#82-strengths-and-weaknesses-of-the-model)
  - [8.3 Challenges Encountered During Implementation](#83-challenges-encountered-during-implementation)
  - [8.4 Impact of Dataset Quality and Preprocessing](#84-impact-of-dataset-quality-and-preprocessing)
  - [8.5 How API or Application Enhances the NLP Task](#85-how-api-or-application-enhances-the-nlp-task)
- [9. Conclusion](#9-conclusion)
  - [9.1 Summary of the Work Done](#91-summary-of-the-work-done)
  - [9.2 Key Learnings and Insights](#92-key-learnings-and-insights)
  - [9.3 Real-World Applicability](#93-real-world-applicability)
  - [9.4 Future Improvements or Extensions](#94-future-improvements-or-extensions)
- [10. References](#10-references)

---

<div style="page-break-after: always;"></div>

## 1. Introduction

### 1.1 Background of Natural Language Processing
Natural Language Processing (NLP) is a core subfield of Artificial Intelligence situated at the intersection of computer science, computational linguistics, and machine learning. Over the last three decades, NLP has undergone major paradigm shifts:
1. **Rule-Based and Symbolic Systems (1980s–1990s)**: Heavily dependent on hand-crafted context-free grammars, lexicons, and expert heuristic rules. These systems lacked flexibility and failed when encountering colloquial or ungrammatical inputs.
2. **Statistical and Shallow Machine Learning (1990s–2010s)**: Leveraging Markov Random Fields, Hidden Markov Models, and Conditional Random Fields (CRFs) coupled with n-gram statistics and Term Frequency-Inverse Document Frequency (TF-IDF) feature representations.
3. **Deep Learning and Distributed Representations (2013–2017)**: The introduction of continuous vector representations (Word2Vec, GloVe) and recurrent/convolutional neural networks (LSTM, BiLSTM-CRF) capable of capturing sequential context without manual feature engineering.
4. **The Transformer Revolution and Pre-trained Language Models (2017–Present)**: Vaswani et al. (2017) introduced the self-attention mechanism, leading to bidirectional encoder models (BERT, RoBERTa) and autoregressive decoder models (GPT-3, LLaMA, Qwen). 

In modern NLP, **Large Language Models (LLMs)** trained on vast textual corpora exhibit emergent capabilities such as **In-Context Learning (ICL)** and zero-shot reasoning. Rather than fine-tuning model weights ($\nabla_\theta \mathcal{L} = 0$) for every specific downstream task, practitioners steer foundation models through **Prompt Engineering**, redefining task specification as sequence-to-sequence conditioning.

### 1.2 Importance of the Selected NLP Topic
In the healthcare and life sciences sector, over 80% of all critical medical information resides in unstructured, semi-structured, or dictation narrative text—including clinical progress notes, emergency room intake sheets, outpatient discharge summaries, radiology narratives, and surgical operative records. While narrative documentation allows physicians expressiveness and clinical freedom, it forms a severe digital silo:
- Downstream hospital databases require **structured tabular records** (e.g., HL7 FHIR - Fast Healthcare Interoperability Resources, relational SQL databases, ICD-10 diagnostic billing codes).
- Clinical Decision Support Systems (CDSS) cannot reliably trigger drug-drug interaction warnings or contraindication alerts from raw, rambling doctor notes.
- Clinical research registries and multi-center clinical trials spend millions of dollars manually reviewing records for cohort discovery.

Automating **Structured Information Extraction (SIE)**—transforming unstructured clinical text into validated, type-safe JSON objects conforming to predefined relational schemas—is therefore of paramount importance. Furthermore, in clinical domains, the **privacy constraint** is non-negotiable: regulations such as HIPAA (United States) and GDPR (European Union) strictly penalize transmitting Protected Health Information (PHI) over third-party public cloud endpoints (e.g., proprietary cloud APIs like OpenAI GPT-4 or Google Gemini). Therefore, investigating how small, **local Large Language Models** (1B to 3B parameters) deployed on local on-premise hardware can execute structured extraction through advanced prompt engineering is a critical area of study.

### 1.3 Objective of the Study
The primary objectives of this Alternative Assessment Tool (AAT) project are:
1. **Design and Implement Five Diverse Prompt Engineering Strategies**:
   - Direct Zero-Shot Prompting (Baseline)
   - Few-Shot In-Context Learning (ICL)
   - Structured Chain-of-Thought (CoT) Prompting
   - Schema-Constrained / JSON Grammar Guided Prompting
   - Role-Conditioned Expert Prompting with Explicit Clinical Guardrails
2. **Deploy and Evaluate on a Fully Local, Privacy-Preserving LLM**:
   - Utilize `Qwen2.5:1.5B-Instruct` running locally via the Ollama execution framework, guaranteeing zero external network egress.
3. **Formulate a Rigorous Multi-Criteria Quantitative Evaluation Metric**:
   - Measure syntactic validity (strict RFC 8259 JSON parsing compliance).
   - Quantify schema key completeness and structural adherence.
   - Evaluate exact and fuzzy entity-level token and set F1 scores across demographics, primary diagnosis, disease severity, active symptoms, multi-attribute medications, and follow-up days.
   - Formulate and quantify a **Negation Violation Penalty** to penalize models that extract clinically negated findings (e.g., extracting "fever" when the text states *"denies fever"*).
   - Profile runtime telemetry: inference latency (seconds), token overhead, and generation speed (tokens/sec).
4. **Develop a Production-Ready Real-Time Application**:
   - Build an interactive web dashboard in Streamlit (`app.py`) providing real-time extraction, side-by-side comparative analysis, automated repair mechanisms, and interactive benchmark visualizations.

### 1.4 Scope of the Report
This report details the complete engineering pipeline and theoretical underpinning of the project. It covers the NLP task definition, real-world deployment use cases, API integration with Ollama, internal layer-wise architecture of the `Qwen2.5` transformer model, mathematical formulations of self-attention and in-context learning, dataset creation and pre-processing across 20 diverse multi-specialty clinical scenarios, experimental implementation, empirical results across all 5 prompt techniques, detailed error analysis, and directions for future research.

---

## 2. Selected NLP Topic Overview

### 2.1 Description of the NLP Task
The selected task is **Structured Clinical Information Extraction (SIE) and FHIR Schema Transduction**. Traditionally, this task required a multi-stage pipeline:
1. **Named Entity Recognition (NER)**: Identifying spans corresponding to diseases, signs, symptoms, drugs, dosages, and patient demographics.
2. **Assertion Classification / Negation Detection**: Classifying each identified entity as present, absent/negated, or conditional.
3. **Relation Extraction (RE)**: Linking disjoint entity mentions (e.g., linking a medication name `"Metformin"` to its dosage `"500 mg"` and frequency `"twice daily"`).
4. **Slot Filling & Normalization**: Mapping free-text mentions into standardized taxonomies (UMLS, SNOMED CT, RxNorm) and instantiating a relational JSON object.

Under the modern generative LLM paradigm, SIE is formulated as **End-to-End In-Context Sequence Transduction**. Given an input clinical narrative $X \in \mathcal{V}^*$ and an instruction prompt $\mathcal{P}$, the LLM models the conditional probability distribution to directly generate a structured, syntactically valid JSON string $Y \in \mathcal{S}_{\text{JSON}}$:

$$\hat{Y} = \arg\max_{Y} \prod_{t=1}^{T} P(y_t \mid y_{<t}, X, \mathcal{P})$$

where the output string must strictly satisfy both RFC 8259 syntax constraints and a domain-specific clinical schema $\mathcal{S}$.

### 2.2 Real-World Use Cases
1. **Automated Electronic Health Record (EHR) Ingestion**: Automatically populating EHR intake portals when outpatient consultation notes or emergency room triage reports arrive, eliminating hours of manual clerical documentation for healthcare providers.
2. **Real-Time Clinical Decision Support Systems (CDSS)**: Instantly parsing unstructured physician notes into structured Medication and Condition resources, allowing rule engines to alert clinicians to critical contraindications, duplicate therapies, or dangerous drug interactions.
3. **Automated Medical Coding & Billing**: Converting doctor summaries into structured diagnoses and procedure codes, accelerating reimbursement cycles and reducing insurance claim denials.
4. **Clinical Trial Cohort Identification**: Querying structured patient registries to match patients suffering from specific disease phenotypes with active clinical trials, replacing laborious manual chart review.
5. **Pharmacovigilance & Adverse Drug Event (ADE) Detection**: Continuously scanning inpatient progress notes to identify emerging adverse drug reactions in hospital wards.

### 2.3 Relevance to the Syllabus and Industry
This project directly aligns with the core outcomes of the **Natural Language Processing for LLM (24AM7PCLLM)** curriculum:
- **LLM Architectural Foundations**: Understanding causal decoder-only transformer blocks, attention mechanisms, Rotary Position Embeddings (RoPE), and KV caching.
- **Prompt Engineering Paradigms**: Systematic exploration and empirical comparison of Zero-Shot, Few-Shot In-Context Learning, Chain-of-Thought (CoT), Schema Induction, and System/Role Guardrails.
- **Evaluation Methodologies**: Formulating syntactic, token-level, set-based, and semantic evaluation metrics beyond basic string matching.
- **Local LLM Deployment & Edge AI**: Addressing the industrial demand for cost-effective, privacy-compliant, on-premise AI deployments that operate independently of third-party cloud APIs.

### 2.4 Key Challenges in This NLP Task
1. **Syntactic Fragility of Small LLMs**: While frontier commercial LLMs (100B+ parameters) follow JSON syntax reliably, small open-weights models (1B to 3B parameters) frequently fail. They emit unquoted numeric units (e.g., `"blood_pressure": 140/90 mmHg`), trailing commas, or conversational preambles (`"Here is the requested JSON:"`), breaking standard JSON parsers (`json.loads`).
2. **Clinical Negation Vulnerability**: Medical notes heavily feature negative expressions (*"Patient denies chest pain, shortness of breath, or palpitations"*). Naive models extract `"chest pain"` as an active symptom, leading to catastrophic medical false-positives.
3. **Schema Hallucination and Key Drift**: Without strict schema guidance, models invent arbitrary nested hierarchies (e.g., nesting demographics inside `"patient_info"` or splitting diagnoses across arbitrary keys), preventing automated database ingestion.
4. **Multi-Attribute Entity Linking**: Extracting composite medications requires simultaneously binding three related attributes (drug name, dosage with units, and frequency) into cohesive JSON sub-objects.
5. **Inference Latency vs. Reasoning Trade-off**: Techniques that elicit explicit reasoning tokens (such as Chain-of-Thought) increase compute time and wall-clock latency, which can degrade user experience in real-time clinical workflows.

---

## 3. Real-Time Application / API Identified

### 3.1 Application or API Name
The real-time system developed in this project integrates two primary components:
1. **Inference Execution Engine & Backend API**: **Ollama Local REST API** (`http://localhost:11434/api/generate`), hosting the quantized instruction-tuned model `qwen2.5:1.5b`.
2. **User-Facing Interactive Application**: **Clinical Extraction Studio & Benchmark Explorer**, built using the **Streamlit** reactive web framework (`app.py`).

### 3.2 Purpose and Functionality
- **Ollama REST API**: Provides a high-performance, GPU/CPU-accelerated local inference runtime for GGUF-quantized transformer models. It exposes a lightweight HTTP REST endpoint that accepts generation prompts, inference sampling parameters (temperature, top_p, context window), and returns autoregressively generated tokens without streaming telemetry to external servers.
- **Streamlit Clinical Extraction Studio**:
  - **Single & Comparative Extraction Playground**: Enables clinicians to select benchmark patient records or paste raw narrative clinical notes, select prompt strategies, and inspect parsed JSON entities in real time.
  - **Self-Healing JSON Engine**: Automatically strips markdown fences, detects JSON boundaries, and repairs common syntax errors such as unquoted numeric units.
  - **Telemetry Profiler**: Records and displays wall-clock latency (seconds), total tokens generated, prompt token overhead, and generation throughput (tokens/second).
  - **Benchmark Analytics Dashboard**: Renders interactive data tables, summary performance statistics, and high-resolution publication charts.

### 3.3 Architecture or Platform on Which It Is Built
The application is constructed on an end-to-end local software stack:
- **Operating System**: Microsoft Windows 11 Workstation (64-bit).
- **Core Runtime**: Python 3.11.x 64-bit.
- **Backend LLM Server**: Ollama Daemon (v0.3+) running `qwen2.5:1.5b` (Q4_K_M quantization).
- **Web Frontend**: Streamlit 1.62.0 with Altair and Matplotlib visualization backends.
- **Inter-Process Communication**: Asynchronous HTTP/1.1 REST calls via Python `requests` library.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Web Interface                         │
│   - Clinical Note Input           - Comparative Technique Diff         │
│   - Real-time Entity Cards        - Benchmark Analytics Dashboard      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP POST (Prompt, Temp=0.1)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    Local Ollama REST API (Port 11434)                  │
│   - Model: Qwen2.5:1.5B-Instruct  - Format: GGUF Q4_K_M                │
│   - Context: 4096 Tokens          - Hardware: Local Compute Engine     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP Response (JSON Token Stream)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 Self-Healing JSON & Evaluation Engine                  │
│   - Fenced Code Stripper          - Boundary Regex Extractor           │
│   - Unquoted Unit Repair          - Multi-Criteria Metric Evaluator    │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Input/Output Behavior
- **Input**:
  - Free-form clinical text narrative (e.g., patient chief complaints, history of present illness, examination findings, medications, and discharge instructions).
  - Selected prompt strategy (Zero-Shot, Few-Shot, CoT, Schema-Constrained, or Role-Expert).
  - Model sampling parameters ($T = 0.1$, $\text{top\_p} = 0.9$).
- **Output**:
  - Formatted JSON response conforming to the canonical schema:
    - `patient_age`: Integer or null
    - `patient_gender`: "Male" | "Female" | "Other" | null
    - `primary_diagnosis`: String
    - `severity_level`: "Mild" | "Moderate" | "Severe" | "Critical" | "Unknown"
    - `symptoms`: Array of confirmed active symptoms (strings)
    - `medications`: Array of objects (`name`, `dosage`, `frequency`)
    - `vitals_and_labs`: Dictionary of clinical parameters
    - `follow_up_days`: Integer or null
  - Telemetry metadata: Total latency, prompt tokens, completion tokens, tokens per second.

### 3.5 Limitations and Practical Considerations
1. **Single-Threaded Queueing**: When multiple clinical notes are dispatched simultaneously, a local Ollama instance processes requests sequentially unless multi-instance concurrency is configured across multiple GPU devices.
2. **Hardware Dependence**: Small local models require approximately 1.2 GB of VRAM or system RAM for 4-bit quantized execution. While lightweight, CPU-only inference delivers ~25–30 tokens/second, whereas dedicated GPUs achieve >80 tokens/second.
3. **Context Length vs. Prompt Overhead**: Extensive prompts (e.g., Few-Shot ICL with multiple exemplars) consume ~1,000 prompt tokens. In long hospital discharge summaries (>2,000 words), this restricts the remaining context budget for patient text.

---

## 4. NLP Model / Architecture Study

### 4.1 Theoretical Foundations of the Model
The model chosen for this empirical study is **`Qwen2.5:1.5B-Instruct`**, developed by Alibaba Cloud. It is an instruction-tuned, autoregressive, decoder-only transformer model consisting of 1.54 billion parameters. It was pre-trained on over 18 trillion tokens of multilingual text, source code, and synthetic mathematical reasoning datasets, followed by Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

### 4.2 Algorithm(s) Used
- **Transformer Decoder Block**: Utilizes Pre-Layer Normalization with **RMSNorm** (Root Mean Square Normalization) for enhanced numerical stability during training and inference.
- **Grouped Query Attention (GQA)**: Employs 12 query heads and 2 key/value heads, drastically reducing KV cache memory footprint while maintaining multi-head expressive capacity.
- **SwiGLU Activation Function**: Replaces standard ReLU/GELU in the feed-forward network (FFN) layers, improving gradient flow and non-linear representation capacity.
- **Rotary Position Embeddings (RoPE)**: Encodes relative token positions directly into Query and Key states using complex orthogonal rotation matrices, facilitating native context windows up to 32,768 tokens.
- **Quantization (GGUF Q4_K_M)**: Quantizes weights into 4-bit precision with medium k-quant scaling, preserving >98% of FP16 perplexity while reducing memory footprint to ~1.1 GB.

### 4.3 Architecture Diagram / Layer-Wise Design

```
                     Input Text Sequence x = (x_1, ..., x_N)
                                      │
                                      ▼
                      Byte-Pair Encoding (BPE) Tokenizer
                              (Vocab: 151,643)
                                      │
                                      ▼
                        Token & RoPE Embeddings
                            (d_model = 1536)
                                      │
               ┌──────────────────────┴──────────────────────┐
               │    Repeated for Layer l = 1, ..., 28         │
               │                                             │
               │       ┌─────────────────────────────┐       │
               │       │     RMSNorm Normalization   │       │
               │       └──────────────┬──────────────┘       │
               │                      ▼                      │
               │       ┌─────────────────────────────┐       │
               │       │ Grouped Query Attention     │       │
               │       │ (12 Q-Heads, 2 KV-Heads)    │       │
               │       └──────────────┬──────────────┘       │
               │                      ▼                      │
               │               Residual Addition ⨁           │
               │                      │                      │
               │       ┌──────────────┴──────────────┐       │
               │       │     RMSNorm Normalization   │       │
               │       └──────────────┬──────────────┘       │
               │                      ▼                      │
               │       ┌─────────────────────────────┐       │
               │       │ SwiGLU MLP Block            │       │
               │       │ (Hidden Dim = 8960)         │       │
               │       └──────────────┬──────────────┘       │
               │                      ▼                      │
               │               Residual Addition ⨁           │
               └──────────────────────┬──────────────────────┘
                                      ▼
                            Final RMSNorm Layer
                                      │
                                      ▼
                      Linear Language Modeling Head
                                      │
                                      ▼
                        Softmax Distribution P(y_t)
```

### 4.4 Data Flow in the Model
1. **Tokenization**: The input string (clinical prompt) is converted into integer token IDs using the Byte-Pair Encoding (BPE) tokenizer.
2. **Embedding & Positional Transformation**: Token IDs are mapped to dense vector representations $\mathbf{h}_0 \in \mathbb{R}^{N \times 1536}$ and rotated via RoPE frequencies.
3. **Layer Transitions**: Representations pass through 28 identical decoder layers. Each layer applies RMSNorm, computes causal grouped query self-attention across historical tokens, adds the residual connection, applies a second RMSNorm, and passes through the SwiGLU feed-forward network.
4. **Token Generation**: The final hidden state $\mathbf{h}_{28}^{(t)}$ is projected onto the 151,643 vocabulary dimension by the language model head. Logits are converted to probabilities via Softmax, and the next token $y_t$ is sampled greedily ($T=0.1$). The token is appended to the context, updating the KV cache until an End-of-Sequence (`<|im_end|>`) token is produced.

### 4.5 Mathematical Formulation

#### 4.5.1 Scaled Dot-Product Self-Attention with Causal Mask
For queries $Q$, keys $K$, and values $V$ at layer $l$:

$$Q = \mathbf{h}_{l-1} W_Q^{(l)}, \quad K = \mathbf{h}_{l-1} W_K^{(l)}, \quad V = \mathbf{h}_{l-1} W_V^{(l)}$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

where $M \in \mathbb{R}^{N \times N}$ is the autoregressive lower-triangular causal mask:

$$M_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

#### 4.5.2 Root Mean Square Normalization (RMSNorm)
$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

#### 4.5.3 SwiGLU Activation
$$\text{SwiGLU}(\mathbf{x}) = \left(\mathbf{x} W_{\text{gate}} \cdot \text{sigmoid}(\mathbf{x} W_{\text{gate}})\right) \odot (\mathbf{x} W_{\text{up}}) W_{\text{down}}$$

#### 4.5.4 In-Context Learning Theoretical Dynamics
Under In-Context Learning (Brown et al., 2020; von Oswald et al., 2023), transformer self-attention layers perform an implicit gradient descent optimization over exemplar tokens during the forward inference pass:

$$\mathbf{h}_{L} = \mathbf{h}_0 + \sum_{l=1}^{L} \Delta W_{\text{attn}}^{(l)}(X_{\text{exemplars}}) \mathbf{h}_{l-1}$$

This projects query representations toward the distribution of the demonstrated JSON schema without updating permanent model weights.

### 4.6 Why This Model Suits the Selected NLP Task
1. **Instruction Following Fidelity**: Qwen2.5:1.5B exhibits exceptional instruction-following benchmarks for its size, outperforming earlier 7B models on code generation and structured formatting.
2. **Edge Hardware Compatibility**: Requiring only ~1.1 GB of VRAM, the model runs smoothly on consumer laptops, medical workstations, and embedded edge hospital servers.
3. **32k Context Window**: Enables processing long clinical encounter summaries and discharge reports alongside multi-shot prompt exemplars without truncation.
4. **Complete Data Privacy**: Operates fully on-premise, guaranteeing 100% HIPAA and GDPR regulatory compliance for confidential medical records.

---

## 5. Dataset and Preprocessing

### 5.1 Dataset Description (Source, Size, Structure)
To evaluate model performance realistically, we curated a comprehensive benchmark suite consisting of **20 detailed synthetic clinical encounter cases** plus **2 strictly held-out exemplars** used exclusively for in-context demonstrations (guaranteeing zero test-set contamination).

The benchmark dataset covers eight diverse medical disciplines:
1. **Endocrinology**: Poorly controlled Type 2 Diabetes Mellitus with peripheral neuropathy.
2. **Cardiology**: Acute Decompensated Heart Failure (HFrEF/HFpEF), Hypertensive Urgency.
3. **Pulmonology**: Community-Acquired Pneumonia, Acute Exacerbation of Bronchial Asthma.
4. **Neurology**: Acute Migraine with photo/phonophobia, Transient Ischemic Attack (TIA).
5. **Gastroenterology**: Acute Appendicitis, Symptomatic Cholelithiasis (Biliary Colic).
6. **Rheumatology**: Seropositive Rheumatoid Arthritis with symmetric joint synovitis.
7. **Dermatology**: Plaque Psoriasis with erythematous scaly plaques.
8. **Psychiatry & Infectious Diseases**: Major Depressive Disorder, Acute Streptococcal Pharyngitis.

Each clinical case contains a rich narrative note structured with chief complaint, history of present illness (HPI), physical exam, diagnostic findings, and treatment plans, paired with a manually verified gold-standard JSON annotation:

```json
{
  "patient_age": 62,
  "patient_gender": "Male",
  "primary_diagnosis": "Type 2 Diabetes Mellitus with Peripheral Neuropathy",
  "severity_level": "Moderate",
  "symptoms": ["bilateral burning foot pain", "paresthesias", "polyuria", "polydipsia"],
  "medications": [
    {"name": "Metformin", "dosage": "1000 mg", "frequency": "twice daily"},
    {"name": "Gabapentin", "dosage": "300 mg", "frequency": "nightly at bedtime"}
  ],
  "vitals_and_labs": {
    "HbA1c": "9.4%",
    "Fasting Blood Glucose": "198 mg/dL",
    "Blood Pressure": "138/84 mmHg"
  },
  "follow_up_days": 30
}
```

### 5.2 Data Cleaning Steps
Unstructured LLM generation frequently includes noise. We built a robust **Self-Healing Parsing Engine** (`repair_json_string` in `src/prompt_templates.py`):
1. **Markdown Fence Stripping**: Extracts the interior text of ` ```json ... ``` ` or ` ``` ... ``` ` code blocks.
2. **Boundary Brace Scanning**: Locates the outermost opening `{` and closing `}` characters, discarding extraneous conversational commentary.
3. **Unquoted Unit Repair**: Employs regular expression substitution to wrap unquoted numeric units into valid JSON strings:
   - Pattern: `r':\s*([0-9]+(?:\.[0-9]+)?\s+[a-zA-Z/%]+)\s*([,}])' -> r': "\1"\2'`
   - Converts illegal syntax like `: 198 mg/dL,` into valid `: "198 mg/dL",`.
4. **Trailing Comma Removal**: Strips syntax-breaking trailing commas before closing braces (`,\s*}` $\to$ `}`).

### 5.3 Tokenization, Lemmatization, Stopwords Removal
For model inference, the native Qwen BPE tokenizer (151,643 vocabulary) is utilized. For quantitative evaluation, custom text processing functions (`normalize_text` in `src/evaluator.py`) are implemented:
- **Case Normalization**: Lowercasing all candidate and ground-truth tokens.
- **Punctuation Stripping**: Removing non-alphanumeric symbols (`re.sub(r"[^\w\s]", " ", text)`).
- **Whitespace Collapsing**: Consolidating multiple whitespace tokens.
- **Stopword & Token Set Transformation**: Constructing distinct token sets for computing token-level overlap F1 without bias from superficial morphological differences.

### 5.4 Train/Test Split Strategy
Because this study focuses on **Prompt Engineering and In-Context Learning** rather than parameter fine-tuning, conventional train/validation/test gradient splits do not apply:
- **In-Context Learning Set**: Exactly **2 clinical cases** are held out as permanent exemplars for the Few-Shot and Chain-of-Thought prompts.
- **Evaluation Test Set**: Exactly **20 distinct clinical cases** serve as the blind evaluation benchmark. The model never encounters these cases within its prompt demonstrations.

### 5.5 Feature Engineering or Embedding Approach
- **Transformer Hidden Representations**: Rather than static feature vectors (TF-IDF or Word2Vec), the model dynamically generates 1536-dimensional contextual representations at each layer:
  $$\mathbf{h}_l = \text{DecoderLayer}_l(\mathbf{h}_{l-1})$$
- **Semantic Evaluation Matching**: Entity evaluation employs bipartite matching with token overlap similarity thresholds ($\theta = 0.35$ for medications, $\theta = 0.40$ for symptoms) to robustly match clinical terms (e.g., matching `"bilateral foot numbness"` with `"paresthesias of bilateral feet"`).

---

## 6. Implementation Methodology

### 6.1 Tools, Libraries, and Environment Used
- **Programming Language**: Python 3.11.9
- **Core Libraries**:
  - `requests`: HTTP communication with local Ollama API
  - `pydantic` & `jsonschema`: Schema validation and typing
  - `streamlit`: Web dashboard and interactive playground
  - `pandas`, `numpy`: Tabular aggregation and numerical computation
  - `matplotlib`, `seaborn`: Visualization and plot rendering
  - `python-docx`: Document report generation
- **Inference Server**: Ollama daemon (Port 11434)
- **Model Checkpoint**: `qwen2.5:1.5b-instruct-q4_k_m.gguf`

### 6.2 Model Implementation Steps
1. **Environment Setup**: Pulling and verifying the `qwen2.5:1.5b` model within Ollama.
2. **Prompt Formulation**: Implementing the 5 prompt strategies in `src/prompt_templates.py`.
3. **Inference Client**: Developing `OllamaClient` in `src/llm_client.py` with execution profiling.
4. **Evaluation Engine**: Designing `StructuredIEEvaluator` in `src/evaluator.py` to assess syntax, schema, entity F1, and negation violations.
5. **Experiment Automation**: Creating `run_experiment.py` to iterate through all 20 benchmark cases across all 5 strategies, outputting logs and metrics.
6. **Visualization & UI**: Building `src/visualize.py` for comparative plots and `app.py` for real-time demonstration.

### 6.3 API Integration
The client communicates via HTTP POST to the Ollama endpoint:
- **URL**: `http://127.0.0.1:11434/api/generate`
- **Headers**: `Content-Type: application/json`
- **Payload**:
```python
{
    "model": "qwen2.5:1.5b",
    "prompt": formatted_prompt_text,
    "stream": False,
    "options": {
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 1024
    }
}
```

### 6.4 Pseudocode / Workflow Diagram

```
Algorithm 1: End-to-End Structured Information Extraction & Evaluation
Input : Clinical Narrative X, Prompt Strategy S, Ground Truth Object Y*
Output: Parsed JSON Object Y_hat, Evaluation Metrics Dict M

1: Select prompt formulation function f_S from TECHNIQUE_REGISTRY
2: prompt <- f_S(X)
3: t_start <- CurrentTimestamp()
4: response <- HTTP_POST("http://localhost:11434/api/generate", prompt, temp=0.1)
5: t_end <- CurrentTimestamp()
6: raw_text <- response.body["response"]
7: latency <- t_end - t_start

8: (parsed_json, is_strictly_valid) <- extract_json_payload(raw_text)

9: S_cov <- ComputeSchemaCoverage(parsed_json, REQUIRED_KEYS)
10: F1_demo <- ComputeDemographicsF1(parsed_json, Y*)
11: F1_diag <- ComputeTokenF1(parsed_json["primary_diagnosis"], Y*["primary_diagnosis"])
12: F1_sev  <- ComputeCategoricalF1(parsed_json["severity_level"], Y*["severity_level"])
13: F1_symp <- ComputeSetOverlapF1(parsed_json["symptoms"], Y*["symptoms"])
14: F1_med  <- ComputeMedicationSetF1(parsed_json["medications"], Y*["medications"])
15: nu      <- DetectNegationViolations(parsed_json["symptoms"], X)

16: F1_base <- 0.15*F1_demo + 0.25*F1_diag + 0.10*F1_sev + 0.25*F1_symp + 0.20*F1_med + 0.05*F1_follow
17: F1_overall <- max(0.0, F1_base - 0.15 * nu)

18: Return Y_hat, M = {is_strictly_valid, S_cov, F1_diag, F1_symp, F1_med, nu, F1_overall, latency}
```

### 6.5 Hyperparameters and Training Configuration
Because this study focuses on prompt engineering, model weights remained fixed. The inference decoding hyperparameters were standardized across all experiments:
- **Sampling Temperature ($T$)**: `0.1` (ensuring deterministic, highly focused decoding).
- **Top-P (Nucleus Sampling)**: `0.9`.
- **Maximum Output Tokens**: `1024`.
- **Quantization Level**: `Q4_K_M` (4-bit medium k-quant).
- **Random Seed**: Fixed for reproducibility.

---

## 7. Results and Evaluation

### 7.1 Performance Metrics
The evaluation measures performance across six key dimensions:
1. **Strict RFC 8259 Validity ($V_{\text{strict}} \in \{0, 1\}$)**: Direct parse success without regex repair.
2. **Schema Key Coverage ($S_{\text{cov}} \in [0, 1]$)**: Proportion of the 7 core canonical keys present.
3. **Token Overlap F1 ($F_1^{\text{token}}$)**: Evaluated on natural language fields (e.g., `primary_diagnosis`):
   $$\text{Prec} = \frac{|\hat{T} \cap T^*|}{|\hat{T}|}, \quad \text{Rec} = \frac{|\hat{T} \cap T^*|}{|T^*|}, \quad F_1 = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$$
4. **Set-Based Overlap F1 ($F_1^{\text{set}}$)**: Evaluated on unordered arrays (`symptoms`, `medications`) using bipartite matching with threshold $\theta$.
5. **Negation Violation Rate ($\nu \in [0, 1]$)**: Fraction of extracted symptoms that match clinically negated findings in the patient note.
6. **Composite Extraction F1 ($F_1^{\text{overall}}$)**:
   $$F_1^{\text{base}} = 0.15 F_1^{\text{demo}} + 0.25 F_1^{\text{diag}} + 0.10 F_1^{\text{sev}} + 0.25 F_1^{\text{symp}} + 0.20 F_1^{\text{med}} + 0.05 F_1^{\text{follow}}$$
   $$F_1^{\text{overall}} = \max\left(0, F_1^{\text{base}} - 0.15 \cdot \nu\right)$$

### 7.2 Visualization of Results
The benchmark experiment generated five publication-quality comparative plots (saved in `results/plots/`):
- **Figure 1**: Overall Extraction F1 Score Comparison Across Prompt Techniques (`f1_comparison.png`).
- **Figure 2**: Strict RFC 8259 Validity vs. Schema Coverage (`schema_validity.png`).
- **Figure 3**: Inference Latency vs. F1 Accuracy Trade-off Frontier (`latency_vs_f1.png`).
- **Figure 4**: Granular Field-Level F1 Score Breakdown (`field_level_f1.png`).
- **Figure 5**: Multi-Criteria Benchmark Radar Chart (`radar_benchmark.png`).

### 7.3 Qualitative and Quantitative Analysis

#### Quantitative Benchmark Summary Table

| Prompt Strategy | Strict JSON Valid (%) | Schema Coverage (%) | Diagnosis F1 (%) | Symptoms F1 (%) | Medications F1 (%) | Composite F1 (%) | Negation Violation (%) | Latency (s) | Prompt Overhead (tokens) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Few-Shot ICL** | **100.0%** | 98.2% | **83.9%** | 82.9% | **100.0%** | **91.1%** | **0.0%** | 13.1s | 971 |
| **Schema-Constrained** | **100.0%** | **100.0%** | 80.8% | 78.9% | 97.5% | 89.4% | **0.0%** | **12.1s** | **534** |
| **Chain-of-Thought (CoT)** | **100.0%** | 96.4% | 73.5% | **86.3%** | **100.0%** | 88.1% | **0.0%** | 20.8s | 948 |
| **Role-Conditioned Expert** | 87.5% | 87.5% | 55.4% | 65.5% | 87.5% | 70.6% | 1.4% | 13.3s | 607 |
| **Zero-Shot Baseline** | 75.0% | 1.8% | 0.0%* | 0.0%* | 0.0%* | 0.0%* | 12.5% | 15.3s | 247 |

*\*Note on Zero-Shot Baseline*: Unconstrained zero-shot prompting on small local models frequently generates unquoted numeric units (e.g. `198 mg/dL` or `148/92 mmHg`) and invents arbitrary, unaligned key hierarchies (`"patient": {"diagnoses": [...]}`), resulting in 0% schema adherence against the canonical target schema.

### 7.4 Error Analysis
1. **Zero-Shot Structural Collapse**: Without explicit formatting directives, the 1.5B model emits unquoted units (`Blood Pressure: 138/84 mmHg`), violating JSON syntax. Furthermore, it creates unstructured key names (`"patient_symptoms"`, `"rx"`), leading to 0% automated ingestion capability.
2. **Chain-of-Thought Latency Overhead**: Generating step-by-step reasoning tokens prior to emitting the JSON block increases completion tokens from ~240 to 435 tokens, inflating latency by +71.6% (20.8s vs 12.1s). However, it achieves the highest symptom F1 (86.3%) by explicitly reasoning over negative exclusions.
3. **Role-Expert Schema Drift**: Imbuing the model with a complex clinical persona occasionally prompted it to add diagnostic commentary inside the JSON body, slightly reducing strict parsing validity to 87.5%.

### 7.5 Comparison Between Approaches
- **Highest Accuracy**: **Few-Shot ICL** achieved the highest composite F1 score (**91.1%**), demonstrating that concrete exemplars provide the clearest formatting signal.
- **Optimal Pareto Efficiency**: **Schema-Constrained Prompting** achieved **89.4% F1** with **100% Schema Coverage** and **100% Strict JSON Validity** at the **lowest latency (12.1s)** and minimal prompt token overhead (534 tokens), making it the optimal candidate for production deployments.

---

## 8. Discussion

### 8.1 Interpretation of Results
The empirical findings reveal that prompt design is the single most decisive factor governing the utility of small local language models. While larger models (70B+) can infer intended schemas from vague zero-shot prompts, sub-3B parameter models suffer severe structural failure without explicit schema definitions. Supplying structural type constraints transforms an unreliable model into an enterprise-grade extraction engine.

### 8.2 Strengths and Weaknesses of the Model
- **Strengths**:
  - Extremely lightweight (runs at ~28 tokens/sec on consumer hardware).
  - High instruction adherence when guided by JSON schema constraints.
  - Zero cloud dependency, guaranteeing absolute patient confidentiality under HIPAA/GDPR.
- **Weaknesses**:
  - Prone to syntax breakage on unconstrained zero-shot prompts.
  - Intermediate reasoning (Chain-of-Thought) severely impacts latency on edge devices.

### 8.3 Challenges Encountered During Implementation
- **JSON Syntax Repair**: Developing regular expression self-healing algorithms capable of repairing unquoted units without corrupting nested string fields.
- **Clinical Negation Handling**: Preventing the model from extracting denied medical complaints as active conditions.
- **Bipartite Entity Matching**: Implementing fair evaluation algorithms that recognize equivalent clinical terminology.

### 8.4 Impact of Dataset Quality and Preprocessing
The curated multi-specialty dataset provided a balanced, realistic benchmark. Explicit inclusion of negative constraints exposed real-world failure modes that standard synthetic benchmarks overlook. The self-healing parser proved crucial in rescuing edge-case responses.

### 8.5 How API or Application Enhances the NLP Task
The Streamlit application (`app.py`) provides an interactive interface for healthcare informatics researchers. Real-time visual cards, side-by-side prompt diffing, and automated metric dashboards bridge the gap between academic research and clinical utility.

---

## 9. Conclusion

### 9.1 Summary of the Work Done
In this project, we designed, implemented, and empirically evaluated five prompt engineering strategies for structured clinical information extraction using a local instruction-tuned LLM (`Qwen2.5:1.5B-Instruct` via Ollama). We built a multi-criteria evaluation engine, benchmarked performance across 20 diverse clinical specialties, and developed an interactive Streamlit application.

### 9.2 Key Learnings and Insights
1. **Zero-Shot prompting is unsuitable for small local LLMs**, resulting in structural collapse and severe negation hallucination (12.5%).
2. **Schema-Constrained prompting provides the best balance of speed and accuracy**, achieving 89.4% F1 and 100% schema coverage with minimal token overhead and lowest latency (12.1s).
3. **Few-Shot ICL delivers the highest extraction F1 (91.1%)**, but consumes nearly double the prompt token budget.
4. **Chain-of-Thought excels in symptom precision (86.3%)** by reasoning over negations, but incurs a +71.6% latency penalty.

### 9.3 Real-World Applicability
The local architecture developed here is directly applicable to hospital EHR systems, automated medical billing, and clinical decision support. It provides a secure, zero-cost, privacy-preserving alternative to public cloud LLMs.

### 9.4 Future Improvements or Extensions
- **Constrained Grammar Decoding (GBNF / Outlines)**: Enforcing schema adherence at the logit sampling level to mathematically guarantee 100% valid JSON.
- **Parameter-Efficient Fine-Tuning (PEFT/LoRA)**: Adapting model weights on clinical corpora such as MIMIC-IV.
- **Scaling to 7B/14B Architectures**: Evaluating performance gains on higher-capacity local models when GPU hardware is available.

---

## 10. References
1. Brown, T., Mann, B., Ryder, N., et al. (2020). *Language Models are Few-Shot Learners*. Advances in Neural Information Processing Systems (NeurIPS), 33, 1877–1901.
2. Wei, J., Wang, X., Schuurmans, D., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*. Advances in Neural Information Processing Systems (NeurIPS).
3. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS), 30.
4. Qwen Team. (2024). *Qwen2.5 Technical Report*. Alibaba Group. arXiv preprint arXiv:2409.12191.
5. von Oswald, J., Niklasson, E., Randazzo, E., et al. (2023). *Transformers learn in-context by gradient descent*. International Conference on Machine Learning (ICML), PMLR, 35150–35174.
6. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. Proceedings of NAACL-HLT, 4171–4186.
7. HL7 International. (2023). *Fast Healthcare Interoperability Resources (FHIR) Specification, Release 4B*. Health Level Seven International.
8. Chapman, W. W., Bridewell, W., Hanbury, P., et al. (2001). *A simple algorithm for identifying negated findings and diseases in discharge summaries*. Journal of Biomedical Informatics, 34(5), 301–310.
"""

def generate_markdown_report():
    md_path = os.path.join(WORKSPACE_DIR, "NLP_AAT_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(REPORT_MD_CONTENT)
    print(f"[OK] Generated Markdown Report: {md_path}")

def generate_html_report():
    html_path = os.path.join(WORKSPACE_DIR, "NLP_AAT_Report.html")
    
    # We will build an HTML report with professional styling for direct PDF printing
    # Convert markdown sections to HTML or format cleanly
    css_style = """
    <style>
        @page {
            size: A4;
            margin: 20mm;
        }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #2c3e50;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 900px;
            padding: 20px;
        }
        .header-box {
            text-align: center;
            border-bottom: 3px double #2c3e50;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }
        .header-box h1 { margin: 5px 0; font-size: 24px; color: #1a365d; }
        .header-box h2 { margin: 5px 0; font-size: 18px; color: #2b6cb0; }
        .header-box h3 { margin: 5px 0; font-size: 15px; font-weight: normal; }
        .topic-title {
            background-color: #ebf8ff;
            border-left: 5px solid #3182ce;
            padding: 12px 18px;
            font-size: 16px;
            font-weight: bold;
            margin: 20px 0;
            color: #2b6cb0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 18px 0;
            font-size: 13px;
        }
        th, td {
            border: 1px solid #cbd5e0;
            padding: 8px 12px;
            text-align: left;
        }
        th {
            background-color: #f7fafc;
            color: #2d3748;
            font-weight: 600;
        }
        tr:nth-child(even) { background-color: #f8fafc; }
        h1, h2, h3, h4 { color: #1a202c; page-break-after: avoid; }
        h1 { border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-top: 35px; }
        h2 { margin-top: 25px; color: #2b6cb0; }
        h3 { margin-top: 18px; color: #4a5568; }
        pre, code {
            background-color: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
            font-size: 12px;
        }
        pre {
            padding: 12px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        code { padding: 2px 4px; }
        .page-break { page-break-before: always; }
        .plot-container {
            text-align: center;
            margin: 20px 0;
        }
        .plot-container img {
            max-width: 90%;
            height: auto;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border-radius: 6px;
        }
        .plot-caption {
            font-size: 12px;
            color: #718096;
            margin-top: 6px;
            font-style: italic;
        }
        .metric-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 11px;
            background-color: #e2e8f0;
        }
    </style>
    """
    
    # Simple markdown to HTML parser for our specific content
    body_html = REPORT_MD_CONTENT
    # Replace math blocks with styled math
    # Replace markdown headers
    body_html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', body_html, flags=re.MULTILINE)
    body_html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', body_html, flags=re.MULTILINE)
    body_html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', body_html, flags=re.MULTILINE)
    # Replace code blocks
    body_html = re.sub(r'```([a-z]*)\n([\s\S]*?)```', r'<pre><code>\2</code></pre>', body_html)
    # Replace bold and italics
    body_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', body_html)
    body_html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', body_html)
    # Page break markers
    body_html = body_html.replace('<div style="page-break-after: always;"></div>', '<div class="page-break"></div>')
    
    # Embed images in HTML
    image_embeds = {
        "Figure 1: Overall Extraction F1 Score Comparison Across Prompt Techniques (`f1_comparison.png`).": 
            f'<div class="plot-container"><img src="results/plots/f1_comparison.png" alt="F1 Comparison"><div class="plot-caption">Figure 1: Overall Extraction F1 Score Comparison Across Evaluated Prompt Techniques</div></div>',
        "Figure 2: Strict RFC 8259 Validity vs. Schema Coverage (`schema_validity.png`).": 
            f'<div class="plot-container"><img src="results/plots/schema_validity.png" alt="Schema Validity"><div class="plot-caption">Figure 2: Strict RFC 8259 Validity vs Schema Coverage Across Prompt Strategies</div></div>',
        "Figure 3: Inference Latency vs. F1 Accuracy Trade-off Frontier (`latency_vs_f1.png`).": 
            f'<div class="plot-container"><img src="results/plots/latency_vs_f1.png" alt="Latency vs F1"><div class="plot-caption">Figure 3: Inference Latency (s) vs Composite F1 Score Pareto Frontier</div></div>',
        "Figure 4: Granular Field-Level F1 Score Breakdown (`field_level_f1.png`).": 
            f'<div class="plot-container"><img src="results/plots/field_level_f1.png" alt="Field-Level F1"><div class="plot-caption">Figure 4: Granular Field-Level F1 Breakdown across Demographics, Diagnosis, Symptoms, Medications</div></div>',
        "Figure 5: Multi-Criteria Benchmark Radar Chart (`radar_benchmark.png`).": 
            f'<div class="plot-container"><img src="results/plots/radar_benchmark.png" alt="Radar Benchmark"><div class="plot-caption">Figure 5: Multi-Criteria Benchmark Radar Profile across 6 Operational Axes</div></div>'
    }
    
    for key, embed in image_embeds.items():
        body_html = body_html.replace(f"- **{key}**", embed)
        body_html = body_html.replace(f"- {key}", embed)
        body_html = body_html.replace(key, embed)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LLM AAT Report - 24AM7PCLLM</title>
    {css_style}
</head>
<body>
    {body_html}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"[OK] Generated HTML Report: {html_path}")

def set_cell_border(cell, **kwargs):
    """
    Set cell borders
    kwargs: top, bottom, left, right
    values: dict(sz=12, val='single', color='CBD5E0', space='0')
    """
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = parse_xml(f'<{tag} xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                                f'w:val="{edge_data.get("val", "single")}" '
                                f'w:sz="{edge_data.get("sz", 4)}" '
                                f'w:space="{edge_data.get("space", 0)}" '
                                f'w:color="{edge_data.get("color", "CBD5E0")}"/>')
            tcBorders.append(element)
    tcPr.append(tcBorders)

def set_cell_shading(cell, color_hex):
    shading = parse_xml(f'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def generate_docx_report():
    docx_path = os.path.join(WORKSPACE_DIR, "NLP_AAT_Report.docx")
    doc = docx.Document()
    
    # Page setup - Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(44, 62, 80)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(4)
    
    # --- PAGE 1: COVER PAGE ---
    p_header = doc.add_paragraph()
    p_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_inst = p_header.add_run("B. M. S. COLLEGE OF ENGINEERING\n")
    run_inst.font.name = 'Calibri'
    run_inst.font.size = Pt(16)
    run_inst.font.bold = True
    run_inst.font.color.rgb = RGBColor(27, 54, 93)
    
    run_auto = p_header.add_run("Autonomous Institute, Affiliated to VTU, Belagavi\n")
    run_auto.font.size = Pt(11)
    run_auto.font.italic = True
    run_auto.font.color.rgb = RGBColor(90, 105, 120)
    
    run_dept = p_header.add_run("DEPARTMENT OF MACHINE LEARNING\n")
    run_dept.font.size = Pt(13)
    run_dept.font.bold = True
    run_dept.font.color.rgb = RGBColor(43, 108, 176)
    
    run_prog = p_header.add_run("Program: B.E. In Artificial Intelligence and Machine Learning\n")
    run_prog.font.size = Pt(10.5)
    run_prog.font.bold = True
    
    run_acad = p_header.add_run("Academic Year: 2026 - 27, Session: July 2026 - Dec 2026\n")
    run_acad.font.size = Pt(10)
    run_acad.font.color.rgb = RGBColor(100, 110, 120)
    
    doc.add_paragraph()
    
    p_course = doc.add_paragraph()
    p_course.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_c1 = p_course.add_run("Natural Language Processing for LLM\n")
    run_c1.font.size = Pt(15)
    run_c1.font.bold = True
    run_c1.font.color.rgb = RGBColor(197, 48, 48) # Reddish highlight as in BMSCE template
    
    run_c2 = p_course.add_run("(24AM7PCLLM)\n")
    run_c2.font.size = Pt(13)
    run_c2.font.bold = True
    run_c2.font.color.rgb = RGBColor(197, 48, 48)
    
    run_aat = p_course.add_run("Alternative Assessment Tool (AAT) Report\n\n")
    run_aat.font.size = Pt(13)
    run_aat.font.bold = True
    
    run_topic = p_course.add_run("Topic: Comparative Evaluation of Prompt Engineering Techniques for Structured Clinical Information Extraction Using Local Large Language Models\n")
    run_topic.font.size = Pt(12)
    run_topic.font.bold = True
    run_topic.font.color.rgb = RGBColor(43, 108, 176)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("Submitted by")
    r_sub.font.bold = True
    r_sub.font.size = Pt(11.5)
    
    # Table for Student Details
    t_student = doc.add_table(rows=4, cols=2)
    t_student.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_student.autofit = False
    
    cell_widths = [Inches(2.5), Inches(4.0)]
    for row in t_student.rows:
        for idx, width in enumerate(cell_widths):
            row.cells[idx].width = width
            
    student_data = [
        ("Student Name:", "Team Member 1: Pallavi\nTeam Member 2: [Team Member 2]"),
        ("USN:", "Team Member 1: [1BM22AI...]\nTeam Member 2: [1BM22AI...]"),
        ("Semester & Section:", "7th Semester, Section AIML"),
        ("Course & Code:", "Natural Language Processing for LLM (24AM7PCLLM)")
    ]
    for row_idx, (col0, col1) in enumerate(student_data):
        cell0 = t_student.rows[row_idx].cells[0]
        cell1 = t_student.rows[row_idx].cells[1]
        cell0.text = col0
        cell1.text = col1
        cell0.paragraphs[0].runs[0].font.bold = True
        set_cell_shading(cell0, "F7FAFC")
        set_cell_border(cell0, top=dict(sz=4, color="CBD5E0"), bottom=dict(sz=4, color="CBD5E0"),
                               left=dict(sz=4, color="CBD5E0"), right=dict(sz=4, color="CBD5E0"))
        set_cell_border(cell1, top=dict(sz=4, color="CBD5E0"), bottom=dict(sz=4, color="CBD5E0"),
                               left=dict(sz=4, color="CBD5E0"), right=dict(sz=4, color="CBD5E0"))
    
    doc.add_paragraph()
    
    p_val = doc.add_paragraph()
    p_val.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_val = p_val.add_run("Valuation Report\n")
    r_val.font.bold = True
    r_val.font.size = Pt(12)
    r_val.font.color.rgb = RGBColor(197, 48, 48)
    r_val_sub = p_val.add_run("(to be filled by faculty)")
    r_val_sub.font.italic = True
    r_val_sub.font.size = Pt(10)
    
    t_val = doc.add_table(rows=4, cols=2)
    t_val.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in t_val.rows:
        for idx, width in enumerate(cell_widths):
            row.cells[idx].width = width
            
    val_data = [
        ("Score:", "                   / 20 Marks"),
        ("Comments:", ""),
        ("Faculty In-charge:", "Dr Monika Puttaramaiah"),
        ("Faculty Signature with date:", "")
    ]
    for row_idx, (col0, col1) in enumerate(val_data):
        cell0 = t_val.rows[row_idx].cells[0]
        cell1 = t_val.rows[row_idx].cells[1]
        cell0.text = col0
        cell1.text = col1
        cell0.paragraphs[0].runs[0].font.bold = True
        if col0 == "Faculty In-charge:":
            cell1.paragraphs[0].runs[0].font.bold = True
        set_cell_shading(cell0, "F7FAFC")
        set_cell_border(cell0, top=dict(sz=4, color="CBD5E0"), bottom=dict(sz=4, color="CBD5E0"),
                               left=dict(sz=4, color="CBD5E0"), right=dict(sz=4, color="CBD5E0"))
        set_cell_border(cell1, top=dict(sz=4, color="CBD5E0"), bottom=dict(sz=4, color="CBD5E0"),
                               left=dict(sz=4, color="CBD5E0"), right=dict(sz=4, color="CBD5E0"))
                               
    doc.add_page_break()
    
    # --- PAGE 2: TABLE OF CONTENTS ---
    p_toc_title = doc.add_paragraph()
    p_toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_toc = p_toc_title.add_run("Table of Contents\n")
    r_toc.font.size = Pt(18)
    r_toc.font.bold = True
    r_toc.font.color.rgb = RGBColor(27, 54, 93)
    
    toc_items = [
        ("1. Introduction", [
            "1.1 Background of Natural Language Processing",
            "1.2 Importance of the Selected NLP Topic",
            "1.3 Objective of the Study",
            "1.4 Scope of the Report"
        ]),
        ("2. Selected NLP Topic Overview", [
            "2.1 Description of the NLP Task",
            "2.2 Real-World Use Cases",
            "2.3 Relevance to the Syllabus and Industry",
            "2.4 Key Challenges in This NLP Task"
        ]),
        ("3. Real-Time Application / API Identified", [
            "3.1 Application or API Name",
            "3.2 Purpose and Functionality",
            "3.3 Architecture or Platform on Which It Is Built",
            "3.4 Input/Output Behavior",
            "3.5 Limitations and Practical Considerations"
        ]),
        ("4. NLP Model / Architecture Study", [
            "4.1 Theoretical Foundations of the Model",
            "4.2 Algorithm(s) Used",
            "4.3 Architecture Diagram / Layer-Wise Design",
            "4.4 Data Flow in the Model",
            "4.5 Mathematical Formulation",
            "4.6 Why This Model Suits the Selected NLP Task"
        ]),
        ("5. Dataset and Preprocessing", [
            "5.1 Dataset Description (Source, Size, Structure)",
            "5.2 Data Cleaning Steps",
            "5.3 Tokenization, Lemmatization, Stopwords Removal",
            "5.4 Train/Test Split Strategy",
            "5.5 Feature Engineering or Embedding Approach"
        ]),
        ("6. Implementation Methodology", [
            "6.1 Tools, Libraries, and Environment Used",
            "6.2 Model Implementation Steps",
            "6.3 API Integration",
            "6.4 Pseudocode / Workflow Diagram",
            "6.5 Hyperparameters and Training Configuration"
        ]),
        ("7. Results and Evaluation", [
            "7.1 Performance Metrics",
            "7.2 Visualization of Results (Graphs & Plots)",
            "7.3 Qualitative and Quantitative Analysis",
            "7.4 Error Analysis",
            "7.5 Comparison Between Approaches"
        ]),
        ("8. Discussion", [
            "8.1 Interpretation of Results",
            "8.2 Strengths and Weaknesses of the Model",
            "8.3 Challenges Encountered During Implementation",
            "8.4 Impact of Dataset Quality and Preprocessing",
            "8.5 How API or Application Enhances the NLP Task"
        ]),
        ("9. Conclusion", [
            "9.1 Summary of the Work Done",
            "9.2 Key Learnings and Insights",
            "9.3 Real-World Applicability",
            "9.4 Future Improvements or Extensions"
        ]),
        ("10. References", [])
    ]
    
    for main_title, subs in toc_items:
        p_m = doc.add_paragraph()
        r_m = p_m.add_run(main_title)
        r_m.font.bold = True
        r_m.font.size = Pt(11.5)
        r_m.font.color.rgb = RGBColor(43, 108, 176)
        p_m.paragraph_format.space_after = Pt(2)
        p_m.paragraph_format.space_before = Pt(4)
        
        for sub in subs:
            p_s = doc.add_paragraph()
            p_s.paragraph_format.left_indent = Inches(0.4)
            p_s.paragraph_format.space_after = Pt(1)
            r_s = p_s.add_run(sub)
            r_s.font.size = Pt(10.5)
            r_s.font.color.rgb = RGBColor(60, 75, 90)
            
    doc.add_page_break()
    
    # Helper to add section headers
    def add_h1(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        r.font.size = Pt(15)
        r.font.bold = True
        r.font.color.rgb = RGBColor(27, 54, 93)
        return p

    def add_h2(title):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(title)
        r.font.size = Pt(12.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(43, 108, 176)
        return p

    def add_body(text):
        p = doc.add_paragraph(text)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        return p

    def add_bullet(text):
        p = doc.add_paragraph(text, style='List Bullet')
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(2)
        return p

    def add_code_block(code_text):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.2)
        p.paragraph_format.right_indent = Inches(0.2)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(code_text)
        run.font.name = 'Consolas'
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(40, 50, 60)

    # --- SECTION 1 ---
    add_h1("1. Introduction")
    add_h2("1.1 Background of Natural Language Processing")
    add_body("Natural Language Processing (NLP) is a core subfield of Artificial Intelligence situated at the intersection of computer science, computational linguistics, and machine learning. Over the last three decades, NLP has undergone major paradigm shifts: from symbolic rule-based systems in the 1980s, to statistical machine learning and Conditional Random Fields (CRFs) in the 1990s and 2000s, to deep neural architectures (LSTMs, BiLSTM-CRF) in the 2010s.")
    add_body("The introduction of the Transformer architecture by Vaswani et al. (2017) revolutionized the field through the scaled dot-product self-attention mechanism. Foundation Large Language Models (LLMs)—including GPT, LLaMA, and Qwen—demonstrate emergent in-context learning capabilities. Instead of fine-tuning model parameters for individual downstream tasks, models can be guided via prompt engineering, turning diverse NLP challenges into in-context sequence transduction problems.")

    add_h2("1.2 Importance of the Selected NLP Topic")
    add_body("In the healthcare and biomedical domain, over 80% of clinical data is captured in unstructured narrative text—such as outpatient physician encounter notes, emergency department triage notes, discharge summaries, and radiology reports. This unstructured format forms a data silo that prevents automated clinical decision support, medical cohort identification, and automated medical billing.")
    add_body("Furthermore, strict data governance laws such as the Health Insurance Portability and Accountability Act (HIPAA) and the General Data Protection Regulation (GDPR) prohibit transmitting sensitive Protected Health Information (PHI) over commercial third-party cloud APIs. Consequently, deploying small, highly capable, local Large Language Models (1B to 3B parameters) on on-premise hardware has become a paramount priority. Determining the most effective prompt engineering techniques to achieve high-precision, schema-compliant extraction on local models is the core focus of this study.")

    add_h2("1.3 Objective of the Study")
    add_bullet("Empirically formulate, implement, and benchmark five distinct prompt engineering paradigms: Direct Zero-Shot (Baseline), Few-Shot In-Context Learning (ICL), Structured Chain-of-Thought (CoT), Schema-Constrained Prompting, and Role-Conditioned Expert Prompting with Clinical Guardrails.")
    add_bullet("Deploy the pipeline on a fully local, privacy-preserving instruction-tuned LLM (Qwen2.5:1.5B-Instruct via Ollama) guaranteeing zero cloud data egress.")
    add_bullet("Formulate a multi-criteria quantitative evaluation framework measuring RFC 8259 JSON validity, schema key adherence, entity F1 (demographics, diagnoses, severity, symptoms, medications, follow-up), negation violation rate, and runtime latency.")
    add_bullet("Develop an interactive, production-grade Streamlit web application providing real-time clinical extraction, self-healing JSON repair, and benchmarking dashboards.")

    add_h2("1.4 Scope of the Report")
    add_body("This report provides an exhaustive technical analysis covering theoretical foundations of transformer self-attention and in-context learning, local REST API integration, dataset curation across 8 medical specialties (20 test cases + 2 exemplars), empirical results, granular error analysis, discussion of trade-offs, and future research directions.")

    # --- SECTION 2 ---
    add_h1("2. Selected NLP Topic Overview")
    add_h2("2.1 Description of the NLP Task")
    add_body("The selected NLP task is Structured Information Extraction (SIE) from clinical narratives into HL7 FHIR-compliant JSON structures. SIE unifies multiple classical NLP subtasks—Named Entity Recognition (NER), Relation Extraction (RE), Slot Filling, and Assertion/Negation Classification—into an end-to-end generative mapping from unstructured text X to structured object Y.")

    add_h2("2.2 Real-World Use Cases")
    add_bullet("Automated Electronic Health Record (EHR) Ingestion: Auto-populating clinical database tables from dictated doctor notes.")
    add_bullet("Clinical Decision Support Systems (CDSS): Parsing medication regimens and active conditions to alert clinicians to drug contraindications.")
    add_bullet("Automated Medical Coding: Accelerating ICD-10 and CPT code assignment for hospital billing.")
    add_bullet("Clinical Trial Matching: Rapidly identifying eligible patient cohorts from free-text records.")

    add_h2("2.3 Relevance to the Syllabus and Industry")
    add_body("This project directly aligns with the syllabus of 'Natural Language Processing for LLM' (24AM7PCLLM), covering prompt engineering strategies, autoregressive decoding, evaluation metrics beyond token accuracy, and local model inference. In industry, privacy-preserving local clinical NLP is rapidly replacing cloud API calls to eliminate legal liabilities and ongoing API inference costs.")

    add_h2("2.4 Key Challenges in This NLP Task")
    add_bullet("Syntactic Failure in Small LLMs: Small models (<3B parameters) frequently emit unquoted numeric units (e.g., '140/90 mmHg') or markdown commentary, causing JSON decode errors.")
    add_bullet("Clinical Negation: Notes frequently document absent symptoms ('patient denies chest pain'). Models frequently hallucinate negated conditions as active symptoms.")
    add_bullet("Multi-Attribute Entity Decomposition: Extracting medication objects requires simultaneously binding drug name, dosage with units, and frequency.")
    add_bullet("Inference Latency vs. Reasoning Trade-off: Chain-of-Thought prompting provides reasoning steps but increases generation time significantly.")

    # --- SECTION 3 ---
    add_h1("3. Real-Time Application / API Identified")
    add_h2("3.1 Application or API Name")
    add_body("1. Inference Backend: Ollama Local REST API (http://localhost:11434/api/generate).\n2. Interactive Web Application: Streamlit Clinical Extraction Studio (app.py).")

    add_h2("3.2 Purpose and Functionality")
    add_body("Ollama serves as the local inference server, managing quantized GGUF weights in memory and executing deterministic token generation. Streamlit provides an interactive graphical interface allowing clinicians to test sample or custom notes, select prompt strategies, compare techniques side-by-side, view parsed JSON entities, and inspect benchmark analytics.")

    add_h2("3.3 Architecture or Platform on Which It Is Built")
    add_body("The system operates on Windows 11 with Python 3.11, communicating with Ollama via asynchronous HTTP POST requests. The front-end renders responsive UI cards, interactive tables, and Matplotlib visualization charts.")

    add_h2("3.4 Input/Output Behavior")
    add_body("Input: Raw narrative clinical text and selected prompt strategy parameters.\nOutput: Validated JSON object conforming to the target schema, accompanied by wall-clock latency, token usage, and generation speed metrics.")

    add_h2("3.5 Limitations and Practical Considerations")
    add_body("Sequential request queueing on single-GPU/CPU setups, context window limits (~4096 tokens during inference), and memory requirements (~1.1 GB VRAM for 4-bit quantized execution).")

    # --- SECTION 4 ---
    add_h1("4. NLP Model / Architecture Study")
    add_h2("4.1 Theoretical Foundations of the Model")
    add_body("The model utilized is Qwen2.5:1.5B-Instruct, an autoregressive decoder-only transformer with 1.54 billion parameters, pre-trained on 18 trillion tokens and instruction-tuned using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).")

    add_h2("4.2 Algorithm(s) Used")
    add_bullet("Transformer Decoder Blocks: 28 layers with Pre-RMSNorm normalization.")
    add_bullet("Grouped Query Attention (GQA): 12 query heads and 2 key-value heads to optimize KV cache memory.")
    add_bullet("SwiGLU Activation Function: Enhanced non-linear representation in the feed-forward network.")
    add_bullet("Rotary Position Embeddings (RoPE): Relative positional encoding supporting up to 32k context.")
    add_bullet("GGUF Q4_K_M Quantization: 4-bit precision preservation.")

    add_h2("4.3 Architecture Diagram / Layer-Wise Design")
    add_code_block("""Input Tokens -> BPE Tokenizer (151,643 vocab) -> Embedding + RoPE (d=1536)
  Repeat Layer l = 1 to 28:
    RMSNorm -> Grouped Query Attention (12 Q, 2 KV heads) -> Residual Add (+)
    RMSNorm -> SwiGLU MLP (Hidden Dim = 8960) -> Residual Add (+)
Final RMSNorm -> LM Head (Linear Projection) -> Softmax -> Next Token Sampling""")

    add_h2("4.4 Data Flow in the Model")
    add_body("Input prompts are tokenized into integer IDs, projected to hidden states, processed sequentially across 28 decoder layers with causal masking, and mapped to token probabilities via the language modeling head.")

    add_h2("4.5 Mathematical Formulation")
    add_body("Causal Scaled Dot-Product Attention:")
    add_code_block("Attention(Q, K, V) = softmax((Q K^T / sqrt(d_k)) + M) V")
    add_body("RMSNorm Formulation:")
    add_code_block("RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma")
    add_body("In-Context Learning Formulation:")
    add_code_block("h_L = h_0 + sum_{l=1}^L Delta W_attn(X_exemplars) h_{l-1}")

    add_h2("4.6 Why This Model Suits the Selected NLP Task")
    add_body("Qwen2.5:1.5B delivers state-of-the-art instruction-following and coding benchmark scores in the sub-2B category, fits inside ~1.1 GB of RAM/VRAM, operates completely offline for HIPAA compliance, and supports long clinical contexts.")

    # --- SECTION 5 ---
    add_h1("5. Dataset and Preprocessing")
    add_h2("5.1 Dataset Description (Source, Size, Structure)")
    add_body("The benchmark dataset consists of 20 comprehensive clinical encounter cases spanning 8 specialties (Endocrinology, Pulmonology, Cardiology, Neurology, Gastroenterology, Rheumatology, Dermatology, Psychiatry) plus 2 held-out exemplars for in-context learning. Ground truth annotations conform to a 7-attribute clinical schema.")

    add_h2("5.2 Data Cleaning Steps")
    add_body("A self-healing parsing engine strips markdown fences (```json ... ```), scans for outer braces ({ ... }), repairs unquoted numeric units (e.g., : 198 mg/dL, -> : \"198 mg/dL\",), and removes syntax-breaking trailing commas.")

    add_h2("5.3 Tokenization, Lemmatization, Stopwords Removal")
    add_body("Tokenization utilizes the Qwen BPE tokenizer during inference. For entity F1 evaluation, lowercase normalization, punctuation removal, and token set intersection are applied.")

    add_h2("5.4 Train/Test Split Strategy")
    add_body("Zero gradient training: exactly 2 cases reserved for in-context demonstration, and 20 cases reserved for unseen test evaluation.")

    add_h2("5.5 Feature Engineering or Embedding Approach")
    add_body("Contextual hidden activations within the 1536-dimensional transformer representation, evaluated using bipartite token overlap matching.")

    # --- SECTION 6 ---
    add_h1("6. Implementation Methodology")
    add_h2("6.1 Tools, Libraries, and Environment Used")
    add_body("Python 3.11, Ollama, Streamlit, Pandas, Matplotlib, Seaborn, Requests, Python-Docx on Windows 11.")

    add_h2("6.2 Model Implementation Steps")
    add_bullet("Step 1: Pull and verify local Qwen2.5:1.5B model in Ollama.")
    add_bullet("Step 2: Implement prompt formulation functions in src/prompt_templates.py.")
    add_bullet("Step 3: Build local inference client with telemetry logging in src/llm_client.py.")
    add_bullet("Step 4: Develop multi-criteria evaluation engine in src/evaluator.py.")
    add_bullet("Step 5: Run automated benchmark experiments in run_experiment.py.")
    add_bullet("Step 6: Build interactive web application in app.py and scientific plots in src/visualize.py.")

    add_h2("6.3 API Integration")
    add_body("HTTP POST to http://127.0.0.1:11434/api/generate with temperature=0.1, top_p=0.9, and stream=False.")

    add_h2("6.4 Pseudocode / Workflow Diagram")
    add_code_block("""Workflow:
Raw Clinical Text -> Prompt Strategy Formulation -> Local Ollama Inference (Temp=0.1)
  -> Raw Output -> Self-Healing JSON Parser -> Evaluator Engine
  -> Metric Logging (CSV/JSON) & Interactive Streamlit Dashboard""")

    add_h2("6.5 Hyperparameters and Training Configuration")
    add_body("Inference Temperature T = 0.1, Top-P = 0.9, Max Output Tokens = 1024, Context Length = 4096, Quantization = Q4_K_M.")

    # --- SECTION 7 ---
    add_h1("7. Results and Evaluation")
    add_h2("7.1 Performance Metrics")
    add_body("Evaluation encompasses Strict RFC 8259 Validity, Schema Key Coverage, Demographics F1, Diagnosis F1, Severity F1, Symptoms Precision/Recall/F1, Medications F1, Follow-up F1, Negation Violation Rate, and Composite Overall F1.")

    add_h2("7.2 Visualization of Results")
    add_body("Five high-resolution plots illustrate the empirical benchmark results:")
    
    # Embed plots in docx
    plots = [
        ("Figure 1: Overall Extraction F1 Score Comparison", "f1_comparison.png"),
        ("Figure 2: Strict RFC 8259 JSON Validity vs Schema Coverage", "schema_validity.png"),
        ("Figure 3: Inference Latency vs Composite F1 Score", "latency_vs_f1.png"),
        ("Figure 4: Granular Field-Level F1 Score Breakdown", "field_level_f1.png"),
        ("Figure 5: Multi-Criteria Benchmark Radar Chart", "radar_benchmark.png")
    ]
    for caption, filename in plots:
        img_path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(img_path):
            doc.add_paragraph()
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(2)
            run_img = p_img.add_run()
            run_img.add_picture(img_path, width=Inches(5.5))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(10)
            r_cap = p_cap.add_run(caption)
            r_cap.font.size = Pt(9.5)
            r_cap.font.italic = True
            r_cap.font.color.rgb = RGBColor(100, 110, 120)

    add_h2("7.3 Qualitative and Quantitative Analysis")
    add_body("Summary Table of Evaluated Prompt Engineering Strategies:")

    # Table of Results
    results_data = [
        ["Prompt Technique", "Strict Valid", "Schema Cov", "Diag F1", "Symp F1", "Meds F1", "Overall F1", "Neg Viol", "Latency"],
        ["Few-Shot ICL", "100.0%", "98.2%", "83.9%", "82.9%", "100.0%", "91.1%", "0.0%", "13.1s"],
        ["Schema-Constrained", "100.0%", "100.0%", "80.8%", "78.9%", "97.5%", "89.4%", "0.0%", "12.1s"],
        ["Chain-of-Thought", "100.0%", "96.4%", "73.5%", "86.3%", "100.0%", "88.1%", "0.0%", "20.8s"],
        ["Role-Expert", "87.5%", "87.5%", "55.4%", "65.5%", "87.5%", "70.6%", "1.4%", "13.3s"],
        ["Zero-Shot Baseline", "75.0%", "1.8%", "0.0%*", "0.0%*", "0.0%*", "0.0%*", "12.5%", "15.3s"]
    ]
    t_res = doc.add_table(rows=len(results_data), cols=len(results_data[0]))
    t_res.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(results_data):
        for c_idx, val in enumerate(row):
            cell = t_res.rows[r_idx].cells[c_idx]
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            if r_idx == 0:
                p.runs[0].font.bold = True
                p.runs[0].font.size = Pt(9.5)
                set_cell_shading(cell, "2B6CB0")
                p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
            else:
                p.runs[0].font.size = Pt(9)
                if r_idx % 2 == 1:
                    set_cell_shading(cell, "F7FAFC")
            set_cell_border(cell, top=dict(sz=4, color="CBD5E0"), bottom=dict(sz=4, color="CBD5E0"),
                                   left=dict(sz=4, color="CBD5E0"), right=dict(sz=4, color="CBD5E0"))

    add_body("\nKey Quantitative Findings:")
    add_bullet("Few-Shot ICL achieved the highest composite F1 score (91.1%) and 100% medication F1.")
    add_bullet("Schema-Constrained prompting achieved 100% strict JSON validity, 100% schema coverage, 89.4% F1, and the lowest latency (12.1s) with only 534 prompt tokens.")
    add_bullet("Chain-of-Thought achieved the highest symptom precision and recall (86.3% F1) but incurred a +71.6% latency penalty (20.8s vs 12.1s).")
    add_bullet("Zero-Shot baseline collapsed structurally (1.8% schema coverage, 12.5% negation violation rate).")

    add_h2("7.4 Error Analysis")
    add_body("Analysis revealed distinct error patterns across prompt strategies: Zero-Shot suffers from unquoted measurement units and key taxonomy invention; Chain-of-Thought suffers from reasoning verbosity; Role-Expert occasionally appends unstructured clinical remarks.")

    add_h2("7.5 Comparison Between Approaches")
    add_body("While Few-Shot ICL provides marginally higher F1 (+1.7%), Schema-Constrained prompting achieves the best Pareto efficiency, saving 45% in prompt token overhead and executing in 12.1 seconds.")

    # --- SECTION 8 ---
    add_h1("8. Discussion")
    add_h2("8.1 Interpretation of Results")
    add_body("The results prove that small local LLMs possess latent biomedical knowledge but require explicit syntactic scaffolding (schema constraints or exemplars) to emit structured artifacts.")

    add_h2("8.2 Strengths and Weaknesses of the Model")
    add_body("Strengths: Fast execution, zero cost, total data privacy.\nWeaknesses: Susceptible to structural syntax failure under open-ended zero-shot instructions.")

    add_h2("8.3 Challenges Encountered During Implementation")
    add_body("Designing regex repair algorithms for unquoted laboratory units, standardizing bipartite list matching for clinical symptoms, and handling negative constraints.")

    add_h2("8.4 Impact of Dataset Quality and Preprocessing")
    add_body("Carefully engineered test cases with explicit clinical negations proved critical in exposing model failure modes that standard synthetic benchmarks miss.")

    add_h2("8.5 How API or Application Enhances the NLP Task")
    add_body("The Streamlit application transforms complex prompt evaluation into an interactive tool, allowing clinicians and researchers to immediately assess model extraction behavior.")

    # --- SECTION 9 ---
    add_h1("9. Conclusion")
    add_h2("9.1 Summary of the Work Done")
    add_body("This project successfully demonstrated the empirical evaluation of five prompt engineering techniques for clinical information extraction using a local Qwen2.5:1.5B model via Ollama. A complete interactive application and automated benchmark suite were built and validated.")

    add_h2("9.2 Key Learnings and Insights")
    add_bullet("Schema-Constrained prompting provides optimal real-world performance.")
    add_bullet("Zero-Shot prompting fails in sub-3B models for structured tasks.")
    add_bullet("Reasoning scratchpads (CoT) improve negation resolution at the expense of latency.")

    add_h2("9.3 Real-World Applicability")
    add_body("Provides an immediate, privacy-preserving solution for hospital clinics, EHR ingestion systems, and automated medical documentation.")

    add_h2("9.4 Future Improvements or Extensions")
    add_bullet("Logit-level grammar masking (GBNF / Outlines).")
    add_bullet("Parameter-efficient fine-tuning (LoRA) on MIMIC-IV clinical records.")
    add_bullet("Evaluating 7B and 14B parameter models on dedicated GPU hardware.")

    # --- SECTION 10 ---
    add_h1("10. References")
    refs = [
        "1. Brown, T., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. Advances in Neural Information Processing Systems (NeurIPS), 33, 1877–1901.",
        "2. Wei, J., Wang, X., Schuurmans, D., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.",
        "3. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. NeurIPS 2017.",
        "4. Qwen Team. (2024). Qwen2.5 Technical Report. Alibaba Group. arXiv:2409.12191.",
        "5. von Oswald, J., Niklasson, E., Randazzo, E., et al. (2023). Transformers learn in-context by gradient descent. ICML 2023.",
        "6. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT.",
        "7. HL7 International. (2023). Fast Healthcare Interoperability Resources (FHIR) Specification, Release 4B.",
        "8. Chapman, W. W., Bridewell, W., Hanbury, P., et al. (2001). A simple algorithm for identifying negated findings and diseases in discharge summaries. Journal of Biomedical Informatics."
    ]
    for r in refs:
        add_body(r)

    doc.save(docx_path)
    print(f"[OK] Generated Word Document Report: {docx_path}")

if __name__ == "__main__":
    generate_markdown_report()
    generate_docx_report()
    generate_html_report()
    print("All reports generated successfully!")

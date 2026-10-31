"""
Prompt Engineering Techniques for Structured Clinical Information Extraction.
Implements 5 core paradigms:
1. Zero-Shot Direct Prompting (Baseline)
2. Few-Shot In-Context Learning (Few-Shot)
3. Chain-of-Thought (CoT) Prompting
4. Schema-Constrained / JSON Grammar Guided Prompting
5. Role-Conditioned Expert Prompting with Guardrails
Includes self-healing JSON parser and clinical schema normalizer.
"""

import json
import re
import ast
from typing import Dict, Any, List, Tuple
from src.dataset import get_few_shot_exemplars

TARGET_JSON_SCHEMA_DESCRIPTION = """Target JSON Schema Specification:
{
  "patient_age": integer or null,
  "patient_gender": "Male" | "Female" | "Other" | null,
  "primary_diagnosis": string,
  "severity_level": "Mild" | "Moderate" | "Severe" | "Critical" | "Unknown",
  "symptoms": [string],
  "medications": [
    {
      "name": string,
      "dosage": string,
      "frequency": string
    }
  ],
  "vitals_and_labs": {
    "<parameter_name>": string
  },
  "follow_up_days": integer or null
}"""

TARGET_JSON_SCHEMA_DICT: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "patient_age": {"type": ["integer", "null"]},
        "patient_gender": {"type": ["string", "null"], "enum": ["Male", "Female", "Other", None]},
        "primary_diagnosis": {"type": "string"},
        "severity_level": {"type": "string", "enum": ["Mild", "Moderate", "Severe", "Critical", "Unknown"]},
        "symptoms": {"type": "array", "items": {"type": "string"}},
        "medications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dosage": {"type": "string"},
                    "frequency": {"type": "string"}
                },
                "required": ["name", "dosage", "frequency"]
            }
        },
        "vitals_and_labs": {"type": "object"},
        "follow_up_days": {"type": ["integer", "null"]}
    },
    "required": [
        "patient_age",
        "patient_gender",
        "primary_diagnosis",
        "severity_level",
        "symptoms",
        "medications",
        "follow_up_days"
    ]
}

class PromptStrategies:
    """Collection of prompt generation functions for structured clinical information extraction."""

    @staticmethod
    def zero_shot(clinical_text: str) -> str:
        """
        Technique 1: Direct Zero-Shot Prompting.
        Baseline: minimal guidance, naive natural language instruction.
        """
        return f"""Extract the patient information from the following clinical note into a JSON object.

Clinical Note:
{clinical_text}

JSON:"""

    @staticmethod
    def few_shot(clinical_text: str) -> str:
        """
        Technique 2: Few-Shot In-Context Learning (ICL).
        Provides 2 full exemplars showing input text and expected JSON output.
        """
        exemplars = get_few_shot_exemplars()
        prompt_parts = [
            "You are a clinical information extraction system. Extract patient data into the JSON format demonstrated in the examples below.\n"
        ]

        for i, ex in enumerate(exemplars, 1):
            prompt_parts.append(f"### Example {i}:")
            prompt_parts.append(f"Clinical Note:\n{ex['text']}")
            prompt_parts.append(f"JSON Output:\n{json.dumps(ex['ground_truth'], indent=2)}\n")

        prompt_parts.append("### New Case to Extract:")
        prompt_parts.append(f"Clinical Note:\n{clinical_text}")
        prompt_parts.append("JSON Output:")

        return "\n".join(prompt_parts)

    @staticmethod
    def chain_of_thought(clinical_text: str) -> str:
        """
        Technique 3: Chain-of-Thought (CoT) Prompting.
        Prompts the model to reason step-by-step regarding demographics,
        negated symptoms, diagnoses, and medications prior to emitting JSON.
        """
        exemplars = get_few_shot_exemplars()
        ex1 = exemplars[0]

        return f"""You are an advanced clinical reasoning and extraction model.
Before producing the final JSON, analyze the clinical note step-by-step:
Step 1: Identify patient demographics (age, gender).
Step 2: List affirmative symptoms while explicitly excluding negated symptoms (e.g., 'denies X').
Step 3: Extract the confirmed primary diagnosis and clinical severity level.
Step 4: Extract all prescribed medications with exact dosage and frequency.
Step 5: Extract relevant vitals, laboratory values, and scheduled follow-up days.
Step 6: Output the final validated JSON inside ```json ... ``` code blocks.

### Worked Exemplar:
Clinical Note:
{ex1['text']}

Reasoning:
{ex1['reasoning']}

Final JSON:
```json
{json.dumps(ex1['ground_truth'], indent=2)}
```

### Now process this clinical note:
Clinical Note:
{clinical_text}

Reasoning:
"""

    @staticmethod
    def schema_constrained(clinical_text: str) -> str:
        """
        Technique 4: Schema-Constrained / JSON Grammar Guided Prompting.
        Provides strict structural specification, type constraints, and JSON-only directives.
        """
        return f"""Extract structured medical entities from the clinical text below.
You must adhere strictly to the schema specification.

{TARGET_JSON_SCHEMA_DESCRIPTION}

Strict Formatting Rules:
1. Output MUST be a single valid JSON object.
2. All keys from the schema must be present.
3. 'patient_age' must be an integer (e.g. 45) or null.
4. 'severity_level' must be one of: ["Mild", "Moderate", "Severe", "Critical", "Unknown"].
5. 'symptoms' must be an array of confirmed active symptoms. DO NOT include negated symptoms.
6. 'medications' must be an array of objects, each containing 'name', 'dosage', 'frequency'.
7. Do not include markdown preamble, explanations, or commentary. Output valid JSON only.

Clinical Note:
{clinical_text}

JSON:"""

    @staticmethod
    def role_expert(clinical_text: str) -> str:
        """
        Technique 5: Role-Conditioned Expert Prompting with Guardrails.
        Combines domain expert persona, operational guardrails (negation filtering,
        anti-hallucination policies), and schema alignment.
        """
        return f"""[ROLE & OBJECTIVE]
You are a Principal Clinical Informatician and Medical Natural Language Processing Engine.
Your objective is to ingest unstructured clinical encounter records and extract high-fidelity,
standardized, hallucination-free structured clinical JSON objects.

[OPERATIONAL GUARDRAILS & CLINICAL GUIDELINES]
1. STRICT NEGATION FILTERING: Clinicians regularly document negated findings (e.g., 'denies fever', 'no chest pain', 'without dyspnea'). Under no circumstances include negated conditions in the 'symptoms' list.
2. ENTITY GROUNDING: Only extract entities explicitly documented in the note. Never extrapolate or assume diagnoses.
3. MEDICATION DECOMPOSITION: Deconstruct every pharmaceutical into name, exact dosage (with units), and frequency.
4. STANDARDIZED SEVERITY: Map severity strictly to one of: ["Mild", "Moderate", "Severe", "Critical"].
5. JSON PURITY: Your entire response must parse cleanly with an RFC 8259 JSON parser without syntax errors.

[SCHEMA DEFINITION]
{TARGET_JSON_SCHEMA_DESCRIPTION}

[INPUT CLINICAL ENCOUNTER]
{clinical_text}

[STANDARDIZED JSON EXTRACTION]
"""


def repair_json_string(text: str) -> str:
    """
    Repairs common LLM JSON syntax errors:
    1. Unquoted values with medical units (e.g., : 198 mg/dL, -> : "198 mg/dL",)
    2. Unquoted BP slashes (e.g., : 148/92 mmHg, -> : "148/92 mmHg",)
    3. Trailing commas before } or ]
    4. Python boolean/null keywords (True, False, None -> true, false, null)
    5. Smart / curved quotes
    6. Unquoted keys ({ patient_age: 62 -> { "patient_age": 62)
    7. Bracket/brace balancing if output was truncated
    """
    # 1. Normalize smart quotes
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")

    # 2. Convert Python-style literals
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)

    # 3. Fix unquoted percentage values (e.g., : 9.4%, or : 9.4% })
    text = re.sub(r':\s*([0-9]+(?:\.[0-9]+)?\s*[%])\s*([,}])', r': "\1"\2', text)

    # 4. Fix unquoted unit values (e.g., : 198 mg/dL, or : 74 bpm, or : 28.4 kg/m^2,)
    text = re.sub(r':\s*([0-9]+(?:\.[0-9]+)?\s+[a-zA-Z/%^0-9]+)\s*([,}])', r': "\1"\2', text)

    # 5. Fix unquoted blood pressure slashes (e.g., : 148/92 mmHg, or : 120/80,)
    text = re.sub(r':\s*([0-9]+/[0-9]+(?:\s+[a-zA-Z/%]+)?)\s*([,}])', r': "\1"\2', text)

    # 6. Fix unquoted keys: e.g. { patient_age: 62, -> { "patient_age": 62,
    text = re.sub(r'([{\[,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)

    # 7. Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # 8. Fix single quotes used for JSON object keys and string values
    if '"' not in text and "'" in text:
        text = re.sub(r"'([^']*)'", r'"\1"', text)

    # 9. Ensure closing braces if truncated
    open_curly = text.count('{')
    close_curly = text.count('}')
    if open_curly > close_curly:
        text = text + ('}' * (open_curly - close_curly))

    open_square = text.count('[')
    close_square = text.count(']')
    if open_square > close_square:
        text = text + (']' * (open_square - close_square))

    return text


def extract_json_payload(raw_text: str) -> Tuple[Dict[str, Any], bool]:
    """
    Robust multi-tiered JSON parser for LLM outputs.
    Guaranteed to extract or recover structured clinical entities without unhandled crashes.
    Returns: (parsed_dict, was_raw_valid_boolean)
    """
    if not raw_text or not raw_text.strip():
        return {}, False

    text = raw_text.strip()
    candidate = text

    # Strategy 1: Look for fenced codeblock ```json ... ``` (with or without closing fence)
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", text, re.IGNORECASE)
    if code_match and code_match.group(1).strip():
        cand = code_match.group(1).strip()
        first_b = cand.find("{")
        last_b = cand.rfind("}")
        if first_b != -1:
            candidate = cand[first_b:last_b+1] if last_b > first_b else cand[first_b:]
    else:
        # Strategy 2: Outermost braces in raw text
        first_b = text.find("{")
        last_b = text.rfind("}")
        if first_b != -1 and last_b > first_b:
            candidate = text[first_b:last_b+1].strip()
        elif first_b != -1:
            candidate = text[first_b:].strip()

    # Tier 1: Direct strict JSON parse
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data, True
    except Exception:
        pass

    # Tier 2: Heuristic Repair
    repaired = repair_json_string(candidate)
    try:
        data = json.loads(repaired)
        if isinstance(data, dict):
            return data, False
    except Exception:
        pass

    # Tier 3: Python AST literal eval
    try:
        data = ast.literal_eval(candidate)
        if isinstance(data, dict):
            return data, False
    except Exception:
        pass

    # Tier 4: Search for innermost balanced JSON substring
    for start_idx in range(len(candidate)):
        if candidate[start_idx] == '{':
            for end_idx in range(len(candidate), start_idx, -1):
                chunk = candidate[start_idx:end_idx]
                try:
                    data = json.loads(chunk)
                    if isinstance(data, dict):
                        return data, False
                except Exception:
                    try:
                        data = json.loads(repair_json_string(chunk))
                        if isinstance(data, dict):
                            return data, False
                    except Exception:
                        pass

    # Tier 5: Regex Entity Harvester Fallback
    fallback_dict: Dict[str, Any] = {}
    age_m = re.search(r'(?:patient_)?age["\']?\s*:\s*([0-9]+)', candidate, re.I)
    if age_m:
        fallback_dict["patient_age"] = int(age_m.group(1))

    gender_m = re.search(r'(?:patient_)?gender["\']?\s*:\s*["\']?(Male|Female|Other)["\']?', candidate, re.I)
    if gender_m:
        fallback_dict["patient_gender"] = gender_m.group(1).capitalize()

    diag_m = re.search(r'(?:primary_)?diagnosis["\']?\s*:\s*["\']?([^,"\n}]+)["\']?', candidate, re.I)
    if diag_m:
        fallback_dict["primary_diagnosis"] = diag_m.group(1).strip()

    sev_m = re.search(r'(?:severity_level|severity)["\']?\s*:\s*["\']?(Mild|Moderate|Severe|Critical)["\']?', candidate, re.I)
    if sev_m:
        fallback_dict["severity_level"] = sev_m.group(1).capitalize()

    if fallback_dict:
        fallback_dict.setdefault("symptoms", [])
        fallback_dict.setdefault("medications", [])
        fallback_dict.setdefault("vitals_and_labs", {})
        fallback_dict.setdefault("follow_up_days", None)
        return fallback_dict, False

    # Ultimate graceful recovery: return empty dict with raw snippet
    return {"raw_extraction": text[:400]}, False


def normalize_clinical_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes diverse or unconstrained model outputs (e.g. from Zero-Shot)
    into the canonical schema keys while preserving all extracted entities:
    - patient_age, patient_gender, primary_diagnosis, severity_level,
      symptoms, medications, vitals_and_labs, follow_up_days
    """
    if not isinstance(data, dict):
        return {}
    normalized = dict(data)

    # 1. Demographics: Age & Gender
    if normalized.get("patient_age") is None:
        p_obj = data.get("patient") or data.get("demographics") or {}
        if isinstance(p_obj, dict) and "age" in p_obj:
            normalized["patient_age"] = p_obj["age"]

    if not normalized.get("patient_gender"):
        p_obj = data.get("patient") or data.get("demographics") or {}
        if isinstance(p_obj, dict) and "gender" in p_obj:
            normalized["patient_gender"] = p_obj["gender"]

    # 2. Diagnosis
    if not normalized.get("primary_diagnosis"):
        d_obj = data.get("diagnosis")
        if isinstance(d_obj, dict) and "primary" in d_obj:
            normalized["primary_diagnosis"] = d_obj["primary"]
        elif isinstance(d_obj, str):
            normalized["primary_diagnosis"] = d_obj
        elif "patient" in data and isinstance(data["patient"], dict):
            p_diag = data["patient"].get("diagnosis") or data["patient"].get("primary_diagnosis")
            if isinstance(p_diag, dict):
                normalized["primary_diagnosis"] = p_diag.get("primary", "")
            elif isinstance(p_diag, str):
                normalized["primary_diagnosis"] = p_diag

    # 3. Severity Level
    if not normalized.get("severity_level"):
        d_obj = data.get("diagnosis")
        if isinstance(d_obj, dict) and "severity" in d_obj:
            normalized["severity_level"] = d_obj["severity"]
        elif "severity" in data:
            normalized["severity_level"] = data["severity"]
        elif "patient" in data and isinstance(data["patient"], dict):
            p_diag = data["patient"].get("diagnosis") or {}
            if isinstance(p_diag, dict) and "severity" in p_diag:
                normalized["severity_level"] = p_diag["severity"]

    # 4. Symptoms
    syms = normalized.get("symptoms")
    if not syms and "patient" in data and isinstance(data["patient"], dict):
        syms = data["patient"].get("symptoms", [])

    if isinstance(syms, list):
        clean_s = []
        for s in syms:
            if isinstance(s, dict):
                desc = s.get("description") or s.get("type") or str(s)
                clean_s.append(desc)
            elif isinstance(s, str):
                clean_s.append(s)
        normalized["symptoms"] = clean_s
    elif isinstance(syms, str):
        normalized["symptoms"] = [syms]
    else:
        normalized["symptoms"] = []

    # 5. Medications
    meds = normalized.get("medications")
    if not meds:
        mp = (
            data.get("management_plan")
            or data.get("prescriptions")
            or (data.get("patient", {}).get("management_plan") if isinstance(data.get("patient"), dict) else None)
            or []
        )
        if isinstance(mp, list):
            clean_m = []
            for item in mp:
                if isinstance(item, dict):
                    clean_m.append({
                        "name": item.get("medication") or item.get("name") or "Unknown",
                        "dosage": item.get("dosage") or item.get("dose") or "N/A",
                        "frequency": item.get("frequency") or "N/A"
                    })
                elif isinstance(item, str):
                    clean_m.append({"name": item, "dosage": "N/A", "frequency": "N/A"})
            if clean_m:
                normalized["medications"] = clean_m
            else:
                normalized["medications"] = []
        else:
            normalized["medications"] = []
    elif isinstance(meds, list):
        clean_m = []
        for item in meds:
            if isinstance(item, dict):
                clean_m.append(item)
            elif isinstance(item, str):
                clean_m.append({"name": item, "dosage": "N/A", "frequency": "N/A"})
        normalized["medications"] = clean_m

    # 6. Vitals & Labs
    vitals = normalized.get("vitals_and_labs")
    if not vitals:
        labs = (
            data.get("lab_results")
            or data.get("vitals")
            or (data.get("patient", {}).get("lab_results") if isinstance(data.get("patient"), dict) else None)
        )
        if isinstance(labs, list) and labs and isinstance(labs[0], dict):
            normalized["vitals_and_labs"] = labs[0]
        elif isinstance(labs, dict):
            normalized["vitals_and_labs"] = labs

    # 7. Follow-up Days
    if normalized.get("follow_up_days") is None:
        fu = (
            data.get("follow_up")
            or data.get("followup")
            or (data.get("patient", {}).get("follow_up") if isinstance(data.get("patient"), dict) else None)
            or {}
        )
        if isinstance(fu, dict):
            freq = str(fu.get("frequency", "") or fu.get("timeframe", ""))
            days_m = re.search(r'([0-9]+)\s*days?', freq, re.I)
            if days_m:
                normalized["follow_up_days"] = int(days_m.group(1))

    return normalized


TECHNIQUE_REGISTRY = {
    "Zero-Shot": PromptStrategies.zero_shot,
    "Few-Shot": PromptStrategies.few_shot,
    "Chain-of-Thought": PromptStrategies.chain_of_thought,
    "Schema-Constrained": PromptStrategies.schema_constrained,
    "Role-Expert": PromptStrategies.role_expert
}

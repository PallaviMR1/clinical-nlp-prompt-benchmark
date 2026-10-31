"""
Evaluation Engine for Structured Information Extraction.
Computes:
1. JSON Syntax Validity
2. Schema Adherence / Key Coverage
3. Field-Level Precision, Recall, and F1 (Demographics, Diagnosis, Severity, Symptoms, Medications, Follow-up)
4. Negation Hallucination Penalty
5. Macro and Micro Information Extraction F1
6. Latency and Token Overhead Metrics
"""

from typing import Dict, Any, List, Tuple
import re

REQUIRED_SCHEMA_KEYS = [
    "patient_age",
    "patient_gender",
    "primary_diagnosis",
    "severity_level",
    "symptoms",
    "medications",
    "follow_up_days"
]

def normalize_text(text: str) -> str:
    """Normalize string for fuzzy comparison."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def token_f1_score(pred: str, target: str) -> float:
    """Compute token-level precision, recall, and F1 between two text strings."""
    pred_tokens = set(normalize_text(pred).split())
    target_tokens = set(normalize_text(target).split())

    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0

    common = pred_tokens.intersection(target_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    return (2.0 * precision * recall) / (precision + recall)

def set_overlap_f1(pred_list: List[str], target_list: List[str], threshold: float = 0.4) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F1 between two lists of entities using token overlap matching.
    """
    if not pred_list and not target_list:
        return 1.0, 1.0, 1.0
    if not pred_list:
        return 0.0, 0.0, 0.0
    if not target_list:
        return 0.0, 0.0, 0.0

    matched_targets = set()
    true_positives = 0

    for p in pred_list:
        p_str = str(p)
        best_f1 = 0.0
        best_t_idx = -1
        for idx, t in enumerate(target_list):
            if idx in matched_targets:
                continue
            sim = token_f1_score(p_str, str(t))
            if sim > best_f1:
                best_f1 = sim
                best_t_idx = idx

        if best_f1 >= threshold and best_t_idx != -1:
            true_positives += 1
            matched_targets.add(best_t_idx)

    precision = true_positives / len(pred_list) if pred_list else 0.0
    recall = true_positives / len(target_list) if target_list else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def evaluate_medications(pred_meds: Any, target_meds: List[Dict[str, str]]) -> Tuple[float, float, float]:
    """Evaluates extracted medication objects against ground truth."""
    if not isinstance(pred_meds, list):
        return 0.0, 0.0, 0.0

    pred_names = []
    for m in pred_meds:
        if isinstance(m, dict):
            name = m.get("name") or m.get("medication") or m.get("drug") or ""
            dosage = m.get("dosage", "")
            pred_names.append(f"{name} {dosage}".strip())
        elif isinstance(m, str):
            pred_names.append(m)

    target_names = [f"{m.get('name', '')} {m.get('dosage', '')}".strip() for m in target_meds]
    return set_overlap_f1(pred_names, target_names, threshold=0.35)

class StructuredIEEvaluator:
    """Evaluates LLM structured extraction outputs against gold ground truth."""

    @staticmethod
    def evaluate_sample(
        extracted_data: Dict[str, Any],
        ground_truth: Dict[str, Any],
        case_text: str = ""
    ) -> Dict[str, float]:
        """
        Evaluates a single structured extraction sample against ground truth.
        """
        if not isinstance(extracted_data, dict):
            return {
                "json_valid": 0.0,
                "schema_coverage": 0.0,
                "demographics_f1": 0.0,
                "diagnosis_f1": 0.0,
                "severity_f1": 0.0,
                "symptoms_precision": 0.0,
                "symptoms_recall": 0.0,
                "symptoms_f1": 0.0,
                "medications_f1": 0.0,
                "followup_f1": 0.0,
                "overall_f1": 0.0,
                "negation_violation_rate": 0.0
            }

        # 1. Schema Key Coverage
        present_keys = sum(1 for k in REQUIRED_SCHEMA_KEYS if k in extracted_data)
        schema_coverage = present_keys / len(REQUIRED_SCHEMA_KEYS)

        # 2. Demographics Score (Age and Gender)
        pred_age = extracted_data.get("patient_age")
        target_age = ground_truth.get("patient_age")
        age_score = 0.0
        try:
            if pred_age is not None and target_age is not None:
                if int(pred_age) == int(target_age):
                    age_score = 1.0
        except (ValueError, TypeError):
            age_score = 0.0

        pred_gender = normalize_text(str(extracted_data.get("patient_gender", "")))
        target_gender = normalize_text(str(ground_truth.get("patient_gender", "")))
        gender_score = 1.0 if pred_gender and (pred_gender in target_gender or target_gender in pred_gender) else 0.0

        demographics_f1 = (age_score + gender_score) / 2.0

        # 3. Diagnosis F1 (Token overlap)
        pred_diag = str(extracted_data.get("primary_diagnosis", ""))
        target_diag = str(ground_truth.get("primary_diagnosis", ""))
        diagnosis_f1 = token_f1_score(pred_diag, target_diag)

        # 4. Severity Level
        pred_sev = normalize_text(str(extracted_data.get("severity_level", "")))
        target_sev = normalize_text(str(ground_truth.get("severity_level", "")))
        severity_f1 = 1.0 if (pred_sev and target_sev and (pred_sev in target_sev or target_sev in pred_sev)) else 0.0

        # 5. Symptoms Precision, Recall, F1
        pred_symptoms = extracted_data.get("symptoms", [])
        if not isinstance(pred_symptoms, list):
            pred_symptoms = [str(pred_symptoms)] if pred_symptoms else []
        target_symptoms = ground_truth.get("symptoms", [])

        s_prec, s_rec, s_f1 = set_overlap_f1(pred_symptoms, target_symptoms, threshold=0.35)

        # 6. Check Negation Violations (did LLM hallucinate denied symptoms?)
        negation_violations = 0
        negation_patterns = [
            r"\b(?:denies|denied|strictly denies|specifically denies|explicitly denies)\b\s+(?:any\s+)?([^,.;]+)",
            r"\b(?:no|without|absence of|negative for)\b\s+(?:history of|evidence of|signs of|prior history of)?\s*([^,.;]+)"
        ]
        denied_phrases = []
        for pat in negation_patterns:
            matches = re.findall(pat, case_text, re.IGNORECASE)
            for m in matches:
                denied_phrases.extend(re.split(r",|\s+or\s+|\s+and\s+", m))

        denied_terms = [normalize_text(d) for d in denied_phrases if len(normalize_text(d)) >= 3]
        for ps in pred_symptoms:
            norm_ps = normalize_text(str(ps))
            for dt in denied_terms:
                if dt in norm_ps or (len(norm_ps) >= 3 and norm_ps in dt):
                    negation_violations += 1
                    break

        negation_violation_rate = (negation_violations / len(pred_symptoms)) if pred_symptoms else 0.0

        # 7. Medications F1
        pred_meds = extracted_data.get("medications", [])
        target_meds = ground_truth.get("medications", [])
        m_prec, m_rec, medications_f1 = evaluate_medications(pred_meds, target_meds)

        # 8. Follow-up Days Accuracy
        pred_fu = extracted_data.get("follow_up_days")
        target_fu = ground_truth.get("follow_up_days")
        followup_f1 = 0.0
        try:
            if pred_fu is not None and target_fu is not None:
                if int(pred_fu) == int(target_fu):
                    followup_f1 = 1.0
                elif abs(int(pred_fu) - int(target_fu)) <= 2:
                    followup_f1 = 0.5
        except (ValueError, TypeError):
            followup_f1 = 0.0

        # Composite Overall Extraction F1 (with penalty for negation violation)
        base_overall_f1 = (
            0.15 * demographics_f1 +
            0.25 * diagnosis_f1 +
            0.10 * severity_f1 +
            0.25 * s_f1 +
            0.20 * medications_f1 +
            0.05 * followup_f1
        )
        overall_f1 = max(0.0, base_overall_f1 - (0.15 * negation_violation_rate))

        return {
            "json_valid": 1.0,
            "schema_coverage": round(schema_coverage, 4),
            "demographics_f1": round(demographics_f1, 4),
            "diagnosis_f1": round(diagnosis_f1, 4),
            "severity_f1": round(severity_f1, 4),
            "symptoms_precision": round(s_prec, 4),
            "symptoms_recall": round(s_rec, 4),
            "symptoms_f1": round(s_f1, 4),
            "medications_f1": round(medications_f1, 4),
            "followup_f1": round(followup_f1, 4),
            "overall_f1": round(overall_f1, 4),
            "negation_violation_rate": round(negation_violation_rate, 4)
        }

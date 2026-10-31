"""
Benchmark dataset for Clinical Structured Information Extraction.
Contains realistic clinical encounter notes with ground-truth structured JSON annotations,
explicit negations, varied disease categories, and edge cases.
"""

from typing import Dict, List, Any, Optional
import json
import os

# Exemplars specifically reserved for In-Context Learning (Few-Shot & CoT Prompts)
# Ensuring strict separation: these are NEVER present in the test evaluation set.
FEW_SHOT_EXEMPLARS: List[Dict[str, Any]] = [
    {
        "id": "EXEMPLAR_01",
        "text": (
            "Patient is a 54-year-old male presenting to the outpatient clinic with complaints of "
            "persistent epigastric pain, frequent nocturnal acid regurgitation, and mild dysphagia lasting 3 weeks. "
            "He denies any hematemesis, melena, unexplained weight loss, or shortness of breath. "
            "Physical exam shows mild epigastric tenderness. Vitals: BP 126/80 mmHg, HR 74 bpm, BMI 28.4. "
            "Assessment: Gastroesophageal Reflux Disease (GERD) with suspected erosive esophagitis, moderate severity. "
            "Plan: Prescribed Omeprazole 40 mg PO once daily before breakfast and Famotidine 20 mg at bedtime prn. "
            "Advised dietary modifications. Schedule follow-up in 14 days."
        ),
        "reasoning": (
            "1. Demographics: 54-year-old male -> age=54, gender='Male'.\n"
            "2. Symptoms: Stated epigastric pain, acid regurgitation, mild dysphagia. "
            "Negated symptoms: hematemesis, melena, weight loss, shortness of breath (must be excluded).\n"
            "3. Diagnosis: Assessment explicitly states 'Gastroesophageal Reflux Disease (GERD)'.\n"
            "4. Severity: 'moderate severity' -> severity_level='Moderate'.\n"
            "5. Medications: Omeprazole 40 mg daily, Famotidine 20 mg bedtime.\n"
            "6. Vitals/Labs: BP 126/80 mmHg, HR 74 bpm, BMI 28.4.\n"
            "7. Follow-up: 'in 14 days' -> follow_up_days=14."
        ),
        "ground_truth": {
            "patient_age": 54,
            "patient_gender": "Male",
            "primary_diagnosis": "Gastroesophageal Reflux Disease (GERD)",
            "severity_level": "Moderate",
            "symptoms": ["epigastric pain", "acid regurgitation", "mild dysphagia"],
            "medications": [
                {"name": "Omeprazole", "dosage": "40 mg", "frequency": "once daily"},
                {"name": "Famotidine", "dosage": "20 mg", "frequency": "at bedtime as needed"}
            ],
            "vitals_and_labs": {
                "blood_pressure": "126/80 mmHg",
                "heart_rate": "74 bpm",
                "bmi": "28.4"
            },
            "follow_up_days": 14
        }
    },
    {
        "id": "EXEMPLAR_02",
        "text": (
            "A 31-year-old female presents to urgent care with acute dysuria, increased urinary frequency, and suprapubic discomfort for 2 days. "
            "She explicitly denies fever, chills, flank pain, nausea, or vaginal discharge. "
            "Urinalysis demonstrates positive nitrites, moderate leukocyte esterase, and WBC 25-30/HPF. "
            "Diagnosis: Acute Uncomplicated Cystitis (mild to moderate severity). "
            "Initiated Nitrofurantoin monohydrate/macrocrystals 100 mg orally twice daily for 5 days. "
            "Instructed to increase hydration. Follow-up in 7 days if symptoms fail to resolve."
        ),
        "reasoning": (
            "1. Demographics: 31-year-old female -> age=31, gender='Female'.\n"
            "2. Symptoms: dysuria, urinary frequency, suprapubic discomfort. "
            "Negated symptoms: fever, chills, flank pain, nausea, vaginal discharge (must be omitted).\n"
            "3. Diagnosis: Acute Uncomplicated Cystitis.\n"
            "4. Severity: 'mild to moderate' -> classify as 'Moderate' or 'Mild'.\n"
            "5. Medications: Nitrofurantoin 100 mg twice daily.\n"
            "6. Vitals/Labs: Urinalysis: positive nitrites, leukocyte esterase, WBC 25-30/HPF.\n"
            "7. Follow-up: 'in 7 days' -> follow_up_days=7."
        ),
        "ground_truth": {
            "patient_age": 31,
            "patient_gender": "Female",
            "primary_diagnosis": "Acute Uncomplicated Cystitis",
            "severity_level": "Moderate",
            "symptoms": ["dysuria", "urinary frequency", "suprapubic discomfort"],
            "medications": [
                {"name": "Nitrofurantoin", "dosage": "100 mg", "frequency": "twice daily"}
            ],
            "vitals_and_labs": {
                "urinalysis": "positive nitrites, moderate leukocyte esterase, WBC 25-30/HPF"
            },
            "follow_up_days": 7
        }
    }
]

# Benchmark Test Dataset (20 comprehensive clinical cases)
BENCHMARK_CASES: List[Dict[str, Any]] = [
    {
        "id": "CASE_001",
        "category": "Endocrinology / Cardiology",
        "text": (
            "The patient is a 62-year-old male with an established history of poorly controlled Type 2 Diabetes Mellitus "
            "and Essential Hypertension. He reports ongoing bilateral lower extremity paresthesia, burning foot sensations, "
            "and blurred vision. Denies chest pain, palpitations, orthopnea, or ankle edema. "
            "Physical exam reveals diminished vibratory sensation in distal lower extremities. "
            "Laboratory results show HbA1c of 9.4%, Fasting Blood Glucose 198 mg/dL, and BP 148/92 mmHg. "
            "Primary diagnosis is Diabetic Peripheral Neuropathy secondary to Type 2 Diabetes Mellitus, assessed as severe. "
            "Management plan: titrate Metformin to 1000 mg PO BID, add Duloxetine 30 mg PO once daily for neuropathic pain, "
            "and continue Lisinopril 20 mg PO daily. Return to clinic in 30 days for repeat glycemic profile."
        ),
        "ground_truth": {
            "patient_age": 62,
            "patient_gender": "Male",
            "primary_diagnosis": "Diabetic Peripheral Neuropathy",
            "severity_level": "Severe",
            "symptoms": ["bilateral lower extremity paresthesia", "burning foot sensations", "blurred vision"],
            "medications": [
                {"name": "Metformin", "dosage": "1000 mg", "frequency": "twice daily"},
                {"name": "Duloxetine", "dosage": "30 mg", "frequency": "once daily"},
                {"name": "Lisinopril", "dosage": "20 mg", "frequency": "once daily"}
            ],
            "vitals_and_labs": {
                "hba1c": "9.4%",
                "fasting_blood_glucose": "198 mg/dL",
                "blood_pressure": "148/92 mmHg"
            },
            "follow_up_days": 30
        }
    },
    {
        "id": "CASE_002",
        "category": "Pulmonology",
        "text": (
            "A 45-year-old female presents with a 4-day history of worsening productive cough with yellow sputum, "
            "pleuritic right-sided chest pain, subjective fever, and moderate dyspnea on exertion. "
            "She specifically denies hemoptysis, calf swelling, wheezing, or confusion. "
            "Vital signs: temperature 38.6 C (101.5 F), BP 118/76 mmHg, pulse 102 bpm, SpO2 93% on room air. "
            "Chest auscultation demonstrates right lower lobe bronchial breath sounds and crackles. "
            "Chest X-ray confirms right lower lobe consolidation consistent with Community-Acquired Pneumonia, moderate severity. "
            "Prescribed Amoxicillin-clavulanate 875/125 mg PO BID for 7 days and Azithromycin 500 mg day 1 then 250 mg daily for 4 days. "
            "Patient instructed to follow up in 5 days or proceed to the ED if SpO2 drops below 92%."
        ),
        "ground_truth": {
            "patient_age": 45,
            "patient_gender": "Female",
            "primary_diagnosis": "Community-Acquired Pneumonia",
            "severity_level": "Moderate",
            "symptoms": ["productive cough", "yellow sputum", "pleuritic chest pain", "fever", "dyspnea on exertion"],
            "medications": [
                {"name": "Amoxicillin-clavulanate", "dosage": "875/125 mg", "frequency": "twice daily"},
                {"name": "Azithromycin", "dosage": "500 mg then 250 mg", "frequency": "daily"}
            ],
            "vitals_and_labs": {
                "temperature": "38.6 C",
                "blood_pressure": "118/76 mmHg",
                "pulse": "102 bpm",
                "spo2": "93%"
            },
            "follow_up_days": 5
        }
    },
    {
        "id": "CASE_003",
        "category": "Cardiology",
        "text": (
            "A 71-year-old female presents to the heart failure clinic complaining of progressive bilateral pedal edema, "
            "orthopnea requiring 3 pillows, and paroxysmal nocturnal dyspnea over the last 10 days. "
            "She reports a recent 4 kg weight gain. She denies acute angina, syncope, fever, or productive cough. "
            "Vitals: BP 154/90 mmHg, HR 88 bpm regular, RR 22/min, SpO2 91% on room air. "
            "Physical examination shows elevated JVP at 8 cm above sternal angle and bilateral bibasilar crackles. "
            "Serum NT-proBNP is severely elevated at 3,450 pg/mL. "
            "Assessment: Acute Decompensated Heart Failure with preserved ejection fraction, severe clinical status. "
            "Adjusted Furosemide to 80 mg PO BID, initiated Spironolactone 25 mg daily, and continuing Metoprolol Succinate 50 mg daily. "
            "Strict fluid restriction (1.5 L/day). Mandatory clinic return in 7 days for electrolyte check."
        ),
        "ground_truth": {
            "patient_age": 71,
            "patient_gender": "Female",
            "primary_diagnosis": "Acute Decompensated Heart Failure",
            "severity_level": "Severe",
            "symptoms": ["bilateral pedal edema", "orthopnea", "paroxysmal nocturnal dyspnea", "weight gain"],
            "medications": [
                {"name": "Furosemide", "dosage": "80 mg", "frequency": "twice daily"},
                {"name": "Spironolactone", "dosage": "25 mg", "frequency": "once daily"},
                {"name": "Metoprolol Succinate", "dosage": "50 mg", "frequency": "once daily"}
            ],
            "vitals_and_labs": {
                "blood_pressure": "154/90 mmHg",
                "heart_rate": "88 bpm",
                "spo2": "91%",
                "nt_probnp": "3450 pg/mL"
            },
            "follow_up_days": 7
        }
    },
    {
        "id": "CASE_004",
        "category": "Neurology",
        "text": (
            "A 28-year-old female presents with throbbing unilateral left frontotemporal headache of 18 hours duration, "
            "accompanied by severe photophobia, phonophobia, and nausea with two episodes of non-bilious emesis. "
            "She denies visual aura, neck stiffness, fever, focal motor weakness, or speech impairment. "
            "Vitals: BP 122/78 mmHg, HR 72 bpm, afebrile. Neurological exam is non-focal. "
            "Diagnosis: Acute Migraine Headache without aura, moderate to severe intensity. "
            "Administered Sumatriptan 50 mg PO single dose with Ondansetron 4 mg ODT. "
            "Prescribed Naproxen sodium 500 mg PRN for breakthrough pain. "
            "Scheduled for neurological review in 60 days to discuss prophylactic Topiramate if attacks exceed 4 per month."
        ),
        "ground_truth": {
            "patient_age": 28,
            "patient_gender": "Female",
            "primary_diagnosis": "Acute Migraine Headache",
            "severity_level": "Moderate",
            "symptoms": ["unilateral headache", "photophobia", "phonophobia", "nausea", "emesis"],
            "medications": [
                {"name": "Sumatriptan", "dosage": "50 mg", "frequency": "single dose"},
                {"name": "Ondansetron", "dosage": "4 mg", "frequency": "as needed"},
                {"name": "Naproxen sodium", "dosage": "500 mg", "frequency": "as needed"}
            ],
            "vitals_and_labs": {
                "blood_pressure": "122/78 mmHg",
                "heart_rate": "72 bpm"
            },
            "follow_up_days": 60
        }
    },
    {
        "id": "CASE_005",
        "category": "Gastroenterology",
        "text": (
            "Patient is an 18-year-old male presenting to the emergency department with acute periumbilical abdominal pain "
            "that migrated to the right lower quadrant over the past 12 hours. Associated with anorexia, low-grade pyrexia, and nausea. "
            "The patient denies diarrhea, hematochezia, dysuria, testicular pain, or prior surgeries. "
            "Vitals: temp 38.1 C, pulse 108 bpm, BP 124/82 mmHg. "
            "Abdominal exam reveals localized McBurney point tenderness with involuntary guarding and positive Rovsing sign. "
            "Ultrasound confirms an aperistaltic, non-compressible tubular structure measuring 8.5 mm with surrounding fat stranding. "
            "Diagnosis: Acute Appendicitis, critical surgical condition. "
            "Surgical consult obtained for emergent laparoscopic appendectomy. IV Ceftriaxone 1 g and Metronidazole 500 mg administered stat. "
            "Post-operative inpatient stay planned; outpatient surgical follow-up in 10 days."
        ),
        "ground_truth": {
            "patient_age": 18,
            "patient_gender": "Male",
            "primary_diagnosis": "Acute Appendicitis",
            "severity_level": "Critical",
            "symptoms": ["periumbilical pain migrating to right lower quadrant", "anorexia", "fever", "nausea"],
            "medications": [
                {"name": "Ceftriaxone", "dosage": "1 g", "frequency": "stat once"},
                {"name": "Metronidazole", "dosage": "500 mg", "frequency": "stat once"}
            ],
            "vitals_and_labs": {
                "temperature": "38.1 C",
                "pulse": "108 bpm",
                "blood_pressure": "124/82 mmHg",
                "appendix_diameter": "8.5 mm"
            },
            "follow_up_days": 10
        }
    },
    {
        "id": "CASE_006",
        "category": "Rheumatology",
        "text": (
            "A 52-year-old female presents with symmetric morning stiffness lasting greater than 90 minutes, "
            "affecting the MCP and PIP joints of both hands, alongside fatigue and low-grade malaise for 4 months. "
            "She denies Raynaud phenomenon, photosensitive facial rash, oral ulcers, or dry eyes. "
            "Joint exam reveals tender, boggy swelling in 8 MCP joints and both wrists. "
            "Serology: Rheumatoid Factor (RF) positive at 128 IU/mL, Anti-CCP antibodies highly positive (>250 U/mL), ESR 48 mm/hr. "
            "Diagnosis: Seropositive Rheumatoid Arthritis, moderate to high disease activity (Moderate severity). "
            "Therapy initiated with Methotrexate 15 mg PO once weekly, supplemented with Folic acid 1 mg PO daily, "
            "and a tapering course of Prednisone 10 mg daily for 4 weeks. "
            "Routine laboratory monitoring (CBC/LFTs) scheduled with rheumatology follow-up in 30 days."
        ),
        "ground_truth": {
            "patient_age": 52,
            "patient_gender": "Female",
            "primary_diagnosis": "Seropositive Rheumatoid Arthritis",
            "severity_level": "Moderate",
            "symptoms": ["symmetric morning stiffness", "joint pain and swelling in hands", "fatigue", "malaise"],
            "medications": [
                {"name": "Methotrexate", "dosage": "15 mg", "frequency": "once weekly"},
                {"name": "Folic acid", "dosage": "1 mg", "frequency": "daily"},
                {"name": "Prednisone", "dosage": "10 mg", "frequency": "daily tapering"}
            ],
            "vitals_and_labs": {
                "rf": "128 IU/mL",
                "anti_ccp": ">250 U/mL",
                "esr": "48 mm/hr"
            },
            "follow_up_days": 30
        }
    },
    {
        "id": "CASE_007",
        "category": "Dermatology",
        "text": (
            "A 36-year-old male presents with sharply demarcated, erythematous plaques covered by silvery-white scales "
            "on the extensor surfaces of both elbows, knees, and scalp with associated pruritus. "
            "He explicitly denies joint pain, nail pitting, sausage digits (dactylitis), or recent streptococcal throat infections. "
            "Physical exam confirms classic plaque psoriasis covering approximately 4% of total body surface area. "
            "Assessment: Chronic Plaque Psoriasis, mild severity. "
            "Prescribed topical Clobetasol propionate 0.05% ointment applied BID for 2 weeks, followed by "
            "Calcipotriene 0.005% cream daily maintenance. Emollient cream recommended liberally. "
            "Advised clinic re-evaluation in 28 days to assess treatment response."
        ),
        "ground_truth": {
            "patient_age": 36,
            "patient_gender": "Male",
            "primary_diagnosis": "Chronic Plaque Psoriasis",
            "severity_level": "Mild",
            "symptoms": ["erythematous plaques", "silvery scales on elbows and knees", "pruritus"],
            "medications": [
                {"name": "Clobetasol propionate", "dosage": "0.05% ointment", "frequency": "twice daily"},
                {"name": "Calcipotriene", "dosage": "0.005% cream", "frequency": "daily maintenance"}
            ],
            "vitals_and_labs": {
                "body_surface_area": "4%"
            },
            "follow_up_days": 28
        }
    },
    {
        "id": "CASE_008",
        "category": "Psychiatry",
        "text": (
            "A 42-year-old female presents with persistent depressed mood, profound anhedonia, terminal insomnia, "
            "difficulty concentrating at work, and severe psychomotor retardation for the past 6 weeks. "
            "Patient strictly denies suicidal ideation, visual or auditory hallucinations, manic episodes, or substance abuse. "
            "PHQ-9 depression screening score is 19 out of 27. "
            "Clinical assessment: Major Depressive Disorder, single episode, severe severity without psychotic features. "
            "Initiated Sertraline hydrochloride 50 mg PO once daily in the morning, with plan to titrate to 100 mg after 2 weeks. "
            "Referred for cognitive behavioral therapy (CBT). Close psychiatric clinic visit arranged in 14 days."
        ),
        "ground_truth": {
            "patient_age": 42,
            "patient_gender": "Female",
            "primary_diagnosis": "Major Depressive Disorder",
            "severity_level": "Severe",
            "symptoms": ["depressed mood", "anhedonia", "terminal insomnia", "poor concentration", "psychomotor retardation"],
            "medications": [
                {"name": "Sertraline hydrochloride", "dosage": "50 mg", "frequency": "once daily"}
            ],
            "vitals_and_labs": {
                "phq9_score": "19/27"
            },
            "follow_up_days": 14
        }
    },
    {
        "id": "CASE_009",
        "category": "Infectious Disease",
        "text": (
            "A 24-year-old male university student presents with sudden onset high fever of 39.4 C, severe odynophagia, "
            "anterior cervical lymphadenopathy, and absence of cough. "
            "Denies rhinorrhea, hoarseness, conjunctivitis, vomiting, or cutaneous eruptions. "
            "Oropharyngeal exam demonstrates marked tonsillar erythema and purulent exudates bilaterally. "
            "Centor criteria score is 4. Rapid Antigen Detection Test (RADT) for Group A Streptococcus is positive. "
            "Diagnosis: Acute Streptococcal Pharyngitis, moderate severity. "
            "Prescribed Penicillin V potassium 500 mg orally four times daily for 10 days and Ibuprofen 400 mg q6h prn fever. "
            "Patient advised isolation for 24 hours post antibiotic initiation. Follow-up in 7 days if no symptomatic improvement."
        ),
        "ground_truth": {
            "patient_age": 24,
            "patient_gender": "Male",
            "primary_diagnosis": "Acute Streptococcal Pharyngitis",
            "severity_level": "Moderate",
            "symptoms": ["high fever", "severe odynophagia", "anterior cervical lymphadenopathy", "tonsillar exudates"],
            "medications": [
                {"name": "Penicillin V potassium", "dosage": "500 mg", "frequency": "four times daily"},
                {"name": "Ibuprofen", "dosage": "400 mg", "frequency": "every 6 hours as needed"}
            ],
            "vitals_and_labs": {
                "temperature": "39.4 C",
                "radt_strep_a": "positive",
                "centor_score": "4"
            },
            "follow_up_days": 7
        }
    },
    {
        "id": "CASE_010",
        "category": "Pulmonology / Allergy",
        "text": (
            "A 12-year-old male with known extrinsic asthma presents with recurrent nocturnal coughing fits, "
            "expiratory wheezing, and chest tightness following exercise. "
            "Mother states the child denies fever, purulent sputum, rhinorrhea, or hemoptysis. "
            "Vitals: RR 24/min, SpO2 96% on room air, HR 90 bpm. Peak expiratory flow (PEF) is 72% of predicted personal best. "
            "Auscultation reveals diffuse bilateral expiratory polyphonic wheezes. "
            "Diagnosis: Moderate Persistent Bronchial Asthma with acute mild exacerbation. "
            "Treatment: Step-up therapy to Fluticasone propionate/Salmeterol 100/50 mcg DPI one inhalation twice daily, "
            "and Albuterol sulfate 90 mcg MDI 2 puffs every 4-6 hours PRN acute wheeze with spacer. "
            "Asthma Action Plan reviewed. Clinic follow-up scheduled in 21 days."
        ),
        "ground_truth": {
            "patient_age": 12,
            "patient_gender": "Male",
            "primary_diagnosis": "Moderate Persistent Bronchial Asthma",
            "severity_level": "Moderate",
            "symptoms": ["nocturnal cough", "expiratory wheezing", "chest tightness with exercise"],
            "medications": [
                {"name": "Fluticasone propionate/Salmeterol", "dosage": "100/50 mcg", "frequency": "twice daily"},
                {"name": "Albuterol sulfate", "dosage": "90 mcg (2 puffs)", "frequency": "every 4-6 hours as needed"}
            ],
            "vitals_and_labs": {
                "spo2": "96%",
                "pef": "72% of predicted",
                "respiratory_rate": "24/min"
            },
            "follow_up_days": 21
        }
    },
    {
        "id": "CASE_011",
        "category": "Nephrology / Urology",
        "text": (
            "A 66-year-old male with chronic kidney disease stage 3b presents for routine nephrology surveillance. "
            "He complains of mild generalized fatigue and nocturnal leg cramps. "
            "Denies gross hematuria, flank pain, dysuria, shortness of breath, or facial puffiness. "
            "Vitals: BP 142/86 mmHg, HR 68 bpm. "
            "Biochemical panel: Serum Creatinine 2.4 mg/dL (eGFR 28 mL/min/1.73m2), Serum Potassium 5.1 mEq/L, "
            "Serum Calcium 8.8 mg/dL, and urine albumin-to-creatinine ratio (uACR) 340 mg/g. "
            "Diagnosis: Chronic Kidney Disease Stage 4 with Macroalbuminuria, severe severity. "
            "Adjusted Losartan 50 mg PO daily, continued Amlodipine 5 mg daily, and initiated Sodium bicarbonate 650 mg BID. "
            "Dietary potassium and phosphorus restrictions emphasized. Return for repeat metabolic panel in 14 days."
        ),
        "ground_truth": {
            "patient_age": 66,
            "patient_gender": "Male",
            "primary_diagnosis": "Chronic Kidney Disease Stage 4",
            "severity_level": "Severe",
            "symptoms": ["generalized fatigue", "nocturnal leg cramps"],
            "medications": [
                {"name": "Losartan", "dosage": "50 mg", "frequency": "once daily"},
                {"name": "Amlodipine", "dosage": "5 mg", "frequency": "once daily"},
                {"name": "Sodium bicarbonate", "dosage": "650 mg", "frequency": "twice daily"}
            ],
            "vitals_and_labs": {
                "serum_creatinine": "2.4 mg/dL",
                "egfr": "28 mL/min/1.73m2",
                "serum_potassium": "5.1 mEq/L",
                "blood_pressure": "142/86 mmHg"
            },
            "follow_up_days": 14
        }
    },
    {
        "id": "CASE_012",
        "category": "Endocrinology",
        "text": (
            "A 38-year-old female presents with unintentional 6 kg weight loss over 2 months, palpitations, "
            "heat intolerance, fine tremor in outstretched hands, and increased bowel frequency (3 loose stools/day). "
            "She denies visual changes, diplopia, neck pain, fever, or dysphagia. "
            "Vitals: BP 136/74 mmHg, resting HR 112 bpm (sinus tachycardia). "
            "Neck exam reveals diffuse, non-tender, symmetrically enlarged thyroid gland with an audible thyroid bruit. "
            "Labs reveal Serum TSH <0.01 uIU/mL, Free T4 3.8 ng/dL, and elevated TRAb titers (12.4 IU/L). "
            "Diagnosis: Graves Disease with hyperthyroidism, moderate to severe clinical severity. "
            "Prescribed Methimazole 20 mg PO daily and Propranolol 20 mg PO TID for symptomatic adrenergic control. "
            "Endocrine clinic review in 30 days with repeat free thyroid hormone panel."
        ),
        "ground_truth": {
            "patient_age": 38,
            "patient_gender": "Female",
            "primary_diagnosis": "Graves Disease",
            "severity_level": "Moderate",
            "symptoms": ["weight loss", "palpitations", "heat intolerance", "fine tremor", "increased bowel frequency"],
            "medications": [
                {"name": "Methimazole", "dosage": "20 mg", "frequency": "once daily"},
                {"name": "Propranolol", "dosage": "20 mg", "frequency": "three times daily"}
            ],
            "vitals_and_labs": {
                "tsh": "<0.01 uIU/mL",
                "free_t4": "3.8 ng/dL",
                "trab": "12.4 IU/L",
                "heart_rate": "112 bpm"
            },
            "follow_up_days": 30
        }
    },
    {
        "id": "CASE_013",
        "category": "Hematology",
        "text": (
            "A 29-year-old female presents to primary care with profound chronic fatigue, dizziness upon standing, "
            "brittle nails, and cold intolerance for 3 months. She also notes heavy menstrual bleeding (menorrhagia). "
            "She denies melena, hematochezia, hematuria, weight loss, or shortness of breath at rest. "
            "Vitals: BP 106/68 mmHg, HR 86 bpm. Physical examination notes marked conjunctival and palmar pallor. "
            "CBC shows Hemoglobin 7.9 g/dL, Hematocrit 25.1%, MCV 66 fL (microcytic hypochromic), and Serum Ferritin 6 ng/mL. "
            "Diagnosis: Severe Iron Deficiency Anemia secondary to menorrhagia. "
            "Prescribed oral Ferrous sulfate 325 mg (65 mg elemental iron) daily with 500 mg Vitamin C on an empty stomach. "
            "Referred to gynecology for menorrhagia evaluation. Follow-up lab check in 30 days."
        ),
        "ground_truth": {
            "patient_age": 29,
            "patient_gender": "Female",
            "primary_diagnosis": "Iron Deficiency Anemia",
            "severity_level": "Severe",
            "symptoms": ["chronic fatigue", "dizziness", "brittle nails", "cold intolerance", "heavy menstrual bleeding"],
            "medications": [
                {"name": "Ferrous sulfate", "dosage": "325 mg", "frequency": "once daily"},
                {"name": "Vitamin C", "dosage": "500 mg", "frequency": "once daily"}
            ],
            "vitals_and_labs": {
                "hemoglobin": "7.9 g/dL",
                "mcv": "66 fL",
                "ferritin": "6 ng/mL"
            },
            "follow_up_days": 30
        }
    },
    {
        "id": "CASE_014",
        "category": "Orthopedics",
        "text": (
            "A 59-year-old male presents with persistent right knee pain and stiffness that worsens with weight-bearing "
            "and prolonged standing, accompanied by coarse crepitus. "
            "He denies joint warmth, erythema, morning stiffness lasting >30 minutes, locking, giving way, or trauma. "
            "Weight 98 kg, BMI 31.2. Knee exam demonstrates medial joint line tenderness and small non-inflammatory effusion. "
            "Weight-bearing bilateral knee radiographs demonstrate Kellgren-Lawrence Grade 3 medial joint space narrowing and osteophytes. "
            "Diagnosis: Primary Osteoarthritis of the Right Knee, moderate severity. "
            "Management: Prescribed Meloxicam 15 mg PO daily with meals and topical Diclofenac 1% gel QID PRN. "
            "Physiotherapy referral for quadriceps strengthening and weight reduction counseling. Review in 45 days."
        ),
        "ground_truth": {
            "patient_age": 59,
            "patient_gender": "Male",
            "primary_diagnosis": "Osteoarthritis of Right Knee",
            "severity_level": "Moderate",
            "symptoms": ["right knee pain", "knee stiffness with weight-bearing", "crepitus"],
            "medications": [
                {"name": "Meloxicam", "dosage": "15 mg", "frequency": "once daily"},
                {"name": "Diclofenac gel", "dosage": "1%", "frequency": "four times daily as needed"}
            ],
            "vitals_and_labs": {
                "bmi": "31.2",
                "radiology_grade": "Kellgren-Lawrence Grade 3"
            },
            "follow_up_days": 45
        }
    },
    {
        "id": "CASE_015",
        "category": "Ophthalmology",
        "text": (
            "A 68-year-old male presents for a comprehensive routine eye evaluation. He has mild ocular dryness and "
            "complains of needing brighter light when reading, but denies eye pain, redness, flashes of light, or visual field defects. "
            "Visual acuity is 20/30 in the right eye and 20/40 in the left eye with correction. "
            "Intraocular pressure (IOP) measured by Goldmann applanation tonometry is 26 mmHg OD and 28 mmHg OS (elevated). "
            "Slit lamp examination shows open anterior chamber angles bilaterally. Funduscopy demonstrates cup-to-disc ratio of 0.75 OD and 0.8 OS with thinning of the neuroretinal rim. "
            "Diagnosis: Primary Open-Angle Glaucoma, moderate severity. "
            "Prescribed Latanoprost 0.005% ophthalmic solution 1 drop in both eyes once daily at bedtime. "
            "Patient scheduled for repeat automated visual field perimetry in 30 days."
        ),
        "ground_truth": {
            "patient_age": 68,
            "patient_gender": "Male",
            "primary_diagnosis": "Primary Open-Angle Glaucoma",
            "severity_level": "Moderate",
            "symptoms": ["mild ocular dryness", "difficulty reading in dim light"],
            "medications": [
                {"name": "Latanoprost", "dosage": "0.005% solution", "frequency": "1 drop both eyes once daily at bedtime"}
            ],
            "vitals_and_labs": {
                "iop_od": "26 mmHg",
                "iop_os": "28 mmHg",
                "cup_to_disc_ratio": "0.75 OD / 0.8 OS"
            },
            "follow_up_days": 30
        }
    },
    {
        "id": "CASE_016",
        "category": "Infectious Disease / ENT",
        "text": (
            "A 33-year-old female presents with 11 days of facial pain, maxilla pressure, thick green nasal discharge, "
            "and impaired sense of smell following an initial viral cold. "
            "She denies visual disturbance, periorbital edema, high fever, stiff neck, or tooth pain. "
            "Anterior rhinoscopy shows erythematous nasal mucosa and purulent drainage in the middle meatus bilaterally. "
            "Vitals: temp 37.3 C, BP 120/78 mmHg. "
            "Diagnosis: Acute Bacterial Rhinosinusitis, mild to moderate severity. "
            "Prescribed Amoxicillin-clavulanate 875/125 mg PO BID for 7 days, Fluticasone furoate nasal spray 2 sprays per nostril daily, "
            "and isotonic saline sinus rinses. Advised to return in 10 days if symptoms worsen or do not remit."
        ),
        "ground_truth": {
            "patient_age": 33,
            "patient_gender": "Female",
            "primary_diagnosis": "Acute Bacterial Rhinosinusitis",
            "severity_level": "Moderate",
            "symptoms": ["facial pain", "maxilla pressure", "thick green nasal discharge", "hyposmia"],
            "medications": [
                {"name": "Amoxicillin-clavulanate", "dosage": "875/125 mg", "frequency": "twice daily"},
                {"name": "Fluticasone furoate spray", "dosage": "2 sprays per nostril", "frequency": "daily"}
            ],
            "vitals_and_labs": {
                "temperature": "37.3 C",
                "blood_pressure": "120/78 mmHg"
            },
            "follow_up_days": 10
        }
    },
    {
        "id": "CASE_017",
        "category": "Gastroenterology",
        "text": (
            "A 49-year-old male presents with recurrent episodes of colicky right upper quadrant abdominal pain triggered "
            "by fatty foods, lasting 2 to 3 hours per episode, with bloating and nausea. "
            "He denies jaundice, fever, chills, tea-colored urine, clay-colored stools, or intractable vomiting. "
            "Physical exam reveals mild tenderness in the right upper quadrant without Murphy sign or peritoneal irritation. "
            "Abdominal ultrasound reveals multiple mobile gallstones within the gallbladder lumen, normal gallbladder wall thickness (2 mm), and common bile duct caliber 4 mm. "
            "Diagnosis: Symptomatic Cholelithiasis (biliary colic), moderate severity. "
            "Prescribed Hyoscine butylbromide 10 mg PO TID PRN for biliary spasms. "
            "Referred for elective laparoscopic cholecystectomy consultation within 21 days."
        ),
        "ground_truth": {
            "patient_age": 49,
            "patient_gender": "Male",
            "primary_diagnosis": "Symptomatic Cholelithiasis",
            "severity_level": "Moderate",
            "symptoms": ["colicky right upper quadrant pain", "postprandial pain", "bloating", "nausea"],
            "medications": [
                {"name": "Hyoscine butylbromide", "dosage": "10 mg", "frequency": "three times daily as needed"}
            ],
            "vitals_and_labs": {
                "gallbladder_wall": "2 mm",
                "cbd_diameter": "4 mm"
            },
            "follow_up_days": 21
        }
    },
    {
        "id": "CASE_018",
        "category": "Neurology",
        "text": (
            "A 64-year-old female presents following an episode of sudden weakness in her right hand and inability to articulate "
            "words clearly (expressive dysphasia) that lasted approximately 35 minutes and completely resolved prior to arrival. "
            "She denies persistent numbness, loss of consciousness, vertigo, visual darkening, or seizure activity. "
            "Vitals: BP 162/96 mmHg, HR 82 bpm regular. Neurological examination is now completely normal (NIHSS score 0). "
            "Brain MRI DWI sequence shows no acute ischemic infarction. Carotid duplex demonstrates 40% stenosis of left internal carotid artery. "
            "Diagnosis: Transient Ischemic Attack (TIA), high risk / severe urgency (ABCD2 score 5). "
            "Initiated dual antiplatelet therapy: Aspirin 81 mg daily and Clopidogrel 75 mg daily for 21 days, plus Atorvastatin 80 mg daily. "
            "Follow-up with vascular neurology in 7 days."
        ),
        "ground_truth": {
            "patient_age": 64,
            "patient_gender": "Female",
            "primary_diagnosis": "Transient Ischemic Attack",
            "severity_level": "Severe",
            "symptoms": ["transient right hand weakness", "expressive dysphasia"],
            "medications": [
                {"name": "Aspirin", "dosage": "81 mg", "frequency": "once daily"},
                {"name": "Clopidogrel", "dosage": "75 mg", "frequency": "once daily"},
                {"name": "Atorvastatin", "dosage": "80 mg", "frequency": "once daily"}
            ],
            "vitals_and_labs": {
                "blood_pressure": "162/96 mmHg",
                "abcd2_score": "5",
                "nihss_score": "0"
            },
            "follow_up_days": 7
        }
    },
    {
        "id": "CASE_019",
        "category": "Cardiology / Vascular",
        "text": (
            "A 55-year-old male presents for cardiovascular risk evaluation. He reports mild exertional shortness of breath "
            "after climbing two flights of stairs, but denies chest tightness, palpitations, syncope, or claudication. "
            "He has a 25 pack-year smoking history. "
            "Physical examination: BP 158/94 mmHg (confirmed on two readings), BMI 29.8 kg/m2, regular pulse 76 bpm. "
            "Lipid panel reveals Total Cholesterol 245 mg/dL, LDL-C 162 mg/dL, HDL-C 38 mg/dL, Triglycerides 225 mg/dL. "
            "Fasting blood glucose 104 mg/dL. "
            "Diagnosis: Stage 2 Essential Hypertension with Mixed Dyslipidemia, moderate severity. "
            "Prescribed Amlodipine 5 mg PO daily, Atorvastatin 40 mg PO nightly, and lifestyle modification counseling. "
            "Return for blood pressure monitoring and liver panel check in 30 days."
        ),
        "ground_truth": {
            "patient_age": 55,
            "patient_gender": "Male",
            "primary_diagnosis": "Stage 2 Essential Hypertension",
            "severity_level": "Moderate",
            "symptoms": ["mild exertional shortness of breath"],
            "medications": [
                {"name": "Amlodipine", "dosage": "5 mg", "frequency": "once daily"},
                {"name": "Atorvastatin", "dosage": "40 mg", "frequency": "once nightly"}
            ],
            "vitals_and_labs": {
                "blood_pressure": "158/94 mmHg",
                "ldl_c": "162 mg/dL",
                "total_cholesterol": "245 mg/dL",
                "triglycerides": "225 mg/dL"
            },
            "follow_up_days": 30
        }
    },
    {
        "id": "CASE_020",
        "category": "Endocrinology / Metabolism",
        "text": (
            "An 8-year-old female is brought by parents to the pediatric clinic due to polydipsia, polyuria, nocturia, "
            "and a 3 kg weight loss over 3 weeks despite increased appetite. "
            "Parents deny vomiting, shortness of breath, abdominal pain, fever, or altered mental status. "
            "Vitals: HR 105 bpm, BP 98/62 mmHg, afebrile, respiration non-labored without Kussmaul breathing. "
            "Point-of-care random blood glucose is 365 mg/dL. Urine dipstick demonstrates 4+ glucose and trace ketones without acidosis. "
            "Venous blood gas reveals normal pH 7.38 and bicarbonate 22 mEq/L (excluding DKA). "
            "Diagnosis: New-Onset Type 1 Diabetes Mellitus without ketoacidosis, severe status. "
            "Immediate pediatric endocrinology consultation initiated. Started Subcutaneous Insulin Glargine 6 units at bedtime "
            "and Insulin Lispro 1 unit per 15 grams carbohydrates. Diabetic education provided. "
            "Daily contact with diabetes team and in-person clinic review in 3 days."
        ),
        "ground_truth": {
            "patient_age": 8,
            "patient_gender": "Female",
            "primary_diagnosis": "Type 1 Diabetes Mellitus",
            "severity_level": "Severe",
            "symptoms": ["polydipsia", "polyuria", "nocturia", "weight loss", "polyphagia"],
            "medications": [
                {"name": "Insulin Glargine", "dosage": "6 units", "frequency": "at bedtime"},
                {"name": "Insulin Lispro", "dosage": "1 unit per 15g carbs", "frequency": "with meals"}
            ],
            "vitals_and_labs": {
                "random_blood_glucose": "365 mg/dL",
                "urine_glucose": "4+",
                "urine_ketones": "trace",
                "vbg_ph": "7.38"
            },
            "follow_up_days": 3
        }
    }
]

def get_benchmark_dataset() -> List[Dict[str, Any]]:
    """Returns the benchmark test dataset."""
    return BENCHMARK_CASES

def get_few_shot_exemplars() -> List[Dict[str, Any]]:
    """Returns the exemplar cases for in-context learning."""
    return FEW_SHOT_EXEMPLARS

def get_case_by_id(case_id: str) -> Optional[Dict[str, Any]]:
    """Returns a specific benchmark case by its unique ID."""
    for case in BENCHMARK_CASES:
        if case["id"] == case_id:
            return case
    return None

def get_categories() -> List[str]:
    """Returns a sorted list of unique clinical specialties in the benchmark."""
    return sorted(list({case["category"] for case in BENCHMARK_CASES}))

def get_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """Returns benchmark cases filtered by clinical specialty."""
    return [case for case in BENCHMARK_CASES if case["category"] == category]

def export_dataset_json(file_path: str = "data/benchmark_cases.json") -> None:
    """Exports benchmark cases to a standardized JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({
            "exemplars": FEW_SHOT_EXEMPLARS,
            "test_cases": BENCHMARK_CASES
        }, f, indent=2)

if __name__ == "__main__":
    export_dataset_json()
    print(f"Exported {len(BENCHMARK_CASES)} benchmark cases and {len(FEW_SHOT_EXEMPLARS)} exemplars to data/benchmark_cases.json")

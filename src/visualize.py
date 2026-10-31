"""
Visualization Engine for Comparative Benchmark Results.
Generates publication-quality charts using Matplotlib and Seaborn:
1. Overall Extraction F1 Comparison (f1_comparison.png)
2. Strict JSON Syntax Validity vs Schema Coverage (schema_validity.png)
3. Latency vs Accuracy Trade-off Curve (latency_vs_f1.png)
4. Granular Field-Level F1 Breakdown (field_level_f1.png)
5. Multi-Criteria Radar Spider Chart (radar_benchmark.png)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS_DIR = "results/plots"
SUMMARY_CSV_PATH = "results/summary_metrics.csv"

# Configure scientific styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14
})

PALETTE = {
    "Zero-Shot": "#e74c3c",          # Red
    "Few-Shot": "#3498db",           # Blue
    "Chain-of-Thought": "#9b59b6",   # Purple
    "Schema-Constrained": "#2ecc71", # Green
    "Role-Expert": "#e67e22"         # Orange
}

def generate_all_plots(summary_csv: str = SUMMARY_CSV_PATH, output_dir: str = PLOTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(summary_csv):
        print(f"Summary metrics file {summary_csv} not found. Run benchmark first.")
        return

    df = pd.read_csv(summary_csv)
    # Ensure ordered techniques
    order = ["Zero-Shot", "Few-Shot", "Chain-of-Thought", "Schema-Constrained", "Role-Expert"]
    df["technique"] = pd.Categorical(df["technique"], categories=order, ordered=True)
    df = df.sort_values("technique").reset_index(drop=True)

    colors = [PALETTE.get(t, "#333333") for t in df["technique"]]

    # 1. Overall Extraction F1 Comparison
    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(df["technique"], df["overall_f1"] * 100, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    plt.ylabel("Extraction F1 Score (%)", fontweight="bold")
    plt.title("Comparative Performance: Overall Information Extraction F1 Score", pad=15, fontweight="bold")
    plt.ylim(0, 105)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 2.0, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_comparison.png"), dpi=300)
    plt.close()

    # 2. Strict JSON Validity vs Schema Coverage
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df["technique"]))
    width = 0.35
    b1 = ax.bar(x - width/2, df["strict_json_valid"] * 100, width, label="Strict RFC 8259 Validity", color="#2980b9", edgecolor="black")
    b2 = ax.bar(x + width/2, df["schema_coverage"] * 100, width, label="Schema Key Adherence", color="#27ae60", edgecolor="black")
    ax.set_ylabel("Compliance Rate (%)", fontweight="bold")
    ax.set_title("Syntactic Integrity: JSON Validity vs Schema Adherence", pad=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["technique"], fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(frameon=True, loc="upper left")
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.0f}%", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "schema_validity.png"), dpi=300)
    plt.close()

    # 3. Latency vs F1 Accuracy Trade-off (Bubble chart)
    plt.figure(figsize=(9, 6))
    for _, row in df.iterrows():
        tech = row["technique"]
        plt.scatter(
            row["latency_sec"],
            row["overall_f1"] * 100,
            s=row["prompt_tokens"] * 1.8, # Bubble size proportional to prompt tokens
            color=PALETTE.get(tech, "#333333"),
            alpha=0.85,
            edgecolors="black",
            linewidth=1.5,
            label=f"{tech} ({int(row['prompt_tokens'])} prompt tok)"
        )
        plt.annotate(
            tech,
            (row["latency_sec"], row["overall_f1"] * 100),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontweight="bold"
        )
    plt.xlabel("Mean Inference Latency (Seconds per Sample)", fontweight="bold")
    plt.ylabel("Extraction F1 Score (%)", fontweight="bold")
    plt.title("Efficiency Frontier: Latency vs Extraction Accuracy (Bubble Size = Prompt Overhead)", pad=15, fontweight="bold")
    plt.ylim(max(0, df["overall_f1"].min() * 100 - 15), 105)
    plt.legend(frameon=True, loc="lower right", title="Technique (Prompt Size)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_vs_f1.png"), dpi=300)
    plt.close()

    # 4. Granular Field-Level F1 Breakdown
    field_cols = {
        "demographics_f1": "Demographics",
        "diagnosis_f1": "Diagnosis",
        "severity_f1": "Severity",
        "symptoms_f1": "Symptoms",
        "medications_f1": "Medications",
        "followup_f1": "Follow-Up"
    }
    melted = df.melt(id_vars=["technique"], value_vars=list(field_cols.keys()), var_name="Field", value_name="F1")
    melted["Field"] = melted["Field"].map(field_cols)

    plt.figure(figsize=(11, 6))
    sns.barplot(data=melted, x="Field", y="F1", hue="technique", palette=PALETTE, edgecolor="black")
    plt.ylabel("Field F1 Score", fontweight="bold")
    plt.title("Granular Field-Level Extraction Performance Across Prompt Techniques", pad=15, fontweight="bold")
    plt.ylim(0, 1.1)
    plt.legend(title="Prompt Technique", frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "field_level_f1.png"), dpi=300)
    plt.close()

    # 5. Radar Chart (Spider Plot)
    categories = ["Schema Integrity", "Diagnosis", "Symptoms", "Medications", "Negation Resistance", "Speed (tok/s)"]
    num_vars = len(categories)

    # Normalize speed between 0 and 1
    max_speed = df["tokens_per_sec"].max() if df["tokens_per_sec"].max() > 0 else 1.0

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # complete loop

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))

    for _, row in df.iterrows():
        tech = row["technique"]
        # Negation resistance = 1.0 - negation_violation_rate
        neg_resist = max(0.0, 1.0 - row["negation_violation_rate"])
        speed_norm = row["tokens_per_sec"] / max_speed

        values = [
            row["schema_coverage"],
            row["diagnosis_f1"],
            row["symptoms_f1"],
            row["medications_f1"],
            neg_resist,
            speed_norm
        ]
        values += values[:1] # complete loop

        ax.plot(angles, values, color=PALETTE.get(tech, "#333333"), linewidth=2, label=tech)
        ax.fill(angles, values, color=PALETTE.get(tech, "#333333"), alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylim(0, 1.05)
    plt.title("Holistic Multi-Dimensional Capability Profile", y=1.08, fontweight="bold")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_benchmark.png"), dpi=300, bbox_inches="tight")
    plt.close('all')
    print(f"Successfully generated all 5 publication-ready plots in '{output_dir}/'", flush=True)

if __name__ == "__main__":
    generate_all_plots()

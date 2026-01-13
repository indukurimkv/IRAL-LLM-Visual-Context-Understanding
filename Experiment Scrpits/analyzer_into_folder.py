import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
RESULT_FILE = "/Users/mohammadeskandari/Documents/Academic/Papers/My Research/Codes/roboflow_hazard_results_20251208_154248.txt"   # ← change this
VALID_LABELS = ["00", "10", "01", "11"]

# Human-readable names for plots
LABEL_MAP = {
    "00": "Safe",
    "10": "Anomalous",
    "01": "Hazardous",
    "11": "An_Hazard",
}
PLOT_LABELS = [LABEL_MAP[l] for l in VALID_LABELS]

# ---- per-run output directory ----
base_dir = os.path.dirname(RESULT_FILE)
runs_root = os.path.join(base_dir, "analysis_runs")
run_dir = os.path.join(runs_root, f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)

OUTPUT_METRICS   = os.path.join(run_dir, f"metrics_report_{timestamp}.txt")
OUTPUT_CSV       = os.path.join(run_dir, f"classification_report_{timestamp}.csv")
OUTPUT_CM        = os.path.join(run_dir, f"confusion_matrix_{timestamp}.png")
OUTPUT_CM_NORM   = os.path.join(run_dir, f"confusion_matrix_normalized_{timestamp}.png")
BAR_METRICS_FILE = os.path.join(run_dir, f"bar_metrics_{timestamp}.png")
SUPPORT_FILE     = os.path.join(run_dir, f"support_{timestamp}.png")
BINARY_BAR_FILE  = os.path.join(run_dir, f"binary_metrics_{timestamp}.png")
ERROR_FILE       = os.path.join(run_dir, f"errors_{timestamp}.png")

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def extract_code(text: str):
    """Extract a 2-bit code from model output (00, 01, 10, 11)."""
    m = re.search(r"\b[01]{2}\b", text)
    return m.group(0) if m else None


# -------------------------------------------------------------
# Parse results
# -------------------------------------------------------------
y_true = []
y_pred = []

with open(RESULT_FILE, "r") as f:
    for line in f:
        line = line.strip()

        if line.startswith("GT:"):
            gt = line.split("GT:")[1].strip()

        if line.startswith("Pred:"):
            pred_raw = line.split("Pred:")[1].strip()
            pred = extract_code(pred_raw)
            if pred is None:
                pred = "??"
            y_true.append(gt)
            y_pred.append(pred)

print(f"\nLoaded {len(y_true)} predictions.\n")

# -------------------------------------------------------------
# Filter invalid predictions (optional)
# -------------------------------------------------------------
clean_true = []
clean_pred = []

for t, p in zip(y_true, y_pred):
    if t in VALID_LABELS:
        clean_true.append(t)
        clean_pred.append(p)

y_true = clean_true
y_pred = clean_pred

# -------------------------------------------------------------
# 4-Class Metrics
# -------------------------------------------------------------
report = classification_report(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    output_dict=True,
    zero_division=0
)

report_text = classification_report(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    zero_division=0
)

# Save text report
with open(OUTPUT_METRICS, "w") as f:
    f.write("FOUR-CLASS REPORT\n")
    f.write(report_text + "\n")

print(report_text)

# Save CSV version
df_report = pd.DataFrame(report).T
df_report.to_csv(OUTPUT_CSV, index=True)

# -------------------------------------------------------------
# Confusion Matrix
# -------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
cm_norm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS, normalize="true")

# Plot raw confusion matrix (axes labeled with names)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=PLOT_LABELS,
    yticklabels=PLOT_LABELS
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.tight_layout()
plt.savefig(OUTPUT_CM)
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=PLOT_LABELS,
    yticklabels=PLOT_LABELS
)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.tight_layout()
plt.savefig(OUTPUT_CM_NORM)
plt.show()

# -------------------------------------------------------------
# Binary Metrics (Anomaly / Hazard)
# -------------------------------------------------------------
true_anom = [int(t[0]) for t in y_true]
true_haz  = [int(t[1]) for t in y_true]
pred_anom = [int(p[0]) if p in VALID_LABELS else 0 for p in y_pred]
pred_haz  = [int(p[1]) if p in VALID_LABELS else 0 for p in y_pred]

prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(
    true_anom, pred_anom, average="binary", zero_division=0
)

prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(
    true_haz, pred_haz, average="binary", zero_division=0
)

binary_report = f"""
===============================
 BINARY ANOMALY DETECTION
===============================
Precision: {prec_a:.3f}
Recall:    {rec_a:.3f}
F1 Score:  {f1_a:.3f}

===============================
 BINARY HAZARD DETECTION
===============================
Precision: {prec_h:.3f}
Recall:    {rec_h:.3f}
F1 Score:  {f1_h:.3f}
"""

print(binary_report)

with open(OUTPUT_METRICS, "a") as f:
    f.write("\n\n" + binary_report + "\n")

# -------------------------------------------------------------
# ADDITIONAL VISUALIZATIONS
# -------------------------------------------------------------

# ===== 1. Bar chart: Precision / Recall / F1 for each class =====
metrics_df = df_report.loc[VALID_LABELS, ["precision", "recall", "f1-score"]]
metrics_df.index = PLOT_LABELS  # use names instead of codes

plt.figure(figsize=(10, 6))
metrics_df.plot(kind="bar", figsize=(10, 6))
plt.title("Per-Class Precision, Recall, F1")
plt.ylabel("Score")
plt.xlabel("Class")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(BAR_METRICS_FILE)
plt.show()

# ===== 2. Bar chart: Support per class =====
support_df = df_report.loc[VALID_LABELS, ["support"]]
support_df.index = PLOT_LABELS  # use names instead of codes

plt.figure(figsize=(8, 5))
support_df.plot(kind="bar", legend=False)
plt.title("Support (Number of Samples) per Class")
plt.ylabel("Count")
plt.xlabel("Class")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(SUPPORT_FILE)
plt.show()

# ===== 3. Binary F1 comparison (Anomaly vs Hazard) =====
binary_scores = pd.DataFrame({
    "task": ["Anomaly", "Hazard"],
    "precision": [prec_a, prec_h],
    "recall": [rec_a, rec_h],
    "f1": [f1_a, f1_h],
})

binary_scores.set_index("task")[["precision", "recall", "f1"]].plot(
    kind="bar", figsize=(8, 6)
)
plt.title("Binary Tasks: Precision / Recall / F1")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(BINARY_BAR_FILE)
plt.show()

# ===== 4. Error distribution =====
errors = [p for t, p in zip(y_true, y_pred) if p != t]
# map codes -> names for plotting
error_df = pd.Series(errors).map(LABEL_MAP).value_counts()

plt.figure(figsize=(6, 4))
error_df.plot(kind="bar", color="salmon")
plt.title("Error Distribution (Wrong Predictions)")
plt.ylabel("Count")
plt.xlabel("Predicted Class")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(ERROR_FILE)
plt.show()

# -------------------------------------------------------------
print(f"\nAll outputs saved under:\n{run_dir}\n")
print("Saved files:")
print(f"- {OUTPUT_METRICS}")
print(f"- {OUTPUT_CSV}")
print(f"- {OUTPUT_CM}")
print(f"- {OUTPUT_CM_NORM}")
print(f"- {BAR_METRICS_FILE}")
print(f"- {SUPPORT_FILE}")
print(f"- {BINARY_BAR_FILE}")
print(f"- {ERROR_FILE}")
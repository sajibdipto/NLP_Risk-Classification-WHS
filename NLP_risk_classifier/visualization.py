# =========================================================
# PLOTTING CODE FOR ROBERTA RESULTS
# Saves each plot separately
# =========================================================

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PATHS
# -----------------------------
BASE = "/content/drive/MyDrive/NLP_risk_classifier"
RISK_RESULT_DIR = os.path.join(BASE, "results", "risk")
UNLABELED_RESULT_DIR = os.path.join(BASE, "results", "filled_master_predictions")
PLOT_DIR = os.path.join(BASE, "results", "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------
# FILE PATHS
# -----------------------------
training_history_file = os.path.join(RISK_RESULT_DIR, "roberta_training_history.csv")
metrics_json_file = os.path.join(RISK_RESULT_DIR, "roberta_metrics.json")
classification_report_file = os.path.join(RISK_RESULT_DIR, "roberta_classification_report.csv")
confusion_matrix_file = os.path.join(RISK_RESULT_DIR, "roberta_confusion_matrix.csv")

filled_master_file = os.path.join(
    UNLABELED_RESULT_DIR,
    "master_data_with_roberta_predictions.xlsx"
)

# =========================================================
# 1. TRAINING CURVES
# =========================================================
history_df = pd.read_csv(training_history_file)

# Plot 1A: Train loss vs Validation loss
plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss")
plt.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RoBERTa Training and Validation Loss")
plt.xticks(history_df["epoch"])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_train_val_loss.png"), dpi=300, bbox_inches="tight")
plt.close()

# Plot 1B: Validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["val_accuracy"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("RoBERTa Validation Accuracy by Epoch")
plt.xticks(history_df["epoch"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_val_accuracy.png"), dpi=300, bbox_inches="tight")
plt.close()

# Plot 1C: Validation Macro F1
plt.figure(figsize=(8, 5))
plt.plot(history_df["epoch"], history_df["val_macro_f1"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Macro F1")
plt.title("RoBERTa Validation Macro F1 by Epoch")
plt.xticks(history_df["epoch"])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_val_macro_f1.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 2. CLASS-WISE F1 BAR CHART
# =========================================================
report_df = pd.read_csv(classification_report_file, index_col=0)

# Keep only actual classes, remove summary rows
summary_rows = ["accuracy", "macro avg", "weighted avg"]
class_report_df = report_df.loc[~report_df.index.isin(summary_rows)].copy()

# Some reports may contain empty rows; drop if needed
class_report_df = class_report_df.dropna(subset=["f1-score"], how="any")

# Sort by F1 for clearer visualization
class_report_df = class_report_df.sort_values("f1-score", ascending=False)

plt.figure(figsize=(12, 7))
plt.bar(class_report_df.index, class_report_df["f1-score"])
plt.xlabel("Risk Category")
plt.ylabel("F1 Score")
plt.title("RoBERTa Class-wise F1 Score")
plt.xticks(rotation=90)
plt.ylim(0, 1.0)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_classwise_f1.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 3. CLASS-WISE PRECISION BAR CHART
# =========================================================
class_report_df_prec = class_report_df.sort_values("precision", ascending=False)

plt.figure(figsize=(12, 7))
plt.bar(class_report_df_prec.index, class_report_df_prec["precision"])
plt.xlabel("Risk Category")
plt.ylabel("Precision")
plt.title("RoBERTa Class-wise Precision")
plt.xticks(rotation=90)
plt.ylim(0, 1.0)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_classwise_precision.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 4. CONFUSION MATRIX HEATMAP-LIKE IMAGE USING MATPLOTLIB
# =========================================================
cm_df = pd.read_csv(confusion_matrix_file, index_col=0)

# If too crowded, keep as full matrix; can also normalize row-wise
cm = cm_df.values.astype(float)

row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

plt.figure(figsize=(12, 10))
plt.imshow(cm_normalized, aspect="auto")
plt.colorbar(label="Normalized Count")
plt.title("RoBERTa Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(len(cm_df.columns)), cm_df.columns, rotation=90, fontsize=8)
plt.yticks(range(len(cm_df.index)), cm_df.index, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_confusion_matrix_normalized.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 5. OVERALL METRICS BAR CHART
# =========================================================
with open(metrics_json_file, "r") as f:
    metrics = json.load(f)

metric_names = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1"
]
metric_values = [metrics[m] for m in metric_names]

plt.figure(figsize=(10, 6))
plt.bar(metric_names, metric_values)
plt.xlabel("Metric")
plt.ylabel("Score")
plt.title("RoBERTa Overall Test Metrics")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1.0)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_overall_metrics.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 6. CONFIDENCE HISTOGRAM (UNLABELED PREDICTIONS)
# =========================================================
filled_df = pd.read_excel(filled_master_file)

# Only rows where model made a prediction
pred_df = filled_df[filled_df["PREDICTION_CONFIDENCE"].notna()].copy()

plt.figure(figsize=(8, 5))
plt.hist(pred_df["PREDICTION_CONFIDENCE"], bins=30)
plt.xlabel("Prediction Confidence")
plt.ylabel("Number of Records")
plt.title("Distribution of RoBERTa Prediction Confidence")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_confidence_histogram.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 7. REVIEW FLAG DISTRIBUTION BAR CHART
# =========================================================
flag_counts = pred_df["PREDICTION_REVIEW_FLAG"].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(flag_counts.index, flag_counts.values)
plt.xlabel("Review Flag")
plt.ylabel("Number of Records")
plt.title("Prediction Review Flag Distribution")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_review_flag_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 8. REVIEW FLAG PIE CHART
# =========================================================
plt.figure(figsize=(6, 6))
plt.pie(
    flag_counts.values,
    labels=flag_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Prediction Review Flag Proportion")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_review_flag_pie.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 9. TOP 15 MOST PREDICTED CLASSES IN UNLABELED DATA
# =========================================================
pred_class_counts = pred_df["PREDICTED_SAFETY_RISK_CATEGORY"].value_counts().head(15)

plt.figure(figsize=(10, 6))
plt.bar(pred_class_counts.index, pred_class_counts.values)
plt.xlabel("Predicted Risk Category")
plt.ylabel("Count")
plt.title("Top 15 Predicted Risk Categories in Unlabeled Data")
plt.xticks(rotation=75, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roberta_top15_predicted_classes.png"), dpi=300, bbox_inches="tight")
plt.close()

print("All plots saved successfully in:")
print(PLOT_DIR)
print("\nSaved files:")
for f in sorted(os.listdir(PLOT_DIR)):
    print("-", f)
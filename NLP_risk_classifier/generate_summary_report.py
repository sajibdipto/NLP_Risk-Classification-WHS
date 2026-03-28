import json
from pathlib import Path
import pandas as pd

BASE = Path("/content/drive/MyDrive/NLP_risk_classifier")

risk_metrics = json.load(open(BASE/"results/risk/metrics.json"))

control_metrics = json.load(open(BASE/"results/controls/metrics.json"))

output = BASE/"results/summary_report.md"

with open(output, "w") as f:
    
    f.write("# 📌 Model Evaluation Summary\n\n")

    f.write("## 1. Risk Category (Single-task RoBERTa)\n")
    f.write(f"- Accuracy: **{risk_metrics['accuracy']:.3f}**\n")
    f.write(f"- Precision: **{risk_metrics['precision']:.3f}**\n")
    f.write(f"- Recall: **{risk_metrics['recall']:.3f}**\n")
    f.write(f"- F1 Score: **{risk_metrics['f1']:.3f}**\n\n")

    f.write("---\n\n## 2. Controls (Multi-task RoBERTa)\n")

    for task, m in control_metrics.items():
        f.write(f"### {task}\n")
        f.write(f"- Accuracy: **{m['accuracy']:.3f}**\n")
        f.write(f"- F1: **{m['f1']:.3f}**\n\n")

    f.write("---\n\n## 3. Key Takeaways\n")
    f.write("- RoBERTa shows strong improvements due to contextual understanding\n")
    f.write("- Multi-task learning shares representations across control tasks\n")
    f.write("- Risk model is performing well with GPU training\n")

print("📄 Summary report created →", output)

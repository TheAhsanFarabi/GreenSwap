"""
GreenSwap — Student 1 (Qwen 2.5 7B)
====================================
Runs the C1/C2/C3 student-phase inferences for Qwen against 24 keyword pairs
(8 per task) on 50 items per dataset, 2 runs per condition.

Total cells:       1,200  (8 pairs x 50 items x 3 tasks)
Per-cell inferences: 6    (3 conditions x 2 runs)
Total inferences:  7,200
Estimated runtime: ~12 hours on 8GB GPU (based on MVP rate)

Outputs (in results/):
    raw_qwen2_5.csv       per-run measurements
    summary_qwen2_5.csv   per-cell medians + deltas (judge columns blank)
    failures_qwen2_5.csv  any errors
    manifest_qwen2_5.json run metadata

Usage:
    ollama serve &
    ollama pull qwen2.5:7b-instruct-q4_K_M
    python run_student1.py
"""

from _student_runner import run_student

MODEL = "qwen2.5:7b-instruct-q4_K_M"


if __name__ == "__main__":
    run_student(MODEL)

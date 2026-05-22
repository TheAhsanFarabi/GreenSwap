"""
GreenSwap — Student 2 (Llama 3.1 8B)
====================================
Runs the C1/C2/C3 student-phase inferences for Llama against 24 keyword pairs
(8 per task) on 50 items per dataset, 2 runs per condition.

Outputs (in results/):
    raw_llama3_1.csv
    summary_llama3_1.csv
    failures_llama3_1.csv
    manifest_llama3_1.json

Usage:
    ollama serve &
    ollama pull llama3.1:8b-instruct-q4_K_M
    python run_student2.py
"""

from _student_runner import run_student

MODEL = "llama3.1:8b-instruct-q4_K_M"


if __name__ == "__main__":
    run_student(MODEL)

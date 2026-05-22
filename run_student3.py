"""
GreenSwap — Student 3 (Mistral 7B v0.3)
========================================
Runs the C1/C2/C3 student-phase inferences for Mistral against 24 keyword pairs
(8 per task) on 50 items per dataset, 2 runs per condition.

Outputs (in results/):
    raw_mistral.csv
    summary_mistral.csv
    failures_mistral.csv
    manifest_mistral.json

Usage:
    ollama serve &
    ollama pull mistral:7b-instruct-v0.3-q4_K_M
    python run_student3.py
"""

from _student_runner import run_student

MODEL = "mistral:7b-instruct-v0.3-q4_K_M"


if __name__ == "__main__":
    run_student(MODEL)

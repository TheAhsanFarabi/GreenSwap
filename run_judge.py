"""
GreenSwap — Judge Runner (Gemma 2 9B)
======================================
Reads each per-student summary CSV (summary_qwen2_5.csv, summary_llama3_1.csv,
summary_mistral.csv), judges all (cell, condition) outputs with Gemma 2 9B,
computes lexical metrics (EM/F1 for QA, accuracy for SST-2), and writes the
results back to the same file.

Robustness:
1. Preflight: Ollama reachable + Gemma pulled + summary files exist + smoke test
2. Fail-fast on first ConnectionError / Timeout / ModelNotFound
3. Resume: skip cells where judge already ran successfully
4. Periodic checkpoints (every 10 rows)
5. Per-student tqdm progress bars
6. Post-run sanity report (score distribution; warns if >20% are score=0)

Outputs (rewrites in place):
    results/summary_qwen2_5.csv
    results/summary_llama3_1.csv
    results/summary_mistral.csv

Usage:
    ollama serve &
    ollama pull gemma2:9b-instruct-q4_K_M
    python run_judge.py
"""

from __future__ import annotations

import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import ollama
import pandas as pd
from tqdm.auto import tqdm

import greenswap_lib as gl


# =============================================================================
# Configuration
# =============================================================================

JUDGE_MODEL = gl.JUDGE_MODEL
OUTPUT_DIR = gl.OUTPUT_DIR
RESUME = True
CHECKPOINT_EVERY = 10
ITEMS_PER_DATASET = gl.ITEMS_PER_DATASET

SUMMARY_FILE_PATTERN = "summary_*.csv"   # discovers all student summaries


# =============================================================================
# Preflight
# =============================================================================

def preflight(summary_files: list[Path]) -> None:
    print("=" * 70)
    print("Preflight checks")
    print("=" * 70)

    # 1. Ollama reachable
    print("[1/4] Ollama reachable... ", end="", flush=True)
    try:
        listed = ollama.list()
    except Exception as e:
        print("FAIL")
        print(f"\n[fatal] Cannot reach Ollama: {type(e).__name__}: {e}")
        print("        Fix: start `ollama serve` and verify with `curl http://localhost:11434/api/tags`")
        sys.exit(1)
    print("ok")

    # 2. Judge model pulled
    print(f"[2/4] {JUDGE_MODEL} pulled... ", end="", flush=True)
    models_field = listed.get("models", []) if isinstance(listed, dict) else getattr(listed, "models", [])
    names = set()
    for m in models_field:
        if isinstance(m, dict):
            names.add(m.get("model") or m.get("name") or "")
        else:
            names.add(getattr(m, "model", None) or getattr(m, "name", None) or "")
    if JUDGE_MODEL not in names:
        print("FAIL")
        print(f"\n[fatal] {JUDGE_MODEL} is not pulled.")
        print(f"        Fix: ollama pull {JUDGE_MODEL}")
        sys.exit(1)
    print("ok")

    # 3. Summary files exist
    print(f"[3/4] Summary files... ", end="", flush=True)
    if not summary_files:
        print("FAIL")
        print(f"\n[fatal] No summary CSVs found in {OUTPUT_DIR}/")
        print("        Run student scripts first (run_student1.py etc.)")
        sys.exit(1)
    print(f"found {len(summary_files)}")
    for sf in summary_files:
        print(f"         - {sf.name}")

    # 4. Smoke test the judge
    print(f"[4/4] {JUDGE_MODEL} smoke test... ", end="", flush=True)
    smoke = "Return ONLY this JSON: {\"score\": 4, \"reason\": \"smoke test\"}"
    try:
        t0 = time.perf_counter()
        resp = ollama.generate(
            model=JUDGE_MODEL,
            prompt=smoke,
            options={"temperature": 0, "num_predict": 60, "num_ctx": gl.NUM_CTX},
            stream=False,
        )
        elapsed = time.perf_counter() - t0
        out = resp.get("response", "")
        score, _ = gl._extract_json_score(out)
        if score == 0:
            print("FAIL")
            print(f"\n[fatal] Judge model reachable but cannot parse its output.")
            print(f"        Raw: {out[:200]}")
            sys.exit(1)
    except Exception as e:
        print("FAIL")
        print(f"\n[fatal] Smoke test threw {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
    print(f"ok  (score={score}, {elapsed:.1f}s)")
    print()


# =============================================================================
# Resume detection
# =============================================================================

def cell_already_judged(row: pd.Series) -> bool:
    for cond in ("c1", "c2", "c3"):
        v = row.get(f"{cond}_judge_score")
        try:
            iv = int(float(v))
            if 1 <= iv <= 5:
                return True
        except (TypeError, ValueError):
            continue
    return False


# =============================================================================
# Per-file judging
# =============================================================================

def judge_one_file(summary_path: Path,
                   question_cache: dict[str, str]) -> dict:
    """Judge one student's summary CSV. Returns a stats dict for the report."""
    print(f"\n[file] {summary_path.name}")
    df = pd.read_csv(summary_path)
    if df.empty:
        print(f"  [skip] empty")
        return {"file": summary_path.name, "cells": 0, "calls": 0, "scores": Counter()}

    # Coerce placeholder columns to object dtype so we can write strings & ints
    placeholder_cols = []
    for c in ("c1", "c2", "c3"):
        placeholder_cols.extend([
            f"{c}_judge_score", f"{c}_judge_reason",
            f"{c}_em", f"{c}_f1", f"{c}_accuracy",
        ])
    for col in placeholder_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")

    already_done = 0
    if RESUME:
        already_done = sum(cell_already_judged(row) for _, row in df.iterrows())
    cells_todo = len(df) - already_done
    calls = cells_todo * 3
    if cells_todo == 0:
        print(f"  [skip] all {len(df)} cells already judged")
        return {"file": summary_path.name, "cells": 0, "calls": 0, "scores": Counter()}

    if already_done > 0:
        print(f"  [resume] {already_done}/{len(df)} cells already judged")

    pbar = tqdm(total=calls, desc=summary_path.stem, unit="call", ncols=110)
    score_hist: Counter = Counter()
    t_start = time.perf_counter()

    for idx, row in df.iterrows():
        if RESUME and cell_already_judged(row):
            continue

        item_id = str(row["item_id"])
        task_type = row["task_type"]
        question = question_cache.get(item_id, "")
        ref_val = row.get("reference")
        reference = (
            "" if (ref_val is None or (isinstance(ref_val, float) and pd.isna(ref_val)))
            else str(ref_val)
        )

        for cond in ("c1", "c2", "c3"):
            out_val = row.get(f"{cond}_output")
            out_text = (
                "" if (out_val is None or (isinstance(out_val, float) and pd.isna(out_val)))
                else str(out_val)
            )

            # Reconstruct an Item-like wrapper for compute_lexical
            class _Item:
                pass
            it = _Item()
            it.task_type = task_type
            it.reference = reference

            j = gl.judge_response(question, reference, out_text, task_type, JUDGE_MODEL)
            df.at[idx, f"{cond}_judge_score"] = j["judge_score"]
            df.at[idx, f"{cond}_judge_reason"] = j["judge_reason"]
            score_hist[j["judge_score"]] += 1

            lex = gl.compute_lexical(out_text, it)
            df.at[idx, f"{cond}_em"] = "" if lex["em"] is None else lex["em"]
            df.at[idx, f"{cond}_f1"] = "" if lex["f1"] is None else lex["f1"]
            df.at[idx, f"{cond}_accuracy"] = "" if lex["accuracy"] is None else lex["accuracy"]

            pbar.update(1)
            pbar.set_postfix(
                zero=score_hist.get(0, 0),
                valid=sum(score_hist.get(s, 0) for s in range(1, 6)),
                latest=j["judge_score"],
            )

        # checkpoint write
        if (int(idx) + 1) % CHECKPOINT_EVERY == 0:
            for col in [f"{c}_judge_score" for c in ("c1", "c2", "c3")]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            df.to_csv(summary_path, index=False)

    pbar.close()

    # final write — coerce score columns cleanly to int
    for col in [f"{c}_judge_score" for c in ("c1", "c2", "c3")]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df.to_csv(summary_path, index=False)
    elapsed_min = (time.perf_counter() - t_start) / 60
    print(f"  [done] {calls} calls in {elapsed_min:.1f} min")

    return {"file": summary_path.name, "cells": cells_todo, "calls": calls, "scores": score_hist}


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    summary_files = sorted(OUTPUT_DIR.glob(SUMMARY_FILE_PATTERN))
    preflight(summary_files)

    # Build the question cache once (same for all student files)
    print("[load] rebuilding question text from datasets")
    qcache: dict[str, str] = {}
    for loader_name, loader_fn in [
        ("TriviaQA", gl.load_triviaqa),
        ("ELI5", gl.load_eli5),
        ("SST-2", gl.load_sst2),
    ]:
        try:
            items = loader_fn(ITEMS_PER_DATASET)
            for it in items:
                qcache[it.item_id] = it.question
        except Exception as e:
            print(f"[warn] {loader_name} reload failed: {e}")
    print(f"[load] {len(qcache)} questions cached")

    # Judge each student's summary file
    t_start = time.perf_counter()
    all_stats = []
    for sf in summary_files:
        try:
            stats = judge_one_file(sf, qcache)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n[fatal] judging {sf.name} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            sys.exit(1)

    total_elapsed_min = (time.perf_counter() - t_start) / 60
    print("\n" + "=" * 70)
    print(f"[done] judging across {len(summary_files)} files in {total_elapsed_min:.1f} min")
    print("=" * 70)

    # Aggregate sanity report
    combined = Counter()
    for s in all_stats:
        combined.update(s["scores"])
    total = sum(combined.values())

    print("\nJudge score distribution (all files combined):")
    if total == 0:
        print("  No new calls (all cells were resume-skipped)")
    else:
        for s in [0, 1, 2, 3, 4, 5]:
            count = combined.get(s, 0)
            pct = count / total * 100
            label = "INVALID (0)" if s == 0 else f"valid {s}"
            print(f"  {label:<15} {count:>5}  ({pct:>5.1f}%)")
        invalid_pct = combined.get(0, 0) / total * 100
        if invalid_pct > 20:
            print(f"\n[WARN] {invalid_pct:.0f}% of judge calls returned score=0.")
            print("       Inspect *_judge_reason columns to diagnose.")

    # Per-file delta summary
    print("\n" + "=" * 70)
    print("Energy + judge medians per student / task")
    print("=" * 70)
    for sf in summary_files:
        try:
            df = pd.read_csv(sf)
            print(f"\n{sf.name}:")
            for task in ["QA", "Generation", "Sentiment"]:
                tsub = df[df["task_type"] == task]
                if len(tsub) == 0:
                    continue
                jc1 = pd.to_numeric(tsub["c1_judge_score"], errors="coerce").median()
                jc2 = pd.to_numeric(tsub["c2_judge_score"], errors="coerce").median()
                jc3 = pd.to_numeric(tsub["c3_judge_score"], errors="coerce").median()
                print(
                    f"  {task:<11} n={len(tsub):>3}  "
                    f"dE(C2vC1)={tsub['delta_energy_c2_vs_c1_pct'].median():+6.1f}%  "
                    f"dE(C3vC1)={tsub['delta_energy_c3_vs_c1_pct'].median():+6.1f}%  "
                    f"dE(C2vC3)={tsub['delta_energy_c2_vs_c3_pct'].median():+6.1f}%  "
                    f"judge: C1={jc1:.1f} C2={jc2:.1f} C3={jc3:.1f}"
                )
        except Exception as e:
            print(f"  [stats error] {e}")


if __name__ == "__main__":
    main()

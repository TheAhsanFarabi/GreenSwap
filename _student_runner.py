"""
Reusable student-runner driver.
Each run_student{1,2,3}.py imports this and calls run_student(MODEL).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

import greenswap_lib as gl


def run_student(model: str,
                pairs_per_task: int | None = None,
                items_per_dataset: int | None = None,
                runs_per_condition: int | None = None,
                keywords_csv: str | None = None,
                output_dir: Path | None = None) -> None:
    """Run student inference for one model. Writes per-model raw/summary CSVs.

    Defaults pull from greenswap_lib (PAIRS_PER_TASK=8, ITEMS_PER_DATASET=50,
    RUNS_PER_CONDITION=2). Override any of them to scale up or down.
    """
    pairs_per_task = pairs_per_task or gl.PAIRS_PER_TASK
    items_per_dataset = items_per_dataset or gl.ITEMS_PER_DATASET
    runs_per_condition = runs_per_condition or gl.RUNS_PER_CONDITION
    keywords_csv = keywords_csv or gl.KEYWORDS_CSV
    output_dir = output_dir or gl.OUTPUT_DIR

    # ---- Per-model output paths ----
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker_dir = output_dir / "_codecarbon_tmp"
    tracker_dir.mkdir(exist_ok=True)

    # Use the model's first token (e.g. "qwen2.5") as a filename-safe slug
    slug = model.split(":")[0].replace(".", "_").replace("/", "_")
    raw_path = output_dir / f"raw_{slug}.csv"
    summary_path = output_dir / f"summary_{slug}.csv"
    failures_path = output_dir / f"failures_{slug}.csv"
    manifest_path = output_dir / f"manifest_{slug}.json"

    for p in (raw_path, summary_path, failures_path):
        if p.exists():
            p.unlink()

    print("=" * 70)
    print(f"GreenSwap student runner: {model}")
    print("=" * 70)

    gl.check_model_available(model)

    pairs = gl.load_keyword_pairs(keywords_csv, pairs_per_task)
    print(f"[load] {len(pairs)} pairs ({pairs_per_task} per task)")

    items_by_task = gl.load_all_items(items_per_dataset)
    print(f"[load] {sum(len(v) for v in items_by_task.values())} items "
          f"({items_per_dataset} per dataset)")

    # ---- Build the cell list so tqdm has the correct total ----
    schedule: list[tuple[gl.KeywordPair, gl.Item]] = []
    for pair in pairs:
        for item in items_by_task[pair.task_type]:
            schedule.append((pair, item))
    total_cells = len(schedule)
    inferences_per_cell = 3 * runs_per_condition  # 3 conditions
    total_inferences = total_cells * inferences_per_cell

    print(f"[plan] {total_cells} cells x {inferences_per_cell} inferences = "
          f"{total_inferences} total student inferences")
    print("=" * 70)

    gl.warmup(model, tracker_dir)

    first_raw = True
    first_summary = True
    first_fail = True
    failed_cells = 0
    t_start = time.perf_counter()

    pbar = tqdm(total=total_cells, desc=slug, unit="cell", ncols=110)

    for (pair, item) in schedule:
        try:
            raw_rows, summary_row = gl.run_cell(
                model, pair, item, tracker_dir,
                runs_per_condition=runs_per_condition,
            )
        except Exception as e:
            failed_cells += 1
            err_row = {
                "student_model": model,
                "pair_id": pair.pair_id,
                "item_id": item.item_id,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[:1000],
            }
            gl.append_rows(failures_path, [err_row],
                           ["student_model", "pair_id", "item_id", "error", "traceback"],
                           first_fail)
            first_fail = False
            tqdm.write(f"  [cell-fail] {pair.pair_id} {item.item_id}: {e}")
            pbar.update(1)
            continue

        gl.append_rows(raw_path, raw_rows, gl.RAW_HEADER, first_raw)
        first_raw = False

        if summary_row is not None:
            gl.append_rows(summary_path, [summary_row], gl.SUMMARY_HEADER, first_summary)
            first_summary = False
            pbar.set_postfix(
                pair=pair.pair_id,
                c1=int(summary_row["c1_tokens_med"]),
                c2=int(summary_row["c2_tokens_med"]),
                c3=int(summary_row["c3_tokens_med"]),
                dE=f'{summary_row["delta_energy_c2_vs_c1_pct"]:+.0f}%',
            )
        else:
            failed_cells += 1

        pbar.update(1)

    pbar.close()
    elapsed_min = (time.perf_counter() - t_start) / 60
    print(f"\n[done] {total_cells - failed_cells}/{total_cells} cells in "
          f"{elapsed_min:.1f} min")

    # ---- Manifest ----
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "student_model": model,
        "pairs_csv": keywords_csv,
        "pairs_per_task": pairs_per_task,
        "items_per_dataset": items_per_dataset,
        "runs_per_condition": runs_per_condition,
        "num_predict": gl.NUM_PREDICT,
        "num_ctx": gl.NUM_CTX,
        "temperature": gl.TEMPERATURE,
        "seed": gl.SEED,
        "cells_total": total_cells,
        "cells_failed": failed_cells,
        "elapsed_min": round(elapsed_min, 2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("=" * 70)
    print(f"[outputs] raw:      {raw_path}")
    print(f"[outputs] summary:  {summary_path}")
    print(f"[outputs] manifest: {manifest_path}")
    if failed_cells > 0:
        print(f"[outputs] failures: {failures_path}")
    print("=" * 70)

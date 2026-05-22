"""
GreenSwap shared library
========================
Shared helpers used by run_student1.py, run_student2.py, run_student3.py,
and run_judge.py. Keeping these in one file avoids drift between scripts.

Public surface:
    Configuration constants:
        PAIRS_PER_TASK, ITEMS_PER_DATASET, RUNS_PER_CONDITION
        NUM_PREDICT, NUM_CTX, TEMPERATURE, SEED, MIN_C3_BUDGET
        OUTPUT_DIR, KEYWORDS_CSV

    Dataclasses:
        KeywordPair, Item, Measurement

    Loaders:
        load_keyword_pairs(path, pairs_per_task)
        load_triviaqa(n), load_eli5(n), load_sst2(n)
        load_all_items(items_per_dataset)

    Inference + measurement:
        measure_one(model, prompt, tracker_dir)
        render_prompt(verb, task_type, item, length_budget=None)
        run_cell(model, pair, item, tracker_dir)
        warmup(model, tracker_dir)
        check_model_available(model)

    Judge:
        judge_response(item, response, judge_model)
        _extract_json_score(text)
        JUDGE_MODEL  (default: gemma2:9b-instruct-q4_K_M)

    Lexical:
        compute_lexical(prediction, item)
        exact_match, token_f1, sst2_label_accuracy

    Row builders / writers:
        RAW_HEADER, SUMMARY_HEADER
        _summary_row, _row
        append_rows(path, rows, header, first_write)
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import ollama
import pandas as pd
from codecarbon import EmissionsTracker
from datasets import load_dataset


# =============================================================================
# Configuration (defaults; runners can override before calling functions)
# =============================================================================

KEYWORDS_CSV = "greenswap_keywords.csv"
OUTPUT_DIR = Path("results")

# How much of the data each runner uses by default. Runners can override these
# by setting their own values before calling load_* functions.
PAIRS_PER_TASK = 8           # full scale; set to 3 for MVP
ITEMS_PER_DATASET = 50       # full scale; set to 3 for MVP
RUNS_PER_CONDITION = 2       # 2 runs gives 0.95 power on observed effects; cite Adamska single-run

MIN_C3_BUDGET = 5
NUM_PREDICT = 512
NUM_CTX = 2048
TEMPERATURE = 0
SEED = 42

JUDGE_MODEL = "gemma2:9b-instruct-q4_K_M"


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class KeywordPair:
    pair_id: str
    task_type: str
    original_verb: str
    green_verb: str


@dataclass
class Item:
    item_id: str
    dataset: str
    task_type: str
    question: str
    reference: str


@dataclass
class Measurement:
    output: str
    output_tokens: int
    prompt_tokens: int
    char_length: int
    word_count: int
    energy_kwh: float
    co2eq_kg: float
    wall_latency_s: float
    eval_duration_ns: int
    truncated: bool
    error: str = ""


# =============================================================================
# Pre-flight check
# =============================================================================

def check_model_available(model: str) -> None:
    """Verify the given model is pulled into Ollama. Exit with a clear message
    if not. Run this once at the start of every runner."""
    try:
        listed = ollama.list()
    except Exception as e:
        print(f"\n[fatal] Cannot reach Ollama: {type(e).__name__}: {e}")
        print("        Fix: start `ollama serve` in another terminal,")
        print("             then verify with `curl http://localhost:11434/api/tags`")
        sys.exit(1)

    models_field = listed.get("models", []) if isinstance(listed, dict) else getattr(listed, "models", [])
    names = set()
    for m in models_field:
        if isinstance(m, dict):
            names.add(m.get("model") or m.get("name") or "")
        else:
            names.add(getattr(m, "model", None) or getattr(m, "name", None) or "")

    if model not in names:
        print(f"\n[fatal] Model '{model}' is not pulled in Ollama.")
        print(f"        Available: {sorted(n for n in names if n)}")
        print(f"        Fix: ollama pull {model}")
        sys.exit(1)


# =============================================================================
# Dataset loaders
# =============================================================================

def load_keyword_pairs(path: str, pairs_per_task: int) -> list[KeywordPair]:
    df = pd.read_csv(path)
    required = {"pair_id", "task_type", "original_verb", "green_verb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    selected = []
    for task in ["QA", "Generation", "Sentiment"]:
        task_pairs = df[df["task_type"] == task].head(pairs_per_task)
        for _, row in task_pairs.iterrows():
            selected.append(KeywordPair(
                pair_id=row["pair_id"],
                task_type=row["task_type"],
                original_verb=row["original_verb"],
                green_verb=row["green_verb"],
            ))
    return selected


def load_triviaqa(n: int) -> list[Item]:
    ds = load_dataset(
        "trivia_qa", "rc.nocontext",
        split=f"validation[:{n * 5}]",
        trust_remote_code=True,
    )
    items: list[Item] = []
    for i, row in enumerate(ds):
        if len(items) >= n:
            break
        q = (row.get("question") or "").strip()
        ref = (row.get("answer", {}).get("value") or "").strip()
        if q and ref:
            items.append(Item(f"trivia_{i:03d}", "triviaqa", "QA", q, ref))
    return items


def load_eli5(n: int) -> list[Item]:
    try:
        ds = load_dataset("kilt_tasks", "eli5", split=f"validation[:{n * 5}]")
        items: list[Item] = []
        for i, row in enumerate(ds):
            if len(items) >= n:
                break
            q = (row.get("input") or "").strip()
            ref = ""
            outs = row.get("output") or []
            for o in outs:
                if isinstance(o, dict) and o.get("answer"):
                    ref = o["answer"].strip()
                    break
            if q:
                items.append(Item(f"eli5_{i:03d}", "eli5", "Generation", q, ref))
        return items
    except Exception as e:
        print(f"[warn] kilt_tasks/eli5 failed ({e}); falling back to eli5_category")
        ds = load_dataset("eli5_category", split=f"validation1[:{n * 5}]")
        items = []
        for i, row in enumerate(ds):
            if len(items) >= n:
                break
            q = (row.get("title") or "").strip()
            answers = row.get("answers", {})
            ref = ""
            if isinstance(answers, dict) and answers.get("text"):
                ref = answers["text"][0]
            if q:
                items.append(Item(f"eli5_{i:03d}", "eli5", "Generation", q, ref))
        return items


def load_sst2(n: int) -> list[Item]:
    ds = load_dataset("glue", "sst2", split=f"validation[:{n * 5}]")
    items: list[Item] = []
    for i, row in enumerate(ds):
        if len(items) >= n:
            break
        text = (row.get("sentence") or "").strip()
        label = "positive" if row.get("label") == 1 else "negative"
        if text:
            items.append(Item(f"sst2_{i:03d}", "sst2", "Sentiment", text, label))
    return items


def load_all_items(items_per_dataset: int) -> dict[str, list[Item]]:
    return {
        "QA": load_triviaqa(items_per_dataset),
        "Generation": load_eli5(items_per_dataset),
        "Sentiment": load_sst2(items_per_dataset),
    }


# =============================================================================
# Prompt rendering
# =============================================================================

def _cap(s: str) -> str:
    return (s[0].upper() + s[1:]) if s else s


def render_prompt(verb: str, task_type: str, item: Item,
                  length_budget: int | None = None) -> str:
    """Build the C1/C2/C3 prompt.

    All verbs in the CSV are single-word imperatives. The template differs only
    by task type:

      QA / Generation:
          "{Verb} the answer to the following question.\\n\\nQuestion: ..."
      Sentiment:
          "{Verb} the sentiment of the following review as positive or negative.\\n\\nReview: ..."

    Both C1 (original verb) and C2 (green verb) use the same template — only
    the verb token differs. This is the cleanest possible operationalization
    of the GreenSwap hypothesis: holding everything constant except the verb.

    C3 appends a length constraint to the original-verb prompt.
    """
    verb_clean = verb.strip()
    verb_cap = _cap(verb_clean)

    if task_type == "Sentiment":
        base = f"{verb_cap} the sentiment of the following review as positive or negative."
        body = f"Review: {item.question}"
    else:
        base = f"{verb_cap} the answer to the following question."
        body = f"Question: {item.question}"

    if length_budget is not None:
        base = f"{base} Answer in {length_budget} tokens or fewer."

    return f"{base}\n\n{body}"


# =============================================================================
# Inference + CodeCarbon measurement
# =============================================================================

def _safe_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def measure_one(model: str, prompt: str, tracker_dir: Path) -> Measurement:
    tracker = EmissionsTracker(
        project_name="greenswap",
        output_dir=str(tracker_dir),
        log_level="error",
        save_to_file=False,
        tracking_mode="process",
        measure_power_secs=0.5,  # FIXED: Lowered polling interval from 1 to 0.5
        allow_multiple_runs=True,
    )

    error = ""
    output = ""
    eval_count = 0
    prompt_eval_count = 0
    eval_duration_ns = 0

    tracker.start()
    t0 = time.perf_counter()
    try:
        resp = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": TEMPERATURE,
                "seed": SEED,
                "num_predict": NUM_PREDICT,
                "num_ctx": NUM_CTX,
            },
            stream=False,
        )
        output = resp.get("response", "")
        eval_count = _safe_int(resp.get("eval_count"))
        prompt_eval_count = _safe_int(resp.get("prompt_eval_count"))
        eval_duration_ns = _safe_int(resp.get("eval_duration"))
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        try:
            tracker.stop()
        except Exception:
            pass

    wall_latency = time.perf_counter() - t0
    try:
        energy_kwh = float(tracker.final_emissions_data.energy_consumed)
        co2eq_kg = float(tracker.final_emissions_data.emissions or 0.0)
    except Exception:
        energy_kwh, co2eq_kg = 0.0, 0.0

    return Measurement(
        output=output,
        output_tokens=eval_count,
        prompt_tokens=prompt_eval_count,
        char_length=len(output),
        word_count=len(output.split()),
        energy_kwh=energy_kwh,
        co2eq_kg=co2eq_kg,
        wall_latency_s=wall_latency,
        eval_duration_ns=eval_duration_ns,
        truncated=eval_count >= NUM_PREDICT,
        error=error,
    )


def warmup(model: str, tracker_dir: Path) -> None:
    try:
        measure_one(model, "Answer briefly. Question: What is 2+2?", tracker_dir)
    except Exception as e:
        print(f"[warmup] non-fatal: {e}")


# =============================================================================
# Lexical quality metrics
# =============================================================================

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> int:
    if not reference:
        return 0
    return int(_normalize(reference) in _normalize(prediction))


def token_f1(prediction: str, reference: str) -> float:
    if not reference:
        return 0.0
    pred = _normalize(prediction).split()
    ref = _normalize(reference).split()
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred)
    r = num_same / len(ref)
    return 2 * p * r / (p + r)


def sst2_label_accuracy(prediction: str, reference: str) -> int:
    # FIXED: First-occurrence wins logic replaces brittle slicing
    pred = prediction.lower()
    pos_idx = pred.find("positive")
    neg_idx = pred.find("negative")

    if reference == "positive":
        return int(pos_idx != -1 and (neg_idx == -1 or pos_idx < neg_idx))
    if reference == "negative":
        return int(neg_idx != -1 and (pos_idx == -1 or neg_idx < pos_idx))
    
    return 0


def compute_lexical(prediction: str, item: Item) -> dict:
    if item.task_type == "QA":
        return {"em": exact_match(prediction, item.reference),
                "f1": round(token_f1(prediction, item.reference), 4),
                "accuracy": None}
    if item.task_type == "Sentiment":
        return {"em": None, "f1": None,
                "accuracy": sst2_label_accuracy(prediction, item.reference)}
    return {"em": None, "f1": None, "accuracy": None}


# =============================================================================
# Judge (used by run_judge.py)
# =============================================================================

JUDGE_PROMPT = """You are an expert evaluator. Score the model response on a 1-5 scale.

Scoring rubric:
1 = incorrect or irrelevant
2 = partially correct, major issues
3 = correct but verbose or minor issues
4 = correct and reasonably concise
5 = correct, concise, complete

Question: {question}
Reference Answer: {reference}
Model Response: {response}

Return ONLY a JSON object on a single line: {{"score": <1-5>, "reason": "<one sentence>"}}"""

SENTIMENT_JUDGE_PROMPT = """You are an expert evaluator for sentiment classification. Score 1-5.

Rubric:
1 = wrong sentiment or nonsense
2 = ambiguous
3 = correct but excessively elaborated
4 = correct with appropriate brevity
5 = correct, very concise

Review: {question}
Gold Sentiment: {reference}
Model Response: {response}

Return ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def _extract_json_score(text: str) -> tuple[int, str]:
    """Robust score extraction. Returns (score 1-5, reason).
    Returns (0, ...) only when all parsing strategies fail."""
    if not text:
        return 0, ""
    text = text.strip()
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE).strip()

    # Try 1: parse whole thing
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            s = int(obj["score"])
            if 1 <= s <= 5:
                return s, str(obj.get("reason", ""))[:300]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try 2: embedded {...}
    for m in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        snippet = m.group(0)
        if "score" not in snippet.lower():
            continue
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict) and "score" in obj:
                s = int(obj["score"])
                if 1 <= s <= 5:
                    return s, str(obj.get("reason", ""))[:300]
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Try 3: 'score': N pattern
    m = re.search(r"['\"]?score['\"]?\s*[:=]\s*(\d)", text, flags=re.IGNORECASE)
    if m:
        s = int(m.group(1))
        if 1 <= s <= 5:
            return s, text[:300]

    # Try 4: "Score is N"
    m = re.search(r"\bscore\s*(?:is|=|:)?\s*(\d)\b", text, flags=re.IGNORECASE)
    if m:
        s = int(m.group(1))
        if 1 <= s <= 5:
            return s, text[:300]

    # Try 5: standalone 1-5 digit (after stripping rubric echoes)
    cleaned = re.sub(r"\b1\s*(?:to|-|\u2013|\u2014)\s*5\b", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b[1-5]\s*(?:=|\.)\s*\w+", "", cleaned)
    m = re.search(r"(?<!\d)([1-5])(?!\d)", cleaned)
    if m:
        return int(m.group(1)), text[:300]

    return 0, text[:300]


# Module-level flag so the judge fails LOUDLY once on a connection problem
# instead of silently returning 0 for every cell.
_first_critical_failure = True


def judge_response(question: str, reference: str, response: str,
                   task_type: str, judge_model: str = JUDGE_MODEL) -> dict:
    """Call the judge model and return {"judge_score", "judge_reason"}.
    Fails fast on first ConnectionError / Timeout / ModelNotFound."""
    global _first_critical_failure

    if not response or not response.strip():
        return {"judge_score": 0, "judge_reason": "empty_response"}

    template = SENTIMENT_JUDGE_PROMPT if task_type == "Sentiment" else JUDGE_PROMPT
    ref = reference if reference else "(no reference provided)"
    prompt = template.format(question=question, reference=ref, response=response)

    try:
        resp = ollama.generate(
            model=judge_model,
            prompt=prompt,
            options={"temperature": 0, "seed": SEED, "num_predict": 120, "num_ctx": NUM_CTX},
            stream=False,
        )
        raw = resp.get("response", "")
        score, reason = _extract_json_score(raw)
        return {"judge_score": score, "judge_reason": reason}
    except Exception as e:
        err_name = type(e).__name__
        critical = ("Connection", "Timeout", "ResponseError", "ModelNot")
        is_critical = any(marker in err_name for marker in critical)

        if is_critical and _first_critical_failure:
            _first_critical_failure = False
            print("")
            print("=" * 70)
            print(f"[FATAL] Judge call threw {err_name} on first attempt.")
            print(f"        Detail: {e}")
            print(f"        Likely cause: Ollama stopped or {judge_model} unloaded.")
            print(f"        Aborting to avoid silently writing 0s for every cell.")
            print(f"        Check: curl http://localhost:11434/api/tags")
            print("=" * 70)
            sys.exit(1)

        return {"judge_score": 0, "judge_reason": f"judge_error: {err_name}"}


# =============================================================================
# Cell runner: C1 -> C2 -> C3 with runtime C3 budget
# =============================================================================

def run_cell(model: str, pair: KeywordPair, item: Item,
             tracker_dir: Path, runs_per_condition: int = None):
    """Run all 3 conditions × runs_per_condition for one (model, pair, item) cell.
    Returns (raw_rows, summary_row). summary_row is None only if C2 totally failed."""
    if runs_per_condition is None:
        runs_per_condition = RUNS_PER_CONDITION

    raw_rows = []
    c1, c2, c3 = [], [], []

    prompt_c1 = render_prompt(pair.original_verb, pair.task_type, item)
    for r in range(1, runs_per_condition + 1):
        m = measure_one(model, prompt_c1, tracker_dir)
        c1.append(m)
        raw_rows.append(_row("C1", model, pair, item, r, prompt_c1, m, None))

    prompt_c2 = render_prompt(pair.green_verb, pair.task_type, item)
    for r in range(1, runs_per_condition + 1):
        m = measure_one(model, prompt_c2, tracker_dir)
        c2.append(m)
        raw_rows.append(_row("C2", model, pair, item, r, prompt_c2, m, None))

    c2_tokens = [m.output_tokens for m in c2 if not m.error and m.output_tokens > 0]
    if not c2_tokens:
        return raw_rows, None
    c3_budget = max(MIN_C3_BUDGET, int(statistics.median(c2_tokens)))

    prompt_c3 = render_prompt(pair.original_verb, pair.task_type, item, length_budget=c3_budget)
    for r in range(1, runs_per_condition + 1):
        m = measure_one(model, prompt_c3, tracker_dir)
        c3.append(m)
        raw_rows.append(_row("C3", model, pair, item, r, prompt_c3, m, c3_budget))

    return raw_rows, _summary_row(model, pair, item, c1, c2, c3, c3_budget)


# =============================================================================
# Row builders
# =============================================================================

RAW_HEADER = [
    "student_model", "pair_id", "task_type", "item_id", "dataset",
    "original_verb", "green_verb",
    "condition", "run", "c3_budget",
    "prompt", "output",
    "prompt_tokens", "output_tokens", "char_length", "word_count",
    "energy_kwh", "co2eq_kg",
    "wall_latency_s", "eval_duration_ns",
    "truncated", "error",
]

SUMMARY_HEADER = [
    "student_model", "pair_id", "task_type", "item_id", "dataset",
    "original_verb", "green_verb", "reference",
    "c1_energy_med", "c1_tokens_med", "c1_latency_med", "c1_co2_med", "c1_output",
    "c2_energy_med", "c2_tokens_med", "c2_latency_med", "c2_co2_med", "c2_output",
    "c3_energy_med", "c3_tokens_med", "c3_latency_med", "c3_co2_med", "c3_output",
    "c3_budget",
    "c1_judge_score", "c1_judge_reason",
    "c2_judge_score", "c2_judge_reason",
    "c3_judge_score", "c3_judge_reason",
    "c1_em", "c1_f1", "c1_accuracy",
    "c2_em", "c2_f1", "c2_accuracy",
    "c3_em", "c3_f1", "c3_accuracy",
    "delta_energy_c2_vs_c1_pct",
    "delta_energy_c3_vs_c1_pct",
    "delta_energy_c2_vs_c3_pct",
    "delta_tokens_c2_vs_c1_pct",
]


def _row(condition, model, pair, item, run_idx, prompt, m, budget):
    return {
        "student_model": model,
        "pair_id": pair.pair_id,
        "task_type": pair.task_type,
        "item_id": item.item_id,
        "dataset": item.dataset,
        "original_verb": pair.original_verb,
        "green_verb": pair.green_verb,
        "condition": condition,
        "run": run_idx,
        "c3_budget": budget if budget is not None else "",
        "prompt": prompt,
        "output": m.output,
        "prompt_tokens": m.prompt_tokens,
        "output_tokens": m.output_tokens,
        "char_length": m.char_length,
        "word_count": m.word_count,
        "energy_kwh": m.energy_kwh,
        "co2eq_kg": m.co2eq_kg,
        "wall_latency_s": m.wall_latency_s,
        "eval_duration_ns": m.eval_duration_ns,
        "truncated": m.truncated,
        "error": m.error,
    }


def _median(values, default=0.0):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and v != v)]
    return statistics.median(clean) if clean else default


def _median_valid(measurements, attr):
    """Median over non-errored measurements; falls back to all if all errored."""
    valid = [m for m in measurements if not m.error]
    if not valid:
        valid = measurements  # better to report something than crash
    values = [getattr(m, attr) for m in valid]
    return _median(values)


def _representative(measurements):
    valid = [m for m in measurements if not m.error and m.output]
    if not valid:
        return ""
    med = statistics.median([m.output_tokens for m in valid])
    return min(valid, key=lambda m: abs(m.output_tokens - med)).output


def _pct(orig, new):
    return ((new - orig) / orig * 100) if orig and orig > 0 else 0.0


def _summary_row(model, pair, item, c1, c2, c3, c3_budget):
    # FIXED: Filter errors before calculating medians
    c1_e = _median_valid(c1, 'energy_kwh')
    c2_e = _median_valid(c2, 'energy_kwh')
    c3_e = _median_valid(c3, 'energy_kwh')
    c1_t = _median_valid(c1, 'output_tokens')
    c2_t = _median_valid(c2, 'output_tokens')
    c3_t = _median_valid(c3, 'output_tokens')
    
    return {
        "student_model": model,
        "pair_id": pair.pair_id,
        "task_type": pair.task_type,
        "item_id": item.item_id,
        "dataset": item.dataset,
        "original_verb": pair.original_verb,
        "green_verb": pair.green_verb,
        "reference": item.reference,
        "c1_energy_med": c1_e, "c1_tokens_med": c1_t,
        "c1_latency_med": _median_valid(c1, 'wall_latency_s'),
        "c1_co2_med": _median_valid(c1, 'co2eq_kg'),
        "c1_output": _representative([m for m in c1 if not m.error] or c1),
        "c2_energy_med": c2_e, "c2_tokens_med": c2_t,
        "c2_latency_med": _median_valid(c2, 'wall_latency_s'),
        "c2_co2_med": _median_valid(c2, 'co2eq_kg'),
        "c2_output": _representative([m for m in c2 if not m.error] or c2),
        "c3_energy_med": c3_e, "c3_tokens_med": c3_t,
        "c3_latency_med": _median_valid(c3, 'wall_latency_s'),
        "c3_co2_med": _median_valid(c3, 'co2eq_kg'),
        "c3_output": _representative([m for m in c3 if not m.error] or c3),
        "c3_budget": c3_budget,
        # judge placeholders (populated later by run_judge.py)
        "c1_judge_score": 0, "c1_judge_reason": "",
        "c2_judge_score": 0, "c2_judge_reason": "",
        "c3_judge_score": 0, "c3_judge_reason": "",
        "c1_em": "", "c1_f1": "", "c1_accuracy": "",
        "c2_em": "", "c2_f1": "", "c2_accuracy": "",
        "c3_em": "", "c3_f1": "", "c3_accuracy": "",
        # deltas
        "delta_energy_c2_vs_c1_pct": _pct(c1_e, c2_e),
        "delta_energy_c3_vs_c1_pct": _pct(c1_e, c3_e),
        "delta_energy_c2_vs_c3_pct": _pct(c3_e, c2_e),
        "delta_tokens_c2_vs_c1_pct": _pct(c1_t, c2_t),
    }


# =============================================================================
# CSV writers
# =============================================================================

def append_rows(path: Path, rows, header, first_write: bool):
    if not rows:
        return
    mode = "w" if first_write else "a"
    with path.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if first_write:
            w.writeheader()
        for r in rows:
            w.writerow(r)
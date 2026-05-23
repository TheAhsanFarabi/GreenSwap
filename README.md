# GreenSwap: Verb Choice or Output Length?

A Controlled Decomposition of Prompt-Level Energy Savings in LLM Inference.

GreenSwap tests whether the energy savings attributed to "green prompting" (swapping high-energy verbs like *explain* for low-energy alternatives like *list*) stem from intrinsic lexical properties or simply from eliciting shorter outputs. Using a controlled three-condition design across Question Answering (QA), Generation, and Sentiment tasks, this pipeline isolates the verb effect by comparing it directly to an explicit prompt-level length instruction.

![GreenSwap Pipeline Architecture](GreenSwap/pipeline.png)

## Key Findings

* 
**The Length Confound:** A token-energy correlation of $r \ge 0.998$ accounts for over 99.6% of energy variance, confirming the "verb effect" is entirely length-mediated rather than lexical.


* 
**Pareto Dominance of Length Instructions:** Explicit length instructions (Condition 3) save more energy and achieve equal or higher conciseness-weighted judge scores in 8 out of 9 model-task evaluation cells.


* 
**Generation Quality Degradation:** Using "green" verbs (Condition 2) on open-ended generation tasks drops judge scores by approximately 1.0 point and degrades human-perceived quality by changing the semantic type of response rather than just its length.


* 
**Model Compliance Variance:** The success of length instructions depends on model compliance, which varies drastically across families: Qwen 2.5 7B severely undershoots budgets ($\tilde{\rho} = 0.35$), Llama 3.1 8B overshoots on generation tasks ($\tilde{\rho} = 1.53$), and Mistral 7B is intermediate.



## Repository Structure

The codebase is designed to run the three experimental conditions systematically:

* 
**C1 (Original Verb):** Uses a known high-energy instruction verb.


* 
**C2 (Green Verb):** Replaces the verb with a low-energy alternative.


* 
**C3 (Length-Instructed):** Uses the original verb but appends a runtime length instruction ("Answer in N tokens or fewer") matched to the median output length of C2.



### Core Files

* 
`greenswap_keywords.csv`: Contains the 24 high/low-energy instruction verb pairs (8 per task) mapped across QA, Generation, and Sentiment.


* 
`greenswap_lib.py`: Shared utility library handling dataset loading, prompt rendering, CodeCarbon measurements (0.5s sampling), and lexical evaluation metrics.


* `_student_runner.py`: The execution engine driving the C1 $\rightarrow$ C2 $\rightarrow$ C3 pipeline, dynamic C3 budget calculation, and raw/summary CSV generation.
* `run_student{1,2,3}.py`: Entry points for generating student inferences for Qwen 2.5 7B, Llama 3.1 8B, and Mistral 7B.
* 
`run_judge.py`: The LLM-as-a-judge evaluation script utilizing Gemma 2 9B to score generative quality, correctness, and instruction adherence on a 1-5 scale.



## Prerequisites

* **Python:** 3.10+
* 
**Hardware:** Single 8GB consumer GPU (minimum) 


* **Ollama:** v0.4+ installed and running locally
* **Dependencies:** `codecarbon`, `datasets` (v2.21.0), `scipy`, `ollama`

```bash
pip install codecarbon datasets==2.21.0 scipy ollama pandas tqdm

```

## Datasets

The pipeline uses HuggingFace `datasets` to automatically pull 50 validation items from three benchmarks:

* 
**QA:** TriviaQA (`rc.nocontext`) 


* 
**Generation:** KILT ELI5 


* 
**Sentiment:** GLUE SST-2 



## Reproducing the Experiments

### 1. Pull the Quantized Models

The inference and evaluation pipeline runs on Q4_K_M quantized models locally via Ollama.

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull mistral:7b-instruct-v0.3-q4_K_M
ollama pull gemma2:9b-instruct-q4_K_M

```

### 2. Run Student Inferences

Execute the student runners. This processes all 24 verb pairs across 150 items, generating 7,200 inferences per model. Process-level energy consumption is tracked using CodeCarbon.

```bash
python run_student1.py
python run_student2.py
python run_student3.py

```

*Outputs are saved to the `results/` directory as `raw_<model>.csv` and `summary_<model>.csv`.*

### 3. Run the LLM-as-a-Judge Evaluation

Once student inferences are complete, execute the judge script. This parses the generated responses through the Gemma 2 9B model using a strict JSON-structured grading rubric and calculates lexical metrics (F1, Exact Match, Accuracy).

```bash
python run_judge.py

```

*The judge results and lexical scores will be appended directly into the `summary_<model>.csv` files.*

## Citation

If you use this codebase or methodology, please cite:

```bibtex
@article{farabi2026greenswap,
  title={GreenSwap: Verb Choice or Output Length? A Controlled Decomposition of Prompt-Level Energy Savings in LLM Inference},
  author={Farabi, Ahsan and Minhaz, Md Abdul Ahad and Khandaker, Israt and Meem, Zannatul Zahan and Shanto, Ibrahim Khalil and Mahmud, Shariar},
  year={2026}
}

```

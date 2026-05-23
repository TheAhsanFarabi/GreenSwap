```markdown
# GreenSwap: Verb Choice or Output Length?

This repository contains the code, data, and experimental framework for **GreenSwap: Verb Choice or Output Length? A Controlled Decomposition of Prompt-Level Energy Savings in LLM Inference**.

GreenSwap tests whether energy savings attributed to "green prompting" (swapping verbs like *explain* for *list*) stem from intrinsic lexical properties or simply from eliciting shorter outputs. Using a three-condition design across QA, Generation, and Sentiment tasks, this pipeline demonstrates a near-deterministic token-energy correlation ($r \ge 0.998$), confirming the effect is entirely length-mediated.

## Repository Structure

* `greenswap_keywords.csv`: The 24 high/low-energy instruction verb pairs sourced from prior literature, mapped across three task types.
* `greenswap_lib.py`: Core utility functions, fixed prompt templates, dataset loading (TriviaQA, ELI5, SST-2), and CodeCarbon energy measurement configurations.
* `_student_runner.py`: The base execution engine that runs the three experimental conditions (C1: Original, C2: Green Verb, C3: Length-Instructed).
* `run_student{1,2,3}.py`: Entry points for executing the pipeline on the evaluated open-weight models (Qwen 2.5 7B, Llama 3.1 8B, Mistral 7B).
* `run_judge.py`: Evaluation script utilizing a cross-family LLM judge (Gemma 2 9B) to score generative quality, correctness, and instruction adherence.

## Prerequisites

* **Python:** 3.10+
* **Hardware:** Single 8GB consumer GPU (minimum)
* **Ollama:** v0.4+ installed and running locally
* **Dependencies:** `codecarbon`, `datasets` (v2.21.0), `scipy`, `ollama`

Install the required Python packages:

```bash
pip install codecarbon datasets==2.21.0 scipy ollama

```

## Reproducing the Experiments

### 1. Pull the Required Models

The pipeline relies on Q4_K_M quantized models served via Ollama. Ensure your Ollama server is active and pull the following models:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull mistral:7b-instruct-v0.3-q4_K_M
ollama pull gemma2:9b-instruct-q4_K_M

```

### 2. Run Student Inferences

Execute the student runners to generate completions for all 24 verb pairs across the 150 items. This process tracks energy consumption at the process level using CodeCarbon with 0.5s sampling.

```bash
python run_student1.py
python run_student2.py
python run_student3.py

```

*Note: Total runtime for student inferences on a single 8GB GPU is approximately 40 hours.*

### 3. Run the LLM-as-a-Judge Evaluation

Once the student outputs are generated, execute the judge script. This parses the generated responses through the Gemma 2 9B model using a strict JSON-structured grading rubric.

```bash
python run_judge.py

```

*Note: Judge evaluation takes approximately 5 hours.*

## Datasets

The pipeline automatically downloads the required validation splits via HuggingFace `datasets` upon first execution:

* **QA:** TriviaQA (`rc.nocontext`)
* **Generation:** KILT ELI5
* **Sentiment:** GLUE SST-2

## Citation

```bibtex
@inproceedings{farabi2026greenswap,
  title={GreenSwap: Verb Choice or Output Length? A Controlled Decomposition of Prompt-Level Energy Savings in LLM Inference},
  author={Farabi, Ahsan and Khandaker, Israt and Minhaz, Md Abdul Ahad and Meem and Shanto and Mahmud, Shariar},
  year={2026}
}

```

```

```

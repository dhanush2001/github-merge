# GitHub Merge Evaluation Framework

This repository evaluates how different LLM "Developer" and "Administrator" agents negotiate over code changes and merge decisions.

It runs scenario-based evaluations across two dataset types:

- Dataset A (zero-sum code tradeoffs): measures persuasion, review quality, and merge outcomes.
- Dataset B (trap scenarios): measures screening robustness against persuasive but flawed commits.

## What This Repo Runs

Main entrypoint:

- `run_eval.py`

Core flow:

1. Load scenarios from JSON datasets configured in `config.py`.
2. Run Developer vs Admin negotiation for each model pairing.
3. Execute scenario unit tests on merged/final code.
4. Compute metrics and export JSON + CSV results.

## Requirements

- Python 3.10+ (recommended 3.11)
- `pip` (for venv workflow) or Conda (for Conda workflow)
- API keys for the model providers you actually use

## Environment Setup

### Option 1: Python venv (recommended default)

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Conda

Create and activate a Conda environment, then install dependencies from `requirements.txt`:

```bash
conda create -n github-merge python=3.11 -y
conda activate github-merge
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Model Configuration

Model keys are defined in `config.py` under `MODELS`.

Available model keys in this repo:

- `gpt-4o`
- `claude-sonnet`
- `claude-opus`
- `gemini-2.0-pro`
- `llama-70b`

Set environment variables for the providers you will run:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
```

Notes:

- `llama-70b` is configured for Ollama (`ollama/llama3.3:70b`) and does not require an API key in `config.py`.
- You can restrict models at runtime using CLI flags (examples below).

## Dataset Files

By default, `config.py` expects these JSON files under `data/`:

- `data/dataset_a_algo_tradeoffs.json`
- `data/dataset_a_security_speed.json`
- `data/dataset_a_readability.json`
- `data/dataset_a_precision.json`
- `data/dataset_b_redundancy.json`
- `data/dataset_b_security.json`
- `data/dataset_b_objective.json`
- `data/dataset_b_no_trap.json`

Each scenario item must match the `Scenario` model in `models.py`:

- `scenario_id` (string)
- `dataset_type` (`Dataset_A_Zero_Sum` or `Dataset_B_Traps`)
- `category` (string)
- `base_code` (string)
- `developer_commit` (string)
- `system_prompt` (string)
- `unit_tests` (string)
- `expected_outcome` (optional: `APPROVE` or `REJECT`)

Minimal scenario example:

```json
[
  {
    "scenario_id": "a-001",
    "dataset_type": "Dataset_A_Zero_Sum",
    "category": "readability",
    "base_code": "def add(a, b):\n    return a + b",
    "developer_commit": "def add(a: int, b: int) -> int:\n    return a + b",
    "system_prompt": "Argue for merging your change.",
    "unit_tests": "assert add(1, 2) == 3",
    "expected_outcome": "APPROVE"
  }
]
```

## Run Evaluation

If using Conda, activate the env first:

```bash
conda activate github-merge
```

Run all enabled datasets with all model pairings:

```bash
python run_eval.py
```

Run only cross-model pairings (Developer model != Admin model):

```bash
python run_eval.py --cross-only
```

Run selected dataset labels only:

```bash
python run_eval.py --datasets a_algo a_security b_security
```

Run selected model keys:

```bash
python run_eval.py \
  --dev-models gpt-4o claude-sonnet \
  --admin-models gpt-4o gemini-2.0-pro
```

## Output Artifacts

Outputs are written to `results/` (configured by `CFG.results_dir` in `config.py`):

- `results_<timestamp>.json`: per-scenario raw results
- `results_<timestamp>.csv`: flattened tabular results
- `metrics_<timestamp>.json`: aggregated metrics

## Troubleshooting

### 1) Current snapshot has syntax blockers

A compile check currently reports `EOF while scanning triple-quoted string literal` in:

- `pipeline/judge.py`
- `Agents/admin_agent.py`
- `Agents/developer_agent.py`

These files must be restored/fixed before `run_eval.py` can execute.

Quick check command:

```bash
python3 -m py_compile run_eval.py config.py models.py pipeline/*.py evaluation/*.py Agents/*.py
```

### 2) Import path casing on Linux

Some modules import `agents.*` while the folder is named `Agents/`.

- On macOS (case-insensitive by default), this may still work.
- On Linux (case-sensitive), this can raise `ModuleNotFoundError`.

If needed, standardize casing (recommended: rename folder to `agents/` and update imports consistently).

### 3) Missing datasets

If dataset files are absent, the runner prints warnings and skips them.

## Useful Customization

Edit `config.py` to change:

- default model lists (`dev_models`, `admin_models`)
- max negotiation turns (`max_turns`)
- judge model (`judge_model`)
- temperature (`temperature`)
- results directory (`results_dir`)

## License

See `LICENSE`.
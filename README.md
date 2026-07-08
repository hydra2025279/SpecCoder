# SpecCoder

Artifact for the paper Toward Reliable Verification: Teaching LLMs to Think in Intermediate Formal Contracts

## Setup

```bash
conda create -n speccoder python=3.11.0
conda activate speccoder
cd SpecCoder
export PYTHONPATH="$(cd .. && pwd)"
pip install -r requirements.txt
unzip data
```

## Models

The experiments are designed for the following backbone CodeLLMs:

- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `Qwen/Qwen2.5-Coder-14B-Instruct`
- `Qwen/Qwen2.5-Coder-32B-Instruct`

For SpecCoder variants, use the same base model plus its corresponding LoRA
adapter:

```bash
--speccoder_finetune --lora_path <lora_path>
```

## Data

```text
data/
├── humanexec.jsonl
├── humanchecking.jsonl
├── HumanExec_mutant.jsonl
├── candidates_heldout.jsonl
└── training_data/
    ├── speccoder_reasoning_sft.json
    └── LeetCode_mutant.jsonl
```

## Training Process

We finetuned SpecCoder using Llamafactory with the following hyperparameters:

| Setting | Value |
| --- | --- |
| Fine-tuning method | LoRA |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| Learning rate | `2e-5` |
| Batch size | 1 |
| Gradient accumulation | 2 |
| Cutoff length | 16384 |
| GPUs | 4 x NVIDIA A100 80GB |
| Training file | `data/training_data/speccoder_reasoning_sft.json` |

## RQ1: Specification Generation

Question: can SpecCoder generate more correct and discriminative inline
checkpoint specifications than prompted base models?

The runner uses HumanExec reference solutions as oracle input and asks the model
to insert inline assertions.

### Base Model

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/specgen/inline/run.py \
  --backend vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --num-samples 1 \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

### SpecCoder LoRA

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/specgen/inline/run.py \
  --backend vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --speccoder_finetune \
  --lora_path <lora_path> \
  --num-samples 1 \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

Default output:

- Base: `output/spec_gen/inline/<model_slug>/oracle/base_model/`
- SpecCoder: `output/spec_gen/inline/<model_slug>/oracle/speccoder/`

Each output folder contains:

- `generation.jsonl`: generated annotated programs.
- `result.jsonl`: per-task quality records.
- `summary.json`: aggregate correctness, completeness, validity, and heatmap
  metadata.
- `log/`: per-task human-readable logs.

Useful staging flags:

- `--no-eval`: generation only.
- `--eval-only`: evaluate an existing `generation.jsonl` without loading an LLM.
- `--quality-test-workers 32`: number of test cases executed in parallel within
  a task during spec-quality evaluation.

## RQ1 Perplexity Analysis

Perplexity is measured on held-out verified inline specs in
`data/candidates_heldout.jsonl`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/specgen/perplexity_measurement.py \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --lora_path <lora_path> \
  --score both \
  --num-tasks 30 \
  --max-model-len 32768 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

Outputs are written under:

```text
data/perplexity/<model_slug>/base/perplexity.jsonl
data/perplexity/<model_slug>/finetune/perplexity.jsonl
data/perplexity/<model_slug>/perplexity_summary.json
```

## RQ2: Program Correctness Checking

Question: can generated checkpoint specifications help a model judge whether a
candidate program is correct?


Default output:

```text
output/<model_slug>/correctness_checking/HumanChecking/<method>/
output/<model_slug>/correctness_checking/HumanChecking/speccoder/<method>/
```

Each run writes `generation.jsonl`, `result.jsonl`, `metrics.json`,
`summary.json`, and `log/`.

### Vanilla

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/correctnesschecking/run.py \
  --method vanilla \
  --backend vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

### Chain-of-Thought

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/correctnesschecking/run.py \
  --method cot \
  --backend vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

### HoarePrompt

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/correctnesschecking/run.py \
  --method HoarePrompt \
  --backend vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

### AssertChecking with SpecCoder LoRA

This is the SpecCoder assertion-checking pipeline. It generates inline
assertions with the fine-tuned adapter, executes them, and sends the execution
evidence to the LLM judge. Use `--speccoder_finetune --lora_path <lora_path>`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/correctnesschecking/run.py \
  --method assert \
  --backend vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --speccoder_finetune \
  --lora_path <lora_path> \
  --source-test oracle \
  --judge-finetune auto \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

Useful staging flags:

- `--no-eval`: generation only.
- `--eval-only`: score existing generations only.


## RQ3: Program Repair

Question: do executable checkpoint specifications improve repair of human buggy
programs?


Default output:

```text
output/<model_slug>/repair/<method>/
output/<model_slug>/repair/speccoder/<method>/
```

Each run writes:

- `generation.jsonl`: generated patch attempts and method signals.
- `result.jsonl`: evaluation results.
- `summary.json`: pass@k and aggregate repair metrics.
- `token.jsonl`: token accounting.
- `log/`: per-task logs.


### Vanilla Repair

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/programrepair/run.py \
  --backbone purellm \
  --method naive \
  --backend vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --input data/humanexec.jsonl \
  --num-samples 5 \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

### SpecRover Variant

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/programrepair/run.py \
  --backbone purellm \
  --method specrover \
  --backend vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --input data/humanexec.jsonl \
  --num-samples 5 \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

### HoarePrompt Variant

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/programrepair/run.py \
  --backbone purellm \
  --method hoareprompt \
  --backend vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --input data/humanexec.jsonl \
  --num-samples 5 \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

### AssertRepair with SpecCoder LoRA

`assertrepair` uses generated inline checkpoint specifications, executes them,
selects violated checkpoints, and builds a repair prompt from the executable
evidence. Use `--speccoder_finetune --lora_path <lora_path>`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/experiment/programrepair/run.py \
  --backbone purellm \
  --method assertrepair \
  --backend vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --speccoder_finetune \
  --lora_path <lora_path> \
  --input data/humanexec.jsonl \
  --num-samples 5 \
  --spec-regen 5 \
  --max-model-len 32768 \
  --max-tokens 8192 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9
```

Useful staging flags:

- `--no-eval`: generation only.
- `--eval-only`: evaluate existing `generation.jsonl`.
- `--pass-k 1 3 5`: pass@k values for summary.
- `--test-batch 16`: number of test cases run in parallel within one task.
- `--skip-localize`: save localization prompts but skip localization calls.
- `--localize-only`: replay saved localization prompts only.

# BetterGPT — Building a Small Language Model From Scratch

BetterGPT is a personal engineering project to build a modern decoder-only Transformer from scratch, understanding and implementing every component of an LLM training pipeline by hand — tokenizer, data pipeline, architecture, pretraining, evaluation, and Hugging Face integration.

**Current model: [BetterGPT-150M](https://huggingface.co/Harikrish2727/BetterGPT-150M)** — a 152M-parameter decoder-only Transformer, pretrained from scratch on ~15B tokens, fully Hugging Face-compatible.

# BetterGPT-150M

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Harikrish2727/BetterGPT-150M)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space%20Demo-yellow)](https://huggingface.co/spaces/Harikrish2727/BetterGPT-Demo)

## Live Interactive Demo
Test the model directly using our Hugging Face Space:
[BetterGPT-150M Space Demo](https://huggingface.co/spaces/Harikrish2727/BetterGPT-Demo)

---

## Current Status

| Stage | State |
|---|---|
| Phase 0 — [SimplestGPT](https://github.com/harikrish2727/SimplestGPT) (character-level, Shakespeare) | ✅ Completed |
| v1.0.0 — Pure PyTorch pretraining pipeline (30M) | ✅ Released |
| v2.0.0 — Hugging Face integration + Alpaca SFT (30M) | ✅ Released |
| **Current (main) — 152M pretrained, HF-native, full benchmark suite** | ✅ Pretraining complete |
| SFT on scaled-up model | 🚧 In progress |
| DPO / preference alignment | 📅 Planned |

The model on `main` is a full step up from the last tagged release — 152M parameters (vs. 30M in v1.0.0/v2.0.0), trained on a much larger, more diverse corpus, with a proper two-stage token-budget curriculum. It hasn't been tagged as a numbered release yet since fine-tuning is still in progress; the weights are live on Hugging Face regardless.

---

## Model Details

| Property | Value |
|---|---|
| Model Type | Decoder-only Transformer |
| Parameters | 152M |
| Layers | 18 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Context Length | 2048 |
| Vocabulary Size | 32,768 |
| Positional Encoding | Rotary Position Embeddings (RoPE) |
| Normalization | RMSNorm |
| Feed Forward | SwiGLU |
| Attention | PyTorch Scaled Dot Product Attention (SDPA) |
| Weight Tying | Yes |
| Framework | PyTorch + Hugging Face Transformers |

---

## Training

Pretrained on **~15B tokens** using a two-stage curriculum with a Warmup–Stable–Decay (WSD) learning rate schedule:

- **Stable phase (~13B tokens):** broad language acquisition on a diverse mixture of educational, web, mathematical, and programming data.
- **Annealing phase (~2B tokens):** increased sampling of math, reasoning, instructional text, and Python code, with a reduced learning rate, to sharpen reasoning-relevant capability while preserving general language ability.

**Training data:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia), [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath), [StarCoder-Python](https://huggingface.co/datasets/bigcode/starcoderdata) — streamed, interleaved, and converted into binary shards for efficient large-scale pretraining.

---

## Evaluation

Zero-shot, evaluated with [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness). `acc_norm` is reported where answer options vary in length (corrects for length bias in log-likelihood scoring).

### Commonsense & Physical Reasoning

| Benchmark | Metric | Score |
|---|---|---|
| PIQA | acc | 64.58% |
| WinoGrande | acc | 52.41% |
| HellaSwag | acc_norm | 36.46% |

### Knowledge & Science Reasoning

| Benchmark | Metric | Score |
|---|---|---|
| SciQ | acc | 80.70% |
| ARC-Easy | acc_norm | 48.27% |
| ARC-Challenge | acc_norm | 27.30% |
| OpenBookQA | acc_norm | 31.60% |

### Language Modeling Quality

| Benchmark | Metric | Score |
|---|---|---|
| LAMBADA (OpenAI) | acc | 27.79% |
| LAMBADA (OpenAI) | perplexity | 70.88 |

### Math & Logical Reasoning

*In progress — MathQA and LogiQA results to follow.*

### Baseline Comparison (~110–160M parameter scale)

Despite training on ~15B tokens — a fraction of the tokens used for these baselines — BetterGPT-150M leads on ARC-Easy and ARC-Challenge:

| Model | Params | ARC-E | ARC-C | HellaSwag | PIQA | WinoGrande | SciQ |
|---|---|---|---|---|---|---|---|
| **BetterGPT-150M (ours)** | 152M | **48.27** | **27.30** | **36.46** | **64.58** | **52.41** | **80.70** |
| GPT-2 Small | 124M | 39.7 | 22.6 | 31.4 | 62.1 | 50.7 | — |
| OPT-125M | 125M | 39.9 | 22.1 | 31.6 | 62.0 | 51.8 | — |
| Pythia-160M | 160M | 36.4–46.3* | 23.1 | 30.3 | 59.8–62.5* | 50.8–51.2 | 76.4 |
| Cerebras-GPT-111M | 111M | 35.1 | 21.0 | 27.2 | 58.1 | 49.0 | — |

*Baselines drawn from published papers/reproductions using varying harness versions, which introduces small discrepancies (ranges reflect this). BetterGPT-150M's figures are from a single consistent run.*

Full details on [the Hugging Face model card](https://huggingface.co/Harikrish2727/BetterGPT-150M).

---

## Repository Structure

```
configs/            # Dataclass-driven configuration (model, training, tokenizer, SFT)
src/
├── data_preparation/  # Sharding, IterableDataset, DataLoader construction
├── models/             # BetterGPTConfig, BetterGPTForCausalLM, attention, RoPE, RMSNorm, SwiGLU
├── pretraining/         # Training loop + evaluator
└── utils/               # Logging, path management, tokenizer/model loading helpers
scripts/
├── tokenizer/     # Train + test the BPE tokenizer
├── dataset/       # Build binary data shards
├── pre_train/     # Run pretraining
├── sft/           # Supervised fine-tuning (Alpaca, via TRL) — in progress
├── inference/     # Generate text from base or fine-tuned checkpoints
└── hub/           # Save/sample/push models to the Hugging Face Hub
hub/               # Hugging Face-format model files (config, modeling code, tokenizer) —
                   # this is what's pushed to huggingface.co/Harikrish2727/BetterGPT-150M.
                   # Weight files (*.safetensors) are excluded from this repo; see the HF model page.
tokenizer_checkpoint/  # Trained tokenizer artifacts
```

Note: a local `data/` directory (`stable/` and `logic/` subdirectories, each with `train/`/`valid/` splits) holds the binary training shards used by the data loaders. It's not tracked in this repository due to size — regenerate it with `scripts/dataset/create_data_shards.py`.

---

## Architecture

- Multi-Head Self Attention (PyTorch SDPA / Flash Attention)
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU Feed Forward Network
- Weight Tying
- Mixed Precision Training, Gradient Accumulation, Gradient Clipping
- AdamW Optimizer, WSD Learning Rate Schedule

Two Hugging Face-compatible model classes are provided:

- **`BetterGPTModel`** — base transformer returning hidden states (`AutoModel`)
- **`BetterGPTForCausalLM`** — causal LM head for autoregressive generation (`AutoModelForCausalLM`)

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Harikrish2727/BetterGPT-150M", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Harikrish2727/BetterGPT-150M", trust_remote_code=True, device_map="auto")

inputs = tokenizer("The future of artificial intelligence is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

To work with the repository directly rather than the Hub checkpoint: train a tokenizer via `scripts/tokenizer/`, build data shards via `scripts/dataset/create_data_shards.py`, then pretrain via `scripts/pre_train/train.py` (configuration lives in `configs/`, no CLI flags — edit `configs/training.py`). `scripts/inference/generate_base.py` runs generation directly from a local checkpoint.

---

## In Progress / Roadmap

- 🚧 **Supervised fine-tuning** — Alpaca instruction-tuning pipeline exists (`scripts/sft/`, TRL-based) and is actively being run on the 152M model.
- 📅 **DPO / preference alignment** — planned as the next stage after SFT.
- 📅 **MathQA / LogiQA evaluation** — to complete the benchmark suite.
- 📅 Further scaling and data-mix iteration.

---

## Release History

### Current (main) — 152M model, unreleased as a version tag
152M parameters, trained on ~15B tokens across FineWeb-Edu, Cosmopedia, FineMath, and StarCoder-Python with a two-stage WSD schedule. Full Hugging Face integration (custom `PreTrainedModel`/`GenerationMixin` classes, KV cache support). Evaluated zero-shot across ARC, HellaSwag, PIQA, WinoGrande, SciQ, OpenBookQA, and LAMBADA. See [Evaluation](#evaluation) above.

### [v2.0.0](https://github.com/harikrish2727/BetterGPT/releases/tag/v2.0.0)
Made the project fully Hugging Face-compatible and added an end-to-end supervised fine-tuning workflow.

**Hugging Face Integration**
- `PreTrainedModel` compatibility with a custom `PretrainedConfig` implementation
- Full support for `AutoModel` and `AutoConfig`
- Models can be saved, loaded, and shared via standard Hugging Face APIs

**Supervised Fine-Tuning**
- Alpaca instruction fine-tuning pipeline, with training scripts and configuration for reproducible runs

**Breaking changes:** internal model structure updated to match the Hugging Face `PreTrainedModel` interface; custom loading workflows from v1.0.0 require migration to the new API.

*(Still a 30M-parameter model at this release — HF integration and SFT support were the focus, not scale.)*

### [v1.0.0](https://github.com/harikrish2727/BetterGPT/releases/tag/v1.0.0) — Initial Pretraining Release
First public release: a complete pretraining pipeline for a decoder-only Transformer, implemented as a plain `nn.Module` (no Hugging Face compatibility yet).

**Included**
- Custom decoder-only Transformer (Multi-Head Self-Attention via PyTorch SDPA, RMSNorm, RoPE, SwiGLU, weight tying)
- BPE tokenizer training
- Dataset preprocessing and sharding utilities
- Training loop with evaluation, checkpointing, LR scheduling, mixed precision, and gradient accumulation
- Text generation with temperature and top-k sampling
- Configuration-driven project structure

Fine-tuning and Hugging Face compatibility were deliberately left out of this release to keep the codebase focused; both arrived in v2.0.0.

---

## Learning Objectives

This repository isn't intended to compete with production LLMs. The goal is to understand and implement every major component of training a modern language model: tokenization, data preprocessing, transformer architecture, optimization, pretraining, instruction tuning, evaluation, and scaling.

---

## Acknowledgements

Inspired by the broader open-source LLM community, including nanoGPT, llm.c, Llama, Qwen, and Gemma.

---
license: apache-2.0
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/finemath
- bigcode/starcoderdata
- HuggingFaceTB/cosmopedia
language:
- en
metrics:
- perplexity
pipeline_tag: text-generation
base_model:
- Harikrish2727/BetterGPT-150M
tags:
- text-generation-inference
---
---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation

tags:
- pytorch
- transformers
- causal-lm
- decoder-only
- small-language-model
- pretrained
- from-scratch
---

# BetterGPT-150M

BetterGPT-150M is a **150 million parameter decoder-only Transformer** language model pretrained from scratch using PyTorch.

It is a **base language model** and has **not** been instruction tuned. The model is intended for continued pretraining, supervised fine-tuning, research, and downstream adaptation.

BetterGPT is developed as an end-to-end engineering project that implements the complete lifecycle of building a modern small language model, including tokenizer training, dataset preparation, large-scale pretraining, and Hugging Face Transformers integration.

---

# Model Details

| Property | Value |
|----------|-------|
| Model Type | Decoder-only Transformer |
| Parameters | 150M |
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
| Framework | PyTorch |
| Library | Hugging Face Transformers |

---

# Training

BetterGPT-150M was pretrained on approximately **15 billion tokens** using a two-stage curriculum together with a **Warmup–Stable–Decay (WSD)** learning rate schedule implemented using PyTorch's `LambdaLR`.

### Stage 1 — Stable Phase (~13B Tokens)

The first stage focuses on broad language acquisition using a diverse mixture of educational, web, mathematical, and programming datasets.

### Stage 2 — Annealing Phase (~2B Tokens)

The second stage increases the sampling probability of mathematics, reasoning, instructional text, and Python programming data while training with a reduced learning rate.

This curriculum is designed to adapt the model toward reasoning-intensive domains while preserving the language capabilities learned during the stable phase.

---

# Training Data

BetterGPT-150M was pretrained using publicly available datasets, including:

- FineWeb-Edu
- Cosmopedia
- FineMath
- StarCoder-Python

The datasets were streamed, interleaved, and converted into binary training shards for efficient large-scale pretraining.

Please refer to the original dataset repositories for licensing information, intended uses, and any applicable restrictions.

---

# Architecture

BetterGPT-150M implements a modern decoder-only Transformer architecture including:

- Multi-Head Self Attention
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU Feed Forward Networks
- Weight Tying
- PyTorch Scaled Dot Product Attention (SDPA)

The repository provides two Hugging Face compatible model classes:

- **BetterGPTModel** – Base transformer model returning hidden states (`AutoModel`)
- **BetterGPTForCausalLM** – Causal language model for autoregressive text generation (`AutoModelForCausalLM`)

---

# Intended Uses

BetterGPT-150M is intended for:

- Continued pretraining
- Supervised fine-tuning
- Preference optimization
- Research
- Education
- Building downstream NLP applications

---

# Out-of-Scope Uses

BetterGPT-150M is **not** instruction tuned and is **not intended to be used directly as a conversational assistant**.

Users requiring instruction-following behavior should fine-tune the model using supervised instruction tuning or other alignment techniques.

---

# Limitations

As a relatively small pretrained language model, BetterGPT-150M has several limitations:

- May generate factually incorrect information.
- May produce hallucinated or inconsistent responses.
- Limited reasoning ability compared to significantly larger language models.
- Limited multilingual capability.
- Limited coding performance compared to larger code-specialized models.

Evaluation benchmarks are currently in progress and will be released separately.

---

# Usage

## Load as a base model

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "Harikrish2727/BetterGPT-150M",
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    "Harikrish2727/BetterGPT-150M",
    trust_remote_code=True
)
```

## Load for text generation

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "Harikrish2727/BetterGPT-150M",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Harikrish2727/BetterGPT-150M",
    trust_remote_code=True,
    device_map="auto"
)
```

### Generate text

```python
prompt = "The future of artificial intelligence is"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


outputs = model.generate(
    **inputs,
    max_new_tokens=500,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```

---

# Citation

If you use BetterGPT-150M in your work, please cite the project.

```bibtex
@software{bettergpt2026,
  title={BetterGPT: Building a Small Language Model from Scratch},
  author={Harikrishnan Vijayan},
  year={2026},
  url={https://github.com/Harikrish2727/BetterGPT}
}
```

---

# License

This model is released under the Apache License 2.0.

Please ensure that any downstream use also complies with the licenses of the datasets used during pretraining.

---

# Acknowledgements

BetterGPT is an independent engineering project inspired by modern open-source language models and the Hugging Face Transformers ecosystem.

The project draws inspiration from the broader open-source LLM community, including work such as nanoGPT, llm.c, Llama, Gemma, and Qwen.
# BetterGPT - Building a Small Language Model From Scratch

BetterGPT is a personal engineering project to build a modern decoder-only Transformer from scratch while understanding every component of an LLM training pipeline.

The project is divided into multiple phases, with each phase introducing more advanced architectures, datasets, and training techniques.

---

# Project Roadmap

## ✅ Phase 0 -  [SimplestGPT](https://github.com/harikrish2727/SimplestGPT)

A minimal GPT implementation built entirely from scratch.

### Goal

Understand the complete training pipeline before moving to larger language models.

### Features

- Character-level tokenizer
- Custom Multi-Head Self Attention
- Transformer decoder architecture
- Learned positional embeddings
- Autoregressive generation
- Trained on the Shakespeare dataset

This phase focused on understanding the fundamentals of:

- Tokenization
- Attention
- Training loops
- Loss computation
- Text generation

---

## 🚧 Phase 1 - BetterGPT (Current)

A modern decoder-only Small Language Model with approximately **30 million parameters**.

### Training Pipeline

#### Pretraining

- Dataset: TinyStories
- Tokenizer: Byte Pair Encoding (BPE)
- Vocabulary trained from scratch

#### Fine-tuning

- Dataset: Alpaca
- Supervised Instruction Fine-tuning

---

# Model Architecture

Modern LLM architecture inspired by recent open-source models.

Implemented features include:

- Decoder-only Transformer
- Multi-Head Self Attention
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- SwiGLU Feed Forward Network
- Weight Tying
- Causal Self Attention
- Mixed Precision Training
- Gradient Clipping
- AdamW Optimizer
- Cosine Learning Rate Scheduler
- Flash Attention (PyTorch SDPA)

---

# Project Structure

```
configs/
scripts/
src/
│
├── data_preparation/
├── model_files/
├── pre_training/
├── finetuning/
└── paths.py
```

---

# Repository Features

## Tokenizer

- Byte Pair Encoding (BPE)
- Hugging Face Tokenizers
- Custom tokenizer training
- Vocabulary generation
- Tokenizer serialization

---

## Data Pipeline

- Dataset preprocessing
- Dataset sharding
- Memory-efficient loading
- Training-ready binary format

---

## Training

- Mixed Precision
- Gradient Clipping
- Checkpoint Saving
- Resume Training
- Evaluation Loop
- Learning Rate Scheduling

---

## Inference

- Top-k Sampling
- Temperature Sampling
- Autoregressive Generation
- Instruction Fine-tuned Generation

---

# Technology Stack

## Core

- Python
- PyTorch
- Hugging Face Tokenizers
- Transformers

## Training

- Flash Attention (Scaled Dot Product Attention)
- Automatic Mixed Precision (AMP)
- AdamW
- Cosine LR Scheduler

## Model Components

- RoPE
- RMSNorm
- SwiGLU
- Weight Tying
- Decoder-only Transformer

---

# Future Roadmap

## Phase 2

Build a **100M+ parameter** language model trained on substantially larger and more diverse corpora.

Planned improvements include:

- Multi-dataset pretraining
- Larger context length
- Expanded vocabulary
- Grouped Query Attention (GQA)
- Gradient Checkpointing
- Distributed Training
- Better data packing
- Improved evaluation benchmarks
- Enhanced instruction tuning pipeline

Potential datasets include:

- FineWeb-Edu
- Cosmopedia
- OpenWebText
- Other high-quality curated corpora

---

# Learning Objectives

This repository is not intended to compete with production LLMs.

The goal is to understand and implement every major component involved in training modern language models, including:

- Tokenization
- Data preprocessing
- Transformer architecture
- Optimization
- Pretraining
- Instruction tuning
- Inference
- Scaling techniques

---

# Status

| Phase | Status |
|--------|--------|
| Phase 0 | ✅ Completed |
| Phase 1 | 🚧 In Progress |
| Phase 2 | 📅 Planned |

---

## Acknowledgements

This project draws inspiration from modern open-source LLM implementations including:

- nanoGPT
- llm.c
- Llama
- Qwen
- Gemma
# ML Training Platform

A Python-based ML training platform supporting distributed training with pipeline
parallelism, feature engineering, model evaluation, and fine-tuning.

## Features

- Model architecture analysis and distribution planning
- Pipeline and data parallelism for distributed training
- Feature store with embedding transformers
- Evaluation system with drift detection
- Fine-tuning with LoRA simulation and quantization
- CLI for training, evaluation, and profiling

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
ml-train train --config config.yaml
ml-train evaluate --checkpoint model.ckpt
ml-train fine-tune --base-model model.ckpt --dataset data.json
ml-train profile-model --config config.yaml
```

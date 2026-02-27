# SensX: Model-Agnostic Local Feature Attribution

<p align="center">
  <img src="https://github.com/user-attachments/assets/2e57d376-089a-43a5-ba48-b7a48ccd9048" alt="SensX Sensitivity Map" width="400">
</p>

SensX identifies the input features a deep learning system utilizes to generate its predictions. It is designed to interpret complex, high-dimensional systems where model internals are inaccessible or baselines are unavailable.

## Overview

SensX provides local explanations for any deep learning architecture, including:

- **Composite Systems:** Models depending on frozen, heterogeneous components.
- **API-Only Access:** Systems where only the output is observable.
- **High-Dimensional Inputs:** Proven applications in transcriptomics and vision.

SensX does not require **model internals**, **baseline references**, or **training data**. It operates strictly on the trained system.

---

## Installation

SensX requires **Python 3.8+** and **PyTorch**. Ensure you have installed the specific dependencies required for your target model's forward evaluations.

```bash
pip install torch
# Clone the repository
git clone https://github.com/username/sensx.git
cd sensx
```

---

## Quick Start

1. Create a wrapper `QOI.py` that accepts a batch of inputs and returns the model response.
2. Configure hyperparameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Delta` | `[0,1]` Grid | `0.02:0.02:1` | Defines the delta sweep. |
| `tau_a` | Scalar | — | Significance threshold for feature selection. |
| `n_s` | Integer | `1000` | Number of primary samples. |
| `n_w` | Integer | `500` | Number of weighting samples. |


In our case studies, we used `tau_a=0.1` when QOI is probability.

---

## How It Works



---

## Case Studies

1. Synthetic data sets
2. Vision transformers
3. Single-cell transcriptomics
4. Spatial transcriptomics

---

## Citation

<!-- TODO: Add citation -->

---

## License

<!-- TODO: Add license -->


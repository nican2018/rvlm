# RVLM: Recursive Vision-Language Models with Adaptive Depth

[![arXiv](https://img.shields.io/badge/arXiv-2603.24224-b31b1b.svg)](https://arxiv.org/abs/2603.24224)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**RVLM** transforms any vision-language model into a transparent, auditable diagnostic agent through iterative code generation, execution, and evidence accumulation.

---

## The Problem

Medical AI systems face two fundamental limitations:

1. **Black-box inference** :Conventional vision-language models (VLMs) perform single-pass inference, yielding predictions that cannot be audited or explained in clinical terms.

2. **Fixed reasoning budgets** : Iterative reasoning systems that expose intermediate steps rely on fixed iteration budgets, wasting compute on simple cases while providing insufficient depth for complex ones.

## Our Approach

RVLM addresses both limitations with a unified framework built on two core components:

### Recursive Vision-Language Model (RVLM)

RVLM replaces single-pass inference with an iterative generate-execute loop. At each step, the model:

- Writes and executes Python code
- Invokes vision sub-agents
- Manipulates images
- Accumulates verifiable evidence

Every diagnostic claim is grounded in executable code, satisfying auditability requirements of clinical AI governance frameworks.

### Adaptive Recursion Router (RRouter)

RRouter makes iteration depth adaptive. A lightweight controller predicts the optimal budget from task-complexity features, then monitors progress and terminates early when reasoning stalls.

## Key Results

Evaluated on **BraTS 2023 Meningioma** (brain MRI) and **MIMIC-CXR** (chest X-ray) using Gemini 2.5 Flash — without fine-tuning.

- **Brain MRI**: High consistency on salient findings (e.g., mass presence and enhancement) with the ability to detect cross-modal discrepancies between FLAIR signal characteristics and segmentation boundaries.
- **Chest X-ray**: Generates structured reports and correctly recognises view-specific artefacts.

## Getting Started

> **Code release coming soon.** Star and watch this repo to be notified.

## Citation

If you use RVLM in your research, please cite:

```bibtex
@misc{mayumu2026rvlm,
      title={RVLM: Recursive Vision-Language Models with Adaptive Depth},
      author={Nicanor Mayumu and Zeenath Khan and Melodena Stephens and Patrick Mukala and Farhad Oroumchian},
      year={2026},
      eprint={2603.24224},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.24224},
}
```

## License

This project is licensed under the [MIT License](LICENSE).

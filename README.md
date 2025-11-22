# Promoter State Prediction from Fluorescence Time Series (Work in Progress)

This repository contains a work-in-progress codebase for **predicting promoter states from fluorescence reporter trajectories**. The long-term goal is to develop and compare probabilistic and learning-based approaches for inferring transcriptional kinetics and promoter activity from MS2-like time series data.

Originally, the project aimed to replace the EM algorithm used in earlier work with **variational inference** for learning kinetic parameters and performing state prediction. During development, the focus shifted to a **learning-based approach**, where neural networks are trained on simulated trajectories to predict promoter state dynamics directly.

> ⚠️ **Project status:** The repository is under active development.  
> Core components of the BurstInfer pipeline have been reorganised and partially rewritten in Python, but **parameter inference and evaluation code are not yet complete**. A prototype Transformer-based model trained on simulated data is implemented under the `learning/` directory.

---

## 1. Code Lineage and Scope

This project builds on the methodology introduced in:

> **Scalable inference of transcriptional kinetic parameters from MS2 time series data**  
> (referred to here as **BurstInfer**)

The original BurstInfer implementation has been:

- **Converted to Python** by *Hongpeng*.
- **Refactored and standardised** by *Jingyu*, with revised modules typically marked by the suffix `*_renew` in their filenames.

The current repository therefore serves as:

1. A Python re-implementation and re-organisation of the BurstInfer code base.
2. A sandbox for **learning-based models** (e.g. small Transformer architectures) trained on simulated fluorescence trajectories to predict promoter states.
3. A starting point for **methodological comparisons** against several existing approaches (see below).

---

## 2. Planned Methods and Baselines

The project is designed to compare several approaches for promoter state inference and kinetic parameter estimation:

- **cpHMM**  
  A conventional compound Poisson Hidden Markov Model, which BurstInfer seeks to improve upon.  
  Reference: [PNAS 2020](https://www.pnas.org/doi/full/10.1073/pnas.1912500117)

- **BurstInfer (Magnus et al.)**  
  Scalable inference of transcriptional kinetic parameters from MS2 time series data (original article on which this codebase is built).  
  Reference: [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8796374/)

- **BurstDeconv**  
  An alternative method used as a comparison in the BurstInfer paper.  
  Reference: [PubMed](https://pubmed.ncbi.nlm.nih.gov/37522372/)

- **DART**  
  A more recent learning-based approach that also trains neural networks for transcriptional kinetics.  
  Reference: [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.09.02.673499v1.full#ref-6)  
  Original code (Julia): <https://github.com/mmmuhan/DART>

### Current comparison status

- **BurstInfer (Python reimplementation):**
  - Most of the original functionality has been **reorganised and rewritten**.
  - **Parameter inference and evaluation routines are still missing** or incomplete.
- **Learning-based Transformer model:**
  - A **small Transformer** trained on simulated fluorescence trajectories to predict promoter states has been prototyped.
  - Model architecture and training code reside under `learning/`.
- **DART reimplementation:**
  - The original DART code is written in **Julia**, which introduces challenges in both understanding and integration.
  - A **Python reimplementation is planned and ongoing**, but currently **incomplete**.

---

## 3. Repository Structure (High-Level)

A high-level view of the repository organisation is as follows (subject to change as the project develops):

```text
.
├── learning/                  # Learning-based models (e.g. Transformer) and training scripts
│   ├── models/                # Model definitions
│   ├── datasets/              # Dataset interfaces / loaders (if applicable)
│   └── ...
├── create_training_dataset.py # Script for generating batches of simulated training data
├── <burstinfer_core_files>.py # Python reimplementation of BurstInfer
├── *_renew.py                 # Revised / refactored versions of original modules
├── requirements.txt           # Python dependencies (if provided)
└── README.md

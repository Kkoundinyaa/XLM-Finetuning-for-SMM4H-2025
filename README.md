# XLM Fine-Tuning Scripts for SMM4H 2025

This repository contains a collection of **fine-tuning and evaluation scripts using XLM / XLM-R based models** developed for tasks associated with **SMM4H 2025**.

The code focuses on applying multilingual transformer models to social media health–related classification tasks.

---

## Overview

The repository includes:
- Fine-tuning scripts for XLM-based models
- Model implementations and configurations
- Utility scripts for tagging, prediction, and comparison

Each script corresponds to a specific experiment or task setup.

---

## Repository Structure

```
XLM-Finetuning/
├── XLM-tuning-scripts/        # Fine-tuning scripts for XLM models
├── model/                     # Model implementations and variants
├── Compare.py                 # Script for comparing model outputs
├── predict_testset_Task1.py   # Test-set prediction script
├── tag.py                     # Tagging / preprocessing utilities
├── .gitignore
└── README.md
```

---

## Usage

Scripts in this repository are intended to be run independently depending on the task and experiment.  
Typical usage involves:
1. Fine-tuning an XLM-based model for a specific SMM4H task
2. Generating predictions on validation or test sets
3. Comparing and analyzing model outputs

Exact parameters and settings vary across scripts.

---

## Notes

This repository reflects completed experimental work prepared for **SMM4H 2025**.

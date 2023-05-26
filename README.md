# model-training

Used Python version 3.11.2 was used along with the modules specified in the requirement.txt

# DVC Guide

Data Version Control (DVC) is an open-source version control system for Machine Learning projects. This guide provides instructions on how to use DVC for running and reproducing experiments.

## Installation

Before using DVC, you need to install it. You can do this using pip:

```bash
pip install dvc
```
## Reproducing Experiments
DVC keeps track of machine learning experiments. You can reproduce any experiment using the following command:
```bash
dvc repro
```
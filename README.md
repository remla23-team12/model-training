# model-training

Used Python version 3.9.2 was used along with the modules specified in the requirement.txt

# DVC Guide
```diff
Note: If you got the `model-training` folder via BrightSpace, there won't be a .git folder (as per submission instructions). Due to the absence of of this .git folder, the execution of the dvc commands below will fail. The pytest command specified below will also fail, because you need to first run `dvc repro` to retrieve the datasets. Without it the tests cannot be run. If you want to follow the instructions below you must first clone our model-training repo, then the cloned folder will contain a .git folder, which allows the commands below to succeed when executed in the terminal.
```
---
---


Data Version Control (DVC) is an open-source version control system for Machine Learning projects. This guide provides instructions on how to use DVC for running and reproducing experiments.

## Installation

Before using DVC, you will need to install it. You can do this using pip:

```bash
pip install dvc
```

However, you can also just run the following command in the terminal when in the model-training directory (we recommend using a python virtual environment first with python 3.9 (you might need to upgrade pip first)):


```bash
pip install -r requirements.txt
```

## Reproducing Experiments

DVC keeps track of machine learning experiments. You can reproduce any experiment using the following command:

```bash
dvc repro
```

## Metrics

You can view metrics difference between latest commit and after some changes using the following command:

```bash
dvc exp run
```

Make a change and execute git add -A, git commit -m "change something"

```bash
dvc exp run
```

```bash
dvc metrics diff
```

To push artifacts/files to remote storage
```bash
dvc push
```

## Pytest
```diff
Note!: You must execute the `dvc repro` command in your terminal first BEFORE running the command below, or else the command below will fail because the tests rely on the dataset retrieved from a remote place using the `dvc repro` command
```
A single command can be executed in the terminal to run all the tests:
```bash
pytest --junitxml=pytest-results.xml -cov=src 
```

## Pylint

All files have been well documented and reach full score using pylint, verify this by executing the following command:

```bash
pylint --load-plugins=dslinter src/<dir>/<testFile>.py
```

## Cookiecutter info

# model-training

REMLA Course

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── ~                  <- dvc local remote
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── tests          <- Scripts to do a simple test
    │   │   └── test_simple.py
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features and datasets for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations (unused)
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Introduction

nirdizati-light is a Python library for predictive process mining that focuses on the following aspects:

- trace encoding
- model training
- hyperparameter optimization
- model evaluation
- explainability

## Demo video

Video demonstration can be found at [this link](https://drive.google.com/file/d/1ZflptyPiuUC84JHncZf78uCR3ZaB-1Jd/view).

## Documentation

Documentation for nirdizati-light can be found at [this link](https://rgraziosi-fbk.github.io/nirdizati-light/nirdizati_light.html).

## Notebook

A Colab notebook with an example pipeline can be found at [this link](https://colab.research.google.com/drive/1_4b7PaNcp9YGhIVxa-TPIqI4qn6-AAAT?usp=sharing).

## Installation

1. Clone the nirdizati-light repository
2. In your project, run `pip install -e <path-to-nirdizati-light-folder>`

## Examples

### Simple pipeline

The [run_simple_pipeline.py](https://github.com/rgraziosi-fbk/nirdizati-light/blob/main/run_simple_pipeline.py) script defines a list of models to train and evaluate. Hyperparameter optimization is performed using default hyperopt spaces. It is the easiest example to get started with nirdizati-light.

### Full pipeline

The [run_full_pipeline.py](https://github.com/rgraziosi-fbk/nirdizati-light/blob/main/run_full_pipeline.py) script extends the simple pipeline by also configuring custom hyperparameter optimization search spaces and defining a custom Pytorch model to train and evaluate. This pipeline is more complex and shows off the full capabilities of nirdizati-light.

## Features

### Encodings

- Simple encoding
- Simple trace encoding
- Frequency encoding
- Complex encoding
- Loreley encoding
- Loreley complex encoding

### Labeling types

- Next activity (classification)
- Attribute string, i.e. outcome (classification)
- Remaining time (regression)
- Duration (regression)

### Predictive models (for classification)

- Random forest (scikit-learn)
- Decision tree (scikit-learn)
- KNN (scikit-learn)
- XGBoost (scikit-learn)
- SGD (scikit-learn)
- SVC (scikit-learn)
- LSTM (PyTorch)
- CustomPytorch, i.e. specify your own custom PyTorch model (PyTorch)

### Predictive models (for regression)

- Random forest (scikit-learn)

### Hyperparameter optimization targets

- F1 score (classification)
- AUC (classification)
- Accuracy (classification)
- MAE (regression)

### Explainers

- ICE
- SHAP
- DiCE
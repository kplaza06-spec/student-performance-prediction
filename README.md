
# Student Performance Prediction System

## Project Description

A Python and Streamlit application that predicts
whether a student is AT RISK or NOT AT RISK of
failing the final Mathematics grade.

## Problem Statement

Schools need ways to identify students who may
require additional academic support.

## Dataset

UCI Student Performance Dataset:
https://archive.ics.uci.edu/dataset/320/student+performance

## Prediction Target

- G3 below 10 = AT RISK
- G3 equal to or above 10 = NOT AT RISK

G3 is excluded from the input features to prevent
target leakage.

## Technologies

- Python
- Pandas
- Scikit-learn
- Random Forest
- Streamlit
- GitHub

## Installation

pip install -r requirements.txt

## Run Training

python train_model.py

## Run Application

streamlit run app.py

## Features

- Student dataset preview
- CSV upload
- Student risk prediction
- Prediction results
- Downloadable prediction CSV

## Limitation

The model uses historical data from one Portuguese
secondary-school dataset. It may not generalize
to other schools or student populations.

Predictions should support, not replace, faculty judgment.

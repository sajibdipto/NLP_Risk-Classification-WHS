
# NLP Risk Classification for Work Health and Safety

## Overview
This project builds an NLP-based hazard classification system for workplace incident reports in the Work Health and Safety domain.

It compares traditional machine learning approaches and transformer-based deep learning models for classifying incident descriptions into safety risk categories.

## Objectives
- classify incident descriptions into WHS risk categories
- compare TF-IDF, FastText, CNN, and RoBERTa-based models
- evaluate performance under class imbalance
- support future deployment for automated incident triage

## Dataset
- Input column: `WHAT_HAPPENED_ENGLISH`
- Target label: `SAFETY_RISK_CATEGORY`
- Multi-class classification problem with severe class imbalance

## Models Implemented
- TF-IDF + Logistic Regression
- Word2Vec + Logistic Regression
- FastText + Logistic Regression
- CNN with FastText embeddings
- RoBERTa-base
- DistilRoBERTa

## Project Structure
```text
NLP_risk_classifier/
├── train_roberta_risk.py
├── train_distilroberta_risk.py
├── inference_roberta.py
├── evaluate_full_dataset.py
├── visualization.py
└── utils.py

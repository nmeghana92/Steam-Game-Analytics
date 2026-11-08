# Model Evaluation

## 1. Project Overview

This project uses machine learning to classify Steam games
into three sentiment categories based on user review percentages.

## 2. Sentiment Classification

The sentiment labels are generated using the following thresholds:

| Positive Review Percentage | Sentiment |
|---|---|
| 0–40% | Negative |
| Above 40–70% | Mixed |
| Above 70–100% | Positive |

## 3. Machine Learning Models

The project evaluates four classification algorithms.

### Logistic Regression
A classification algorithm used to predict sentiment categories.

### Decision Tree
A tree-based algorithm that learns decision rules from the data.

### Random Forest
An ensemble learning algorithm that combines multiple decision trees.

### Dummy Classifier
A baseline model using the most frequent class for predictions.

## 4. Feature Processing

- TF-IDF vectorization for game names and genres.
- Median imputation for numerical price values.
- Standard scaling for numerical features.

## 5. Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion Matrix

## 6. Train-Test Split

The dataset is divided into:

- 80% Training Data
- 20% Testing Data

The split uses a fixed random state of 42 and stratified sampling.

## 7. Objective

The objective is to compare machine learning models
and identify their effectiveness in classifying
Steam game review sentiment.

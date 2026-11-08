import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

LOCAL_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "games_march2025_cleaned.csv"
)

@st.cache_data
def prepare_training_data():
    raw_df = pd.read_csv(
        LOCAL_DATA_PATH,
        engine="python",
        on_bad_lines="skip"
    )

    mandatory_cols = [
        "name", "genres", "price", 
        "positive", "negative", "pct_pos_total"
    ]

    missing_cols = [c for c in mandatory_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    processed_df = raw_df[mandatory_cols].copy()

    for col in ["price", "positive", "negative", "pct_pos_total"]:
        processed_df[col] = pd.to_numeric(processed_df[col], errors="coerce")

    processed_df = processed_df.dropna(
        subset=["name", "genres", "price", "pct_pos_total"]
    )

    processed_df = processed_df[
        processed_df["pct_pos_total"].between(0, 100) & (processed_df["price"] >= 0)
    ].copy()

    processed_df["sentiment_label"] = pd.cut(
        processed_df["pct_pos_total"],
        bins=[-0.01, 40, 70, 100],
        labels=["Negative", "Mixed", "Positive"],
        include_lowest=True
    )

    processed_df["features_text"] = (
        processed_df["name"].astype(str) + " " + processed_df["genres"].astype(str)
    )

    processed_df = processed_df.dropna(subset=["sentiment_label"])
    return processed_df.reset_index(drop=True)


@st.cache_data
def analyze_price_features():
    data = prepare_training_data()

    price_stats = (
        data.groupby("sentiment_label", observed=False)["price"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )

    price_stats.columns = [
        "Sentiment", "Total Games", "Mean Price", "Median Price"
    ]
    return price_stats


@st.cache_data
def run_model_eval():
    dataset = prepare_training_data()

    X_features = dataset[["features_text", "price"]].copy()
    y_target = dataset["sentiment_label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_target,
        test_size=0.20,
        random_state=42,
        stratify=y_target
    )

    text_transformer = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, stop_words="english"))
    ])

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("text_pipe", text_transformer, "features_text"),
        ("num_pipe", num_transformer, ["price"])
    ])

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1),
        "Dummy": DummyClassifier(strategy="most_frequent")
    }

    eval_metrics = []
    class_reports = {}
    conf_matrices = {}

    for name, clf in classifiers.items():
        model_pipeline = Pipeline([
            ("prep", preprocessor),
            ("classifier", clf)
        ])

        model_pipeline.fit(X_train, y_train)
        predictions = model_pipeline.predict(X_test)

        eval_metrics.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Macro Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
            "Macro F1": f1_score(y_test, predictions, average="macro", zero_division=0)
        })

        class_reports[name] = classification_report(
            y_test, predictions,
            labels=["Negative", "Mixed", "Positive"],
            zero_division=0, output_dict=True
        )

        conf_matrices[name] = confusion_matrix(
            y_test, predictions, labels=["Negative", "Mixed", "Positive"]
        )

    return pd.DataFrame(eval_metrics), class_reports, conf_matrices
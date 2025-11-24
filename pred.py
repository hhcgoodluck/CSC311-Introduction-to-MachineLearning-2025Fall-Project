import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RATING_COLS = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?",
]

BEST_TASKS_COL = "Which types of tasks do you feel this model handles best? (Select all that apply.)"
SUBOPT_TASKS_COL = "For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)"

TARGET_TASKS = [
    "Math computations",
    "Writing or debugging code",
    "Data processing or analysis",
    "Explaining complex concepts simply",
    "Writing or editing essays/reports",
    "Drafting professional text",
    "Brainstorming or generating creative ideas",
    "Converting content between formats",
]

TEXT_COLS = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?",
]

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def load_artifacts():
    with open(ARTIFACT_DIR / "word_vocab.json", "r") as f:
        word_vocab = json.load(f)
    word_idf = np.load(ARTIFACT_DIR / "word_idf.npy")

    with open(ARTIFACT_DIR / "char_vocab.json", "r") as f:
        char_vocab = json.load(f)
    char_idf = np.load(ARTIFACT_DIR / "char_idf.npy")

    svd_components = np.load(ARTIFACT_DIR / "svd_components.npy")
    scaler_mean = np.load(ARTIFACT_DIR / "scaler_mean.npy")
    scaler_scale = np.load(ARTIFACT_DIR / "scaler_scale.npy")

    lr_W = np.load(ARTIFACT_DIR / "lr_W.npy")
    lr_b = np.load(ARTIFACT_DIR / "lr_b.npy")
    class_labels = np.load(ARTIFACT_DIR / "class_labels.npy", allow_pickle=True)

    return {
        "word_vocab": word_vocab,
        "word_idf": word_idf.astype(np.float32),
        "char_vocab": char_vocab,
        "char_idf": char_idf.astype(np.float32),
        "svd_components": svd_components.astype(np.float32),
        "scaler_mean": scaler_mean.astype(np.float32),
        "scaler_scale": scaler_scale.astype(np.float32),
        "lr_W": lr_W.astype(np.float32),
        "lr_b": lr_b.astype(np.float32),
        "class_labels": class_labels,
    }

ARTIFACTS = load_artifacts()


def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).lower())


def generate_ngrams(tokens, n):
    if n == 1:
        return tokens
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def build_word_tfidf(texts):
    vocab = ARTIFACTS["word_vocab"]
    idf = ARTIFACTS["word_idf"]
    D = len(idf)
    X = np.zeros((len(texts), D), dtype=np.float32)

    for i, text in enumerate(texts):
        counts = {}
        tokens = tokenize(text)
        for n in (1, 2):
            if len(tokens) < n:
                continue
            for gram in generate_ngrams(tokens, n):
                idx = vocab.get(gram)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
        for idx, count in counts.items():
            X[i, idx] = 1.0 + np.log(count)

    X *= idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def build_char_tfidf(texts):
    vocab = ARTIFACTS["char_vocab"]
    idf = ARTIFACTS["char_idf"]
    D = len(idf)
    X = np.zeros((len(texts), D), dtype=np.float32)

    for i, text in enumerate(texts):
        counts = {}
        s = str(text).lower()
        L = len(s)
        for n in (3, 4, 5):
            if L < n:
                continue
            for start in range(L - n + 1):
                gram = s[start : start + n]
                idx = vocab.get(gram)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
        for idx, count in counts.items():
            X[i, idx] = 1.0 + np.log(count)

    X *= idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms
    return X @ ARTIFACTS["svd_components"].T


def encode_tasks(series):
    arr = np.zeros((len(series), len(TARGET_TASKS)), dtype=np.float32)
    for i, entry in enumerate(series.fillna("")):
        text = str(entry)
        for j, task in enumerate(TARGET_TASKS):
            if task in text:
                arr[i, j] = 1.0
    return arr


def build_structured(df):
    ratings = []
    for col in RATING_COLS:
        ratings.append(df[col].apply(extract_rating).to_numpy(dtype=np.float32))
    ratings = np.column_stack(ratings)

    best = encode_tasks(df[BEST_TASKS_COL])
    subopt = encode_tasks(df[SUBOPT_TASKS_COL])

    structured = np.hstack([ratings, best, subopt])
    return (structured - ARTIFACTS["scaler_mean"]) / ARTIFACTS["scaler_scale"]


def extract_rating(text):
    if pd.isna(text):
        return 3
    if isinstance(text, (int, float)):
        return int(text)
    match = re.match(r"^(\d+)", str(text))
    return int(match.group(1)) if match else 3


def build_features(df):
    combined_text = df[TEXT_COLS].fillna(" ").agg(" ".join, axis=1).tolist()
    X_word = build_word_tfidf(combined_text)
    X_char = build_char_tfidf(combined_text)
    X_struct = build_structured(df)
    return np.hstack([X_struct, X_word, X_char]).astype(np.float32)


def predict_lr(X):
    scores = X @ ARTIFACTS["lr_W"].T + ARTIFACTS["lr_b"]
    idx = np.argmax(scores, axis=1)
    return ARTIFACTS["class_labels"][idx]


def predict_all(csv_path):
    df = pd.read_csv(csv_path)
    X = build_features(df)
    return predict_lr(X)


def main():
    import sys

    preds = predict_all(sys.argv[1])
    for label in preds:
        print(label)


if __name__ == "__main__":
    main()

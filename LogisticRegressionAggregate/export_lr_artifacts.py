import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler

FILE_NAME = "DataSet/training_data_clean.csv"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

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

WORD_VECTORIZER_PARAMS = dict(
    max_features=12000,
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    stop_words=None,
)

CHAR_VECTORIZER_PARAMS = dict(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=20000,
    min_df=1,
    sublinear_tf=True,
    lowercase=True,
)

CHAR_SVD_COMPONENTS = 200


def extract_rating(value) -> int:
    if isinstance(value, str):
        match = re.match(r"^(\d+)", value)
        if match:
            return int(match.group(1))
    if value is not None and not (isinstance(value, float) and np.isnan(value)):
        try:
            return int(value)
        except Exception:
            pass
    return 3


def collect_tasks(series: pd.Series):
    processed = []
    for entry in series.fillna(""):
        entry = str(entry)
        tasks = [task for task in TARGET_TASKS if task in entry]
        processed.append(tasks)
    return processed


def build_features(df: pd.DataFrame):
    ratings = np.column_stack([df[col].apply(extract_rating).to_numpy(dtype=np.float32) for col in RATING_COLS])

    best_lists = collect_tasks(df[BEST_TASKS_COL])
    sub_lists = collect_tasks(df[SUBOPT_TASKS_COL])

    mlb_best = MultiLabelBinarizer(classes=TARGET_TASKS)
    mlb_sub = MultiLabelBinarizer(classes=TARGET_TASKS)
    mlb_best.fit([TARGET_TASKS])
    mlb_sub.fit([TARGET_TASKS])
    best = mlb_best.transform(best_lists)
    subopt = mlb_sub.transform(sub_lists)
    structured = np.hstack([ratings, best, subopt]).astype(np.float32)

    scaler = StandardScaler()
    structured_scaled = scaler.fit_transform(structured)

    combined_text = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)

    word_vectorizer = TfidfVectorizer(**WORD_VECTORIZER_PARAMS)
    X_word = word_vectorizer.fit_transform(combined_text)

    char_vectorizer = TfidfVectorizer(**CHAR_VECTORIZER_PARAMS)
    X_char = char_vectorizer.fit_transform(combined_text)

    svd = TruncatedSVD(n_components=CHAR_SVD_COMPONENTS, random_state=0)
    X_char_svd = svd.fit_transform(X_char)

    X_full = np.hstack([
        structured_scaled,
        X_word.toarray(),
        X_char_svd,
    ])

    artifacts = {
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "word_vocab": word_vectorizer.vocabulary_,
        "word_idf": word_vectorizer.idf_.astype(np.float32),
        "char_vocab": char_vectorizer.vocabulary_,
        "char_idf": char_vectorizer.idf_.astype(np.float32),
        "svd_components": svd.components_.astype(np.float32),
    }

    return X_full, artifacts


def main():
    df = pd.read_csv(FILE_NAME)
    X_full, feats = build_features(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"].values)

    lr = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        C=1.0,
        max_iter=5000,
        n_jobs=-1,
    )
    lr.fit(X_full, y)

    np.save(ARTIFACT_DIR / "lr_W.npy", lr.coef_.astype(np.float32))
    np.save(ARTIFACT_DIR / "lr_b.npy", lr.intercept_.astype(np.float32))
    np.save(ARTIFACT_DIR / "class_labels.npy", label_encoder.classes_)

    word_vocab_serializable = {k: int(v) for k, v in feats["word_vocab"].items()}
    with open(ARTIFACT_DIR / "word_vocab.json", "w") as f:
        json.dump(word_vocab_serializable, f)
    np.save(ARTIFACT_DIR / "word_idf.npy", feats["word_idf"])

    char_vocab_serializable = {k: int(v) for k, v in feats["char_vocab"].items()}
    with open(ARTIFACT_DIR / "char_vocab.json", "w") as f:
        json.dump(char_vocab_serializable, f)
    np.save(ARTIFACT_DIR / "char_idf.npy", feats["char_idf"])
    np.save(ARTIFACT_DIR / "svd_components.npy", feats["svd_components"])

    np.save(ARTIFACT_DIR / "scaler_mean.npy", feats["scaler_mean"])
    np.save(ARTIFACT_DIR / "scaler_scale.npy", feats["scaler_scale"])

    print("Artifacts exported to", ARTIFACT_DIR.resolve())


if __name__ == "__main__":
    main()

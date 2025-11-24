"""
Error analysis for the best RF configuration (Iter5).

Steps:
- Student-wise 80/20 split (same as training).
- Rebuild structured + CountVectorizer features (max_features=3000).
- Train RF with params from iter5 (n_estimators=600, max_depth=8, max_features=0.2,
  min_samples_split=2, min_samples_leaf=3, criterion='gini').
- Compute confusion matrix, per-class accuracy, and list of misclassified samples.
- Save top misclassified rows to CSV for manual inspection.
"""
import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

file_name = "../DataSet/training_data_clean.csv"
MISCLASS_PATH = Path("artifacts/misclassified_iter5.csv")

RATING_COLS = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?",
]

BEST_TASKS_COL = 'Which types of tasks do you feel this model handles best? (Select all that apply.)'
SUBOPT_TASKS_COL = 'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'

target_tasks = [
    'Math computations',
    'Writing or debugging code',
    'Data processing or analysis',
    'Explaining complex concepts simply',
    'Writing or editing essays/reports',
    'Drafting professional text',
    'Brainstorming or generating creative ideas',
    'Converting content between formats',
]

TEXT_COLS = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?",
]

NEUTRAL_RATING_VALUE = 3


def process_multiselect(series, target_tasks):
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            text = str(response)
            present_tasks = [task for task in target_tasks if task in text]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None


def build_rating_matrix(df, rating_cols, neutral_value=3):
    columns = []
    for col in rating_cols:
        raw = df[col].apply(extract_rating)
        filled = raw.fillna(neutral_value).astype(int)
        columns.append(filled.to_numpy().reshape(-1, 1))
    return np.hstack(columns)


def build_text_series(df, text_cols):
    combined = df[text_cols[0]].fillna('')
    for col in text_cols[1:]:
        combined = combined + " " + df[col].fillna('')
    return combined


def student_wise_split(df, train_ratio=0.8, seed=0):
    student_ids = df['student_id'].values
    unique_ids = np.unique(student_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    n_train_students = int(train_ratio * len(unique_ids))
    train_ids = set(unique_ids[:n_train_students])
    is_train = np.array([sid in train_ids for sid in student_ids])
    return df[is_train].copy(), df[~is_train].copy()


def build_features(df_train, df_test, max_text_features=3000):
    X_ratings_train = build_rating_matrix(df_train, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)
    X_ratings_test = build_rating_matrix(df_test, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)

    best_tasks_train = process_multiselect(df_train[BEST_TASKS_COL], target_tasks)
    subopt_tasks_train = process_multiselect(df_train[SUBOPT_TASKS_COL], target_tasks)

    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()

    best_encoded_train = mlb_best.fit_transform(best_tasks_train)
    subopt_encoded_train = mlb_subopt.fit_transform(subopt_tasks_train)

    best_encoded_test = mlb_best.transform(process_multiselect(df_test[BEST_TASKS_COL], target_tasks))
    subopt_encoded_test = mlb_subopt.transform(process_multiselect(df_test[SUBOPT_TASKS_COL], target_tasks))

    X_struct_train = np.hstack([X_ratings_train, best_encoded_train, subopt_encoded_train])
    X_struct_test = np.hstack([X_ratings_test, best_encoded_test, subopt_encoded_test])

    train_text = build_text_series(df_train, TEXT_COLS)
    test_text = build_text_series(df_test, TEXT_COLS)

    vectorizer = CountVectorizer(
        max_features=max_text_features,
        lowercase=True,
    )
    X_text_train = vectorizer.fit_transform(train_text).toarray()
    X_text_test = vectorizer.transform(test_text).toarray()

    X_train = np.hstack([X_struct_train, X_text_train])
    X_test = np.hstack([X_struct_test, X_text_test])

    return X_train, X_test, vectorizer, mlb_best, mlb_subopt


def main():
    df = pd.read_csv(file_name)
    df_train, df_test = student_wise_split(df, train_ratio=0.8, seed=0)
    print(f"Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")

    X_train, X_test, vectorizer, mlb_best, mlb_subopt = build_features(df_train, df_test, max_text_features=3000)
    y_train = df_train['label'].values
    y_test = df_test['label'].values

    params = {
        'n_estimators': 600,
        'max_depth': 8,
        'max_features': 0.2,
        'min_samples_split': 2,
        'min_samples_leaf': 3,
        'criterion': 'gini',
        'random_state': 0,
        'n_jobs': -1,
    }

    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nTest accuracy: {acc:.3f}, macro F1: {f1:.3f}")

    labels = sorted(df['label'].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

    misclassified = df_test.copy()
    misclassified['pred'] = y_pred
    misclassified = misclassified[misclassified['label'] != misclassified['pred']]
    print(f"\nTotal misclassified: {len(misclassified)} / {len(df_test)}")
    sample = misclassified.head(100)
    MISCLASS_PATH.parent.mkdir(exist_ok=True)
    sample.to_csv(MISCLASS_PATH, index=False)
    print(f"Saved first {len(sample)} misclassified samples to {MISCLASS_PATH}")


if __name__ == "__main__":
    main()

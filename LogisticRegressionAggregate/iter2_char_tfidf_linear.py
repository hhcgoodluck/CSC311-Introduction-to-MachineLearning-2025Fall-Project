"""
Character-level TF-IDF baseline using LinearSVC.

Configuration:
- analyzer='char', ngram_range=(3,5)
- max_features=20000
- min_df=1, sublinear_tf=True
- Structured features (20 dims) standardized and concatenated with TF-IDF.
- Student-wise 80/20 split + 4-fold StratifiedKFold for CV accuracy.
"""
import numpy as np
import pandas as pd
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

file_name = "../DataSet/training_data_clean.csv"

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
TFIDF_MAX_FEATURES = 20000
LINEAR_SVC_C_VALUES = [0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]


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


def build_features(df_train, df_test, max_features=TFIDF_MAX_FEATURES):
    X_ratings_train = build_rating_matrix(df_train, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)
    X_ratings_test = build_rating_matrix(df_test, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)

    best_train = process_multiselect(df_train[BEST_TASKS_COL], target_tasks)
    subopt_train = process_multiselect(df_train[SUBOPT_TASKS_COL], target_tasks)

    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()

    best_encoded_train = mlb_best.fit_transform(best_train)
    subopt_encoded_train = mlb_subopt.fit_transform(subopt_train)

    best_encoded_test = mlb_best.transform(process_multiselect(df_test[BEST_TASKS_COL], target_tasks))
    subopt_encoded_test = mlb_subopt.transform(process_multiselect(df_test[SUBOPT_TASKS_COL], target_tasks))

    X_struct_train_dense = np.hstack([X_ratings_train, best_encoded_train, subopt_encoded_train]).astype(float)
    X_struct_test_dense = np.hstack([X_ratings_test, best_encoded_test, subopt_encoded_test]).astype(float)

    scaler = StandardScaler()
    X_struct_train = csr_matrix(scaler.fit_transform(X_struct_train_dense))
    X_struct_test = csr_matrix(scaler.transform(X_struct_test_dense))

    train_text = build_text_series(df_train, TEXT_COLS)
    test_text = build_text_series(df_test, TEXT_COLS)

    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=max_features,
        min_df=1,
        lowercase=True,
        sublinear_tf=True,
    )
    X_text_train = vectorizer.fit_transform(train_text)
    X_text_test = vectorizer.transform(test_text)

    X_train = hstack([X_struct_train, X_text_train], format='csr')
    X_test = hstack([X_struct_test, X_text_test], format='csr')

    return X_train, df_train['label'].values, X_test, df_test['label'].values, vectorizer


def evaluate_classifier(clf, X, y, cv_splits=4):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    accs, f1s = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[val_idx])
        acc = accuracy_score(y[val_idx], pred)
        f1 = f1_score(y[val_idx], pred, average='macro')
        accs.append(acc)
        f1s.append(f1)
        print(f"    Fold {fold}: acc={acc:.3f}, f1={f1:.3f}")
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


def main():
    df = pd.read_csv(file_name)
    df_train, df_test = student_wise_split(df, train_ratio=0.8, seed=0)
    print(f"Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")

    X_train, y_train, X_test, y_test, vectorizer = build_features(df_train, df_test)
    print(f"Feature shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

    best_score = -1
    best_C = None

    print("\n=== Char-level LinearSVC Configurations ===")
    for C in LINEAR_SVC_C_VALUES:
        clf = LinearSVC(C=C, max_iter=5000)
        mean_acc, std_acc, mean_f1 = evaluate_classifier(clf, X_train, y_train)
        print(f"  C={C}: CV acc={mean_acc:.3f} ± {std_acc:.3f}, CV F1={mean_f1:.3f}")
        if mean_acc > best_score:
            best_score = mean_acc
            best_C = C

    print(f"\nBest CV configuration: LinearSVC(C={best_C}), CV accuracy={best_score:.3f}")

    final_clf = LinearSVC(C=best_C, max_iter=5000)
    final_clf.fit(X_train, y_train)

    train_pred = final_clf.predict(X_train)
    test_pred = final_clf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')

    print("\n=== Char-level Model Performance ===")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Training macro F1: {train_f1:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test macro F1: {test_f1:.3f}")


if __name__ == "__main__":
    main()

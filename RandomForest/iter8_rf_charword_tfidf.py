"""
Random Forest with combined word + character TF-IDF features.

Motivation:
- Word-level TF-IDF captures semantics.
- Character-level TF-IDF captures stylistic cues (capitalization, punctuation, spelling),
  which may be important for distinguishing between models.
- Apply SelectKBest to keep the feature space compact and reduce noise.

Pipeline:
1. Student-wise 80/20 split.
2. Build structured features (ratings + multi-select) identical to prior iterations.
3. Build word TF-IDF (ngram (1,2)) and char TF-IDF (ngram (3,5)); concatenate.
4. Apply SelectKBest (f_classif) to top 4000 combined TF-IDF features.
5. Concatenate structured + selected TF-IDF features -> feed into RandomForest.
6. Random search hyperparameters (focus on promising ranges) with 3-fold StratifiedKFold.
7. Train final RF on best config and evaluate accuracy/F1.
"""
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import hstack
import random

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


def build_features(df_train, df_test, k_best=4000):
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

    word_vectorizer = TfidfVectorizer(
        max_features=5000,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=5000,
        min_df=2,
    )

    X_word_train = word_vectorizer.fit_transform(train_text)
    X_word_test = word_vectorizer.transform(test_text)
    X_char_train = char_vectorizer.fit_transform(train_text)
    X_char_test = char_vectorizer.transform(test_text)

    X_text_train = hstack([X_word_train, X_char_train])
    X_text_test = hstack([X_word_test, X_char_test])

    selector = SelectKBest(score_func=f_classif, k=min(k_best, X_text_train.shape[1]))
    X_text_train_sel = selector.fit_transform(X_text_train, df_train['label'].values)
    X_text_test_sel = selector.transform(X_text_test)

    X_train_final = np.hstack([X_struct_train, X_text_train_sel.toarray()])
    X_test_final = np.hstack([X_struct_test, X_text_test_sel.toarray()])

    return {
        'X_train': X_train_final,
        'X_test': X_test_final,
        'y_train': df_train['label'].values,
        'y_test': df_test['label'].values,
        'word_vectorizer': word_vectorizer,
        'char_vectorizer': char_vectorizer,
        'selector': selector,
        'mlb_best': mlb_best,
        'mlb_subopt': mlb_subopt,
    }


def evaluate_params(X_train, y_train, params, cv_folds=3):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    scores = []
    f1_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        rf = RandomForestClassifier(**params)
        rf.fit(X_train[train_idx], y_train[train_idx])
        preds = rf.predict(X_train[val_idx])
        acc = accuracy_score(y_train[val_idx], preds)
        f1 = f1_score(y_train[val_idx], preds, average='macro')
        scores.append(acc)
        f1_scores.append(f1)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(f1_scores))


def random_params(n_samples=60):
    param_space = {
        'n_estimators': [600, 700, 800, 900, 1000, 1200],
        'max_depth': [None, 10, 12, 14, 16],
        'max_features': [0.2, 0.25, 0.3, 'sqrt'],
        'min_samples_split': [2, 3, 5, 7],
        'min_samples_leaf': [1, 2, 3],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
    }
    combinations = []
    random.seed(0)
    for _ in range(n_samples):
        params = {
            'n_estimators': random.choice(param_space['n_estimators']),
            'max_depth': random.choice(param_space['max_depth']),
            'max_features': random.choice(param_space['max_features']),
            'min_samples_split': random.choice(param_space['min_samples_split']),
            'min_samples_leaf': random.choice(param_space['min_samples_leaf']),
            'criterion': random.choice(param_space['criterion']),
            'class_weight': random.choice(param_space['class_weight']),
            'random_state': 0,
            'n_jobs': -1,
        }
        combinations.append(params)
    return combinations


def main():
    df = pd.read_csv(file_name)
    df_train, df_test = student_wise_split(df, train_ratio=0.8, seed=0)
    print(f"Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")

    features = build_features(df_train, df_test, k_best=4000)
    X_train = features['X_train']
    y_train = features['y_train']
    X_test = features['X_test']
    y_test = features['y_test']

    print(f"Final X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    best_score = -1
    best_params = None
    best_result = None

    print("\n=== Random search with combined TF-IDF features ===")
    param_combos = random_params(n_samples=60)
    for idx, params in enumerate(param_combos, 1):
        mean_acc, std_acc, mean_f1 = evaluate_params(X_train, y_train, params, cv_folds=3)
        if mean_acc > best_score:
            best_score = mean_acc
            best_params = params.copy()
            best_result = (mean_acc, std_acc, mean_f1)
        if idx % 10 == 0 or mean_acc > 0.68:
            print(f"{idx:02d} - CV acc: {mean_acc:.3f} ± {std_acc:.3f}, F1: {mean_f1:.3f}, params: {params}")

    print("\nBest params:", best_params)
    print(f"Best CV acc: {best_result[0]:.3f} ± {best_result[1]:.3f}, F1: {best_result[2]:.3f}")

    rf = RandomForestClassifier(**best_params)
    rf.fit(X_train, y_train)

    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    test_pred = rf.predict(X_test)
    test_f1 = f1_score(y_test, test_pred, average='macro')

    print("\n=== Final Performance ===")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test macro F1: {test_f1:.3f}")


if __name__ == "__main__":
    main()

"""
Stacking ensemble combining:
- Word+Char LinearSVC (best from iter3)
- MultinomialNB (word TF-IDF)
- RF ensemble predictions (iter5 best config)

Strategy:
1. Rebuild features (word TF-IDF, char TF-IDF SVD, structured).
2. Train base models:
    a) LinearSVC (word+char).
    b) MultinomialNB (word-level TF-IDF only).
    c) RandomForest with iter5 params.
3. Generate out-of-fold predictions (probabilities or scores) for stacking.
4. Train meta LogisticRegression on stacked features.
5. Evaluate on hold-out test split.
"""
import numpy as np
import pandas as pd
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

WORD_VECTORIZER_PARAMS = dict(
    max_features=12000,
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    stop_words=None,
)

CHAR_VECTORIZER_PARAMS = dict(
    analyzer='char',
    ngram_range=(3, 5),
    max_features=20000,
    min_df=1,
    sublinear_tf=True,
    lowercase=True,
)

CHAR_SVD_COMPONENTS = 200
LINEAR_C = 0.05
RF_PARAMS = {
    'n_estimators': 600,
    'max_depth': 8,
    'max_features': 0.2,
    'min_samples_split': 2,
    'min_samples_leaf': 3,
    'criterion': 'gini',
    'random_state': 0,
    'n_jobs': -1,
}


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


def build_features(df_train, df_test):
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

    X_struct_train = np.hstack([X_ratings_train, best_encoded_train, subopt_encoded_train]).astype(float)
    X_struct_test = np.hstack([X_ratings_test, best_encoded_test, subopt_encoded_test]).astype(float)

    scaler = StandardScaler()
    X_struct_train = csr_matrix(scaler.fit_transform(X_struct_train))
    X_struct_test = csr_matrix(scaler.transform(X_struct_test))

    train_text = build_text_series(df_train, TEXT_COLS)
    test_text = build_text_series(df_test, TEXT_COLS)

    word_vectorizer = TfidfVectorizer(**WORD_VECTORIZER_PARAMS)
    char_vectorizer = TfidfVectorizer(**CHAR_VECTORIZER_PARAMS)

    X_word_train = word_vectorizer.fit_transform(train_text)
    X_word_test = word_vectorizer.transform(test_text)

    X_char_train = char_vectorizer.fit_transform(train_text)
    X_char_test = char_vectorizer.transform(test_text)

    svd = TruncatedSVD(n_components=CHAR_SVD_COMPONENTS, random_state=0)
    X_char_train_svd = svd.fit_transform(X_char_train)
    X_char_test_svd = svd.transform(X_char_test)

    X_char_train_svd = csr_matrix(X_char_train_svd)
    X_char_test_svd = csr_matrix(X_char_test_svd)

    X_word_char_train = hstack([X_word_train, X_char_train_svd], format='csr')
    X_word_char_test = hstack([X_word_test, X_char_test_svd], format='csr')

    X_train = hstack([X_struct_train, X_word_char_train], format='csr')
    X_test = hstack([X_struct_test, X_word_char_test], format='csr')

    return {
        'X_struct_train': X_struct_train,
        'X_struct_test': X_struct_test,
        'X_word_train': X_word_train,
        'X_word_test': X_word_test,
        'X_word_char_train': X_word_char_train,
        'X_word_char_test': X_word_char_test,
        'X_train_final': X_train,
        'X_test_final': X_test,
        'y_train': df_train['label'].values,
        'y_test': df_test['label'].values,
    }


def get_oof_predictions(clf, X, y, cv_splits=4, decision_function=False):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    oof = np.zeros((X.shape[0], len(np.unique(y))))
    for train_idx, val_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        if decision_function and hasattr(clf, "decision_function"):
            preds = clf.decision_function(X[val_idx])
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
        else:
            preds = clf.predict_proba(X[val_idx])
        oof[val_idx] = preds
    clf.fit(X, y)
    return clf, oof


def main():
    df = pd.read_csv(file_name)
    df_train, df_test = student_wise_split(df, train_ratio=0.8, seed=0)
    features = build_features(df_train, df_test)

    y_train = features['y_train']
    y_test = features['y_test']
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    n_classes = len(label_encoder.classes_)

    print("Training base models and generating OOF predictions...")

    linear_svc = LinearSVC(C=LINEAR_C, max_iter=6000)
    linear_svc, oof_linear = get_oof_predictions(
        linear_svc,
        features['X_train_final'],
        y_train_enc,
        cv_splits=4,
        decision_function=True,
    )

    nb = MultinomialNB()
    nb, oof_nb = get_oof_predictions(
        nb,
        features['X_word_train'],
        y_train_enc,
        cv_splits=4,
        decision_function=False,
    )

    rf = RandomForestClassifier(**RF_PARAMS)
    rf, oof_rf = get_oof_predictions(
        rf,
        features['X_train_final'].toarray(),
        y_train_enc,
        cv_splits=4,
        decision_function=False,
    )

    stack_train = np.hstack([oof_linear, oof_nb, oof_rf])
    stack_model = LogisticRegression(
        max_iter=2000,
        solver='lbfgs',
        multi_class='multinomial',
    )
    stack_model.fit(stack_train, y_train_enc)

    print("Evaluating on hold-out test split...")
    linear_scores = linear_svc.decision_function(features['X_test_final'])
    if linear_scores.ndim == 1:
        linear_scores = linear_scores.reshape(-1, 1)
    nb_probs = nb.predict_proba(features['X_word_test'])
    rf_probs = rf.predict_proba(features['X_test_final'].toarray())
    stack_test = np.hstack([linear_scores, nb_probs, rf_probs])

    final_pred_enc = stack_model.predict(stack_test)
    final_pred = label_encoder.inverse_transform(final_pred_enc)

    test_acc = accuracy_score(y_test, final_pred)
    test_f1 = f1_score(y_test, final_pred, average='macro')

    print("\n=== Stacking Ensemble Performance ===")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test macro F1: {test_f1:.3f}")


if __name__ == "__main__":
    main()

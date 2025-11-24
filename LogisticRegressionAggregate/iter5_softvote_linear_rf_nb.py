"""
Soft voting ensemble:
- Calibrated LinearSVC (word+char TF-IDF)
- RandomForest (iter5 best params)
- MultinomialNB (word TF-IDF)

Weights: LinearSVC 0.7, RF 0.2, NB 0.1
"""
import numpy as np
import pandas as pd
import re
import os
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

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
default_weights = {'linear': 0.7, 'rf': 0.2, 'nb': 0.1}
env_weights = os.environ.get("SOFT_VOTE_WEIGHTS")
if env_weights:
    vals = [float(x) for x in env_weights.split(",")]
    if len(vals) == 3 and np.isclose(sum(vals), 1.0):
        default_weights = {'linear': vals[0], 'rf': vals[1], 'nb': vals[2]}
VOTE_WEIGHTS = default_weights


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

    X_train_final = hstack([X_struct_train, X_word_char_train], format='csr')
    X_test_final = hstack([X_struct_test, X_word_char_test], format='csr')

    return {
        'X_word_char_train': X_word_char_train,
        'X_word_char_test': X_word_char_test,
        'X_word_train': X_word_train,
        'X_word_test': X_word_test,
        'X_struct_train': X_struct_train,
        'X_struct_test': X_struct_test,
        'X_train_final': X_train_final,
        'X_test_final': X_test_final,
    }


def main():
    df = pd.read_csv(file_name)
    df_train, df_test = student_wise_split(df, train_ratio=0.8, seed=0)
    features = build_features(df_train, df_test)
    y_train = df_train['label'].values
    y_test = df_test['label'].values
    classes = np.unique(y_train)

    linear = LinearSVC(C=LINEAR_C, max_iter=6000)
    calib_linear = CalibratedClassifierCV(linear, method='sigmoid', cv=4)
    calib_linear.fit(features['X_train_final'], y_train)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(features['X_train_final'].toarray(), y_train)

    nb = MultinomialNB()
    nb.fit(features['X_word_train'], y_train)

    linear_probs = calib_linear.predict_proba(features['X_test_final'])
    rf_probs = rf.predict_proba(features['X_test_final'].toarray())
    nb_probs = nb.predict_proba(features['X_word_test'])

    combined_probs = (
        VOTE_WEIGHTS['linear'] * linear_probs +
        VOTE_WEIGHTS['rf'] * rf_probs +
        VOTE_WEIGHTS['nb'] * nb_probs
    )
    final_pred = classes[np.argmax(combined_probs, axis=1)]

    test_acc = accuracy_score(y_test, final_pred)
    test_f1 = f1_score(y_test, final_pred, average='macro')

    print("\n=== Soft Voting Ensemble ===")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test macro F1: {test_f1:.3f}")


if __name__ == "__main__":
    main()

"""
Train a Random Forest model using the same feature engineering as logreg/nb:
    - 4 rating (Likert) features
    - 2 multi-select questions, each encoded using 8 target tasks -> 16 binary features total
    - 3 free-text questions combined into a bag-of-words with up to 3000 features

Model:
    - Random Forest with student-wise split to avoid data leakage

This is a baseline to compare with logistic regression (test acc: 0.683) and naive bayes.
"""
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

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

MAX_TEXT_FEATURES = 3000

def process_multiselect(series, target_tasks):
    """
    Convert multiselect strings to lists, keeping only specified features.
    If a response is missing, we treat it as selecting nothing.
    """
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
    """
    Extract numeric rating from responses like '3 - Sometimes'.
    Returns None for missing or unparsable responses.
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None

def build_rating_matrix(df, rating_cols, neutral_value=3):
    """
    Build a numeric feature matrix from multiple rating columns.
    """
    columns = []
    for col in rating_cols:
        raw = df[col].apply(extract_rating)
        filled = raw.fillna(neutral_value).astype(int)
        columns.append(filled.to_numpy().reshape(-1, 1))
    X_ratings = np.hstack(columns)
    return X_ratings

def build_text_series(df, text_cols):
    """
    Combine several text columns into a single string per row.
    """
    combined = df[text_cols[0]].fillna('')
    for col in text_cols[1:]:
        combined = combined + " " + df[col].fillna('')
    return combined

def main():
    df = pd.read_csv(file_name)
    n_rows = df.shape[0]
    print(f"Loaded {n_rows} rows from {file_name}")

    student_ids = df['student_id'].values
    unique_ids = np.unique(student_ids)
    np.random.seed(0)
    np.random.shuffle(unique_ids)
    
    n_train_students = int(0.8 * len(unique_ids))
    train_ids = set(unique_ids[:n_train_students])
    test_ids = set(unique_ids[n_train_students:])
    
    is_train = np.array([sid in train_ids for sid in student_ids])
    is_test = np.array([sid in test_ids for sid in student_ids])
    
    df_train = df[is_train].copy()
    df_test = df[is_test].copy()
    
    print(f"Train students: {len(train_ids)}, Test students: {len(test_ids)}")
    print(f"Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")

    X_ratings_train = build_rating_matrix(df_train, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)
    
    best_tasks_lists_train = process_multiselect(df_train[BEST_TASKS_COL], target_tasks)
    subopt_tasks_lists_train = process_multiselect(df_train[SUBOPT_TASKS_COL], target_tasks)
    
    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()
    
    best_tasks_encoded_train = mlb_best.fit_transform(best_tasks_lists_train)
    suboptimal_tasks_encoded_train = mlb_subopt.fit_transform(subopt_tasks_lists_train)
    
    X_structured_train = np.hstack([
        X_ratings_train,
        best_tasks_encoded_train,
        suboptimal_tasks_encoded_train,
    ])
    
    all_text_train = build_text_series(df_train, TEXT_COLS)
    
    vectorizer = CountVectorizer(
        max_features=MAX_TEXT_FEATURES,
        lowercase=True,
    )
    
    X_text_train_sparse = vectorizer.fit_transform(all_text_train)
    X_text_train = X_text_train_sparse.toarray()
    
    X_train = np.hstack([X_structured_train, X_text_train])
    y_train = df_train['label'].values
    
    print("Train feature matrix shape:", X_train.shape)
    
    X_ratings_test = build_rating_matrix(df_test, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)
    
    best_tasks_lists_test = process_multiselect(df_test[BEST_TASKS_COL], target_tasks)
    subopt_tasks_lists_test = process_multiselect(df_test[SUBOPT_TASKS_COL], target_tasks)
    
    best_tasks_encoded_test = mlb_best.transform(best_tasks_lists_test)
    suboptimal_tasks_encoded_test = mlb_subopt.transform(subopt_tasks_lists_test)
    
    X_structured_test = np.hstack([
        X_ratings_test,
        best_tasks_encoded_test,
        suboptimal_tasks_encoded_test,
    ])
    
    all_text_test = build_text_series(df_test, TEXT_COLS)
    X_text_test_sparse = vectorizer.transform(all_text_test)
    X_text_test = X_text_test_sparse.toarray()
    
    X_test = np.hstack([X_structured_test, X_text_test])
    y_test = df_test['label'].values
    
    print("Test feature matrix shape:", X_test.shape)
    print("Distinct labels:", np.unique(y_train))

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features='sqrt',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=0,
        n_jobs=-1,
    )
    
    rf.fit(X_train, y_train)
    
    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    
    print(f"\nTraining accuracy (RF baseline): {train_acc:.3f}")
    print(f"Test accuracy (RF baseline): {test_acc:.3f}")
    
    print(f"\nComparison:")
    print(f"  Logistic Regression test acc: 0.683")
    print(f"  Random Forest test acc: {test_acc:.3f}")
    print(f"  Difference: {test_acc - 0.683:+.3f}")

if __name__ == "__main__":
    main()


import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

file_name = "../DataSet/training_data_clean.csv"

# All rating / Likert-style questions we want to use (4 columns)
RATING_COLS = [
    "How likely are you to use this model for academic tasks?",
    "Based on your experience, how often has this model given you a response that felt suboptimal?",
    "How often do you expect this model to provide responses with references or supporting evidence?",
    "How often do you verify this model's responses?",
]

# Multi-select columns
BEST_TASKS_COL = 'Which types of tasks do you feel this model handles best? (Select all that apply.)'
SUBOPT_TASKS_COL = 'For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'

# Target tasks we will track for both multi-select questions
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

# Text columns to combine into one bag-of-words text field
TEXT_COLS = [
    "In your own words, what kinds of tasks would you use this model for?",
    "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
    "When you verify a response from this model, how do you usually go about it?",
]

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

    For each column:
        - Apply extract_rating() to get numbers.
        - Replace missing values (None/NaN) with `neutral_value`.
        - Cast to int and reshape to a column vector.

    X_ratings : np.ndarray of shape (n_samples, len(rating_cols))
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

    For each row:
        combined_text = col1 + " " + col2 + " " + col3
    
    Missing values are treated as empty strings.
    """
    combined = df[text_cols[0]].fillna('')
    for col in text_cols[1:]:
        combined = combined + " " + df[col].fillna('')
    return combined

def main():
    df = pd.read_csv(file_name)

    X_ratings = build_rating_matrix(df, RATING_COLS, neutral_value=3)

    best_tasks_lists = process_multiselect(df[BEST_TASKS_COL], target_tasks)
    suboptimal_tasks_lists = process_multiselect(df[SUBOPT_TASKS_COL], target_tasks)

    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()

    best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists) # shape (n_samples, 8)
    suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists) # shape (n_samples, 8)

    # 4 rating columns + 8 best task columns + 8 suboptimal task columns = 20 structured features
    X_structured = np.hstack([
        X_ratings,
        best_tasks_encoded,
        suboptimal_tasks_encoded,
    ])

    y = df['label'].values
    student_ids = df['student_id'].values

    # split by student_id to avoid data leakage (same student might appear in both train and test)
    unique_ids = np.unique(student_ids)
    rng = np.random.default_rng(seed=0) # seed -> reproducable
    rng.shuffle(unique_ids)

    n_train_ids = int(0.7 * len(unique_ids))
    train_ids = set(unique_ids[:n_train_ids])
    test_ids = set(unique_ids[n_train_ids:])

    is_train = np.array([sid in train_ids for sid in student_ids])
    is_test = np.array([sid in test_ids for sid in student_ids])

    X_struct_train = X_structured[is_train]
    X_struct_test = X_structured[is_test]
    y_train = y[is_train]
    y_test = y[is_test]

    train_text = build_text_series(df[is_train], TEXT_COLS)
    test_text = build_text_series(df[is_test], TEXT_COLS)

    vectorizer = CountVectorizer(
        max_features=2000, # avoid overfitting
        lowercase=True,
    )

    # Only fit on train_text to avoid data leakage
    X_text_train = vectorizer.fit_transform(train_text) # shape (n_train, vocab_size)
    X_text_test = vectorizer.transform(test_text) # shape (n_test, vocab_size)

    # Convert sparse matrices to dense NumPy arrays
    X_text_train_dense = X_text_train.toarray()
    X_text_test_dense = X_text_test.toarray()

    # Final feature matrices: structured + text
    X_train = np.hstack([X_struct_train, X_text_train_dense])
    X_test = np.hstack([X_struct_test, X_text_test_dense])

    print("Structured feature shape (train):", X_struct_train.shape)
    print("Text feature shape (train):      ", X_text_train_dense.shape)
    print("Final X_train shape:             ", X_train.shape)
    print("Final X_test shape:              ", X_test.shape)

    print(f"Number of unique students: {len(unique_ids)}")
    print(f"Train students: {len(train_ids)}, Test students: {len(test_ids)}")
    print(f"Train rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")

    logreg = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        random_state=0
    )
    
    logreg.fit(X_train, y_train)

    train_acc = logreg.score(X_train, y_train)
    test_acc = logreg.score(X_test, y_test)

    print(f"Training accuracy (full features): {train_acc:.3f}")
    print(f"Test accuracy (full features):     {test_acc:.3f}")

if __name__ == "__main__":
    main()
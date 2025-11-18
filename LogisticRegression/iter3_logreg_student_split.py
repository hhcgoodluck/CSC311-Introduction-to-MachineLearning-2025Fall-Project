import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

file_name = "../DataSet/training_data_clean.csv"

def process_multiselect(series, target_tasks):
    """
    Convert multiselect strings to lists, keeping only specified features
    """
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            # Check which of the target tasks are present in the response
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    """
    Extract numeric rating from responses like '3 - Sometimes'.
    Returns None for missing responses
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None

def main():
    df = pd.read_csv(file_name)
    df.dropna(inplace=True)

    target_tasks = [
        'Math computations',
        'Writing or debugging code',
        'Data processing or analysis',
        'Explaining complex concepts simply',
    ]

    best_tasks_lists = process_multiselect(
        df['Which types of tasks do you feel this model handles best? (Select all that apply.)'],
        target_tasks
    )

    suboptimal_tasks_lists = process_multiselect(
        df['For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'],
        target_tasks
    )

    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()

    best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
    suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)

    academic_numeric = df['How likely are you to use this model for academic tasks?'].apply(extract_rating)
    subopt_numeric = df['Based on your experience, how often has this model given you a response that felt suboptimal?'].apply(extract_rating)

    X = np.hstack([
        academic_numeric.values.reshape(-1, 1),
        subopt_numeric.values.reshape(-1, 1),
        best_tasks_encoded,
        suboptimal_tasks_encoded
    ])

    y = df['label'].values
    
    # split by student_id to avoid data leakage (same student might appear in both train and test)
    student_ids = df['student_id'].values
    unique_ids = np.unique(student_ids)

    # Shuffle the unique student IDs to randomize which students go to train vs test
    rng = np.random.default_rng(seed=0) # seed for reproducibility
    rng.shuffle(unique_ids)

    # Use 70% of students for training, 30% for testing
    n_train_ids = int(0.7 * len(unique_ids))
    train_ids = set(unique_ids[:n_train_ids])
    test_ids = set(unique_ids[n_train_ids:])

    is_train = np.array([sid in train_ids for sid in student_ids])
    is_test = np.array([sid in test_ids for sid in student_ids])

    assert np.all(~(is_train & is_test)), "A row is both train and test, which should not happen."
    assert np.all(is_train | is_test), "Some rows are neither in train nor test."

    X_train, y_train = X[is_train], y[is_train]
    X_test, y_test = X[is_test], y[is_test]

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

    print(f"Training accuracy (LogReg, student-wise split): {train_acc:.3f}")
    print(f"Test accuracy (LogReg, student-wise split): {test_acc:.3f}")

if __name__ == "__main__":
    main()
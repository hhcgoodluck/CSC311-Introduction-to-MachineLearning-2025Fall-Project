"""
Neural network baseline using TF-IDF bigram text features + structured data.

Pipeline:
1. Build structured features (ratings + multi-select) identical to other models.
2. Build TF-IDF (ngram 1-2, max 5000, min_df=2) on combined text columns.
3. Optionally reduce text dimensionality using TruncatedSVD (default: 400 comps).
4. Standardize final feature matrix and train an MLPClassifier.
5. Evaluate hyperparameters with student-wise CV, then train/test split.
"""
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
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
    is_test = ~is_train

    return df[is_train].copy(), df[is_test].copy(), train_ids


def build_features(df_train, df_test, svd_components=400):
    # Structured features
    X_ratings_train = build_rating_matrix(df_train, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)
    X_ratings_test = build_rating_matrix(df_test, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)

    best_tasks_lists_train = process_multiselect(df_train[BEST_TASKS_COL], target_tasks)
    subopt_tasks_lists_train = process_multiselect(df_train[SUBOPT_TASKS_COL], target_tasks)

    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()

    best_tasks_encoded_train = mlb_best.fit_transform(best_tasks_lists_train)
    suboptimal_tasks_encoded_train = mlb_subopt.fit_transform(subopt_tasks_lists_train)

    best_tasks_encoded_test = mlb_best.transform(process_multiselect(df_test[BEST_TASKS_COL], target_tasks))
    suboptimal_tasks_encoded_test = mlb_subopt.transform(process_multiselect(df_test[SUBOPT_TASKS_COL], target_tasks))

    X_struct_train = np.hstack([X_ratings_train, best_tasks_encoded_train, suboptimal_tasks_encoded_train])
    X_struct_test = np.hstack([X_ratings_test, best_tasks_encoded_test, suboptimal_tasks_encoded_test])

    # Text features
    train_text = build_text_series(df_train, TEXT_COLS)
    test_text = build_text_series(df_test, TEXT_COLS)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_text_train_sparse = vectorizer.fit_transform(train_text)
    X_text_test_sparse = vectorizer.transform(test_text)

    if svd_components:
        svd = TruncatedSVD(n_components=svd_components, random_state=0)
        X_text_train = svd.fit_transform(X_text_train_sparse)
        X_text_test = svd.transform(X_text_test_sparse)
    else:
        # Convert to dense; may be large but manageable (~5000 dims)
        X_text_train = X_text_train_sparse.toarray()
        X_text_test = X_text_test_sparse.toarray()
        svd = None

    X_train = np.hstack([X_struct_train, X_text_train])
    X_test = np.hstack([X_struct_test, X_text_test])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, vectorizer, svd, scaler


def evaluate_params(X, y, params, cv_splits=4):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    cv_scores = []
    cv_f1 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        mlp = MLPClassifier(
            hidden_layer_sizes=params['hidden_layer_sizes'],
            activation='relu',
            solver='adam',
            alpha=params['alpha'],
            learning_rate_init=params['learning_rate_init'],
            batch_size=64,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=params['seed_offset'] + fold,
            verbose=False,
        )
        mlp.fit(X_train_fold, y_train_fold)

        val_pred = mlp.predict(X_val_fold)
        acc = accuracy_score(y_val_fold, val_pred)
        f1 = f1_score(y_val_fold, val_pred, average='macro')
        cv_scores.append(acc)
        cv_f1.append(f1)
        print(f"    Fold {fold}: acc={acc:.3f}, f1={f1:.3f}")

    return float(np.mean(cv_scores)), float(np.std(cv_scores)), float(np.mean(cv_f1))


def main():
    df = pd.read_csv(file_name)
    df_train, df_test, train_ids = student_wise_split(df, train_ratio=0.8, seed=0)

    print(f"Total rows: {df.shape[0]}")
    print(f"Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")
    print(f"Unique students: {df['student_id'].nunique()}, train students: {len(train_ids)}")

    X_train, X_test, vectorizer, svd, scaler = build_features(
        df_train, df_test, svd_components=400
    )
    print(f"Feature shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['label'].values)
    y_test = label_encoder.transform(df_test['label'].values)

    param_grid = [
        {'hidden_layer_sizes': (512,), 'alpha': 1e-4, 'learning_rate_init': 1e-3, 'seed_offset': 0},
        {'hidden_layer_sizes': (512, 256), 'alpha': 3e-4, 'learning_rate_init': 1e-3, 'seed_offset': 5},
        {'hidden_layer_sizes': (512, 256, 128), 'alpha': 3e-4, 'learning_rate_init': 5e-4, 'seed_offset': 10},
        {'hidden_layer_sizes': (256, 128), 'alpha': 1e-4, 'learning_rate_init': 5e-4, 'seed_offset': 15},
        {'hidden_layer_sizes': (512, 256), 'alpha': 1e-3, 'learning_rate_init': 1e-3, 'seed_offset': 20},
    ]

    best_score = -1
    best_params = None
    results = []

    print("\n=== Hyperparameter Evaluation (Stratified 4-fold CV) ===")
    for idx, params in enumerate(param_grid, 1):
        print(f"\nConfig {idx}: {params}")
        mean_acc, std_acc, mean_f1 = evaluate_params(X_train, y_train, params)
        print(f"  -> CV acc: {mean_acc:.3f} ± {std_acc:.3f}, CV macro F1: {mean_f1:.3f}")
        results.append({'params': params, 'mean_acc': mean_acc, 'std_acc': std_acc, 'mean_f1': mean_f1})
        if mean_acc > best_score:
            best_score = mean_acc
            best_params = params

    print("\nBest CV configuration:", best_params)
    print(f"Best CV accuracy: {best_score:.3f}")

    # Train final model with best params
    final_mlp = MLPClassifier(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        activation='relu',
        solver='adam',
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        batch_size=64,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=0,
        verbose=False,
    )

    final_mlp.fit(X_train, y_train)
    train_pred = final_mlp.predict(X_train)
    test_pred = final_mlp.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')

    print("\n=== Final Model Performance ===")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Training macro F1: {train_f1:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test macro F1: {test_f1:.3f}")
    print(f"Gap to Logistic Regression baseline (0.683): {test_acc - 0.683:+.3f}")
    print(f"Gap to RF ensemble (0.691): {test_acc - 0.691:+.3f}")


if __name__ == "__main__":
    main()

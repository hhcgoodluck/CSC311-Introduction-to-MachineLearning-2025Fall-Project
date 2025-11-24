"""
Hyperparameter tuning for Random Forest based on proposal ranges:
    - n_estimators: {200, 400, 800}
    - max_depth: {None, 12, 20, 30}
    - max_features: {"sqrt", "log2", 0.2}
    - min_samples_split: {2, 5, 10}
    - min_samples_leaf: {1, 2, 4}

Uses student-wise split and 5-fold cross-validation for robust evaluation.
"""
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
import itertools

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
    """Convert multiselect strings to lists, keeping only specified features."""
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
    """Extract numeric rating from responses like '3 - Sometimes'."""
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None

def build_rating_matrix(df, rating_cols, neutral_value=3):
    """Build a numeric feature matrix from multiple rating columns."""
    columns = []
    for col in rating_cols:
        raw = df[col].apply(extract_rating)
        filled = raw.fillna(neutral_value).astype(int)
        columns.append(filled.to_numpy().reshape(-1, 1))
    X_ratings = np.hstack(columns)
    return X_ratings

def build_text_series(df, text_cols):
    """Combine several text columns into a single string per row."""
    combined = df[text_cols[0]].fillna('')
    for col in text_cols[1:]:
        combined = combined + " " + df[col].fillna('')
    return combined

def build_features(df_train, df_test=None):
    """
    Build features for training and optionally test set.
    If df_test is None, only returns training features.
    """
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
    
    if df_test is None:
        return X_train, y_train, mlb_best, mlb_subopt, vectorizer
    
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
    
    return X_train, y_train, X_test, y_test, mlb_best, mlb_subopt, vectorizer

def main():
    df = pd.read_csv(file_name)
    print(f"Loaded {df.shape[0]} rows from {file_name}")

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
    print(f"Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}\n")

    X_train, y_train, X_test, y_test, _, _, _ = build_features(df_train, df_test)
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Distinct labels: {np.unique(y_train)}\n")

    param_grid = {
        'n_estimators': [200, 400, 800],
        'max_depth': [None, 12, 20, 30],
        'max_features': ['sqrt', 'log2', 0.2],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    best_score = -1
    best_params = None
    best_cv_scores = None
    

    print("Starting hyperparameter tuning with 5-fold CV...")
    print("=" * 80)
    
    n_estimators_list = [200, 400, 800]
    max_depth_list = [None, 12, 20]
    max_features_list = ['sqrt', 'log2']
    min_samples_split_list = [2, 5]
    min_samples_leaf_list = [1, 2]
    
    total_combinations = len(n_estimators_list) * len(max_depth_list) * len(max_features_list) * len(min_samples_split_list) * len(min_samples_leaf_list)
    print(f"Total combinations to try: {total_combinations}")
    print()
    
    results = []
    
    for n_est in n_estimators_list:
        for max_d in max_depth_list:
            for max_f in max_features_list:
                for min_split in min_samples_split_list:
                    for min_leaf in min_samples_leaf_list:
                        params = {
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'max_features': max_f,
                            'min_samples_split': min_split,
                            'min_samples_leaf': min_leaf,
                            'random_state': 0,
                            'n_jobs': -1,
                        }
                        
                        cv_scores = []
                        for train_idx, val_idx in skf.split(X_train, y_train):
                            X_train_fold = X_train[train_idx]
                            y_train_fold = y_train[train_idx]
                            X_val_fold = X_train[val_idx]
                            y_val_fold = y_train[val_idx]
                            
                            rf = RandomForestClassifier(**params)
                            rf.fit(X_train_fold, y_train_fold)
                            val_acc = rf.score(X_val_fold, y_val_fold)
                            cv_scores.append(val_acc)
                        
                        mean_cv = np.mean(cv_scores)
                        std_cv = np.std(cv_scores)
                        
                        results.append({
                            'params': params,
                            'mean_cv': mean_cv,
                            'std_cv': std_cv,
                            'cv_scores': cv_scores,
                        })
                        
                        if mean_cv > best_score:
                            best_score = mean_cv
                            best_params = params.copy()
                            best_cv_scores = cv_scores
                        
                        print(f"n_est={n_est:3d}, max_d={str(max_d):>4s}, max_f={str(max_f):>4s}, "
                              f"min_split={min_split:2d}, min_leaf={min_leaf:2d} | "
                              f"CV: {mean_cv:.3f} ± {std_cv:.3f}")
    
    print("\n" + "=" * 80)
    print("Best hyperparameters:")
    print(f"  n_estimators: {best_params['n_estimators']}")
    print(f"  max_depth: {best_params['max_depth']}")
    print(f"  max_features: {best_params['max_features']}")
    print(f"  min_samples_split: {best_params['min_samples_split']}")
    print(f"  min_samples_leaf: {best_params['min_samples_leaf']}")
    print(f"\nBest CV score: {best_score:.3f} ± {np.std(best_cv_scores):.3f}")
    
    print("\nTraining final model on full training set...")
    rf_final = RandomForestClassifier(**best_params)
    rf_final.fit(X_train, y_train)
    
    train_acc = rf_final.score(X_train, y_train)
    test_acc = rf_final.score(X_test, y_test)
    
    print(f"\nFinal model performance:")
    print(f"  Training accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(f"\nComparison:")
    print(f"  Logistic Regression test acc: 0.683")
    print(f"  Random Forest test acc: {test_acc:.3f}")
    print(f"  Difference: {test_acc - 0.683:+.3f}")
    
    results_sorted = sorted(results, key=lambda x: x['mean_cv'], reverse=True)
    print("\n" + "=" * 80)
    print("Top 5 configurations:")
    for i, result in enumerate(results_sorted[:5], 1):
        p = result['params']
        print(f"{i}. CV: {result['mean_cv']:.3f} ± {result['std_cv']:.3f} | "
              f"n_est={p['n_estimators']}, max_d={p['max_depth']}, "
              f"max_f={p['max_features']}, min_split={p['min_samples_split']}, "
              f"min_leaf={p['min_samples_leaf']}")

if __name__ == "__main__":
    main()


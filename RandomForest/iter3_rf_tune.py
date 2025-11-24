"""
Extensive hyperparameter tuning for Random Forest to improve accuracy.

Improvements:
1. Full hyperparameter grid search (all combinations from proposal)
2. Try TF-IDF instead of CountVectorizer
3. Try different text feature counts
4. More granular hyperparameter ranges
5. Try different criteria (Gini vs entropy)
6. Try class_weight options
"""
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

def build_features(df_train, df_test=None, text_method='count', max_text_features=3000):
    """
    Build features for training and optionally test set.
    
    Parameters:
        text_method: 'count' for CountVectorizer, 'tfidf' for TfidfVectorizer
        max_text_features: maximum number of text features
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
    
    if text_method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_text_features,
            lowercase=True,
        )
    else:
        vectorizer = CountVectorizer(
            max_features=max_text_features,
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

def evaluate_params(X_train, y_train, params, cv_folds=5):
    """Evaluate hyperparameters using cross-validation."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
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
    
    return np.mean(cv_scores), np.std(cv_scores), cv_scores

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

    feature_configs = [
        {'text_method': 'count', 'max_text_features': 3000},
        {'text_method': 'tfidf', 'max_text_features': 3000},
        {'text_method': 'count', 'max_text_features': 5000},
        {'text_method': 'tfidf', 'max_text_features': 5000},
    ]
    
    param_grids = [
        {
            'n_estimators': [200, 400, 800],
            'max_depth': [None, 12, 20, 30],
            'max_features': ['sqrt', 'log2', 0.2],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini'],
            'bootstrap': [True],
        },
        {
            'n_estimators': [300, 400, 500, 600],
            'max_depth': [10, 12, 15, 18, 20],
            'max_features': ['sqrt', 0.3, 0.4],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2],
            'criterion': ['gini', 'entropy'],
            'bootstrap': [True],
        },
        {
            'n_estimators': [500, 800, 1000],
            'max_depth': [15, 20, 25, None],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4],
            'criterion': ['gini'],
            'bootstrap': [True],
        },
    ]
    
    best_overall_score = -1
    best_overall_config = None
    best_overall_params = None
    all_results = []
    
    for feat_config in feature_configs:
        print("=" * 80)
        print(f"Feature Config: {feat_config}")
        print("=" * 80)
        
        X_train, y_train, X_test, y_test, _, _, _ = build_features(
            df_train, df_test, 
            text_method=feat_config['text_method'],
            max_text_features=feat_config['max_text_features']
        )
        
        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Text method: {feat_config['text_method']}, Max features: {feat_config['max_text_features']}\n")
        
        for grid_idx, param_grid in enumerate(param_grids):
            print(f"\n--- Parameter Grid {grid_idx + 1} ---")
            
            keys = param_grid.keys()
            values = param_grid.values()
            combinations = list(itertools.product(*values))
            
            print(f"Testing {len(combinations)} combinations...")
            
            for combo in combinations:
                params_dict = dict(zip(keys, combo))
                params_dict['random_state'] = 0
                params_dict['n_jobs'] = -1
                
                try:
                    mean_cv, std_cv, cv_scores = evaluate_params(X_train, y_train, params_dict)
                    
                    result = {
                        'feature_config': feat_config,
                        'params': params_dict,
                        'mean_cv': mean_cv,
                        'std_cv': std_cv,
                        'cv_scores': cv_scores,
                    }
                    all_results.append(result)
                    
                    if mean_cv > best_overall_score:
                        best_overall_score = mean_cv
                        best_overall_config = feat_config
                        best_overall_params = params_dict.copy()
                    
                    if mean_cv > 0.66:
                        print(f"  CV: {mean_cv:.3f} ± {std_cv:.3f} | "
                              f"n_est={params_dict['n_estimators']}, "
                              f"max_d={params_dict['max_depth']}, "
                              f"max_f={params_dict['max_features']}, "
                              f"min_split={params_dict['min_samples_split']}, "
                              f"min_leaf={params_dict['min_samples_leaf']}, "
                              f"criterion={params_dict['criterion']}")
                except Exception as e:
                    print(f"  Error with params {params_dict}: {e}")
                    continue
    
    print("\n" + "=" * 80)
    print("BEST OVERALL CONFIGURATION")
    print("=" * 80)
    print(f"Feature Config: {best_overall_config}")
    print(f"Best Parameters:")
    for key, value in best_overall_params.items():
        if key not in ['random_state', 'n_jobs']:
            print(f"  {key}: {value}")
    print(f"\nBest CV Score: {best_overall_score:.3f}")
    
    print("\nTraining final model with best configuration...")
    X_train_final, y_train_final, X_test_final, y_test_final, _, _, _ = build_features(
        df_train, df_test,
        text_method=best_overall_config['text_method'],
        max_text_features=best_overall_config['max_text_features']
    )
    
    rf_final = RandomForestClassifier(**best_overall_params)
    rf_final.fit(X_train_final, y_train_final)
    
    train_acc = rf_final.score(X_train_final, y_train_final)
    test_acc = rf_final.score(X_test_final, y_test_final)
    
    print(f"\nFinal Model Performance:")
    print(f"  Training accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(f"\nComparison:")
    print(f"  Logistic Regression test acc: 0.683")
    print(f"  Random Forest test acc: {test_acc:.3f}")
    print(f"  Difference: {test_acc - 0.683:+.3f}")
    
    all_results_sorted = sorted(all_results, key=lambda x: x['mean_cv'], reverse=True)
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS:")
    print("=" * 80)
    for i, result in enumerate(all_results_sorted[:10], 1):
        p = result['params']
        fc = result['feature_config']
        print(f"{i}. CV: {result['mean_cv']:.3f} ± {result['std_cv']:.3f}")
        print(f"   Features: {fc['text_method']}, max={fc['max_text_features']}")
        print(f"   Params: n_est={p['n_estimators']}, max_d={p['max_depth']}, "
              f"max_f={p['max_features']}, min_split={p['min_samples_split']}, "
              f"min_leaf={p['min_samples_leaf']}, crit={p['criterion']}")
        print()

if __name__ == "__main__":
    main()


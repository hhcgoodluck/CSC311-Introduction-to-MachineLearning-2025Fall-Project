"""
Strategies:
1. Larger hyperparameter search space
2. Multiple feature engineering approaches
3. Different text feature counts
4. Feature selection
5. More trees and deeper search
"""
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
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
    X_ratings = np.hstack(columns)
    return X_ratings

def build_text_series(df, text_cols):
    combined = df[text_cols[0]].fillna('')
    for col in text_cols[1:]:
        combined = combined + " " + df[col].fillna('')
    return combined

def build_features(df_train, df_test=None, text_method='count', max_text_features=3000, use_feature_selection=False, k_best=None):
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
            ngram_range=(1, 2),  # Try bigrams too
        )
    else:
        vectorizer = CountVectorizer(
            max_features=max_text_features,
            lowercase=True,
            ngram_range=(1, 2),  # Try bigrams too
        )
    
    X_text_train_sparse = vectorizer.fit_transform(all_text_train)
    X_text_train = X_text_train_sparse.toarray()
    
    X_train = np.hstack([X_structured_train, X_text_train])
    y_train = df_train['label'].values
    
    selector = None
    if use_feature_selection and k_best:
        selector = SelectKBest(f_classif, k=k_best)
        X_train = selector.fit_transform(X_train, y_train)
    
    if df_test is None:
        return X_train, y_train, mlb_best, mlb_subopt, vectorizer, selector
    
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
    
    if selector:
        X_test = selector.transform(X_test)
    
    return X_train, y_train, X_test, y_test, mlb_best, mlb_subopt, vectorizer, selector

def evaluate_params(X_train, y_train, params, cv_folds=5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    cv_scores = []
    cv_f1_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        
        rf = RandomForestClassifier(**params)
        rf.fit(X_train_fold, y_train_fold)
        y_pred = rf.predict(X_val_fold)
        
        val_acc = accuracy_score(y_val_fold, y_pred)
        val_f1 = f1_score(y_val_fold, y_pred, average='macro')
        
        cv_scores.append(val_acc)
        cv_f1_scores.append(val_f1)
    
    return np.mean(cv_scores), np.std(cv_scores), np.mean(cv_f1_scores), cv_scores

def generate_promising_params(n_samples=100):
    """Generate parameter combinations focusing on promising regions."""
    combinations = []
    random.seed(0)
    np.random.seed(0)
    
    for _ in range(n_samples):
        n_est = random.choice([400, 500, 600, 700, 800, 1000, 1200])
        
        max_d = random.choice([None, 10, 12, 15, 18, 20, 25, 30])
        
        max_f = random.choice(['sqrt', 'log2', 0.2, 0.25, 0.3, 0.35, 0.4])
        
        min_split = random.choice([2, 3, 5, 7, 10])
        min_leaf = random.choice([1, 2, 3, 4, 5])
        
        criterion = random.choice(['gini', 'entropy'])
        
        params = {
            'n_estimators': n_est,
            'max_depth': max_d,
            'max_features': max_f,
            'min_samples_split': min_split,
            'min_samples_leaf': min_leaf,
            'criterion': criterion,
            'random_state': 0,
            'n_jobs': -1,
        }
        combinations.append(params)
    
    return combinations

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
        {'text_method': 'count', 'max_text_features': 3000, 'use_feature_selection': False},
        {'text_method': 'tfidf', 'max_text_features': 3000, 'use_feature_selection': False},
        {'text_method': 'count', 'max_text_features': 5000, 'use_feature_selection': False},
        {'text_method': 'tfidf', 'max_text_features': 5000, 'use_feature_selection': False},
        {'text_method': 'count', 'max_text_features': 3000, 'use_feature_selection': True, 'k_best': 2000},
        {'text_method': 'tfidf', 'max_text_features': 5000, 'use_feature_selection': True, 'k_best': 3000},
    ]
    
    best_overall_score = -1
    best_overall_config = None
    best_overall_params = None
    all_results = []
    
    for feat_config in feature_configs:
        print("=" * 80)
        print(f"Feature Config: {feat_config}")
        print("=" * 80)
        
        use_fs = feat_config.get('use_feature_selection', False)
        k_best = feat_config.get('k_best', None)
        
        X_train, y_train, X_test, y_test, _, _, _, _ = build_features(
            df_train, df_test, 
            text_method=feat_config['text_method'],
            max_text_features=feat_config['max_text_features'],
            use_feature_selection=use_fs,
            k_best=k_best
        )
        
        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Text method: {feat_config['text_method']}, Max features: {feat_config['max_text_features']}")
        if use_fs:
            print(f"Feature selection: Top {k_best} features")
        print()
        
        param_combinations = generate_promising_params(n_samples=100)
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, params_dict in enumerate(param_combinations, 1):
            try:
                mean_cv, std_cv, mean_f1, cv_scores = evaluate_params(X_train, y_train, params_dict)
                
                result = {
                    'feature_config': feat_config,
                    'params': params_dict,
                    'mean_cv': mean_cv,
                    'std_cv': std_cv,
                    'mean_f1': mean_f1,
                }
                all_results.append(result)
                
                if mean_cv > best_overall_score:
                    best_overall_score = mean_cv
                    best_overall_config = feat_config
                    best_overall_params = params_dict.copy()
                
                if i % 20 == 0 or mean_cv > 0.70:
                    print(f"  [{i:3d}/100] CV: {mean_cv:.3f} ± {std_cv:.3f}, F1: {mean_f1:.3f} | "
                          f"n_est={params_dict['n_estimators']}, max_d={params_dict['max_depth']}, "
                          f"max_f={params_dict['max_features']}")
            except Exception as e:
                continue
        
        print()
    
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
    use_fs = best_overall_config.get('use_feature_selection', False)
    k_best = best_overall_config.get('k_best', None)
    
    X_train_final, y_train_final, X_test_final, y_test_final, _, _, _, _ = build_features(
        df_train, df_test,
        text_method=best_overall_config['text_method'],
        max_text_features=best_overall_config['max_text_features'],
        use_feature_selection=use_fs,
        k_best=k_best
    )
    
    rf_final = RandomForestClassifier(**best_overall_params)
    rf_final.fit(X_train_final, y_train_final)
    
    train_acc = rf_final.score(X_train_final, y_train_final)
    test_acc = rf_final.score(X_test_final, y_test_final)
    
    y_pred = rf_final.predict(X_test_final)
    test_f1 = f1_score(y_test_final, y_pred, average='macro')
    
    print(f"\nFinal Model Performance:")
    print(f"  Training accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(f"  Test F1 (macro): {test_f1:.3f}")
    print(f"\nComparison:")
    print(f"  Logistic Regression test acc: 0.683")
    print(f"  Previous RF best: 0.685")
    print(f"  Current RF test acc: {test_acc:.3f}")
    print(f"  Target: 0.85+")
    print(f"  Gap to target: {0.85 - test_acc:.3f}")
    
    all_results_sorted = sorted(all_results, key=lambda x: x['mean_cv'], reverse=True)
    print("\n" + "=" * 80)
    print("TOP 15 CONFIGURATIONS:")
    print("=" * 80)
    for i, result in enumerate(all_results_sorted[:15], 1):
        p = result['params']
        fc = result['feature_config']
        print(f"{i:2d}. CV: {result['mean_cv']:.3f} ± {result['std_cv']:.3f}, F1: {result['mean_f1']:.3f}")
        print(f"    Features: {fc['text_method']}, max={fc['max_text_features']}, "
              f"FS={'Yes' if fc.get('use_feature_selection') else 'No'}")
        print(f"    Params: n_est={p['n_estimators']}, max_d={p['max_depth']}, "
              f"max_f={p['max_features']}, min_split={p['min_samples_split']}, "
              f"min_leaf={p['min_samples_leaf']}, crit={p['criterion']}")
        print()

if __name__ == "__main__":
    main()


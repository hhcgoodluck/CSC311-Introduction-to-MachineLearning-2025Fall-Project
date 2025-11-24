"""
Random Forest ensemble experiments targeting higher test accuracy.

Approach:
- Reuse best-performing feature pipelines from earlier iterations.
- Train small ensembles (multiple seeds) for each hyperparameter set.
- Average predicted probabilities to form an ensemble prediction.
- Compare ensemble accuracy/F1 across configurations.
"""
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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


def build_features(
    df_train,
    df_test,
    text_method='count',
    max_text_features=3000,
    use_bigrams=False,
    vectorizer_kwargs=None,
):
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

    vectorizer_kwargs = vectorizer_kwargs or {}
    all_text_train = build_text_series(df_train, TEXT_COLS)

    ngram_range = (1, 2) if use_bigrams else (1, 1)
    base_vectorizer_params = {
        'max_features': max_text_features,
        'lowercase': True,
        'ngram_range': ngram_range,
    }
    base_vectorizer_params.update(vectorizer_kwargs)

    if text_method == 'tfidf':
        vectorizer = TfidfVectorizer(**base_vectorizer_params)
    else:
        vectorizer = CountVectorizer(**base_vectorizer_params)

    X_text_train = vectorizer.fit_transform(all_text_train).toarray()

    X_train = np.hstack([X_structured_train, X_text_train])
    y_train = df_train['label'].values

    # Apply same transformers to test split
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
    X_text_test = vectorizer.transform(all_text_test).toarray()

    X_test = np.hstack([X_structured_test, X_text_test])
    y_test = df_test['label'].values

    return X_train, y_train, X_test, y_test


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


def train_ensemble(X_train, y_train, params, seeds):
    models = []
    individual_metrics = []

    for seed in seeds:
        model_params = params.copy()
        model_params['random_state'] = seed
        model_params['n_jobs'] = -1

        rf = RandomForestClassifier(**model_params)
        rf.fit(X_train, y_train)

        train_acc = rf.score(X_train, y_train)
        individual_metrics.append({
            'seed': seed,
            'train_acc': train_acc,
        })
        models.append(rf)

    return models, individual_metrics


def ensemble_predictions(models, X_test):
    probas = None
    for model in models:
        current = model.predict_proba(X_test)
        if probas is None:
            probas = current
        else:
            probas += current
    probas /= len(models)

    classes = models[0].classes_
    predictions_idx = np.argmax(probas, axis=1)
    predictions = classes[predictions_idx]
    return predictions


def main():
    df = pd.read_csv(file_name)
    df_train, df_test, train_ids = student_wise_split(df, train_ratio=0.8, seed=0)

    print(f"Loaded {df.shape[0]} rows. Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")
    print(f"Unique students: {df['student_id'].nunique()}, train students: {len(train_ids)}")

    feature_configs = [
        {
            'name': 'count_3000_unigrams',
            'text_method': 'count',
            'max_text_features': 3000,
            'use_bigrams': False,
        },
        {
            'name': 'count_5000_bigrams',
            'text_method': 'count',
            'max_text_features': 5000,
            'use_bigrams': True,
        },
        {
            'name': 'tfidf_5000_bigrams_stopwords',
            'text_method': 'tfidf',
            'max_text_features': 5000,
            'use_bigrams': True,
            'vectorizer_kwargs': {'stop_words': 'english', 'min_df': 2},
        },
    ]

    param_configs = [
        {
            'name': 'iter5_best',
            'params': {
                'n_estimators': 600,
                'max_depth': 8,
                'max_features': 0.2,
                'min_samples_split': 2,
                'min_samples_leaf': 3,
                'criterion': 'gini',
                'class_weight': None,
            },
        },
        {
            'name': 'deeper_regularized',
            'params': {
                'n_estimators': 1000,
                'max_depth': 12,
                'max_features': 0.25,
                'min_samples_split': 7,
                'min_samples_leaf': 2,
                'criterion': 'gini',
                'class_weight': None,
            },
        },
        {
            'name': 'wider_forest_balanced',
            'params': {
                'n_estimators': 1200,
                'max_depth': 14,
                'max_features': 0.25,
                'min_samples_split': 10,
                'min_samples_leaf': 3,
                'criterion': 'gini',
                'class_weight': 'balanced',
            },
        },
        {
            'name': 'medium_depth_entropy',
            'params': {
                'n_estimators': 800,
                'max_depth': 10,
                'max_features': 0.25,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'criterion': 'entropy',
                'class_weight': None,
            },
        },
    ]

    ensemble_seeds = [0, 1, 2, 3, 4]
    results = []

    for feat_cfg in feature_configs:
        print("\n" + "=" * 80)
        print(f"Feature config: {feat_cfg['name']}")
        print("=" * 80)

        X_train, y_train, X_test, y_test = build_features(
            df_train,
            df_test,
            text_method=feat_cfg['text_method'],
            max_text_features=feat_cfg['max_text_features'],
            use_bigrams=feat_cfg.get('use_bigrams', False),
            vectorizer_kwargs=feat_cfg.get('vectorizer_kwargs'),
        )

        print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}")

        for param_cfg in param_configs:
            print(f"\n--- Params: {param_cfg['name']} ---")
            print(f"Hyperparameters: {param_cfg['params']}")

            models, individual_metrics = train_ensemble(X_train, y_train, param_cfg['params'], ensemble_seeds)

            ensemble_pred = ensemble_predictions(models, X_test)
            ensemble_acc = accuracy_score(y_test, ensemble_pred)
            ensemble_f1 = f1_score(y_test, ensemble_pred, average='macro')

            print("Individual model train accuracies:")
            for metric in individual_metrics:
                print(f"  Seed {metric['seed']}: train acc = {metric['train_acc']:.3f}")

            print(f"Ensemble Test Accuracy: {ensemble_acc:.3f}")
            print(f"Ensemble Test Macro F1: {ensemble_f1:.3f}")

            results.append({
                'feature_config': feat_cfg['name'],
                'param_config': param_cfg['name'],
                'test_acc': ensemble_acc,
                'test_f1': ensemble_f1,
            })

    results_sorted = sorted(results, key=lambda x: x['test_acc'], reverse=True)

    print("\n" + "=" * 80)
    print("RESULT SUMMARY (sorted by test accuracy)")
    print("=" * 80)
    for res in results_sorted:
        print(f"{res['feature_config']} + {res['param_config']}: "
              f"Test Acc = {res['test_acc']:.3f}, Test F1 = {res['test_f1']:.3f}")


if __name__ == "__main__":
    main()

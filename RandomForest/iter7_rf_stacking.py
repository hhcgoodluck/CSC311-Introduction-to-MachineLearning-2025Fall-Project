"""
Stacking experiment: combine the two strongest Random Forest pipelines by
training a logistic regression meta-classifier on cross-validated probability
outputs. Goal: improve accuracy beyond the standalone ensembles (≈0.691).
"""
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

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
ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]


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

    return df[is_train].copy(), df[is_test].copy()


def build_feature_matrices(
    df_train,
    df_test,
    text_method='count',
    max_text_features=3000,
    use_bigrams=False,
    vectorizer_kwargs=None,
):
    vectorizer_kwargs = vectorizer_kwargs or {}

    best_tasks_lists_train = process_multiselect(df_train[BEST_TASKS_COL], target_tasks)
    subopt_tasks_lists_train = process_multiselect(df_train[SUBOPT_TASKS_COL], target_tasks)

    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()

    best_tasks_encoded_train = mlb_best.fit_transform(best_tasks_lists_train)
    suboptimal_tasks_encoded_train = mlb_subopt.fit_transform(subopt_tasks_lists_train)

    X_ratings_train = build_rating_matrix(df_train, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)

    X_structured_train = np.hstack([
        X_ratings_train,
        best_tasks_encoded_train,
        suboptimal_tasks_encoded_train,
    ])

    all_text_train = build_text_series(df_train, TEXT_COLS)
    vectorizer_params = {
        'max_features': max_text_features,
        'lowercase': True,
        'ngram_range': (1, 2) if use_bigrams else (1, 1),
    }
    vectorizer_params.update(vectorizer_kwargs)

    if text_method == 'tfidf':
        vectorizer = TfidfVectorizer(**vectorizer_params)
    else:
        vectorizer = CountVectorizer(**vectorizer_params)

    X_text_train = vectorizer.fit_transform(all_text_train).toarray()

    X_train = np.hstack([X_structured_train, X_text_train])

    # Build test matrices
    best_tasks_lists_test = process_multiselect(df_test[BEST_TASKS_COL], target_tasks)
    subopt_tasks_lists_test = process_multiselect(df_test[SUBOPT_TASKS_COL], target_tasks)

    X_ratings_test = build_rating_matrix(df_test, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)
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

    return X_train, X_test


def generate_oof_probabilities(X, y, params, n_classes, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.zeros((X.shape[0], n_classes))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        model_params = params.copy()
        model_params['random_state'] = fold
        model_params['n_jobs'] = -1

        rf = RandomForestClassifier(**model_params)
        rf.fit(X[train_idx], y[train_idx])
        oof[val_idx] = rf.predict_proba(X[val_idx])

        fold_acc = rf.score(X[val_idx], y[val_idx])
        print(f"    Fold {fold} accuracy: {fold_acc:.3f}")

    return oof


def train_ensemble(X, y, params, seeds):
    models = []
    for seed in seeds:
        model_params = params.copy()
        model_params['random_state'] = seed
        model_params['n_jobs'] = -1
        rf = RandomForestClassifier(**model_params)
        rf.fit(X, y)
        models.append(rf)
    return models


def ensemble_probabilities(models, X, n_classes):
    proba = np.zeros((X.shape[0], n_classes))
    for model in models:
        proba += model.predict_proba(X)
    proba /= len(models)
    return proba


def main():
    df = pd.read_csv(file_name)
    df_train, df_test = student_wise_split(df, train_ratio=0.8, seed=0)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['label'].values)
    y_test = label_encoder.transform(df_test['label'].values)
    class_names = label_encoder.classes_
    n_classes = len(class_names)

    print(f"Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")
    print(f"Classes: {class_names}")

    feature_configs = {
        'count_bigram': {
            'text_method': 'count',
            'max_text_features': 5000,
            'use_bigrams': True,
            'vectorizer_kwargs': {},
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
        'tfidf_bigram_stopwords': {
            'text_method': 'tfidf',
            'max_text_features': 5000,
            'use_bigrams': True,
            'vectorizer_kwargs': {'stop_words': 'english', 'min_df': 2},
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
    }

    pipeline_data = {}
    for name, cfg in feature_configs.items():
        print("\n" + "=" * 80)
        print(f"Building features for pipeline: {name}")
        print("=" * 80)
        X_train, X_test = build_feature_matrices(
            df_train,
            df_test,
            text_method=cfg['text_method'],
            max_text_features=cfg['max_text_features'],
            use_bigrams=cfg.get('use_bigrams', False),
            vectorizer_kwargs=cfg.get('vectorizer_kwargs'),
        )
        print(f"Feature shapes: train={X_train.shape}, test={X_test.shape}")

        print("  Generating cross-validated probabilities...")
        oof_probs = generate_oof_probabilities(X_train, y_train, cfg['params'], n_classes)

        pipeline_data[name] = {
            'X_train': X_train,
            'X_test': X_test,
            'oof_probs': oof_probs,
        }

    stack_train = np.hstack([pipeline_data['count_bigram']['oof_probs'],
                             pipeline_data['tfidf_bigram_stopwords']['oof_probs']])

    print("\nTraining logistic regression meta-classifier...")
    meta_clf = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        random_state=0,
    )
    meta_clf.fit(stack_train, y_train)

    meta_train_preds = meta_clf.predict(stack_train)
    train_acc = accuracy_score(y_train, meta_train_preds)
    train_f1 = f1_score(y_train, meta_train_preds, average='macro')
    print(f"Meta-classifier training accuracy: {train_acc:.3f}")
    print(f"Meta-classifier training macro F1: {train_f1:.3f}")

    print("\nTraining ensembles on full training data...")
    for name, cfg in feature_configs.items():
        models = train_ensemble(pipeline_data[name]['X_train'], y_train, cfg['params'], ENSEMBLE_SEEDS)
        pipeline_data[name]['ensemble_models'] = models
        test_probs = ensemble_probabilities(models, pipeline_data[name]['X_test'], n_classes)
        pipeline_data[name]['test_probs'] = test_probs
        test_preds = np.argmax(test_probs, axis=1)
        acc = accuracy_score(y_test, test_preds)
        f1 = f1_score(y_test, test_preds, average='macro')
        print(f"  {name} ensemble -- Test Acc: {acc:.3f}, Test F1: {f1:.3f}")

    stack_test = np.hstack([pipeline_data['count_bigram']['test_probs'],
                            pipeline_data['tfidf_bigram_stopwords']['test_probs']])

    final_preds = meta_clf.predict(stack_test)
    final_acc = accuracy_score(y_test, final_preds)
    final_f1 = f1_score(y_test, final_preds, average='macro')

    print("\n" + "=" * 80)
    print("STACKING RESULTS")
    print("=" * 80)
    print(f"Test accuracy: {final_acc:.3f}")
    print(f"Test macro F1: {final_f1:.3f}")
    print("Class mapping:", dict(enumerate(class_names)))


if __name__ == "__main__":
    main()

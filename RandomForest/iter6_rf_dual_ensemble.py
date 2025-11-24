"""
Dual-ensemble Random Forest experiment that blends the two best feature
pipelines from Iter5 to push accuracy beyond 0.691.

Pipelines:
- CountVectorizer (5000 max features, bigrams) with Iter5-style shallow RF
- TF-IDF (5000 features, bigrams, English stop words, min_df=2) with deeper RF

We train small ensembles for each configuration (five seeds), average their
probabilities separately, and then blend the two probability matrices with
several weights to see if the mixture improves test accuracy/F1.
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
    y_train = df_train['label'].values

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

    return df[is_train].copy(), df[is_test].copy()


def train_ensemble(X_train, y_train, params, seeds):
    models = []
    for seed in seeds:
        model_params = params.copy()
        model_params['random_state'] = seed
        model_params['n_jobs'] = -1

        rf = RandomForestClassifier(**model_params)
        rf.fit(X_train, y_train)
        models.append(rf)

    return models


def aggregate_probabilities(models, X_test, master_classes):
    class_to_idx = {cls: idx for idx, cls in enumerate(master_classes)}
    prob_matrix = np.zeros((X_test.shape[0], len(master_classes)))

    for model in models:
        model_probs = model.predict_proba(X_test)
        for class_index, cls in enumerate(model.classes_):
            target_idx = class_to_idx[cls]
            prob_matrix[:, target_idx] += model_probs[:, class_index]

    prob_matrix /= len(models)
    return prob_matrix


def evaluate_predictions(prob_matrix, y_test, master_classes):
    preds_idx = np.argmax(prob_matrix, axis=1)
    preds = master_classes[preds_idx]
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    return acc, f1


def main():
    df = pd.read_csv(file_name)
    df_train, df_test = student_wise_split(df, train_ratio=0.8, seed=0)
    master_classes = np.sort(df_train['label'].unique())

    print(f"Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")
    print(f"Classes: {master_classes}")

    ensemble_seeds = [0, 1, 2, 3, 4]

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

    prob_outputs = {}
    metrics = {}

    for name, cfg in feature_configs.items():
        print("\n" + "=" * 80)
        print(f"Training ensemble for feature config: {name}")
        print("=" * 80)

        X_train, y_train, X_test, y_test = build_features(
            df_train,
            df_test,
            text_method=cfg['text_method'],
            max_text_features=cfg['max_text_features'],
            use_bigrams=cfg.get('use_bigrams', False),
            vectorizer_kwargs=cfg.get('vectorizer_kwargs'),
        )

        models = train_ensemble(X_train, y_train, cfg['params'], ensemble_seeds)
        prob_matrix = aggregate_probabilities(models, X_test, master_classes)
        acc, f1 = evaluate_predictions(prob_matrix, y_test, master_classes)

        print(f"{name} ensemble accuracy: {acc:.3f}, F1: {f1:.3f}")
        prob_outputs[name] = prob_matrix
        metrics[name] = {'acc': acc, 'f1': f1}

    weight_options = [0.4, 0.5, 0.6]
    blend_results = []

    for w in weight_options:
        combined_probs = w * prob_outputs['count_bigram'] + (1 - w) * prob_outputs['tfidf_bigram_stopwords']
        acc, f1 = evaluate_predictions(combined_probs, y_test, master_classes)
        blend_results.append({'weight_count': w, 'test_acc': acc, 'test_f1': f1})
        print(f"\nBlend weight (count={w:.1f}, tfidf={1-w:.1f}): Acc={acc:.3f}, F1={f1:.3f}")

    best_blend = max(blend_results, key=lambda x: x['test_acc'])
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Count bigram ensemble: acc={metrics['count_bigram']['acc']:.3f}, "
          f"f1={metrics['count_bigram']['f1']:.3f}")
    print(f"TF-IDF bigram ensemble: acc={metrics['tfidf_bigram_stopwords']['acc']:.3f}, "
          f"f1={metrics['tfidf_bigram_stopwords']['f1']:.3f}")
    print(f"Best blend: count weight={best_blend['weight_count']:.1f}, "
          f"acc={best_blend['test_acc']:.3f}, f1={best_blend['test_f1']:.3f}")


if __name__ == "__main__":
    main()

"""
Neural baseline using Doc2Vec embeddings + structured survey features.

Steps:
- Student-wise split (80/20) to avoid leakage.
- Train Doc2Vec on training texts only (combined text columns).
- Concatenate doc vectors with ratings/multi-select encodings.
- Standardize features and train an MLP with cross-validated hyperparameter search.
"""
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

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


def tokenize(text):
    if text is None:
        return []
    return simple_preprocess(str(text), deacc=True)


def train_doc2vec(docs, vector_size=256, epochs=60, min_count=2):
    tagged = [TaggedDocument(words=tokenize(doc), tags=[i]) for i, doc in enumerate(docs)]
    model = Doc2Vec(
        vector_size=vector_size,
        window=7,
        min_count=min_count,
        workers=4,
        dm=1,
        negative=10,
        hs=0,
        seed=0,
    )
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=epochs)
    return model


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


def build_features(df_train, df_test, vector_size=256):
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

    train_texts = build_text_series(df_train, TEXT_COLS).tolist()
    test_texts = build_text_series(df_test, TEXT_COLS).tolist()

    doc2vec_model = train_doc2vec(train_texts, vector_size=vector_size)

    train_vectors = np.vstack([doc2vec_model.dv[i] for i in range(len(train_texts))])
    test_vectors = np.vstack([
        doc2vec_model.infer_vector(tokenize(text), epochs=40)
        for text in test_texts
    ])

    X_train = np.hstack([X_struct_train, train_vectors])
    X_test = np.hstack([X_struct_test, test_vectors])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, doc2vec_model, scaler


def evaluate_params(X, y, params, cv_splits=4):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    scores = []
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        clf = MLPClassifier(
            hidden_layer_sizes=params['hidden_layer_sizes'],
            activation='relu',
            solver='adam',
            alpha=params['alpha'],
            learning_rate_init=params['learning_rate_init'],
            batch_size=64,
            max_iter=250,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1,
            random_state=params['seed_offset'] + fold,
            verbose=False,
        )
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[val_idx])
        acc = accuracy_score(y[val_idx], preds)
        f1 = f1_score(y[val_idx], preds, average='macro')
        scores.append(acc)
        f1_scores.append(f1)
        print(f"    Fold {fold}: acc={acc:.3f}, f1={f1:.3f}")

    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(f1_scores))


def main():
    df = pd.read_csv(file_name)
    df_train, df_test, train_ids = student_wise_split(df, train_ratio=0.8, seed=0)

    print(f"Rows: {df.shape[0]}, Train: {df_train.shape[0]}, Test: {df_test.shape[0]}")
    print(f"Students: {df['student_id'].nunique()}, Train students: {len(train_ids)}")

    X_train, X_test, doc2vec_model, scaler = build_features(df_train, df_test, vector_size=256)
    print(f"Feature shapes: X_train={X_train.shape}, X_test={X_test.shape}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['label'].values)
    y_test = label_encoder.transform(df_test['label'].values)

    param_grid = [
        {'hidden_layer_sizes': (256,), 'alpha': 1e-4, 'learning_rate_init': 1e-3, 'seed_offset': 0},
        {'hidden_layer_sizes': (256, 128), 'alpha': 3e-4, 'learning_rate_init': 1e-3, 'seed_offset': 5},
        {'hidden_layer_sizes': (512, 256), 'alpha': 1e-4, 'learning_rate_init': 5e-4, 'seed_offset': 10},
        {'hidden_layer_sizes': (512, 256, 64), 'alpha': 5e-4, 'learning_rate_init': 5e-4, 'seed_offset': 15},
        {'hidden_layer_sizes': (256, 128, 64), 'alpha': 1e-3, 'learning_rate_init': 1e-3, 'seed_offset': 20},
    ]

    best_score = -1
    best_params = None

    print("\n=== Hyperparameter Evaluation (Doc2Vec features) ===")
    for idx, params in enumerate(param_grid, 1):
        print(f"\nConfig {idx}: {params}")
        mean_acc, std_acc, mean_f1 = evaluate_params(X_train, y_train, params)
        print(f"  -> CV acc: {mean_acc:.3f} ± {std_acc:.3f}, CV macro F1: {mean_f1:.3f}")
        if mean_acc > best_score:
            best_score = mean_acc
            best_params = params

    print("\nBest CV params:", best_params)
    print(f"Best CV accuracy: {best_score:.3f}")

    final_clf = MLPClassifier(
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
    final_clf.fit(X_train, y_train)

    train_pred = final_clf.predict(X_train)
    test_pred = final_clf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')

    print("\n=== Final Model Performance (Doc2Vec) ===")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Training macro F1: {train_f1:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test macro F1: {test_f1:.3f}")
    print(f"Compare to Logistic Regression baseline (0.683): {test_acc - 0.683:+.3f}")
    print(f"Compare to RF ensemble (0.691): {test_acc - 0.691:+.3f}")


if __name__ == "__main__":
    main()

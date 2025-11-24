"""
Logistic regression using averaged GloVe embeddings + structured features.

Steps:
- Load pretrained glove-wiki-gigaword-50 embeddings via gensim (cached locally).
- Student-wise 80/20 split to prevent leakage.
- Build structured features (ratings + multi-select) identical to prior models.
- For each example, compute the mean of available word embeddings across TEXT_COLS.
- Concatenate structured + embedding features, standardize, and train multinomial logistic regression.
- Evaluate accuracy on the held-out test split and save artifacts (scaler, coefficients, embedding vocab).
"""
import numpy as np
import pandas as pd
import re
from pathlib import Path
from gensim.utils import simple_preprocess
import gensim.downloader as api
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

file_name = "../DataSet/training_data_clean.csv"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

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
    return df[is_train].copy(), df[~is_train].copy(), train_ids


def build_structured_features(df_train, df_test):
    X_ratings_train = build_rating_matrix(df_train, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)
    X_ratings_test = build_rating_matrix(df_test, RATING_COLS, neutral_value=NEUTRAL_RATING_VALUE)

    best_tasks_train = process_multiselect(df_train[BEST_TASKS_COL], target_tasks)
    subopt_tasks_train = process_multiselect(df_train[SUBOPT_TASKS_COL], target_tasks)

    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()

    best_encoded_train = mlb_best.fit_transform(best_tasks_train)
    subopt_encoded_train = mlb_subopt.fit_transform(subopt_tasks_train)

    best_encoded_test = mlb_best.transform(process_multiselect(df_test[BEST_TASKS_COL], target_tasks))
    subopt_encoded_test = mlb_subopt.transform(process_multiselect(df_test[SUBOPT_TASKS_COL], target_tasks))

    X_struct_train = np.hstack([X_ratings_train, best_encoded_train, subopt_encoded_train])
    X_struct_test = np.hstack([X_ratings_test, best_encoded_test, subopt_encoded_test])
    return X_struct_train, X_struct_test


def tokenize_texts(series):
    return [simple_preprocess(str(text), deacc=True) for text in series]


def build_embedding_dictionary(token_lists, model):
    vocab = set()
    for tokens in token_lists:
        vocab.update(tokens)
    embedding_dim = model.vector_size
    token_to_vec = {}
    for token in vocab:
        if token in model:
            token_to_vec[token] = model[token]
    print(f"Embedding vocab size retained: {len(token_to_vec)}")
    return token_to_vec, embedding_dim


def texts_to_embeddings(token_lists, token_to_vec, embedding_dim):
    vectors = []
    for tokens in token_lists:
        collected = [token_to_vec[token] for token in tokens if token in token_to_vec]
        if collected:
            vectors.append(np.mean(collected, axis=0))
        else:
            vectors.append(np.zeros(embedding_dim))
    return np.vstack(vectors)


def main():
    df = pd.read_csv(file_name)
    df_train, df_test, train_ids = student_wise_split(df, train_ratio=0.8, seed=0)
    print(f"Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")

    X_struct_train, X_struct_test = build_structured_features(df_train, df_test)
    print(f"Structured feature shape: {X_struct_train.shape}")

    train_texts = build_text_series(df_train, TEXT_COLS)
    test_texts = build_text_series(df_test, TEXT_COLS)

    train_tokens = tokenize_texts(train_texts)
    test_tokens = tokenize_texts(test_texts)

    print("Loading GloVe embeddings via gensim...")
    glove_model = api.load("glove-wiki-gigaword-50")
    token_to_vec, embedding_dim = build_embedding_dictionary(train_tokens, glove_model)

    X_emb_train = texts_to_embeddings(train_tokens, token_to_vec, embedding_dim)
    X_emb_test = texts_to_embeddings(test_tokens, token_to_vec, embedding_dim)
    print(f"Embedding feature shape: {X_emb_train.shape}")

    X_train = np.hstack([X_struct_train, X_emb_train])
    X_test = np.hstack([X_struct_test, X_emb_test])
    print(f"Final feature shape: {X_train.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['label'].values)
    y_test = label_encoder.transform(df_test['label'].values)

    logreg = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=200,
        C=1.0,
        n_jobs=-1,
        verbose=1,
    )
    logreg.fit(X_train_scaled, y_train)

    train_pred = logreg.predict(X_train_scaled)
    test_pred = logreg.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')

    print("\n=== GloVe Logistic Regression Performance ===")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Training macro F1: {train_f1:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test macro F1: {test_f1:.3f}")

    artifact_path = ARTIFACT_DIR / "glove_logreg.npz"
    np.savez_compressed(
        artifact_path,
        coef=logreg.coef_,
        intercept=logreg.intercept_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        label_encoder_classes=label_encoder.classes_,
        tokens=np.array(list(token_to_vec.keys())),
        embeddings=np.stack(list(token_to_vec.values())),
        structured_shape=X_struct_train.shape[1],
    )
    print(f"Saved artifact to {artifact_path}")


if __name__ == "__main__":
    main()
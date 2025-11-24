"""
PyTorch MLP on TF-IDF + structured features with dropout/batch norm.

Workflow:
- Build structured + TF-IDF (ngram (1,2), max 5000, min_df=2) features.
- Compress text via TruncatedSVD (400 components) to keep final feature size manageable.
- Standardize features (mean 0, std 1).
- Use student-wise StratifiedKFold (4 folds) to tune hyperparameters based on validation accuracy.
- Train final MLP on full train split and evaluate on held-out test split (student-wise 80/20).
- Export weights/biases to .npz for future use in numpy-only pred script.
"""
import numpy as np
import pandas as pd
import re
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

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


def build_features(df_train, df_test, svd_components=400):
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

    svd = TruncatedSVD(n_components=svd_components, random_state=0)
    X_text_train = svd.fit_transform(X_text_train_sparse)
    X_text_test = svd.transform(X_text_test_sparse)

    X_train = np.hstack([X_struct_train, X_text_train])
    X_test = np.hstack([X_struct_test, X_text_test])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, vectorizer, svd, scaler


class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_layers=(512, 256), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 3))  # 3 classes
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one_model(X_train, y_train, params, device):
    dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    train_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    model = MLPNet(
        input_dim=X_train.shape[1],
        hidden_layers=params['hidden_layers'],
        dropout=params['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    best_state = None
    best_acc = -1
    patience = params.get('patience', 10)
    wait = 0

    for epoch in range(params['max_epochs']):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        # Evaluate on training set for quick monitoring
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_train).float().to(device)
            logits = model(X_tensor)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            acc = accuracy_score(y_train, pred)
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    model.load_state_dict(best_state)
    return model


def evaluate_cv(X, y, params, device, n_splits=4):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    f1s = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        model = train_one_model(X_train, y_train, params, device)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(X_val).float().to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average='macro')
        accs.append(acc)
        f1s.append(f1)
        print(f"    Fold {fold}: acc={acc:.3f}, f1={f1:.3f}")
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s))


def export_model(model, scaler, vectorizer, svd, label_encoder, artifact_path):
    artifacts = {
        'state_dict': {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'vectorizer_vocab': vectorizer.vocabulary_,
        'vectorizer_idf': vectorizer.idf_,
        'svd_components': svd.components_,
        'svd_mean': getattr(svd, 'mean_', None),
        'svd_explained_variance': svd.explained_variance_,
        'label_encoder_classes': label_encoder.classes_,
    }
    np.savez_compressed(artifact_path, **artifacts)
    print(f"Saved artifacts to {artifact_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(file_name)
    df_train, df_test, train_ids = student_wise_split(df, train_ratio=0.8, seed=0)
    print(f"Train rows: {df_train.shape[0]}, Test rows: {df_test.shape[0]}")

    X_train, X_test, vectorizer, svd, scaler = build_features(df_train, df_test, svd_components=400)
    print(f"Feature shapes -> train: {X_train.shape}, test: {X_test.shape}")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['label'].values)
    y_test = label_encoder.transform(df_test['label'].values)

    param_grid = [
        {'hidden_layers': (512, 256), 'dropout': 0.3, 'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 64, 'max_epochs': 80},
        {'hidden_layers': (512, 256), 'dropout': 0.4, 'lr': 1e-3, 'weight_decay': 3e-4, 'batch_size': 64, 'max_epochs': 80},
        {'hidden_layers': (512, 256, 128), 'dropout': 0.4, 'lr': 5e-4, 'weight_decay': 1e-4, 'batch_size': 64, 'max_epochs': 100},
        {'hidden_layers': (640, 320), 'dropout': 0.3, 'lr': 1e-3, 'weight_decay': 1e-4, 'batch_size': 64, 'max_epochs': 80},
        {'hidden_layers': (512, 256), 'dropout': 0.5, 'lr': 5e-4, 'weight_decay': 5e-4, 'batch_size': 64, 'max_epochs': 120},
    ]

    best_params = None
    best_acc = -1

    print("\n=== Hyperparameter Evaluation (PyTorch MLP) ===")
    for idx, params in enumerate(param_grid, 1):
        print(f"\nConfig {idx}: {params}")
        mean_acc, std_acc, mean_f1 = evaluate_cv(X_train, y_train, params, device, n_splits=4)
        print(f"  -> CV acc: {mean_acc:.3f} ± {std_acc:.3f}, CV macro F1: {mean_f1:.3f}")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_params = params

    print("\nBest params:", best_params)
    print(f"Best CV accuracy: {best_acc:.3f}")

    final_model = train_one_model(X_train, y_train, {**best_params, 'patience': 15}, device)
    final_model.eval()

    with torch.no_grad():
        train_logits = final_model(torch.from_numpy(X_train).float().to(device))
        test_logits = final_model(torch.from_numpy(X_test).float().to(device))
        train_pred = torch.argmax(train_logits, dim=1).cpu().numpy()
        test_pred = torch.argmax(test_logits, dim=1).cpu().numpy()

    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')

    print("\n=== Final Model Performance ===")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Training macro F1: {train_f1:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print(f"Test macro F1: {test_f1:.3f}")

    export_model(
        final_model,
        scaler,
        vectorizer,
        svd,
        label_encoder,
        ARTIFACT_DIR / "pytorch_mlp_tfidf.npz"
    )


if __name__ == "__main__":
    main()

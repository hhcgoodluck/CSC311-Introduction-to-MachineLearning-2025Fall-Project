import numpy as np
import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer

file_name = "training_data_clean.csv"

def process_multiselect(series, target_tasks):
    """
    Convert multiselect strings to lists, keeping only specified features
    """
    processed = []
    for response in series:
        if pd.isna(response) or response == '':
            processed.append([])
        else:
            # Check which of the target tasks are present in the response
            present_tasks = [task for task in target_tasks if task in str(response)]
            processed.append(present_tasks)
    return processed


def extract_rating(response):
    """
    Extract numeric rating from responses like '3 - Sometimes'.
    Returns None for missing / unparsable responses.
    """
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None


def main():
    df = pd.read_csv(file_name)

    # ---- 1. 构造多选题特征 ----
    target_tasks = [
        'Math computations',
        'Writing or debugging code',
        'Data processing or analysis',
        'Explaining complex concepts simply',
    ]

    best_tasks_lists = process_multiselect(
        df['Which types of tasks do you feel this model handles best? (Select all that apply.)'],
        target_tasks
    )

    suboptimal_tasks_lists = process_multiselect(
        df['For which types of tasks do you feel this model tends to give suboptimal responses? (Select all that apply.)'],
        target_tasks
    )

    mlb_best = MultiLabelBinarizer()
    mlb_subopt = MultiLabelBinarizer()

    best_tasks_encoded = mlb_best.fit_transform(best_tasks_lists)
    suboptimal_tasks_encoded = mlb_subopt.fit_transform(suboptimal_tasks_lists)

    # ---- 2. Likert 评分题：提取数字 1–5 ----
    academic_numeric = df['How likely are you to use this model for academic tasks?'].apply(extract_rating)
    subopt_numeric = df['Based on your experience, how often has this model given you a response that felt suboptimal?'].apply(extract_rating)

    # 如果有解析失败（None），用 0 填充，保证都是非负数
    academic_numeric = academic_numeric.fillna(0)
    subopt_numeric = subopt_numeric.fillna(0)

    # ---- 3. 拼成特征矩阵 X ----
    X = np.hstack([
        academic_numeric.values.reshape(-1, 1),
        subopt_numeric.values.reshape(-1, 1),
        best_tasks_encoded,
        suboptimal_tasks_encoded
    ])

    y = df['label'].values

    # ---- 4. 简单 70/30 随机划分（暂时不做 student_id 分组）----
    n_train = int(0.7 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # ---- 5. 训练 Multinomial Naive Bayes ----
    nb = MultinomialNB(
        alpha=1.0  # Laplace smoothing
    )

    nb.fit(X_train, y_train)

    train_acc = nb.score(X_train, y_train)
    test_acc = nb.score(X_test, y_test)

    print(f"Training accuracy (Multinomial NB): {train_acc:.3f}")
    print(f"Test accuracy (Multinomial NB): {test_acc:.3f}")


if __name__ == "__main__":
    main()

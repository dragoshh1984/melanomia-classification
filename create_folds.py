import os

import numpy as np
import pandas as pd
import random

from collections import Counter, defaultdict
from sklearn import model_selection

def create_stratified_folds():
    input_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/"

    df = pd.read_csv(os.path.join(input_path, "train.csv"))
    
    # Insert kfold column
    df["kfold"] = -1

    # Shuffle the dataset:
    df = df.sample(frac=1).reset_index(drop=True)

    # Create folds
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=5)

    # Add the fold number to each row
    for fold, (train_index, test_index) in enumerate(kf.split(X=df, y=y)):
        df.loc[test_index, "kfold"] = fold
    
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def create_group_k_folds():
    input_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/"

    df_folds = pd.read_csv(os.path.join(input_path, "train.csv"))
    
    df_folds['patient_id'] = df_folds['patient_id'].fillna(df_folds['image_name'])
    df_folds['sex'] = df_folds['sex'].fillna('unknown')
    df_folds['anatom_site_general_challenge'] = df_folds['anatom_site_general_challenge'].fillna('unknown')
    df_folds['age_approx'] = df_folds['age_approx'].fillna(round(df_folds['age_approx'].mean()))
    df_folds = df_folds.set_index('image_name')

    def get_stratify_group(row):
        stratify_group = row['sex']
        stratify_group += f'_{row["anatom_site_general_challenge"]}'
        stratify_group += f'_{row["target"]}'
        return stratify_group

    df_folds['stratify_group'] = df_folds.apply(get_stratify_group, axis=1)
    df_folds['stratify_group'] = df_folds['stratify_group'].astype('category').cat.codes

    df_folds.loc[:, 'fold'] = 0

    skf = stratified_group_k_fold(X=df_folds.index, y=df_folds['stratify_group'], groups=df_folds['patient_id'], k=5, seed=42)

    for fold_number, (train_index, val_index) in enumerate(skf):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    
    df_folds.to_csv(os.path.join(input_path, "group_folds.csv"))

if __name__ == "__main__":
    create_group_k_folds()
import os

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
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
        df.loc[:, "kfold"] = fold
    
    df.to_csv(os.path.join(input_path, "train_folds.csv"), index=False)
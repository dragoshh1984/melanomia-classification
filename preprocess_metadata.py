import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

if __name__=="__main__":
    input_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/"

    df_train = pd.read_csv(os.path.join(input_path, "group_folds.csv"))
    df_test = pd.read_csv(os.path.join(input_path, "test.csv"))

    df_train = df_train.drop(["patient_id", "benign_malignant", "diagnosis", "stratify_group"], axis=1)
    df_train['sex'] = df_train['sex'].fillna('unknown')
    df_train['anatom_site_general_challenge'] = df_train['anatom_site_general_challenge'].fillna('unknown')
    df_train['age_approx'] = df_train['age_approx'].fillna(round(df_train['age_approx'].mean()))

    df_test = df_test.drop(["patient_id"], axis=1)
    df_test['sex'] = df_test['sex'].fillna('unknown')
    df_test['anatom_site_general_challenge'] = df_test['anatom_site_general_challenge'].fillna('unknown')
    df_test['age_approx'] = df_test['age_approx'].fillna(round(df_test['age_approx'].mean()))
    
    le = LabelEncoder()
    types = df_train['anatom_site_general_challenge'].unique()
    df_train['anatom_site_general_challenge'] = le.fit_transform(df_train['anatom_site_general_challenge'])

    types = df_train['anatom_site_general_challenge'].unique()
    df_test['anatom_site_general_challenge'] = le.fit_transform(df_test['anatom_site_general_challenge'])
    
    types = df_train['sex'].unique()
    df_train['sex'] = le.fit_transform(df_train['sex'])

    types = df_test['sex'].unique()
    df_test['sex'] = le.fit_transform(df_test['sex'])

    df_train.to_csv(os.path.join(input_path, "train_metadata.csv"), index=False)
    df_test.to_csv(os.path.join(input_path, "test_metadata.csv"), index=False)
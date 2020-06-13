import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm

if __name__=="__main__":
    input_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/"
    train_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/512x512-dataset-melanoma/512x512-dataset-melanoma"
    test_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/512x512-test/512x512-test"

    df_train = pd.read_csv(os.path.join(input_path, "folds_big.csv"))
    df_test = pd.read_csv(os.path.join(input_path, "test.csv"))
    df_test = df_test.rename(columns={"image_name": "image_id"})

    df_train['sex'] = df_train['sex'].fillna('unknown')
    df_train['anatom_site_general_challenge'] = df_train['anatom_site_general_challenge'].fillna('unknown')
    df_train['age_approx'] = df_train['age_approx'].fillna(round(df_train['age_approx'].mean()))

    df_test['sex'] = df_test['sex'].fillna('unknown')
    df_test['anatom_site_general_challenge'] = df_test['anatom_site_general_challenge'].fillna('unknown')
    df_test['age_approx'] = df_test['age_approx'].fillna(round(df_test['age_approx'].mean()))

    # get color mean
    train_means = []
    for image_name in tqdm(df_train.image_id):
        color_mean = np.array(Image.open(os.path.join(train_path, f"{image_name}.jpg"))).mean()
        train_means.append(color_mean)
    df_train.loc[:, 'color_mean'] = train_means

    test_means = []
    for image_name in tqdm(df_test.image_id):
        color_mean = np.array(Image.open(os.path.join(test_path, f"{image_name}.jpg"))).mean()
        test_means.append(color_mean)
    df_test.loc[:, 'color_mean'] = test_means

    # label anatom_challenge
    le = LabelEncoder()
    types = df_train['anatom_site_general_challenge'].unique()
    df_train['anatom_site_general_challenge'] = le.fit_transform(df_train['anatom_site_general_challenge'])

    types = df_test['anatom_site_general_challenge'].unique()
    df_test['anatom_site_general_challenge'] = le.fit_transform(df_test['anatom_site_general_challenge'])

    # encode sex
    types = df_train['sex'].unique()
    df_train['sex'] = le.fit_transform(df_train['sex'])

    types = df_test['sex'].unique()
    df_test['sex'] = le.fit_transform(df_test['sex'])

    df_train.to_csv(os.path.join(input_path, "new_train.csv"), index=False)
    df_test.to_csv(os.path.join(input_path, "new_test.csv"), index=False)
# df_folds.loc[:, 'fold'] = 0
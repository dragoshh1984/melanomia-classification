import os
import torch

import pretrainedmodels
import albumentations

import numpy as np
import pandas as pd
import torch.nn as nn

from sklearn import metrics
from torch.nn import functional as F

from engine import Engine
from early_stopping import EarlyStopping
from loader import ClassificationLoader


class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        
        self.model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        self.out = nn.Linear(2048, 1)
    
    # depending on what the engine returns
    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(
            out, targets.view(-1, 1).type_as(x)
        )

        return out, loss

def train(fold):
    training_data_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/train_224/"
    model_path = "/home/dragoshh1984/repos/kaggle/melanomia-classification"
    df = pd.read_csv("/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/train_folds.csv")

    # defines
    device = "cuda"
    epochs = 50
    train_bs = 32
    valid_bs = 16

    # for this model
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # data for training
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # augmentations
    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".png") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".png") for i in valid_images]
    valid_targets = df_valid.target.values

    # datasets
    training_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )

    # loaders
    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=4
    )

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # max for auc metric
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"
    )

    # early stopping
    es = EarlyStopping(patience=5, mode="max")
    # import pdb; pdb.set_trace()
    for epoch in range(epochs):
        training_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            device
        )
        predictions, valid_loss = Engine.evaluate(
            valid_loader,
            model,
            device
        )

        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)

        print(f"epoch={epoch}, auc={auc}")
        es(auc, model, os.path.join(model_path, f"model{fold}.bin"))

        if es.early_stop:
            print("early stopping")
            break

def predict(fold):
    test_data_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/test_224/"
    model_path = "/home/dragoshh1984/repos/kaggle/melanomia-classification"
    df_test = pd.read_csv("/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/test.csv")
    df_test.loc[:, "target"] = 0

    # defines
    device = "cuda"
    epochs = 50
    test_bs = 32
    valid_bs = 16

    # for this model
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # augmentations
    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    test_images = df_test.image_name.values.tolist()
    test_images = [os.path.join(test_data_path, i + ".png") for i in test_images]
    test_targets = df_test.target.values

    # datasets
    test_dataset = ClassificationLoader(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )

    # loaders  
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=4
    )

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.load_state_dict(torch.load(os.path.join(model_path, f"model{fold}.bin")))
    model.to(device)

    predictions = Engine.predict(
        test_loader,
        model,
        device
    )

    return np.vstack((predictions)).ravel()

if __name__ == "__main__":
    train(fold=0)
    predict(fold=0)

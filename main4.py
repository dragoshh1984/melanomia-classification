import os
import torch

import albumentations

import numpy as np
import pandas as pd
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from sklearn import metrics
from torch.nn import functional as F
from catalyst.data.sampler import BalanceClassSampler
from torch.utils.data.sampler import SequentialSampler

from engine import Engine
from early_stopping import EarlyStopping
from loader3 import ClassificationLoader


class EfficientNet_tabular(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(EfficientNet_tabular, self).__init__()
        
        self.model_image = EfficientNet.from_pretrained('efficientnet-b1')
        self.model_image._fc = nn.Linear(1280, 512, bias=True)

        self.model_tabular = nn.Sequential(
            nn.Linear(4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.model_out = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.out = nn.Linear(1024, 1)
    
    def forward(self, image, metadata, targets):
        bs, _, _, _ = image.shape
        
        x1 = self.model_image(image)
        x1 = x1.view(x1.size(0), -1)
        
        x2 = self.model_tabular(metadata)
        x2 = x2.view(x2.size(0), -1)
        
        x3 = torch.cat((x1, x2), 1)
        x4 = self.model_out(x3)

        out = self.out(x4)
        weight = 0.7*torch.ones([1]).cuda()
        weight.to('cuda')
        loss = nn.BCEWithLogitsLoss(weight=weight)(
            out, targets.view(-1, 1).type_as(x4)
        )

        return out, loss


def train(fold):
    training_data_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/512x512-dataset-melanoma/512x512-dataset-melanoma"
    model_path = "/home/dragoshh1984/repos/kaggle/melanomia-classification"
    df = pd.read_csv("/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/new_train.csv")

    # defines
    device = "cuda"
    epochs = 10
    train_bs = 16
    valid_bs = 16

    # for this model
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # data for training
    df_train = df[df.fold != fold].reset_index(drop=True)
    df_valid = df[(df.fold == fold) & (df.source == 'ISIC20')].reset_index(drop=True)

    # augmentations
    train_aug = albumentations.Compose(
        [
            albumentations.RandomResizedCrop(224, 224, (0.7, 1.0)),
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.Cutout(),
            albumentations.RandomBrightness(),
            albumentations.RandomContrast(),
            # albumentations.Rotate(),
            albumentations.RandomScale(),
            albumentations.PadIfNeeded(330, 330),
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.RandomResizedCrop(224, 224, (0.7, 1.0)),
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.Cutout(),
            albumentations.RandomBrightness(),
            albumentations.RandomContrast(),
            # albumentations.Rotate(),
            albumentations.RandomScale(),
            albumentations.PadIfNeeded(330, 330),
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    train_images = df_train.image_id.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
    train_metada = df_train.drop(["fold", "target", "image_id", "patient_id", "source", "stratify_group"], axis=1).values.tolist()
    train_targets = df_train.target.values

    valid_images = df_valid.image_id.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_metadata = df_valid.drop(["fold", "target", "image_id", "patient_id", "source", "stratify_group"], axis=1).values.tolist()
    valid_targets = df_valid.target.values

    # datasets
    training_dataset = ClassificationLoader(
        image_paths=train_images,
        metadata=train_metada,
        targets=train_targets,
        resize=None,
        augmentations=train_aug
    )

    # loaders
    train_loader = torch.utils.data.DataLoader(
        training_dataset,
        # sampler=BalanceClassSampler(labels=train_targets, mode="downsampling"),
        batch_size=train_bs,
        drop_last=True,
        pin_memory=False,
        num_workers=4
    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        metadata=valid_metadata,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        # sampler=SequentialSampler(valid_dataset),
        batch_size=valid_bs,
        shuffle=False,
        pin_memory=False,
        num_workers=4
    )

    model = EfficientNet_tabular(pretrained="imagenet")
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
        # import pdb; pdb.set_trace()
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)

        print(f"epoch={epoch}, auc={auc}")
        es(auc, model, os.path.join(model_path, f"model_v2_{fold}.bin"))

        if es.early_stop:
            print("early stopping")
            break

def predict(fold):
    test_data_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/512x512-test/512x512-test"
    model_path = "/home/dragoshh1984/repos/kaggle/melanomia-classification"
    df_test = pd.read_csv("/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/new_test.csv")
    df_test.loc[:, "target"] = 0

    # defines
    device = "cuda"
    test_bs = 16

    # for this model
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # augmentations
    test_aug = albumentations.Compose(
        [
            albumentations.RandomResizedCrop(224, 224, (0.7, 1.0)),
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.Cutout(),
            albumentations.RandomBrightness(),
            albumentations.RandomContrast(),
            # albumentations.Rotate(),
            albumentations.RandomScale(),
            albumentations.PadIfNeeded(330, 330),
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    test_images = df_test.image_id.values.tolist()
    test_images = [os.path.join(test_data_path, i + ".jpg") for i in test_images]
    test_metadata = df_test.drop(["image_id", "target", "patient_id"], axis=1).values.tolist()
    test_targets = df_test.target.values

    # datasets
    test_dataset = ClassificationLoader(
        image_paths=test_images,
        metadata=test_metadata,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
    )

    # loaders  
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=4
    )

    model = EfficientNet_tabular(pretrained="imagenet")
    model.load_state_dict(torch.load(os.path.join(model_path, f"model_v2_{fold}.bin")))
    model.to(device)

    predictions = Engine.predict(
        test_loader,
        model,
        device
    )

    return np.vstack((predictions)).ravel()

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    # training
    for fold in range(0,5):
        train(fold)
    
    #predicting
    all_predictions = []
    for fold in range(0,5):
        all_predictions.append(predict(fold))
    
    predictions = sum(all_predictions) / 5
    # import pdb; pdb.set_trace()
    submission = pd.read_csv("/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/sample_submission.csv")
    submission.loc[:, "target"] = predictions
    submission.to_csv("submission_v2.csv", index=False)

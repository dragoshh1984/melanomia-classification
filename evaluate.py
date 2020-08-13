import os
import torch

import albumentations

import numpy as np
import pandas as pd
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from sklearn import metrics
from torch.nn import functional as F


from engine import Engine
from early_stopping import EarlyStopping
from loader3 import ClassificationLoader
from loss import FocalLoss

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
            nn.Dropout(0.5)
        )
        self.model_out = nn.Sequential(
            nn.Linear(768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
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
        loss = nn.BCEWithLogitsLoss()(
            out, targets.view(-1, 1).type_as(x4)
        )

        # loss = FocalLoss(
        #     alpha = 0.75,
        #     gamma=2,
        #     logits=True
        # )(
        #     out, targets.view(-1, 1).type_as(x4)
        # )
        return out, loss

def confusion_matrix(preds, targets, conf_matrix):
    # import pdb; pdb.set_trace()
    # preds = np.argmax(preds, 1)
    preds = np.array([0 if x <= 0 else 1 for x in preds])
    for p, t in zip(preds, targets):
        conf_matrix[p, t] += 1
    
    return conf_matrix

def evaluate(fold):
    training_data_path = "/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/512x512-dataset-melanoma/512x512-dataset-melanoma"
    model_path = "/home/dragoshh1984/repos/kaggle/melanomia-classification"
    df = pd.read_csv("/home/dragoshh1984/repos/kaggle/datasets/melanomia_classification/new_train.csv")

    # confusion matrix
    conf_matrix = torch.zeros(2,2)

    # defines
    device = "cuda"
    valid_bs = 16

    # for this model
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # data for training
    df_valid = df[df.fold == fold].reset_index(drop=True)

    valid_aug = albumentations.Compose(
        [
            albumentations.RandomResizedCrop(224, 224, (0.7, 1.0)),
            albumentations.HorizontalFlip(),
            albumentations.VerticalFlip(),
            albumentations.Cutout(),
            albumentations.RandomBrightness(),
            albumentations.RandomContrast(),
            albumentations.Rotate(),
            albumentations.RandomScale(),
            albumentations.PadIfNeeded(300, 300),
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        ]
    )

    valid_images = df_valid.image_id.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
    valid_metadata = df_valid.drop(["fold", "target", "image_id", "patient_id", "source", "stratify_group"], axis=1).values.tolist()
    valid_targets = df_valid.target.values

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        metadata=valid_metadata,
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

    model = EfficientNet_tabular(pretrained="imagenet")
    model.load_state_dict(torch.load(os.path.join(model_path, f"model_v2_{fold}.bin")))
    model.to(device)
    predictions = Engine.predict(
        valid_loader,
        model,
        device
    )

    predictions = np.vstack((predictions)).ravel()
    conf_matrix = confusion_matrix(predictions, valid_targets, conf_matrix)

    # import pdb; pdb.set_trace()
    auc = metrics.roc_auc_score(valid_targets, predictions)

    print(f"auc={auc}")

    return conf_matrix

if __name__ == "__main__":
    # confusion matrix avg
    conf_matrix = torch.zeros(2, 2)
    # training
    for fold in range(0,5):
        new_conf_matrix = evaluate(fold)
        conf_matrix += new_conf_matrix

    true_neg = conf_matrix[0,0].tolist()
    false_poz = conf_matrix[0,1].tolist()
    false_neg = conf_matrix[1,0].tolist()
    true_poz = conf_matrix[1,1].tolist()

    sensitivity = true_poz / (false_neg + true_poz)
    specificity = true_neg / (true_neg + false_poz)

    print("sensitivity {}\nspecificity {}".format(sensitivity, specificity))

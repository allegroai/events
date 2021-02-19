import argparse
from pathlib import Path
import os
import tqdm  # temp
import glob
import warnings
from dataclasses import dataclass

import albumentations
import pandas as pd
import tez
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import metrics, preprocessing
from tez.callbacks import EarlyStopping
from tez.datasets import ImageDataset
from torch.nn import functional as F

from clearml import Task, Dataset

# INPUT_PATH = str(Path("~/datasets/flowers/").expanduser())
# MODEL_PATH = "../../models/"
MODEL_NAME = os.path.basename(__file__)[:-3]
# TRAIN_BATCH_SIZE = 32
# VALID_BATCH_SIZE = 32
# IMAGE_SIZE = 192
# EPOCHS = 20

@dataclass
class FlowerTrainingConfig:
    # For dataset, go to: https://www.kaggle.com/msheriey/104-flowers-garden-of-eden
    # original flower dataset id
    dataset_id: str = "86895530658c47a4918bda4f0d92c3e8"
    # just in case you need to access models locally
    model_path: str = "models/"
    # currently base name is fixed
    model_name: str = MODEL_NAME
    fp16mode: bool = True
    train_batch_size: int = 32
    valid_batch_size: int = 32
    # can only be 192, 224, 331, 512 if using the garden dataset
    image_size: int = 192 # this will be removed soon
    num_epochs: int = 20
    data_loader_n_jobs: int = 1
    efficient_model_type: str = "efficientnet-b0"
    early_stopping_patience: int = 3
    # don't change
    adam_lr: float = 1e-4


class CustomTensorBoardLogger(tez.callbacks.TensorBoardLogger):
    def __init__(self, log_dir=".logs/"):
        super().__init__(log_dir)

    def on_train_step_end(self, model: tez.Model):
        for metric in model.metrics["train"]:
            if "step" in metric:
                self.writer.add_scalar(
                    f"train_step/{metric}", model.metrics["train"][metric], model.current_train_step
                )

class FlowerModel(tez.Model):
    def __init__(self,
                 num_classes,
                 efficientnet_model: str ="efficientnet-b0",
                 adam_lr: float = 1e-4
                 ):
        super().__init__()

        self.effnet = EfficientNet.from_pretrained(efficientnet_model)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(1280, num_classes)
        self.lr = adam_lr
        self.step_report_every_n: int = 5

    def monitor_metrics(self, outputs, targets):
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt

    def train_one_step(self, data):
        loss, metrics = super().train_one_step(data)
        if (self.current_train_step % self.step_report_every_n) == 0:
            self.metrics[self._model_state.value].update({"step_loss": loss.item()})
        return loss, metrics

    def forward(self, image, targets=None):
        batch_size, _, _, _ = image.shape

        x = self.effnet.extract_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        outputs = self.out(self.dropout(x))

        if targets is not None:
            loss = nn.CrossEntropyLoss()(outputs, targets)
            metrics = self.monitor_metrics(outputs, targets)
            return outputs, loss, metrics
        return outputs, 0, {}


if __name__ == "__main__":

    # Track everything on ClearML Free
    task = Task.init(project_name='R|D?R&D! Webinar 01',
                     task_name='remote control interface',
                     output_uri=True, # auto save everything to Clearml Free
                     )

    task.connect(FlowerTrainingConfig, 'config')
    cfg = FlowerTrainingConfig()

    # Need to run on cpu only?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn('GPU not available!, using CPU mode')
        warnings.filterwarnings("ignore", module='torch.cuda.amp.autocast')

    train_aug = albumentations.Compose(
        [
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )

    # download dataset (cached!)
    dataset_folder = Dataset.get(dataset_id=cfg.dataset_id).get_local_copy()

    train_image_paths = glob.glob(
        os.path.join(
            dataset_folder, f"jpeg-{cfg.image_size}x{cfg.image_size}", "train", "**", "*.jpeg"
        ),
        recursive=True,
    )

    valid_image_paths = glob.glob(
        os.path.join(
            dataset_folder, f"jpeg-{cfg.image_size}x{cfg.image_size}", "val", "**", "*.jpeg"
        ),
        recursive=True,
    )

    train_targets = [x.split("/")[-2] for x in train_image_paths]
    valid_targets = [x.split("/")[-2] for x in valid_image_paths]

    lbl_enc = preprocessing.LabelEncoder()
    train_targets = lbl_enc.fit_transform(train_targets)
    valid_targets = lbl_enc.transform(valid_targets)

    # track model labels
    task.set_model_label_enumeration({
        lbl: n for n, lbl in enumerate(lbl_enc.classes_)
    })

    train_dataset = ImageDataset(
        image_paths=train_image_paths,
        targets=train_targets,
        augmentations=train_aug,
    )

    valid_dataset = ImageDataset(
        image_paths=valid_image_paths,
        targets=valid_targets,
        augmentations=valid_aug,
    )

    model = FlowerModel(
        num_classes=len(lbl_enc.classes_),
        adam_lr=cfg.adam_lr)

    # temporary, model pathname here, and make sure directory exists
    model_path = os.path.join(MODEL_PATH, MODEL_NAME + ".bin")
    Path(MODEL_PATH).mkdir(exist_ok=True)

    tb = CustomTensorBoardLogger()

    es = EarlyStopping(
        monitor="valid_loss",
        model_path=model_path,
        patience=cfg.early_stopping_patience,
        mode="min",
    )
    model.fit(
        train_dataset,
        valid_dataset=valid_dataset,
        train_bs=cfg.train_batch_size,
        valid_bs=cfg.valid_batch_size,
        device=device,
        epochs=cfg.num_epochs,
        callbacks=[es, tb],
        n_jobs=cfg.data_loader_n_jobs,
        fp16=cfg.fp16mode,
    )

    print("Goodbye")

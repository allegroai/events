# For dataset, go to: https://www.kaggle.com/msheriey/104-flowers-garden-of-eden
# Update INPUT_PATH and MODEL_PATH
import os
import tqdm  # temp
import glob
import warnings
from dataclasses import dataclass

import albumentations
import tez
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn import metrics, preprocessing
from tez.callbacks import EarlyStopping
from tez.datasets import ImageDataset
from torch.nn import functional as F

from clearml import Task, Dataset

#temporary - we are going to remove this later
DATASET_ID = '86895530658c47a4918bda4f0d92c3e8'
INPUT_PATH = "../../input/"
MODEL_PATH = "models/"
MODEL_NAME = os.path.basename(__file__)[:-3]
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
IMAGE_SIZE = 192
EPOCHS = 20

@dataclass
class FlowerTrainingConfig:
    # we are going to get rid of this
    input_path: str = "../../input/"
    # we are going to get rid of this
    model_path: str = "models/"
    # currently base name is fixed
    model_name: str = MODEL_NAME
    fp16mode: bool = True
    train_batch_size: int = 32
    valid_batch_size: int = 32
    # can only be 192, 224, 331, 512 if using the garden dataset
    image_size: int = 192 # this should be an enum!
    num_epochs: int = 20
    data_loader_n_jobs: int = 1
    efficient_model_type: str = "efficientnet-b0"
    early_stopping_patience: int = 3


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
    def __init__(self, num_classes, efficentnet_model: str ="efficientnet-b0"):
        super().__init__()

        self.effnet = EfficientNet.from_pretrained(efficentnet_model)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(1280, num_classes)
        self.step_report_every_n: int = 5

    def monitor_metrics(self, outputs, targets):
        outputs = torch.argmax(outputs, dim=1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy": accuracy}

    def fetch_optimizer(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
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


@dataclass
class AugConfig():
    transpose: float = 0.5
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.5
    shift_scale_rotate: float = 0.5
    hue_sat_val: float = 0.5
    hue_shift_limit: int = 20
    sat_shift_limit: int = 20
    val_shift_limit: int = 20
    random_bright_contrast: float = 0.5
    random_bright_mag: float = 0.1
    random_contrast_mag: float = 0.1


def get_train_augmentations(augment_config: AugConfig=None, train_dataset_id=None):
    augment_config = AugConfig() if augment_config is None else augment_config
    train_aug = albumentations.Compose(
        [
            albumentations.Transpose(p=augment_config.transpose),
            albumentations.HorizontalFlip(p=augment_config.horizontal_flip),
            albumentations.VerticalFlip(p=augment_config.vertical_flip),
            albumentations.ShiftScaleRotate(p=augment_config.shift_scale_rotate),
            albumentations.HueSaturationValue(
                hue_shift_limit=augment_config.hue_shift_limit,
                sat_shift_limit=augment_config.sat_shift_limit,
                val_shift_limit=augment_config.val_shift_limit,
                p=augment_config.hue_sat_val,
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-augment_config.random_bright_mag, augment_config.random_bright_mag),
                contrast_limit=(-augment_config.random_contrast_mag, augment_config.random_contrast_mag),
                p=augment_config.random_bright_contrast
            ),
            train_based_normalize(train_dataset_id)
        ],
        p=1.0,
    )


def get_valid_augmentations(augment_config: AugConfig, train_dataset_id=None):
    # augment_config left as an option if we will ever do TTA
    if augment_config is not None:
        return get_train_augmentations(augment_config=augment_config, train_dataset_id=train_dataset_id)
    return albumentations.Compose(
        [
            train_based_normalize(train_dataset_id)
        ],
        p=1.0,
    )


def get_normalization_info(train_dataset_id):
    """
    queries the dataset for the norm info config
    """
    raise NotImplementedError('need to implement data preprocessing first')
    return {}


def train_based_normalize(train_dataset_id=None):
    # the dataset will have a dict containing these
    default_values = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
    )
    values = default_values.copy() if train_dataset_id is None \
        else get_normalization_info(train_dataset_id)

    default_values.update({"p": 1.0})
    return albumentations.Normalize(**default_values)



if __name__ == "__main__":

    task = Task.init(project_name='tez Flower Detection',
                     task_name='remove all hardcoding',
                     # upload models to your free community storage
                     output_uri=True)

    task.connect(FlowerTrainingConfig, 'config')
    task.connect(AugConfig, 'augmentation_config')
    cfg = FlowerTrainingConfig()
    aug_cfg = AugConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn('GPU not available!, using CPU mode')
        warnings.filterwarnings("ignore", module='torch.cuda.amp.autocast')


    train_aug = get_train_augmentations(aug_cfg, train_dataset_id=None)
    valid_aug = get_valid_augmentations(None, train_dataset_id=None)

    dataset_folder = Dataset.get(dataset_id=DATASET_ID).get_local_copy()

    train_image_paths = glob.glob(
        os.path.join(
            dataset_folder, f"jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}", "train", "**", "*.jpeg"
        ),
        recursive=True,
    )

    valid_image_paths = glob.glob(
        os.path.join(
            dataset_folder, f"jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}", "val", "**", "*.jpeg"
        ),
        recursive=True,
    )

    train_targets = [x.split("/")[-2] for x in train_image_paths]
    valid_targets = [x.split("/")[-2] for x in valid_image_paths]

    lbl_enc = preprocessing.LabelEncoder()
    train_targets = lbl_enc.fit_transform(train_targets)
    valid_targets = lbl_enc.transform(valid_targets)

    task.set_model_label_enumeration({lbl: n for n, lbl in enumerate(lbl_enc.classes_)})

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

    model = FlowerModel(num_classes=len(lbl_enc.classes_))

    # temporary, model pathname here, and make sure directory exists
    model_path = os.path.join(MODEL_PATH, MODEL_NAME + ".bin")
    from pathlib import Path
    Path.mkdir(Path(MODEL_PATH), exist_ok=True)

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

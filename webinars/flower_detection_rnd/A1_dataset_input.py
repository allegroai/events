import argparse
from pathlib import Path
import tqdm  # temp
import warnings
from dataclasses import dataclass, asdict

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

MODEL_NAME = "FlowerDetector_{}"

@dataclass
class FlowerTrainingConfig:
    # need just the image size and the artifact generated when splitting
    # can only be 192, 224, 311, 512 if using the garden dataset
    image_size: int = 192
    dataset_metadata_id: str = "50a8767573a34b97820f82cc34daa34c"
    dataset_metadata_artifact_name: str = 'dataset_metadata'
    # just in case you need to access models locally
    model_path: str = "models/"
    fp16mode: bool = True
    train_batch_size: int = 32
    valid_batch_size: int = 32
    num_epochs: int = 20
    data_loader_n_jobs: int = 1
    early_stopping_patience: int = 3
    # relevant for executing remotely
    cloud_queue: str = 'colab'

@dataclass
class ModelConfig:
    model_name: str = MODEL_NAME
    efficient_model_type: str = "efficientnet-b0"
    p_dropout: float = 0.1
    linear_dim: int = 1280
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
                 adam_lr: float = 1e-4,
                 p_dropout: float = 0.1,
                 linear_dim: int = 1280,
                 ):
        super().__init__()

        self.effnet = EfficientNet.from_pretrained(efficientnet_model)
        self.dropout = nn.Dropout(p_dropout)
        self.out = nn.Linear(linear_dim, num_classes)
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


def get_train_augmentations(augment_config: AugConfig = None, norm_setting=None):
    augment_config = AugConfig() if augment_config is None else augment_config
    return albumentations.Compose(
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
                brightness_limit=
                (-augment_config.random_bright_mag, augment_config.random_bright_mag),
                contrast_limit=
                (-augment_config.random_contrast_mag, augment_config.random_contrast_mag),
                p=augment_config.random_bright_contrast
            ),
            train_based_normalize(norm_setting)
        ],
        p=1.0,
    )


def get_valid_augmentations(augment_config: AugConfig, norm_setting=None):
    # augment_config left as an option if we will ever do TTA
    if augment_config is not None:
        return get_train_augmentations(augment_config=augment_config, norm_setting=norm_setting)
    return albumentations.Compose([train_based_normalize(norm_setting)], p=1.0,)


def train_based_normalize(norm_setting=None):
    default_values = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0,
    )
    values = default_values.copy() if norm_setting is None \
        else norm_setting.copy()
    values.update({"p": 1.0})
    return albumentations.Normalize(**values)


if __name__ == "__main__":
    # force colab to get dataclasses
    Task.add_requirements('dataclasses')
    # override numpy version for colab
    Task.add_requirements('numpy', '1.19.5')
    # Track everything on ClearML Free
    task = Task.init(project_name='R|D?R&D! Webinar 01',
                     task_name='Full integration',
                     output_uri=True,  # auto save everything to Clearml Free
                     )

    # Need to run on cpu only?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn('GPU not available!, using CPU mode')
        warnings.filterwarnings("ignore", module='torch.cuda.amp.autocast')

    # configs
    cfg = FlowerTrainingConfig()
    aug_cfg = AugConfig()
    task.connect(cfg, 'config')
    task.connect(aug_cfg, 'augmentation_config')
    # default model config
    task.set_model_config(config_dict=asdict(ModelConfig()))
    model_params = ModelConfig(**task.get_model_config_dict())

    if cfg.cloud_queue is not None and len(cfg.cloud_queue):
        task.execute_remotely(cfg.cloud_queue)
    
    # get artifact
    datasets_metadata_task = Task.get_task(cfg.dataset_metadata_id)
    artifact = datasets_metadata_task.artifacts[cfg.dataset_metadata_artifact_name]
    metadata = artifact.get()

    dataset_metadata = metadata[str(cfg.image_size)]

    # get augmentations - including mean pixel value
    norm_info = dataset_metadata['norm_info']
    train_aug = get_train_augmentations(aug_cfg, norm_setting=norm_info)
    valid_aug = get_valid_augmentations(None, norm_setting=norm_info)
    # get dataset id's
    train_dataset_id = dataset_metadata.get('train',"")
    valid_dataset_id = dataset_metadata.get('val',"")
    if not len(train_dataset_id) or not len(valid_dataset_id):
        raise ValueError('Preprocess error: could not find'
                         f' datasets for image size {cfg.image_size}')
    # download dataset (cached!)
    try:
        train_dataset_folder = Dataset.get(dataset_id=train_dataset_id).get_local_copy()
        valid_dataset_folder = Dataset.get(dataset_id=valid_dataset_id).get_local_copy()
    except ValueError as ex:
        raise ValueError(f'Preprocess error for datasets for image size {cfg.image_size}\n{ex}')


    train_image_paths = [f for f in Path(train_dataset_folder).glob('**/*.jp*g')]
    valid_image_paths = [f for f in Path(valid_dataset_folder).glob('**/*.jp*g')]

    train_targets = [x.parts[-2] for x in train_image_paths]
    valid_targets = [x.parts[-2] for x in valid_image_paths]

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
        efficientnet_model=model_params.efficient_model_type,
        adam_lr=model_params.adam_lr,
        p_dropout=model_params.p_dropout,
        linear_dim=model_params.linear_dim,
    )

    # model pathname here, and make sure directory exists
    model_path = Path(cfg.model_path)
    model_path.mkdir(exist_ok=True)
    model_name = model_params.model_name.format(cfg.image_size) + ".bin"
    model_path = model_path / model_name

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

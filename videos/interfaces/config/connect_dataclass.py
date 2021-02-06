from dataclasses import dataclass
from clearml import Task

@dataclass
class TrainingConfig:
    # we are going to get rid of this
    input_path: str = "../../input/"
    # we are going to get rid of this
    model_path: str = "models/"
    # currently base name is fixed
    model_name: str = 'hello'
    train_batch_size: int = 32
    valid_batch_size: int = 32
    # can only be 192, 224, 331, 512 if using the garden dataset
    image_size: int = 192 # this should be an enum!
    num_epochs: int = 20
    data_loader_n_jobs: int = 1
    efficient_model_type: str = "efficientnet-b0"


if __name__ == '__main__':
    task = Task.init('reproducers','connect_dataclass')

    cfg_but_as_dict = task.connect(TrainingConfig, 'train_config')
    cfg = TrainingConfig(**cfg_but_as_dict)

    print(f"Image size {cfg.image_size}")

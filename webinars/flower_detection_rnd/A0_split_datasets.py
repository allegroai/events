from clearml import Task, Dataset
from dataclasses import dataclass


def extract_relevant_filenames(dataset_path, image_size):
    import os
    import glob
    train_image_paths = glob.glob(
        os.path.join(
            dataset_path, f"jpeg-{image_size}x{image_size}", "train", "**", "*.jpeg"
        ),
        recursive=True,
    )

    valid_image_paths = glob.glob(
        os.path.join(
            dataset_path, f"jpeg-{image_size}x{image_size}", "val", "**", "*.jpeg"
        ),
        recursive=True,
    )
    return train_image_paths, valid_image_paths

@dataclass
class DataSplitConf:
    input_dataset_id: str = "86895530658c47a4918bda4f0d92c3e8"
    image_size_values: set = (192, 224, 331, 512)
    dataset_name: str = "flower_detection"



if __name__ == '__main__':
    task = Task.init(
        project_name='R|D?R&D! Webinar 01',
        task_name='dataset split to sizes',
        output_uri=True,  # auto save everything to Clearml Free
        task_type=Task.TaskTypes.data_processing
    )

    cfg = DataSplitConf()
    task.connect(cfg,'dataset split config')

    input_dataset = Dataset.get(dataset_id=cfg.input_dataset_id)
    input_dataset_folder = input_dataset.get_local_copy()

    for image_size in cfg.image_size_values:
        # if dataset exists skip creatino

        test_if_exists = Dataset.get()

        # test that there are files train/val

        dataset_name = f"{cfg.dataset_name}_{image_size}x{image_size}_"

        new_dataset_train = Dataset.create(
            dataset_name=dataset_name+"train",
            dataset_project=task.project,
            parent_datasets=[cfg.input_dataset_id]
        )
        new_dataset_val = Dataset.create(
            dataset_name=dataset_name+"val",
            dataset_project=task.project,
            parent_datasets=[cfg.input_dataset_id]
        )



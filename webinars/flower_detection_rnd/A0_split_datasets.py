from clearml import Task, Dataset
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataSplitConf:
    # already exists - "someone already uploaded via cli"
    input_dataset_id: str = "86895530658c47a4918bda4f0d92c3e8"
    image_size_values: set = (192, 224, 331, 512)
    dataset_name: str = "flower_detection"
    folder_name_prefix = "jpeg-"
    delete_target_new_dataset_if_exists = True


def extract_relevant_filenames(dataset_path, im_size, folder_name_pattern=None):
    # original dataset glob patterns
    import os
    import glob
    folder_name_pattern = f"jpeg-{im_size}x{im_size}" if\
        folder_name_pattern is None else folder_name_pattern
    train_image_paths = glob.glob(
        os.path.join(
            dataset_path, folder_name_pattern, "train", "**", "*.jpeg"
        ),
        recursive=True,
    )

    valid_image_paths = glob.glob(
        os.path.join(
            dataset_path, folder_name_pattern, "val", "**", "*.jpeg"
        ),
        recursive=True,
    )
    return train_image_paths, valid_image_paths


if __name__ == '__main__':
    project_name = 'R|D?R&D! Webinar 01'
    task = Task.init(
        project_name=project_name,
        task_name='Orig dataset split to sizes',
        output_uri=True,  # auto save everything to ClearML Free
        task_type=Task.TaskTypes.data_processing
    )

    cfg = DataSplitConf()
    task.connect(cfg, 'dataset split config')

    # Uncomment to force run remotely
    # task.execute_remotely(queue_name='colab')

    input_dataset = Dataset.get(dataset_id=cfg.input_dataset_id)
    input_dataset_folder = input_dataset.get_local_copy()

    for image_size in cfg.image_size_values:
        # if dataset exists skip creating
        dataset_name = f"{cfg.dataset_name}_{image_size}x{image_size}_"

        try:
            test_if_exists = Dataset.list_datasets(
                dataset_project=project_name,
                partial_name=dataset_name,
                only_completed=False,
            )

            if len(test_if_exists):
                if cfg.delete_target_new_dataset_if_exists:
                    print(f'found datasets in the project with image size {image_size}')
                    for t in test_if_exists:
                        try:
                            Dataset.delete(t['id'])
                        except ValueError:
                            print(f'could not delete dataset - has children?')
                else:
                    continue

        except ValueError:
            print(f'Did not find {dataset_name}, creating!')

        train_files, validation_files = \
            extract_relevant_filenames(input_dataset_folder, image_size)

        # test that there are files train/val
        if not len(train_files):
            raise ValueError(f'No files found for image size {image_size} on folder {input_dataset_folder}')
        if not len(validation_files):
            raise NotImplementedError('No validation files - option for train only not supported')

        # train
        path_for_image_size = Path(input_dataset_folder) / (cfg.folder_name_prefix + f"{image_size}x{image_size}")
        for stage in ['train', 'val']:  # TODO 'test'
            file_folder = path_for_image_size / stage

            new_dataset = Dataset.create(
                dataset_name=dataset_name+stage,
                dataset_project=project_name,
                parent_datasets=[cfg.input_dataset_id]
            )

            new_dataset.add_files(file_folder, wildcard='*.jp*g')
            new_dataset.upload(show_progress=True, verbose=True)
            new_dataset.finalize()
            new_dataset.publish()



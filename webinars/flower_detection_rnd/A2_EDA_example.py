from dataclasses import dataclass
from pathlib import Path
from random import sample

import pandas as pd
import plotly.express as px

from clearml import Task, Dataset


@dataclass
class EDAConf:
    dataset_metadata_id: str = "5b3da654bb1c4b9c81acfcf4d75063ea"
    dataset_metadata_artifact_name: str = 'dataset_metadata'
    # put graphics options here
    ...


if __name__ == '__main__':
    Task.add_requirements('dataclasses')
    Task.add_requirements('plotly')
    # force colab to get dataclasses
    Task.add_requirements('dataclasses','0.4')
    # override versions for colab
    Task.add_requirements('pandas', '1.1.5')
    Task.add_requirements('numpy', '1.19.5')
    # Track everything on ClearML Free
    task = Task.init(project_name='R|D?R&D! Webinar 01 - demo',
                     task_name='EDA example',
                     output_uri=True,  # auto save everything to Clearml Free
                     )

    cfg = EDAConf()
    task.connect(cfg, 'EDA Config')

    task.execute_remotely('colab')

    datasets_metadata_task = Task.get_task(cfg.dataset_metadata_id)
    artifact = datasets_metadata_task.artifacts[cfg.dataset_metadata_artifact_name]
    metadata = artifact.get()

    for image_size, meta in metadata.items():
        print(f'processing {image_size}...')
        # get augmentations - including mean pixel value
        norm_info = meta['norm_info']
        # get dataset id's
        train_dataset_id = meta.get('train', "")
        valid_dataset_id = meta.get('val', "")
        if not len(train_dataset_id) or not len(valid_dataset_id):
            raise ValueError('Preprocess error: could not find'
                             f' datasets for image size {image_size}')
        # download dataset (cached!)
        try:
            train_dataset_folder = Dataset.get(dataset_id=train_dataset_id).get_local_copy()
            valid_dataset_folder = Dataset.get(dataset_id=valid_dataset_id).get_local_copy()
        except ValueError as ex:
            raise ValueError(f'Preprocess error for datasets for image size {image_size}\n{ex}')

        train_image_paths = [f for f in Path(train_dataset_folder).glob('**/*.jp*g')]
        valid_image_paths = [f for f in Path(valid_dataset_folder).glob('**/*.jp*g')]

        # show some images

        some_images = sample(train_image_paths, 5)
        for im in some_images:
            task.logger.report_image(f'example images {image_size}x{image_size}', 'train', 0, str(im))
        some_images_val = sample(valid_image_paths, 5)
        for im in some_images_val:
            task.logger.report_image(f'example images {image_size}x{image_size}', 'val', 0, str(im))

        train_targets = [x.parts[-2] for x in train_image_paths]
        valid_targets = [x.parts[-2] for x in valid_image_paths]

        labels = set(train_targets)
        labels_val = set(valid_targets)
        labels.update(labels_val)

        train_df = pd.DataFrame(train_targets, columns=['label'])
        train_df["stage"] = "train"
        val_df = pd.DataFrame(valid_targets, columns=['label'])
        val_df["stage"] = "val"

        label_df = pd.concat([train_df, val_df], axis=0)

        fig = px.histogram(label_df,
                           x="label",
                           title=f'Flower Distribution in dataset {image_size}x{image_size}',
                           labels={'label': 'Flower Type'},  # can specify one label per df column
                           color="stage",
                           barmode="overlay",
                           nbins=len(labels)
                           )

        task.logger.report_plotly(title='Label density',
                                  series=f'{image_size}x{image_size}',
                                  iteration=0,
                                  figure=fig,
                                  )

from clearml import Task, Dataset
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import plotly.express as px


@dataclass
class EDAConf:
    dataset_metadata_id: str = "466f3798cb0041a3801bd904e7cf3631"
    dataset_metadata_artifact_name: str = 'dataset_metadata'
    # put graphics options here



if __name__ == '__main__':
    Task.add_requirements('dataclasses')
    Task.add_requirements('pandas')
    Task.add_requirements('plotly')
    # force colab to get dataclasses
    Task.add_requirements('dataclasses')
    # override numpy version for colab
    Task.add_requirements('numpy', '1.19.5')
    # Track everything on ClearML Free
    task = Task.init(project_name='R|D?R&D! Webinar 01',
                     task_name='Full integration',
                     output_uri=True,  # auto save everything to Clearml Free
                     )

    cfg = EDAConf()
    task.connect(cfg, 'EDA Config')


    datasets_metadata_task = Task.get_task(cfg.dataset_metadata_id)
    artifact = datasets_metadata_task.artifacts[cfg.dataset_metadata_artifact_name]
    metadata = artifact.get()
    dataset_metadata = metadata[str(cfg.image_size)]

    for image_size in dataset_metadata.keys():
        # get augmentations - including mean pixel value
        norm_info = dataset_metadata['norm_info']
        # get dataset id's
        train_dataset_id = dataset_metadata.get('train', "")
        valid_dataset_id = dataset_metadata.get('val', "")
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

        train_targets = [x.parts[-2] for x in train_image_paths]
        valid_targets = [x.parts[-2] for x in valid_image_paths]




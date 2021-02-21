from clearml import Task, Dataset
from dataclasses import dataclass
from plotly import graph_objs as go

@dataclass
class EDAConf:
    # need just the image size and the artifact generated when splitting
    # can only be 192, 224, 311, 512 if using the garden dataset
    image_size: int = 192
    dataset_metadata_id: str = "677645c9afd843ecb40f77ca119eb85a"
    dataset_metadata_artifact_name: str = 'dataset_metadata'



if __name__ == '__main__':
    Task.add_requirements('dataclasses')
    Task.add_requirements('plotly')

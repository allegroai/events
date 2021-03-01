from clearml import Task
from clearml.automation.controller import PipelineController
from dataclasses import dataclass


@dataclass
class PipeConfig:
    input_dataset_id: str = "86895530658c47a4918bda4f0d92c3e8"
    image_size_values: set = (192, 224, 311, 512)


if __name__ == "__main__":
    # force colab to get dataclasses
    Task.add_requirements('dataclasses')
    # Track everything on ClearML Free
    base_project_name = 'R|D?R&D! Webinar 01'
    task = Task.init(
        project_name=base_project_name + "_automations",
        task_name='Pipeline example',
        output_uri=True,  # auto save everything to Clearml Free
    )

    pipe_cfg = PipeConfig()
    task.connect(pipe_cfg, 'pipeline config')
    # possibly control everything from here:
    # train_cfg = FlowerTrainingConfig()
    # aug_cfg = AugConfig()
    # task.connect(train_cfg, 'pipeline config')
    # task.connect(aug_cfg, 'augmentation config')

    # TODO: build a parameter override for training tasks

    pipe = PipelineController(default_execution_queue='colab')

    # step 1 - split data
    pipe.add_step(name='split_dataset',
                  base_task_id='5b3da654bb1c4b9c81acfcf4d75063ea',
                  parameter_override={
                      'dataset split config/image_size_values': pipe_cfg.image_size_values,
                      'dataset split config/input_dataset_id': pipe_cfg.input_dataset_id,
                  })
    pipe.add_step(name='EDA',
                  base_task_id='f0b86d8e288143019cea0c41898133c7',
                  execution_queue='laptop',  # don't need gpu for this one ;)
                  parents=['split_dataset', ],
                  parameter_override={
                      'EDA Config/dataset_metadata_id': '${split_dataset.id}',
                  })
    for image_size in sorted(pipe_cfg.image_size_values):
        pipe.add_step(name=f'train_{image_size}x{image_size}',
                      base_task_id='cf51631735c9426cae3e09879123f269',
                      parents=['split_dataset', ],
                      parameter_override={'config/image_size': image_size,
                                          'config/dataset_metadata_id': '${split_dataset.id}',
                      })

    # Starting the pipeline (in the background)
    print('starting...')
    pipe.start()
    # Wait until pipeline terminates
    pipe.wait()
    # cleanup everything
    pipe.stop()

    print('done')

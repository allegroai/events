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
                  base_task_id='466f3798cb0041a3801bd904e7cf3631',
                  parameter_override={
                      'dataset split config/image_size_values': pipe_cfg.image_size_values,
                      'dataset split config/input_dataset_id': pipe_cfg.input_dataset_id,
                  })
    pipe.add_step(name='EDA',
                  base_task_id='d547006792784ea0933fa93e4afb822d',
                  execution_queue='laptop',  # don't need gpu for this one ;)
                  parents=['split_dataset', ],
                  parameter_override={
                      'EDA Config/dataset_metadata_id': '${split_dataset.id}',
                  })
    for image_size in sorted(pipe_cfg.image_size_values):
        pipe.add_step(name=f'train_{image_size}x{image_size}',
                      base_task_id='508da62a7132454ca185ae7fe940d3b5',
                      execution_queue='laptop',  # don't need gpu for this one ;)
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

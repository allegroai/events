from simple_parsing import ArgumentParser
from dataclasses import dataclass
from typing import *
from clearml import Task
from fastcore.utils import dict2obj


@dataclass
class TaskHyperParameters():
    """
    HyperParameters for a task-specific model
    """
    # name of the task
    name: str
    # number of dense layers
    num_layers: int = 1
    # units per layer
    num_units: int = 8
    # activation function
    activation: str = "tanh"
    # wether or not to use batch normalization after each dense layer
    use_batchnorm: bool = False
    # wether or not to use dropout after each dense layer
    use_dropout: bool = True
    # the dropout rate
    dropout_rate: float = 0.1
    # wether or not image features should be used as input
    use_image_features: bool = True
    # wether or not 'likes' features should be used as input
    use_likes: bool = True
    # L1 regularization coefficient
    l1_reg: float = 0.005
    # L2 regularization coefficient
    l2_reg: float = 0.005
    # Wether or not a task-specific Embedding layer should be used on the 'likes' features.
    # When set to 'True', it is expected that there no shared embedding is used.
    embed_likes: bool = False

@dataclass
class HyperParameters():
    """Hyperparameters of our model."""
    # the batch size
    batch_size: int = 128
    # Which optimizer to use during training.
    optimizer: str = "sgd"
    # Learning Rate
    learning_rate: float = 0.001

    # number of individual 'pages' that were kept during preprocessing of the 'likes'.
    # This corresponds to the number of entries in the multi-hot like vector.
    num_like_pages: int = 10_000

    gender_loss_weight: float   = 1.0
    age_loss_weight: float      = 1.0

    num_text_features: ClassVar[int] = 91
    num_image_features: ClassVar[int] = 65

    max_number_of_likes: int = 2000
    embedding_dim: int = 8

    shared_likes_embedding: bool = True

    # Wether or not to use Rémi's better kept like pages
    use_custom_likes: bool = True

    # Gender model settings:
    gender: TaskHyperParameters = TaskHyperParameters(
        "gender",
        num_layers=1,
        num_units=32,
        use_batchnorm=False,
        use_dropout=True,
        dropout_rate=0.1,
        use_image_features=True,
        use_likes=True,
    )

    # Age Group Model settings:
    age_group: TaskHyperParameters = TaskHyperParameters(
        "age_group",
        num_layers=2,
        num_units=64,
        use_batchnorm=False,
        use_dropout=True,
        dropout_rate=0.1,
        use_image_features=True,
        use_likes=True,
    )

    # Personality Model(s) settings:
    personality: TaskHyperParameters = TaskHyperParameters(
        "personality",
        num_layers=1,
        num_units=8,
        use_batchnorm=False,
        use_dropout=True,
        dropout_rate=0.1,
        use_image_features=False,
        use_likes=False,
    )


@dataclass
class MyFeatureConfig():
    """Config for my new feature"""
    # the word size
    word_size: int = 128


parser = ArgumentParser()
parser.add_arguments(HyperParameters, dest="hparams")
args = parser.parse_args()

if __name__ == '__main__':
    task = Task.init(project_name='simple_parse',
                     task_name='nested using simple-parsing',
                     auto_connect_arg_parser=False,
                     reuse_last_task_id=False)

    task.connect(parser,name='command line')
    extra_args = task.connect(MyFeatureConfig, name='my_feature1')

    hparams: HyperParameters = args.hparams
    my_feature_conf : MyFeatureConfig
    print(hparams)
    task.close()
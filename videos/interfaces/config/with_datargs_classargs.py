from clearml import Task

import typing, logging
from datargs import argsclass, arg, parse

@argsclass(description="dataset config")
class DatasetConfig:
    dataset: str = arg(positional=True, help="package to install")

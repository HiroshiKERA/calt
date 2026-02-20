from importlib.metadata import version as _version

__version__ = _version("calt-x")

from . import dataset  # ensure calt.dataset subpackage is loadable
from . import io
from . import models
from . import trainer
from .dataset import DatasetPipeline
from .dataset.sympy.dataset_generator import DatasetGenerator
from .dataset.sympy.utils.polynomial_sampler import PolynomialSampler
from .dataset.utils.dataset_writer import DatasetWriter
from .dataset.utils.statistics_calculator import (
    BaseStatisticsCalculator,
)
from .io import IOPipeline
from .io.base import StandardDataCollator, StandardDataset
from .io.preprocessor import (
    AbstractPreProcessor,
    NumberPolicy,
    PreProcessorChain,
    UnifiedLexer,
)
from .io.tokenizer import get_tokenizer
from .models import ModelPipeline
from .trainer import TrainerPipeline
from .trainer.trainer import Trainer
from .trainer.utils import count_cuda_devices

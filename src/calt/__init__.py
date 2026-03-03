from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("calt-x")
except PackageNotFoundError:
    # Source-only environments (e.g., Kaggle dataset bundles) may not include wheel metadata.
    __version__ = "0+unknown"

from . import (
    dataset,  # ensure calt.dataset subpackage is loadable
    io,
    models,
    remote,
    trainer,
)
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
from .remote import (
    RemoteJobError,
    RemoteRunConfig,
    RemoteRunResult,
    run_remote_job,
)
from .trainer import TrainerPipeline
from .trainer.trainer import Trainer
from .trainer.utils import count_cuda_devices

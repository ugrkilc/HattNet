"""
HattNet — A Multi-Stream Deep Learning Architecture
for Arabic Calligraphy Style Classification.
"""

from .model import HattNet
from .train import run_mode, train_fold
from .config import AVAILABLE_MODES, SEED
from .utils import set_seed

__version__ = "1.0.0"
__all__ = [
    "HattNet",
    "run_mode",
    "train_fold",
    "AVAILABLE_MODES",
    "SEED",
    "set_seed",
]

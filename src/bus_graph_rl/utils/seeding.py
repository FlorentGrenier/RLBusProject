from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

def seed_everything(seed: Optional[int] = None) -> int:
    """Seed python, numpy (and env var) for reproducibility.

    Returns the seed used (generated if None).
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed

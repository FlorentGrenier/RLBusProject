from __future__ import annotations

import random

import numpy as np

from bus_graph_rl.utils.seeding import seed_everything


def test_seed_everything_is_reproducible():
    seed_everything(7)
    first_random = random.random()
    first_numpy = np.random.rand()

    seed_everything(7)
    second_random = random.random()
    second_numpy = np.random.rand()

    assert first_random == second_random
    assert first_numpy == second_numpy

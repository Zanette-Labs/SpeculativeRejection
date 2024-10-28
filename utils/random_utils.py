import numpy as np
import time


def reset_numpy_seed() -> None:
    current_ms_time = int(time.time() * 1000) % (2 ** 32)
    np.random.seed(current_ms_time)

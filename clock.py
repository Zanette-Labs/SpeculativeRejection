import numpy as np
from pprint import pprint
from time import sleep, time


class Clock:
    def __init__(self) -> None:
        self.running_time = 0.0
        self.time_chunks: list[tuple[str, float]] = []

    def start(self) -> None:
        self.start_time = time()

    def stop(self, chunk_name: str = "") -> None:
        if self.start_time is None:
            raise Exception("Attempted to stop clock that was not running.")
        elapsed_time = time() - self.start_time
        self.running_time += elapsed_time
        self.start_time = None
        self.time_chunks.append((chunk_name, elapsed_time))

    def get_time(self) -> float:
        return self.running_time

    def get_chunks(self) -> list[tuple[str, float]]:
        return self.time_chunks

    def reset(self) -> None:
        self.running_time = 0.0
        self.time_chunks = []


def test_clock() -> None:
    clock = Clock()
    clock.start()
    sleep(0.3)
    clock.stop("generation pass")
    clock.start()
    sleep(0.1)
    clock.stop("reward pass")

    elapsed_time = clock.get_time()
    chunks = clock.get_chunks()
    assert np.isclose(elapsed_time, 0.4, atol=0.05)
    assert len(chunks) == 2
    assert chunks[0][0] == "generation pass"
    assert chunks[1][0] == "reward pass"
    assert np.isclose(chunks[0][1], 0.3, atol=0.05)
    assert np.isclose(chunks[1][1], 0.1, atol=0.05)

    clock.reset()
    assert clock.get_time() == 0.0
    assert clock.get_chunks() == []


if __name__ == "__main__":
    test_clock()

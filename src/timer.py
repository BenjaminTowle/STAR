"""
This class is used for debugging purposes. It is used to compare the performance of the different parts of the program to identify
where speed-ups can be obtained.
"""
from contextlib import contextmanager
from time import perf_counter


class Timer:
    """
    The class should maintain a dictionary of the different parts of the program that are being timed. This can
    be updated cumulatively during each loop of the program. Finally, it should be able to print out the results.
    """
    def __init__(self) -> None:
        self.lap_times = {}
        self.last = perf_counter()

    def start_lap(self):
        self.last = perf_counter()

    def stop_lap(self, name: str):
        time = perf_counter() - self.last
        if name not in self.lap_times:
            self.lap_times[name] = 0
        self.lap_times[name] += time
        self.last = perf_counter()

    @contextmanager
    def lap(self, name: str):
        self.start_lap()
        yield
        self.stop_lap(name)

    def print(self, logger, num_samples: int = 1):
        # Print results showing both absolute time for each item and proportion of total time
        total_time = sum(self.lap_times.values())
        for name, time in self.lap_times.items():
            logger.info(f"{name}: {time/num_samples:.3f} ({time/total_time:.3%})")
        # Log total time
        logger.info(f"Latency per sample: {total_time/num_samples:.3f}")


timer = Timer()

import time


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.start = None
        self.end = None
        self.interval = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"Elapsed time {self.name}: {round(self.interval, 4)}s")

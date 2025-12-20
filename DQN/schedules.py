class LinearEpsilonSchedule:
    def __init__(self, start: float, end: float, decay_steps: int):
        self.start = float(start)
        self.end = float(end)
        self.decay_steps = int(decay_steps)

    def __call__(self, step: int) -> float:
        if self.decay_steps <= 0:
            return self.end
        frac = min(1.0, step / float(self.decay_steps))
        return self.start + frac * (self.end - self.start)

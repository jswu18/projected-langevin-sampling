import math


class LogBisectionSearch:
    def __init__(self, lower: float, upper: float, soft_update: bool = False):
        self.lower = lower
        self.upper = upper
        self.soft_update = soft_update

    @property
    def current(self) -> float:
        return math.exp((math.log(self.lower) + math.log(self.upper)) / 2)

    def update_upper(self) -> None:
        if self.soft_update:
            self.upper = math.exp((math.log(self.upper) + math.log(self.current)) / 2)
        else:
            self.upper = self.current

    def update_lower(self) -> None:
        if self.soft_update:
            self.lower = math.exp((math.log(self.lower) + math.log(self.current)) / 2)
        else:
            self.lower = self.current

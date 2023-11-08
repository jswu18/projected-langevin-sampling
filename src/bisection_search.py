import math


class LogBisectionSearch:
    """
    Bisection search in log space where
        current = exp((log(lower) + log(upper))/2)/2

    Lower and upper bounds are updated with the current value, splitting the (log) search space in half each update.
    Additionally, the option for a soft update would update the bound following
        lower = exp((log(lower) + log(current))/2)/2
        or
        upper = exp((log(upper) + log(current))/2)/2
    the (log) midpoint between the current and bound.

    """

    def __init__(self, lower: float, upper: float, soft_update: bool = False):
        """
        Constructor for Bisection search.

        :param lower: lower bound for search
        :param upper: upper bound for search
        :param soft_update: update bounds with the (log) midpoint between the current and current bound
        """
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

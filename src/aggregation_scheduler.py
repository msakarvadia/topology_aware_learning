""" Define Aggregation coefficient (aka. softmax tempterature) scheduler"""

import math


class ScheduledOptim:
    """A simple class for anealing the aggregation coefficient"""

    def __init__(self, softmax_coeff, n_warmup_steps):

        self.softmax_coeff = softmax_coeff
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def _get_softmax_scale(self):
        # d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
        # return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def get_softmax_coeff(self):
        self.softmax_coeff -= 1 * self._get_softmax_scale()
        return self.softmax_coeff

    def step(self, round_idx=None):
        """Step could be called after every round."""
        self.n_steps += 1


class CosineAnnealingWarmRestarts:

    def __init__(
        self,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_round: int = -1,
        softmax_coeff: float = 100,
    ):  # noqa: D107
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if not isinstance(eta_min, (float, int)):
            raise ValueError(
                f"Expected float or int eta_min, but got {eta_min} of type {type(eta_min)}"
            )
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_round
        self.softmax_coeff = softmax_coeff

    def get_softmax_coeff(self):
        return (
            self.eta_min
            + (self.softmax_coeff - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
        )

    def step(self, round_idx=None):
        """Step could be called after every round."""
        if round_idx is None and self.last_round < 0:
            round_idx = 0

        if round_idx is None:
            round_idx = self.last_round + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if round_idx < 0:
                raise ValueError(f"Expected non-negative round, but got {round_idx}")
            if round_idx >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = round_idx % self.T_0
                else:
                    n = int(
                        math.log(
                            (round_idx / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = round_idx - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = round_idx
        self.last_round = math.floor(round_idx)


class FakeScheduler:
    """Decays softmax coefficient gamma every round."""

    def __init__(
        self,
        softmax_coeff: float = 100,
    ):  # noqa: D107
        self.softmax_coeff = softmax_coeff

    def get_softmax_coeff(self):
        return self.softmax_coeff

    def step(self, round_idx=None):
        """Compute the learning rate of each parameter group."""
        return


class ExponentialScheduler:
    """Decays softmax coefficient gamma every round."""

    def __init__(
        self,
        gamma: float,
        eta_min: float = 1,  # min value for softmax coeff
        softmax_coeff: float = 100,
    ):  # noqa: D107
        self.gamma = gamma
        self.softmax_coeff = softmax_coeff
        self.eta_min = eta_min

    def get_softmax_coeff(self):
        if self.softmax_coeff < self.eta_min:
            return self.eta_min
        return self.softmax_coeff

    def step(self, round_idx=None):
        """Compute the learning rate of each parameter group."""

        self.softmax_coeff *= self.gamma
        return


"""
if __name__ == "__main__":
    softmax_coeff_scheduler = ScheduledOptim(softmax_coeff=100, n_warmup_steps=20)

    CA_schedule = CosineAnnealingWarmRestarts(
        T_0=20,
        T_mult=1,
        eta_min=1,
        last_round=-1,
        softmax_coeff=100,
    )

    exponent_schedule = ExponentialScheduler(
        softmax_coeff=100,
        gamma=0.95,
    )
    for i in range(100):
        softmax_coeff_scheduler.step(round_idx=i)
        print(softmax_coeff_scheduler.get_softmax_coeff())

        CA_schedule.step(round_idx=i)
        print(CA_schedule.get_softmax_coeff())

        exponent_schedule.step(round_idx=i)
        print(exponent_schedule.get_softmax_coeff())
"""

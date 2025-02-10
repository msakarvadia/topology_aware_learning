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

    def update_softmax_coeff(self):
        """Softmax coeff scheduling per step"""

        self.n_steps += 1
        self.softmax_coeff -= 1 * self._get_softmax_scale()


class CosineAnnealingWarmRestarts:

    def __init__(
        self,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
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
        self.T_cur = last_epoch
        self.softmax_coeff = softmax_coeff

    def get_softmax_coeff(self):

        return (
            self.eta_min
            + (self.softmax_coeff - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
        )

    def step(self, epoch=None):
        """Step could be called after every batch update.

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)

        This function can be called in an interleaved way.

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)


if __name__ == "__main__":
    softmax_coeff_scheduler = ScheduledOptim(softmax_coeff=50, n_warmup_steps=20)

    CA_schedule = CosineAnnealingWarmRestarts(
        T_0=20,
        T_mult=1,
        eta_min=1,
        last_epoch=-1,
        softmax_coeff=100,
    )
    for i in range(100):
        # print(softmax_coeff_scheduler.softmax_coeff)
        # softmax_coeff_scheduler.update_softmax_coeff()

        CA_schedule.step(epoch=i)
        print(CA_schedule.get_softmax_coeff())

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_iters,
        max_iters,
        warmup_start_factor=1e-6,
        min_lr=1e-6,
        last_epoch=-1,
    ):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.warmup_start_factor = warmup_start_factor

        if isinstance(min_lr, list):
            self.min_lrs = min_lr
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        # --------------- Warmup Phase ---------------
        if step < self.warmup_iters:
            warmup_factor = self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * (step / self.warmup_iters)

            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # --------------- Cosine Decay Phase ---------------
        progress = (step - self.warmup_iters) / max(
            1, self.max_iters - self.warmup_iters
        )
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.min_lrs[i] + (base_lr - self.min_lrs[i]) * cosine_factor
            for i, base_lr in enumerate(self.base_lrs)
        ]

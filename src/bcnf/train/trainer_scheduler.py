import torch

class TrainerScheduler():
    def __init__(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        self.lr_scheduler_kwargs = {
            "mode": "min",
            "factor": 0.5,
            "patience": 250,
            "threshold_mode": "abs",
            "threshold": 1e-1,
        }

    def _create_scheduler(self,
                          optimizer: torch.optim.Optimizer):
        return self.scheduler(optimizer, **self.lr_scheduler_kwargs)
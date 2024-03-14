

class TrainerLossHandler():
    def __init__(self,
                 val_loss_alpha: float,
                 val_loss_patience: int | None = None,
                 val_loss_tolerance_mode: str = "rel",
                 val_loss_tolerance: float = 1e-3) -> None:
        if val_loss_tolerance_mode not in ["rel", "abs"]:
            raise ValueError("val_loss_tolerance_mode must be either 'rel' or 'abs'")
        else:
            self.val_loss_tolerance_mode = val_loss_tolerance_mode

        self.best_val_loss = float('inf')
        self.best_val_epoch = 0
        self.val_loss_rolling_avg = None
        self.val_loss_alpha = val_loss_alpha
        self.val_loss_patience = val_loss_patience
        self.val_loss_tolerance = val_loss_tolerance

        self.loss_history: dict = {
            "train": [],
            "val": [],
            "lr": [],
            "early_stop_counter": [],
            "time": []
        }

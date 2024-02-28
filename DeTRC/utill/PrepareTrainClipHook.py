from mmcv.runner import Hook


class PrepareTrainClipHook(Hook):
    """
    control the dataloader prepare train clip for every epoch.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader

    def after_train_epoch(self, runner):
        """Called after every training epoch to change the train clip."""
        if getattr(self.dataloader.dataset, "temporal_aug_type", None) is None:
            return
        if self.dataloader.dataset.temporal_aug_type == "part_exchange_and_accelerate":
            self.dataloader.dataset.prepare_train_clip()

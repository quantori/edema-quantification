from pytorch_lightning.callbacks import TQDMProgressBar


class PNetProgressBar(TQDMProgressBar):
    def __init__(self, process_position: int = 1):
        super().__init__(process_position=process_position)

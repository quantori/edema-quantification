from torch import nn


class LastLayer(nn.Linear):
    def warm(self) -> None:
        self.requires_grad_(True)

    def joint(self) -> None:
        self.requires_grad_(True)

    def last(self) -> None:
        self.requires_grad_(True)

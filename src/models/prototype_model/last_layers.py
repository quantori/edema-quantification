from torch import nn


class ILastLayers(nn.Linear):
    """Abstract class for the layers of the edema model."""

    def warm(self) -> None:
        """Sets grad policy for the warm training stage."""
        raise NotImplementedError

    def joint(self) -> None:
        """Sets grad policy for the joint training stage."""
        raise NotImplementedError

    def last(self) -> None:
        """Sets grad policy for the last training stage."""
        raise NotImplementedError


class LastLayers(ILastLayers):
    """Last layers."""

    def warm(self) -> None:
        self.requires_grad_(True)

    def joint(self) -> None:
        self.requires_grad_(True)

    def last(self) -> None:
        self.requires_grad_(True)

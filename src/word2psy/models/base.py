"""Base model interface for text feature extraction."""

from abc import ABC, abstractmethod

import torch

from word2psy.exceptions import DeviceError


def _get_default_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BaseModel(ABC):
    """Common interface for text feature extraction models.

    Subclasses must set the class attributes ``name`` and ``level``,
    and implement ``load()`` and ``predict()``.
    """

    name: str = ""
    level: str = ""  # "word" or "chunk"

    def __init__(self, device: str | None = None):
        if device:
            try:
                self.device = torch.device(device)
            except RuntimeError as e:
                raise DeviceError(device, str(e)) from e
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise DeviceError(device, "CUDA is not available on this system")
            if self.device.type == "mps" and not torch.backends.mps.is_available():
                raise DeviceError(device, "MPS is not available on this system")
        else:
            self.device = _get_default_device()
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights and any required resources."""

    @abstractmethod
    def predict(self, text: str) -> dict[str, float]:
        """Return named scores for a single text input.

        For word-level models, ``text`` is a single word.
        For chunk-level models, ``text`` is a text chunk (sentence, paragraph, etc.).
        """

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Return named scores for a list of text inputs.

        Default implementation loops over ``predict()``; subclasses can override
        with a more efficient batched version.
        """
        return [self.predict(t) for t in texts]

    def to(self, device: torch.device) -> "BaseModel":
        """Move the underlying torch model to the given device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self

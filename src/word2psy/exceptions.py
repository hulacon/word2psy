"""Custom exceptions for word2psy."""


class Word2PsyError(Exception):
    """Base exception for all word2psy errors."""


class TextLoadError(Word2PsyError):
    """Raised when a text file cannot be loaded or decoded."""

    def __init__(self, path, reason=None):
        self.path = path
        msg = f"Failed to load text: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class ModelLoadError(Word2PsyError):
    """Raised when a model fails to load weights or initialize."""

    def __init__(self, model_name, reason=None):
        self.model_name = model_name
        msg = f"Failed to load model '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class DeviceError(Word2PsyError):
    """Raised when a requested compute device is unavailable or invalid."""

    def __init__(self, device, reason=None):
        self.device = device
        msg = f"Invalid or unavailable device '{device}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class InferenceError(Word2PsyError):
    """Raised when model inference fails."""

    def __init__(self, model_name, reason=None):
        self.model_name = model_name
        msg = f"Inference failed for model '{model_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class NormDataError(Word2PsyError):
    """Raised when norm data cannot be downloaded or parsed."""

    def __init__(self, norm_name, reason=None):
        self.norm_name = norm_name
        msg = f"Failed to load norm data '{norm_name}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)

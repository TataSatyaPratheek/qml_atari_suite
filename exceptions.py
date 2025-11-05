# Custom exceptions for the QML Atari suite

class ExperimentTimeoutError(Exception):
    """Raised when an experiment exceeds the timeout limit."""
    pass
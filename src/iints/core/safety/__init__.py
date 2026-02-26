from .config import SafetyConfig
from .input_validator import InputValidator

__all__ = ["SafetyConfig", "SafetySupervisor", "InputValidator"]


def __getattr__(name: str):
    if name == "SafetySupervisor":
        from .supervisor import SafetySupervisor

        return SafetySupervisor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

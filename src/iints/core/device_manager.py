import torch
import sys

class DeviceManager:
    """
    Manages hardware device detection for cross-platform compatibility.
    Detects MPS (Apple Silicon), CUDA (NVIDIA GPUs), or falls back to CPU.
    """
    def __init__(self):
        self._device = self._detect_device()

    def _detect_device(self):
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            print("Detected Apple Silicon (MPS) for accelerated computing.")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print(f"Detected NVIDIA GPU (CUDA) with {torch.cuda.device_count()} device(s) for accelerated computing.")
            return torch.device("cuda")
        else:
            print("No GPU detected. Falling back to CPU for computing.")
            return torch.device("cpu")

    def get_device(self):
        """
        Returns the detected torch.device object.
        """
        return self._device

# Example usage (for testing purposes, remove in final SDK if not needed)
if __name__ == "__main__":
    device_manager = DeviceManager()
    device = device_manager.get_device()
    print(f"Using device: {device}")

    # Small test to ensure device is working
    try:
        x = torch.randn(10, 10, device=device)
        y = torch.randn(10, 10, device=device)
        z = x @ y
        print(f"Successfully performed a matrix multiplication on {device}.")
    except Exception as e:
        print(f"Error performing test on {device}: {e}")

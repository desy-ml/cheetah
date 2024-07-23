import torch


# Check if MPS is available and functional (for GitHub Actions)
def is_mps_available_and_functional():
    if not torch.backends.mps.is_available():
        return False
    try:
        # Try to allocate a small tensor on the MPS device
        torch.tensor([1.0], device="mps")
        return True
    except RuntimeError:
        return False

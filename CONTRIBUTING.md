# Contribution Guidelines

## How to write fast PyTorch code

### Creating new tensors

```python
torch.tensor(0.0, device=a.device, dtype=a.dtype)
torch.zeros((), device=a.device, dtype=a.dtype)
torch.zeros_like(a)   # <-- This is fastest for same shape (see #561)
a.new_zeros(())   # <-- This is fastest for compatible constants (see #561)
a.new_zeros(a.shape)
```

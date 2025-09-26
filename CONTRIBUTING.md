# Contribution Guidelines

## How to write fast PyTorch code

### Transpose

```python
torch.transpose(x, -2, -1)
x.transpose(-2, -1)
x.mT   # <-- This is fastest (see #558)
```

### Creating new tensors

```python
torch.tensor(0.0, device=a.device, dtype=a.dtype)
torch.zeros((), device=a.device, dtype=a.dtype)
torch.zeros_like(a)   # <-- This is fastest for same shape (see #561)
a.new_zeros(())   # <-- This is fastest for compatible constants (see #561)
a.new_zeros(a.shape)
```

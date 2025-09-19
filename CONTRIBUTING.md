# Contribution Guidelines

## How to write fast PyTorch code

### Dividing one by tensor

```python
1 / x
torch.reciprocal(x)
x.reciprocal()   # <-- This is fastest (see #563)
```

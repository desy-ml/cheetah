# Contribution Guidelines

## How to write fast PyTorch code

### Transpose

```python
torch.transpose(x, -2, -1)
x.transpose(-2, -1)
x.mT   # <-- This is fastest (see #558)
```

### Square

```python
x**2
x * x
torch.square(x)
x.square()   # <-- This is fastest (see #555)
```

### Dividing 1 by a tensor

```python
1 / x
torch.reciprocal(x)
x.reciprocal()   # <-- This is fastest (see #563)
```

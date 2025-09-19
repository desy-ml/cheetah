# Contribution Guidelines

## How to write fast PyTorch code

### `x.op()` vs `torch.op(x)`

```python
torch.any(x < 0)
(x < 0).any()   # <-- This is faster (see #556)
```

```python
torch.square(x)
x.square()   # <-- This is faster (see #556)
```

```python
torch.sum(x)
x.sum()   # <-- This is faster (see #556)
```

# Contribution Guidelines

## How to write fast PyTorch code

### Square

```python
x**2
x * x
x.square()   # <-- This is fastest (see #555)
```

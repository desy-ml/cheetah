# Contribution Guidelines

## Verify an element change

Run the element's tests first. For example, run
`pytest tests/test_quadrupole.py` for changes to `Quadrupole`. If a failure
depends on random inputs, use the seed printed in the pytest header to reproduce
it, for example `pytest tests/test_quadrupole.py --seed=721`.

When adding an `Element` subclass, register representative constructor arguments
in `ELEMENT_SUBCLASSES_ARGS` in [`tests/conftest.py`](tests/conftest.py).
This includes the new class in tests marked with `for_every_element`. Run
`pytest tests/test_elements.py` to check the shared element behaviour.

When changing differentiable tracking, check a physical parameter's gradient
in `torch.float64` against a centred finite difference. Use the same incoming
beam for both evaluations. For example, the [zero-strength quadrupole
test](tests/test_quadrupole.py) compares the autograd derivative of `sigma_x`
with respect to `k1` against the tracked beam sizes at `k1 ± h`.

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

### Dividing 1 by a tensor

```python
1 / x
torch.reciprocal(x)
x.reciprocal()   # <-- This is fastest (see #563)
```

### Creating new tensors

```python
torch.tensor(0.0, device=a.device, dtype=a.dtype)
torch.zeros((), device=a.device, dtype=a.dtype)
torch.zeros_like(a)   # <-- This is fastest for same shape (see #561)
a.new_zeros(())   # <-- This is fastest for compatible constants (see #561)
a.new_zeros(a.shape)
```

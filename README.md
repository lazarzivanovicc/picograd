<p align="center">
  <img src="assets/picograd.svg" alt="picograd banner"/>
</p>

# picograd

A minimal deep learning framework built from scratch. Inspired by [micrograd](https://github.com/karpathy/micrograd), [tinygrad](https://github.com/tinygrad/tinygrad) and [PyTorch](https://github.com/pytorch/pytorch).

Implements a small autograd engine with a PyTorch-like API on top of NumPy.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from picograd.tensor.tensor_base import Tensor
from picograd.nn.layer.linear import Linear

x = Tensor([[1.0, 2.0], [3.0, 4.0]])
layer = Linear(2, 1)
out = layer(x).sum()
out.backward()
```

## Tests

```bash
python -m picograd.tests.test
```

## License

MIT
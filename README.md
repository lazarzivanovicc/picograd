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

## Examples

```bash
python -m picograd.examples.xor
```

```python
from picograd.nn.module import Module
from picograd.nn.layer.linear import Linear
from picograd.nn.optim import SGD
from picograd.tensor.tensor_base import Tensor
import numpy as np
from matplotlib import pyplot as plt


class NeuralNet(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(2, 15)
        self.layer2 = Linear(15, 1)
    
    def forward(self, x: Tensor):
        x = self.layer1(x)
        x = x.relu()
        x = self.layer2(x)
        return x
    
    def __call__(self, x: Tensor):
        return self.forward(x)
    

if __name__ == "__main__":

    x: Tensor = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]) 
    y: Tensor = Tensor([[0], [1], [1], [0]])

    model = NeuralNet()
    optimizer = SGD(model.parameters(), lr=0.001)

    running_loss: list[float] = []
    for epoch in range(2500):
        optimizer.zero_grad()
        predictions = model(x)
        loss = ((predictions - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.data)
        print(f"Epoch - {epoch}, loss - {loss.data}")
```

## License

MIT
# Micrograd

Scalar autograd engine that allows backpropagation over a dynamically built Directed Acyclic Graph (DAG), with a PyTorch-like API.

Based on [karpathy/micrograd](https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py) and [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&ab_channel=AndrejKarpathy).

## Architecture

Micrograd is a **tensor-centric** (or more precisely, scalar-centric) autograd engine. The nodes in the computational graph are `Value` objects that wrap individual scalar numbers, not operations. This is the opposite of frameworks like tinygrad and PyTorch internals, which are **operation-centric** -- their graph nodes represent functions/operations, and tensors flow between them.

The entire engine is built on two core ideas:

1. Every arithmetic operation produces a new `Value` node that remembers its parents (`_prev`) and what operation created it (`_op`).
2. Every node carries a closure (`_backward`) that knows how to propagate gradients back to its parents using the chain rule.

### Computational Graph

When you write `c = a + b`, micrograd doesn't just compute the sum -- it builds a graph:

```
a ──┐
    ├──(+)──> c
b ──┘
```

`c._prev = {a, b}` and `c._op = '+'`. The graph is constructed lazily, on-the-fly, as Python executes forward operations. This is what "dynamically built" means -- there is no separate graph compilation step. You build it by running code, which is identical to PyTorch's eager mode.

### The Value Class

`Value` is both the data container and the graph node. Each instance holds:

| Field | Purpose |
|---|---|
| `data` | The scalar float value |
| `grad` | Accumulated gradient (initialized to 0) |
| `_backward` | Closure that computes local gradient contribution |
| `_prev` | Set of parent `Value` nodes (the inputs that produced this value) |
| `_op` | String label of the operation (for debugging/visualization) |

This design means the graph is implicit in the object references -- there is no separate `Graph` data structure. The topology is encoded in `_prev` pointers.

### Backward Pass

`backward()` does two things:

1. **Topological sort** -- walks `_prev` pointers recursively (DFS) to produce a linear ordering where every node appears after all of its children.
2. **Reverse-order gradient propagation** -- iterates the sorted list in reverse, calling each node's `_backward()` closure.

The topological sort guarantees that by the time we call `v._backward()`, `v.grad` already contains the fully accumulated gradient from all paths downstream. This is critical because a single node can feed into multiple consumers, and each consumer adds its contribution via `+=`.

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1  # dL/dL = 1
    for v in reversed(topo):
        v._backward()
```

Setting `self.grad = 1` seeds the process: the derivative of the loss with respect to itself is 1.

### Gradient Accumulation with +=

Every `_backward` closure uses `+=`, not `=`:

```python
def _backward():
    self.grad += out.grad
    other.grad += out.grad
```

This is essential. If a `Value` is used in multiple operations (e.g., `x + x`, or `x` feeding into two different branches), each usage contributes a gradient. The multivariate chain rule says these contributions sum. Without `+=`, later backward calls would overwrite earlier ones, producing wrong gradients.

This is also why `zero_grad()` exists -- gradients accumulate across backward passes, so they must be manually reset before each new pass.

### Local Gradient Rules

Each operation defines how the upstream gradient (`out.grad`) distributes to its inputs:

| Operation | `d(out)/d(self)` | `d(out)/d(other)` |
|---|---|---|
| `self + other` | 1 | 1 |
| `self * other` | `other.data` | `self.data` |
| `self ** n` | `n * self.data^(n-1)` | -- |
| `relu(self)` | 1 if `self.data > 0`, else 0 | -- |

All other operations are composed from these primitives:
- Negation: `self * -1`
- Subtraction: `self + (-other)`
- Division: `self * other^(-1)`

This is a key design choice: a minimal set of differentiable primitives, with everything else derived. It keeps the backward logic simple and correct.

## Neural Network Layers

Built on top of the autograd engine, the `Module` / `Neuron` / `Layer` / `MLP` classes follow PyTorch conventions:

### Module

Base class that provides:
- `parameters()` -- returns all learnable `Value` objects (overridden by subclasses)
- `zero_grad()` -- resets all parameter gradients to 0

### Neuron

A single neuron: `y = relu(w . x + b)` or `y = w . x + b` (if `nonlin=False`).

- `weights`: list of `Value` objects, one per input, initialized uniformly in `[-1, 1]`
- `bias`: a single `Value`, initialized to 0
- `__call__`: computes the dot product + bias, optionally applies ReLU

The dot product is computed with Python's `sum()` over a zip of weights and inputs. Each `wi * xi` produces a new `Value` node, and `sum()` chains `__add__` calls, building the full forward graph for one neuron.

### Layer

A collection of `Neuron` objects. `Layer(3, 4)` creates 4 neurons, each expecting 3 inputs. Calling the layer runs all neurons on the same input and returns a list (or a single `Value` if there's only one neuron -- convenience for the output layer).

### MLP

Multi-Layer Perceptron. `MLP(3, [4, 4, 1])` creates:
- Layer(3, 4) with ReLU
- Layer(4, 4) with ReLU
- Layer(4, 1) **without** ReLU (linear output)

The last layer is always linear (`nonlin=False`), which is the standard setup for regression or pre-softmax output. The `nonlin=i!=len(number_of_outputs)-1` expression handles this.

## Training Loop (not included in micrograd.ipynb, but follows naturally)

```python
for epoch in range(epochs):
    # Forward pass
    predictions = [model(x) for x in X]
    loss = sum((pred - y)**2 for pred, y in zip(predictions, Y))

    # Backward pass
    model.zero_grad()
    loss.backward()

    # SGD update
    for p in model.parameters():
        p.data -= learning_rate * p.grad
```

This is pure SGD. The parameter update modifies `.data` directly, not `.grad`. The graph is rebuilt from scratch each forward pass (dynamic graph).

## Comparison with mangrad/

The `/mangrad` directory shows the pre-autograd approach: each layer manually implements both `forward_propagation` and `backward_propagation`. The `FCLayer.backward_propagation` explicitly computes `dL/dW`, `dL/dB`, and `dL/dX` using matrix math, and immediately applies the weight update.

Micrograd eliminates this by:
1. Defining local gradient rules once per operation (not per layer type)
2. Letting the topological sort + chain rule compose them automatically
3. Separating gradient computation from parameter updates

This is the fundamental insight of autograd: you only need to define forward operations with their local derivatives, and the framework handles the rest.

## Limitations

- **Scalar only**: every value is a single float. An MLP with 1000 parameters has 1000 `Value` objects and thousands of graph nodes. No vectorization, no BLAS, no GPU.
- **Python overhead**: every operation is a Python function call with closure allocation. This is orders of magnitude slower than C/CUDA tensor ops.
- **No batching**: inputs are processed one at a time. Real frameworks batch inputs into tensors and parallelize via matrix multiplication.
- **Limited ops**: only +, *, **, ReLU. No exp, log, matmul, conv, softmax, etc.

These limitations are by design -- micrograd exists to teach autograd, not to train models. The path from here to a real framework (like what `/src` is building with the C tensor library) involves replacing scalar `Value` with n-dimensional tensors and replacing Python closures with compiled kernels.

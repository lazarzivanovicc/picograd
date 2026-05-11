from picograd.tensor.tensor_base import Tensor
import numpy as np


class Optim:
    def __init__(self, params: list[Tensor], lr: float):
        self.params: list[Tensor] = params
        self.lr = lr

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        raise NotImplementedError()


class SGD(Optim):
    def __init__(self, params: list[Tensor], lr: float):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data, dtype=np.float32)
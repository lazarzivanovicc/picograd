from picograd.tensor.tensor_base import Tensor, Function
import numpy as np


class Add(Function):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        tensors: list[Tensor] = [x, y]
        out_data: np.ndarray = x.data + y.data
        self.ctx.save_for_backward(*tensors)
        return out_data

    def backward(self) -> None:
        x, y, z = self.ctx.saved_tensors
        x.grad += z.grad
        y.grad += z.grad


# Think about receiving grad in backward and how does that influence my walk in backward
class Mul(Function):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        tensors: list[Tensor] = [x, y]
        out_data: np.ndarray = x.data * y.data
        self.ctx.save_for_backward(*tensors)
        return out_data
    
    def backward(self) -> None:
        x, y, z = self.ctx.saved_tensors
        x.grad += z.grad * y.data
        y.grad += z.grad * x.data


class MatMul(Function):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        tensors: list[Tensor] = [x, y]
        out_data: np.ndarray = x.data @ y.data
        self.ctx.save_for_backward(*tensors)
        return out_data
    
    def backward(self) -> None:
        x, y, z = self.ctx.saved_tensors
        x.grad += z.grad @ y.data.T
        y.grad += x.data.T @ z.grad


class ReLU(Function):
    def forward(self, x: Tensor) -> np.ndarray:
        tensors: list[Tensor] = [x]
        out_data: np.ndarray = np.clip(x.data, 0, None)
        self.ctx.save_for_backward(*tensors)
        return out_data
    
    def backward(self) -> None:
        x, z = self.ctx.saved_tensors
        z_grad_c = z.grad.copy()
        z_grad_c[x.data < 0] = 0
        x.grad += z_grad_c
        


# def register(fn_name):
#     setattr(Tensor, fn_name: str, partialmethod())
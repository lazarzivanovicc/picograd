from src.tensor.tensor_base import Tensor
import numpy as np

class Context:
    def __init__(self):
        self.saved_tensors: list[Tensor] = []

    def save_for_backward(self, *tensors: Tensor) -> None:
        self.saved_tensors.extend(tensors)


class Function:
    def __init__(self):
        self.parents: list[Function] = []
        self.ctx: Context | None = None

    def apply(self, *tensors: Tensor) -> Tensor:
        self.parents = [tensor.fn for tensor in tensors] 
        self.ctx: Context = Context()
        output: Tensor = self.forward(*tensors)
        output.fn = self
        return output


class Add(Function):
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        tensors: list[Tensor] = [x, y]
        out_data: np.ndarray = x.data + y.data
        z: Tensor = Tensor(out_data)
        tensors.append(z)
        self.ctx.save_for_backward(*tensors)
        return z

    def backward(self) -> None:
        x, y, z = self.ctx.saved_tensors
        x.grad += z.grad
        y.grad += z.grad



class Mul(Function):
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        tensors: list[Tensor] = [x, y]
        out_data: np.ndarray = x.data * y.data
        z: Tensor = Tensor(out_data)
        tensors.append(z)
        self.ctx.save_for_backward(*tensors)
        return z
    
    def backward(self):
        x, y, z = self.ctx.saved_tensors
        x.grad += z.grad * y.data
        y.grad += z.grad * x.data
        


# def register(fn_name):
#     setattr(Tensor, fn_name: str, partialmethod())
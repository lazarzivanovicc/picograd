from src.nn.module import Module
from src.tensor.tensor_base import Tensor
import numpy as np

# Most basic module used for building DNN
# Since does not support bias yet since I can't deal with gradients if broadcast happens in the fw

class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W: Tensor = Tensor(np.random.randn(in_dim, out_dim)) # pytorch saves this as out_dim, in_dim

    def forward(self, x: Tensor):
        return x @ self.W
    
    def __call__(self, x: Tensor):
        return self.forward(x)
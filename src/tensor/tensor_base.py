import numpy as np

# TODO: Finish the transition to function based graph
class Tensor:
    def __init__(self, data: np.ndarray | list, prev: set = None) -> None:
        self.data: np.ndarray = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float32)
        # These two can be deleted _b and _p
        self._backward = lambda: None
        self._prev: set = set() if prev is None else prev
        # requires_grad boolean - if it is not true pytorch wont allow backward call
        # additionally pytorch does not allow grads for ints, only floats and complex
        # device
        # dtype
        self.fn: "Function" | None = None

    
    def __add__(self, other) -> Tensor:
        out: Tensor = Tensor(self.data + other.data, {self, other})
        def backward() -> None:
            self.grad += out.grad 
            other.grad += out.grad 
        out._backward = backward
        return out


    def __matmul__(self, other) -> Tensor:
        # assert self.data.shape[-1] == other.data.shape[-2] / "Error dimensions not match"
        out: Tensor = Tensor(self.data @ other.data, {self, other})
        def backward() -> None:
            # X @ W = Z
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = backward
        return out
    

    def __mul__(self, other: Tensor) -> Tensor:
        from src.autograd.function import Mul
        return Mul().apply(self, other)
        
    
    def __div__(self, other) -> Tensor:
        pass

    def __pow__(self, other) -> Tensor:
        pass
    
    def sum(self) -> Tensor:
        pass

    def mean(self) -> Tensor:
        pass

    def relu(self) -> Tensor:
        out: Tensor = Tensor(np.clip(self.data, 0, None), {self})
        def backward() -> None:
            self.grad = [grad if val > 0 else 0 for val, grad in zip(self.data, out.grad)]
        out._backward = backward
        return out

    
    def backward(self) -> None:
        # This is my initialization step
        # TODO: Create graph of fns
        self.grad: np.ndarray = np.ones_like(self.data, dtype=np.float32)
        visited: set = set()
        topo: list[Tensor] = []
        def build_topo(t: Tensor) -> None:
            if t not in visited:
                visited.add(t)
                for p in t._prev:
                    build_topo(p)
                topo.append(t)
        build_topo(self)
        for t in reversed(topo):
            t._backward()


    @staticmethod
    def standard_normal(shape: tuple):
        data: np.ndarray = np.random.standard_normal(shape)
        return Tensor(data)
    
    @staticmethod
    def uniform(shape: tuple):
        pass

# Separate Function as an independent class, this clojure approach is not good and this will not scale well
# Each new function should extends Function and define it's forward and backward and it should save it's parents along with the context needed for backward pass
# Tensor will hold only API for interacting with these fn's
# Once we change that we will have FN-centric graph and not Tensor-centric like now


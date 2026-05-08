import numpy as np


class Context:
    """
    Context represent the storage of each Function that is used to comupte gradients of input Tensor/s that participated in Function. 
    Once function performs it's dedicated operation in forward pass it stores Tensors that will be necessary for the backward pass.
    Context is modified by both forward method of concrete Function implementation and apply method from the base class since Function needs 
    reference to it's output Tensor from it's forward function in order to compute gradients succesfully.
    """
    def __init__(self):
        self.saved_tensors: list[Tensor] = []

    def save_for_backward(self, *tensors: Tensor) -> None:
        self.saved_tensors.extend(tensors)


class Function:
    """
    Function represents the most fundamental block of picograd's autodiff engine.
    It represent a node in a computational graph.

    Parents field of Function class is used in order to make connections to other nodes (Functions).
    Context field of Function class is used to store variables necessary for backward pass and gradient computation.

    Apply method of a Function class gets called by a Tensor. It receives Tensor(s) as input(s) and produces new Tensor as a result.
    This method performs forward pass of a concrete Function class on given Tensor(s) input(s) and stores the references to the input Tensor(s) inside the ctx field. 
    Additionally once done with the forward pass, reference to output Tensor is also saved in the ctx since Function will need it's gradient during the backward pass. 
    Lastly, Tensor that was produced by the function gets the reference to this Function, since the backward pass procedure needs to be intialized from Tensor object.

    Prior to creation of the output Tensor we check if any of the input tensors requires grad, if yes, we must create output Tensor with requires_grad property marked as True
    in order for it to accumulate gradients of loss with respect to itself's data which will be used as upstream grads when we call this Functions backward needed
    for computation of gradient of give input tensors who require their own gradients.
    """
    def __init__(self):
        self.parents: list[Function] = []
        self.ctx: Context | None = None

    def apply(self, *tensors: Tensor) -> Tensor:
        self.parents = [tensor.fn for tensor in tensors] 
        self.ctx: Context = Context()
        requires_grad = any(tensor.requires_grad for tensor in tensors)
        output: Tensor = Tensor(self.forward(*tensors), requires_grad=requires_grad) # Foward also modifies the context as it saves the data needed for backward based on fn type
        self.ctx.save_for_backward(output) # As this will be used in the backward pass rest of the context is set in self.forward method
        output.fn = self
        return output


# Helper I need
def as_tensor(obj) -> Tensor:
    if isinstance(obj, Tensor):
        return obj
    elif isinstance(obj, list) or isinstance(obj, np.ndarray):
        return Tensor(obj)
    return Tensor(np.array([obj]))

# Inputs (X) won't need gradients in X @ W figure out how to skip calculation of grads for them
class Tensor:
    """
    The most basic data container in picograd.
    
    Tensor represents an input(s) to a Function(s) which form a computation graph.
    Stores gradients of computation graph output(s) with respect to data it holds.
    requires_grad properity is a flag that marks if the given instance of a Tensor class requires gradients.

    Exposes methods which act as an API, that invoke concrete Function implementations forming computation graph.
    """
    def __init__(self, data: np.ndarray | list, requires_grad: bool = False) -> None:
        self.data: np.ndarray = data if isinstance(data, np.ndarray) else np.array(data)
        self.fn: Function | None = None
        self.requires_grad: bool = requires_grad
        if self.requires_grad:
            self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float32)
        else:
            self.grad = None # Nice memory savings if Tensor does not need grads and additionally required_grad allowed me to implement some primitive guards in ops

        # additionally pytorch does not allow grads for ints, only floats and complex
        # device
        # dtype

    
    def __add__(self, other) -> Tensor:
        from picograd.autograd.ops import Add # Check if registration can help me with breaking the circular deps
        return Add().apply(self, as_tensor(other))


    def __matmul__(self, other) -> Tensor:
        from picograd.autograd.ops import MatMul
        return MatMul().apply(self, other)
    

    def __mul__(self, other) -> Tensor:
        from picograd.autograd.ops import Mul
        return Mul().apply(self, as_tensor(other))
    
    # Feels a bit dirty but ok for now I will add an Op later
    def __sub__(self, other) -> Tensor:
        return self + (other * -1)
    
    
    def __truediv__(self, other) -> Tensor:
        return self * (other ** -1)

    def __pow__(self, other) -> Tensor:
        from picograd.autograd.ops import Pow
        return Pow(other).apply(self)
        
    # Axis and keep_dims passed to this fn
    def sum(self) -> Tensor:
        from picograd.autograd.ops import Sum
        return Sum().apply(self)

    def mean(self) -> Tensor:
        divisor: Tensor = Tensor([1 / self.data.size]) # 1 / n
        return self.sum() * divisor
        

    def relu(self) -> Tensor:
        from picograd.autograd.ops import ReLU
        return ReLU().apply(self)
    

    def backward(self) -> None:
        if not self.requires_grad:
            raise RuntimeError("Tensor used for backward intialization does not require grad.")
        self.grad: np.ndarray = np.ones_like(self.data, dtype=np.float32)
        if self.fn is None:
            return # Case of calling backward on leaf node which has no parents
        visited: set = set()
        topo: list[Function] = []
        
        def build_topo(f: Function) -> None:
            if f is None or f in visited:
                return
            visited.add(f)
            for p in f.parents:
                build_topo(p)
            topo.append(f)
        
        build_topo(self.fn)
        
        for f in reversed(topo):
            f.backward()

    def shape(self) -> tuple:
        return self.data.shape
    

    def numel(self) -> int:
        return len(self.data)

    # Static methods of Tensor class that can be used for initialization of new Tensors with certain properties

    @staticmethod
    def standard_normal(shape: tuple):
        data: np.ndarray = np.random.standard_normal(shape)
        return Tensor(data)
    
    @staticmethod
    def uniform(shape: tuple):
        pass

    @staticmethod
    def zeros(shape: tuple):
        pass

    @staticmethod
    def ones(shape: tuple):
        pass

    @staticmethod
    def ones_like(tensor: Tensor):
        pass

    @staticmethod
    def zeros_like(tensor: Tensor):
        pass

# Separate Function as an independent class, this clojure approach is not good and this will not scale well
# Each new function should extends Function and define it's forward and backward and it should save it's parents along with the context needed for backward pass
# Tensor will hold only API for interacting with these fn's
# Once we change that we will have FN-centric graph and not Tensor-centric like now


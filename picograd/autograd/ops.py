from picograd.tensor.tensor_base import Tensor, Function
import numpy as np

# READ THIS, I PLAN TO CREATE REAL DOCUMENTATION OUT OF THIS

# Forward pass defines set of rules that are used to create new output from input(s)
# Backward pass defines set of rules to compute gradients (derivatives) of Function's input(s) with respect to the outputs
# Backward pass leverages upstream gradient coming from the Tensor that was the output of current Function from which backward is being invoked
# It defines the local gradient [it does't do this every time since this can be really computationally expensive :)] and multiplies it with
# upstream gradient in order to obtain gradients of loss with respect to each of the inputs

# Additional explanation about not computing the local derivatives explicitly for every function
# If you have * operation on 2 scalars, a, b and result is scalar c which gets feed in loss calculation l (for the simplification purposes l = c^2)
# We are interested in how a and b affect l in order to perform optimization
# Keep in mind that by our desing we have a and b as inputs to * Function whose output is c that becomes input to ^2 Function node whose output is l
# dl/dl = 1, dl/dc = 2c and this is what I refered to as upstream gradient when looking at the * Function node 
# Now backward method of * Function node will have a rule how to compute derivatives of loss w.r.t it's inputs a and b, dl/da, dl/db
# dl/da = dl/dl * dl/dc * dc/da = 1 * dl/dc[upstram gradient stored on c.grad] * dc/da[local gradient calculated by a simple math rule, for c = a * b, dc/da = b]
# dl/da = dl/dl * dl/dc * dc/da = 1 * 2c * b
# Following the same logic dl/db = 1 * 2c * a
# Now, moment of truth, once we move to matrices and tensors, thing get bit more complex
# Imagine matrix multiplication of A (2 x 2) and B (2 x 3) -> A @ B = C (2 x 3) and L = sum of all elements in C (scalar value)! It is important that L is scalar because than we know dL/dA has the same dimension as A (how does L scalar change with respect to change in any value from A)
# Now dL/dL (1), dL/dC (2 x 3) all ones, and now this is our upstream gradient flowing in @ Function node
# We need our backward to calculate dL/dA and dL/dB, and we know that dL/dA = dL/dC * dC/dA
# HERE COMES THE PLOT dC/dA is a Jacobian and it has form of a tensor (2 x 3 x 2 x 2) - dC/dA asks how does each value in C change with respect to each value in A
# So this Jacobian would have 2 x 3 x 2 x 2 = 24 elements and we are working with extreamly small matrices
# And guess what else? Most of these values are 0. 
# Why? Looking again at the statement on what is dC/dA - dC/dA asks how does each value in C change with respect to each value in A.
# So element in Jacobian at dCdA[1,1,1,1] says how much does element C[1,1] change with respect to A[1, 1]
# We know that for matmul operation,  C[1, 1] = SUM OVER K A[1, K] * B[K, 1], so for the making of C[1, 1] if we chosen A[1, 1], we can see that k=1 so A[1, 1] gets paired and mutliplied with B[1, 1]
# So influence A[1, 1] has on C[1, 1] is B[1, 1] => dCdA[1, 1, 1, 1] = B[1, 1]
# What is the influence of A[1, 1] on C[2, 2]? C[2, 2] = SUM OVER K A[2, K] * B[K, 2], so we see A[1, 1] does not participate in the making of C[2, 2] so influence is 0 and
# dCdA[2, 2, 1, 1] = 0
# WE CAN SEE THAT MOST OF OUR JACOBIAN dC/dA are just ZEROS AND THEY GET HUGE REALLY FAST
# SAME GOES FOR dC/dB
# NOTICE that A[1, 1] participates in the buidling of all C[1, :], and it's influence total influence is basically the sum of B[1, :] and we know that
# dCdA is full of zeros and that we would like to go around it and we also know that L is a scalar and we are interested in dL/dA - how much does that scalar change w.r.t each element in A
# So I previously said that participates in the building of all C[1, :] so THAT MEANS IT PARTICIPATES IN THE LOSS THROUGH EACH ELEMENT IN a row C[1, :]
# So total influence A[1, 1] has on scalar L is dL/dA[1, 1] = SUM OVER J dL/dC[1, J] * B[1, J] - since there are J elements in row of C and each participated in C.sum() to form scalar L, we have to account for the influence of A[1, 1] on all of those elements
# AND FORM HERE WE SEE THAT dL/dA (2 x 2) = dL/dC (2 x 3) @ B^T (3 x 2) and this is the shortened form we can see in line 82
# WE COMPUTED GRADIENT OF SCALAR LOSS WITH RESPECT TO @ FUNCTION NODE INPUT WITHOUT CALCULATING REAL LOCAL GRADIENT BUT BY BEING SMART AND USING UPSTREAM GRADIENT AND A BIT OF MATH

# Here we leverage numpy's standard operations on numpy ndarray to produce new numpy ndarray
# We would like to get rid of numpy at one point
# With numpy we are tied to CPU which is not good for machine learning lib
# This goes back to Tensor - since it is the data structure that holds our data
# GPU Ops need to be performed on Tensors that live on GPU for example
# So our tensor has to have reference of where it stores memory and how a function can access it perform an op and return the reference

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
# We will also have to support broadcasting
# If broadcasting happens on forward pass
# We also need to be clever what we will do on the backward pass 
# I mean it seams obvious to me that we will have to acumulate grad across the broadcasted dimension but I am not sure completely ATM
# Also we will have to take care about Tensors that do not store grads because they do not need the computation of grads
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
        z_grad_c[x.data <= 0] = 0
        x.grad += z_grad_c

class Pow(Function):
    def __init__(self, exponent: int|float):
        self.exponent = exponent

    def forward(self, x: Tensor) -> np.ndarray:
        tensors: list[Tensor] = [x]
        out_data: np.ndarray = np.pow(x.data, self.exponent)
        self.ctx.save_for_backward(*tensors)
        return out_data
    
    def backward(self) -> None:
        x, z = self.ctx.saved_tensors
        x.grad += np.pow(self.exponent * x.data, self.exponent - 1) * z.grad


class Sum(Function):
    def __init__(self, axis: int, keep_dims: bool):
        self.axis = axis
        self.keep_dims

    def forward(self, x: Tensor) -> np.ndarray:
        tensors: list[Tensor] = [x]
        out_data: np.ndarray = np.sum(x.data, self.axis) # This will collapse 1D list do scalar for example by default so I need to recover that in backward or it will keep it if keep_dims true
        self.ctx.save_for_backward(*tensors)
        return out_data
    
    # def backward(self) -> None:
    #     x, z = self.ctx.saved_tensors
    #     x.grad += 
    

        


# def register(fn_name):
#     setattr(Tensor, fn_name: str, partialmethod())
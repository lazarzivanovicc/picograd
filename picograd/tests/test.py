from typing import Callable
import numpy as np
import torch
 
from picograd.tensor.tensor_base import Tensor
from picograd.nn.module import Module
from picograd.nn.layer.linear import Linear


def add_test() -> bool:
    a: Tensor = Tensor([1, 2, 3])
    b: Tensor = Tensor([1, 1, 1])
    c: Tensor = a + b
    c_exp: np.ndarray = np.array([2, 3, 4])
    if np.array_equal(c.data, c_exp):
        print("ADD TEST PASSED")
        return True
    else:
        print("ADD TEST FAILED")
        return False
    

def mat_mul_1D() -> bool:
    a: Tensor = Tensor([1, 2, 3])
    b: Tensor = Tensor([1, 2, 1])
    c: Tensor = a @ b
    c_exp: np.float32 = 8.0
    if np.array_equal(c.data, c_exp):
        print("MAT_MUL_1D TEST PASSED")
        return True
    else:
        print("MAT_MUL_1D TEST FAILED")
        return False
    

def mat_mul_2D() -> bool:
    a: Tensor = Tensor([[1, 2, 3], [1, 1, 1]]) 
    b: Tensor = Tensor([[1, 2, 1], [1, 2, 3], [1, 2, 2]]) 
    c: Tensor = a @ b
    c_exp: np.ndarray = np.array([[6, 12, 13], [3, 6, 6]])
    if np.array_equal(c.data, c_exp):
        print("MAT_MUL_2D TEST PASSED")
        return True
    else:
        print("MAT_MUL_2D TEST FAILED")
        return False
    

def mat_vec() -> bool:
    a: Tensor = Tensor([1, 2, 3])
    b: Tensor = Tensor([[1, 2, 1], [1, 2, 3], [1, 2, 2]]) 
    c: Tensor = a @ b
    c_exp: np.ndarray = np.array([6, 12, 13])
    if np.array_equal(c.data, c_exp):
        print("MAT_VEC TEST PASSED")
        return True
    else:
        print("MAT_VEC TEST FAILED")
        return False
    

def relu() -> bool:
    a: Tensor = Tensor([-1, 2, 3])
    b: Tensor = Tensor([[-50, 20, 0], [14, -23, 12]])
    a_r: Tensor = a.relu()
    b_r: Tensor = b.relu()

    if np.array_equal(a_r.data, [0, 2, 3]) and np.array_equal(b_r.data, [[0, 20, 0], [14, 0, 12]]):
        print("ReLU TEST PASSED")
        return True
    else:
        print("ReLU TEST FAILED")
        return False
    

def backward_relu() -> bool:
    a: Tensor = Tensor([-1, 2, 3])
    a_r: Tensor = a.relu()
    a_r.backward()

    a_t: torch.Tensor = torch.tensor([-1, 2, 3], dtype=torch.float32, requires_grad=True)
    a_t_r: torch.Tensor = torch.nn.functional.relu(a_t)
    a_t_r.backward(torch.ones_like(a_t_r))

    if torch.allclose(a_t.grad, torch.tensor(a.grad, dtype=torch.float32)):
        print("BACKWARD_RELU TEST PASSED")
        return True
    else:
        print("BACKWARD_RELU TEST FAILED")
        return False


def backward_add() -> bool:
    a: Tensor = Tensor([1, 2, 3])
    b: Tensor = Tensor([1, 1, 1])
    c: Tensor = a + b
    c.backward()

    a_t: torch.Tensor = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    b_t: torch.Tensor = torch.tensor([1, 2, 1], dtype=torch.float32, requires_grad=True) 
    c_t = a_t + b_t
    c_t.backward(torch.ones_like(c_t))

    if torch.allclose(a_t.grad, torch.tensor(a.grad, dtype=torch.float32)) and torch.allclose(b_t.grad, torch.tensor(b.grad, dtype=torch.float32)):
        print("BACKWARD_ADD TEST PASSED")
        return True
    else:
        print("BACKWARD_ADD TEST FAILED")
        return False
                    

def backward_mat_mul() -> bool:
    a: Tensor = Tensor([[1, 2, 3], [1, 1, 1]]) 
    b: Tensor = Tensor([[1, 2, 1], [1, 2, 3], [1, 2, 2]]) 
    c: Tensor = a @ b
    c.backward()

    a_t: torch.Tensor = torch.tensor([[1, 2, 3], [1, 1, 1]], dtype=torch.float32, requires_grad=True)
    b_t: torch.Tensor = torch.tensor([[1, 2, 1], [1, 2, 3], [1, 2, 2]], dtype=torch.float32, requires_grad=True) 
    c_t = a_t @ b_t
    c_t.backward(torch.ones_like(c_t))

    if torch.allclose(a_t.grad, torch.tensor(a.grad, dtype=torch.float32)) and torch.allclose(b_t.grad, torch.tensor(b.grad, dtype=torch.float32)):
        print("BACKWARD_MAT_MUL TEST PASSED")
        return True
    else:
        print("BACKWARD_MAT_MUL TEST FAILED")
        return False
    

def module_test() -> bool:
    layer: Linear = Linear(3, 5)
    tensors: Tensor = layer.parameters()
    if tensors[0].data.shape == (3, 5):
        print("MODULE TEST PASSED")
        return True
    else:
        print("MODULE TEST FAILED")
        return False
    

def module_nested_test() -> bool:
    class PicoNet(Module):
        def __init__(self) -> None:
            self.layer1 = Linear(3, 5)
            self.layer2 = Linear(5, 3)
            
        def forward(self, x: Tensor) -> Tensor:
            return self.layer2(self.layer1(x).relu())
            
        def __call__(self, x: Tensor) -> Tensor:
            return self.forward(x)
        
    net: PicoNet = PicoNet()
    net_parameters: list[Tensor] = net.parameters()

    if net_parameters[0].data.shape == (3, 5) and net_parameters[1].data.shape == (5, 3):
        print("MODULE NESTED TEST PASSED")
        return True
    else:
        print("MODULE NESTED TEST FAILED")
        return False
    

def standard_normal_generation() -> bool:
    a: Tensor = Tensor.standard_normal((10000, 10))
    if np.abs(a.data.mean() - 0.0) <= 1e-2 and np.abs(a.data.var() - 1.0) <= 1e-2:
        print("STANDARD NORMAL GENERATION TEST PASSED")
        return True
    else:
        print("STANDARD NORMAL GENERATION TEST FAILED")
        return False
    

def elementwise_mul() -> bool:
    a: Tensor = Tensor([1, 1, 1])
    b: Tensor = Tensor([2, 2, 2])
    c: Tensor = a * b
    
    if np.array_equal([2, 2, 2], c.data):
        print("ELEMENTWISE MUL TEST PASSED")
        return True
    else:
        print("ELEMENTWISE MUL TEST FAILED")
        return False       


def piconet_test():
    pass

def test(torch_fn: Callable, picograd_fn: Callable, atol: float) -> bool:
    pass

def run_tests() -> bool:
    tests: list = [
        add_test, 
        mat_mul_1D, 
        mat_mul_2D, 
        mat_vec,
        elementwise_mul,
        relu,
        module_test,
        module_nested_test,
        backward_add,
        backward_mat_mul,
        backward_relu,
        standard_normal_generation
    ]
    passed_cnt = 0
    failed_cnt = 0
    print(50 * '=')
    for test in tests:
        res = test()
        if res:
            passed_cnt += 1
        else:
            failed_cnt += 1
    print(50 * '=')
    print(f"\nNUMBER OF TESTS PASSED: {passed_cnt}")
    print(f"NUMBER OF TESTS FAILED: {failed_cnt}")
    
    
    return True


if __name__ == "__main__":
    run_tests()




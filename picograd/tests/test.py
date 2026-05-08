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
    

def sub_test() -> bool:
    a: Tensor = Tensor([1, 2, 3])
    b: Tensor = Tensor([1, 1, 1])
    c: Tensor = a - b
    c_exp: np.ndarray = np.array([0, 1, 2])

    d: Tensor = Tensor([1, 2, 3])
    e: int = 3
    f: Tensor = d - e
    f_exp: np.ndarray = np.array([-2, -1, 0])

    if np.array_equal(c.data, c_exp) and np.array_equal(f.data, f_exp):
        print("SUB TEST PASSED")
        return True
    else:
        print("SUB TEST FAILED")
        return False
    

def scalar_vec_mul() -> bool:
    a: Tensor = Tensor([1, 2, 3])
    b: int = 2
    c: Tensor = a * b

    if np.array_equal(c.data, [2, 4, 6]):
        print("SCALAR VEC MUL TEST PASSED")
        return True
    else:
        print("SCALAR VEC MUL TEST FAILED")
        return False
    

def scalar_vec_div() -> bool:
    a: Tensor = Tensor([1, 2, 3])
    b: int = 2
    c: Tensor = a / b

    if np.array_equal(c.data, [0.5, 1, 1.5]):
        print("SCALAR VEC DIV TEST PASSED")
        return True
    else:
        print("SCALAR VEC DIV TEST FAILED")
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
    
    
def power() -> bool:
    a: Tensor = Tensor([-1, 2, 3])
    b: Tensor = Tensor([[-50, 20, 0], [14, -23, 12]])
    a_p: Tensor = a ** 2
    b_p: Tensor = b ** 3

    if np.array_equal(a_p.data, np.pow([-1, 2, 3], 2)) and np.array_equal(b_p.data, np.pow([[-50, 20, 0], [14, -23, 12]], 3)):
        print("POWER TEST PASSED")
        return True
    else:
        print("POWER TEST FAILED")
        return False
    

def backward_relu() -> bool:
    a: Tensor = Tensor([-1, 2, 3], requires_grad=True)
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
    a: Tensor = Tensor([1, 2, 3], requires_grad=True)
    b: Tensor = Tensor([1, 1, 1], requires_grad=True)
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

# CURRENT IMPLEMENTATION FAILS THE TEST BECAUSE OF THE BRODCASTING -1 gets converted to (1,) and it tries to accumulate grad by using += from (3,) so i would need to collaps
# MUL AND ADD would also have problems with similar stuff I guess
def backward_sub() -> bool:
    a: Tensor = Tensor([1, 2, 3], requires_grad=True)
    b: Tensor = Tensor([1, 1, 1], requires_grad=True)
    c: Tensor = a - b
    c.backward()

    a_t: torch.Tensor = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    b_t: torch.Tensor = torch.tensor([1, 2, 1], dtype=torch.float32, requires_grad=True) 
    c_t = a_t - b_t
    c_t.backward(torch.ones_like(c_t))

    if torch.allclose(a_t.grad, torch.tensor(a.grad, dtype=torch.float32)) and torch.allclose(b_t.grad, torch.tensor(b.grad, dtype=torch.float32)):
        print("BACKWARD_SUB TEST PASSED")
        return True
    else:
        print("BACKWARD_SUB TEST FAILED")
        return False
                    

def backward_mat_mul() -> bool:
    a: Tensor = Tensor([[1, 2, 3], [1, 1, 1]], requires_grad=True) 
    b: Tensor = Tensor([[1, 2, 1], [1, 2, 3], [1, 2, 2]], requires_grad=True) 
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
    

def backward_pow() -> bool:
    a: Tensor = Tensor([-1, 2, 3], requires_grad=True)
    a_p: Tensor = a ** 2
    a_p.backward()

    a_t: torch.Tensor = torch.tensor([-1, 2, 3], dtype=torch.float32, requires_grad=True)
    a_t_p: torch.Tensor = a_t.pow(2)
    a_t_p.backward(torch.ones_like(a_t_p))

    if torch.allclose(a_t.grad, torch.tensor(a.grad, dtype=torch.float32)):
        print("BACKWARD_POW TEST PASSED")
        return True
    else:
        print("BACKWARD_POW TEST FAILED")
        return False
    

def backward_scalar_vec_mul() -> bool:
    a: Tensor = Tensor([1, 2, 3], requires_grad=True)
    b: int = 2
    c: Tensor = a * b
    c.backward()

    a_t: torch.Tensor = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    b_t: int = 2
    c_t: torch.Tensor = a_t * b_t
    c_t.backward(torch.ones_like(c_t))

    if torch.allclose(a_t.grad, torch.tensor(a.grad, dtype=torch.float32)):
        print("BACKWARD  SCALAR VEC MUL TEST PASSED")
        return True
    else:
        print("BACKWARD SCALAR VEC MUL TEST FAILED")
        return False
    

def backward_scalar_vec_div() -> bool:
    a: Tensor = Tensor([1, 2, 3], requires_grad=True)
    b: int = 2
    c: Tensor = a / b
    c.backward()

    a_t: torch.Tensor = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    b_t: int = 2
    c_t: torch.Tensor = a_t / b_t
    c_t.backward(torch.ones_like(c_t))

    if torch.allclose(a_t.grad, torch.tensor(a.grad, dtype=torch.float32)):
        print("BACKWARD SCALAR VEC MUL TEST PASSED")
        return True
    else:
        print("BACKWARD SCALAR VEC MUL TEST FAILED")
        return False
    
def backward_elementwise_mul() -> bool:
    pass
    

def backward_sum() -> bool:
    a: Tensor = Tensor([1, 2, 3, 4], requires_grad=True)
    b: Tensor = a.sum()
    b.backward()

    a_t: torch.Tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True)
    b_t: torch.Tensor = a_t.sum()
    b_t.backward()

    if torch.allclose(torch.tensor(a.grad, dtype=torch.float32), a_t.grad):
        print("BACKWARD SUM TEST PASSED")
        return True
    else:
        print("BACKWARD SUM TEST FAILED")
        return False
    

def backward_mse() -> bool:
    a: Tensor = Tensor([1, 2, 3, 4], requires_grad=True)
    b: Tensor = Tensor([1, 0, 1, 4])
    c: Tensor = ((a - b) ** 2).mean()
    c.backward()

    a_t: torch.Tensor = torch.tensor([1, 2, 3, 4], dtype=torch.float32, requires_grad=True)
    b_t: torch.Tensor = torch.tensor([1, 0, 1, 4], dtype=torch.float32)
    c_t: torch.Tensor = ((a_t - b_t) ** 2).mean()
    c_t.backward()

    if torch.allclose(torch.tensor(a.grad, dtype=torch.float32), a_t.grad):
        print("BACKWARD MSE TEST PASSED")
        return True
    else:
        print("BACKWARD MSE TEST FAILED")
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
    if np.abs(a.data.mean() - 0.0) <= 1e-1 and np.abs(a.data.var() - 1.0) <= 1e-1:
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
        sub_test, 
        scalar_vec_mul,
        scalar_vec_div,
        mat_mul_1D, 
        mat_mul_2D, 
        mat_vec,
        elementwise_mul,
        relu,
        power,
        module_test,
        module_nested_test,
        backward_add,
        backward_sub,
        backward_mat_mul,
        backward_relu,
        backward_pow,
        backward_scalar_vec_mul,
        backward_scalar_vec_div,
        backward_sum,
        backward_mse,
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




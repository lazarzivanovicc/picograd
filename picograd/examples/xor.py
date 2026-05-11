from picograd.nn.module import Module
from picograd.nn.layer.linear import Linear
from picograd.nn.optim import SGD
from picograd.tensor.tensor_base import Tensor
import numpy as np
from matplotlib import pyplot as plt


def plot_loss(loss: list[float], epochs: int) -> None:
    x: np.ndarray = np.arange(0, epochs, 1)
    y: np.ndarray = np.array(loss)
    plt.plot(x, y)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('XOR Example')
    plt.show()


class NeuralNet(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(2, 15)
        self.layer2 = Linear(15, 1)
    
    def forward(self, x: Tensor):
        x = self.layer1(x)
        x = x.relu()
        x = self.layer2(x)
        return x
    
    def __call__(self, x: Tensor):
        return self.forward(x)
    

if __name__ == "__main__":

    x: Tensor = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]) # data
    y: Tensor = Tensor([[0], [1], [1], [0]]) # targets

    model = NeuralNet()
    optimizer = SGD(model.parameters(), lr=0.001)

    untrained_model = NeuralNet()

    running_loss: list[float] = []
    for epoch in range(2500):
        optimizer.zero_grad()
        predictions = model(x)
        loss = ((predictions - y) ** 2).mean() # I shall wrap this in MSELoss class
        loss.backward()
        optimizer.step()
        running_loss.append(loss.data)
        print(f"Epoch - {epoch}, loss - {loss.data}")



    print(f"\nCOMPARISON BETWEEN TRAINED AND UNTRAINED NEURAL NET:\n")
    print(f"TRAINED NEURAL NET RESULTS")
    print(f"Data:\n {x.data},\nResults:\n {model.forward(x).data}\n")
    print(f"UNTRAINED NEURAL NET RESULTS")
    print(f"Data:\n {x.data},\nResults:\n {untrained_model.forward(x).data}\n")

    plot_loss(running_loss, epochs=2500)
        

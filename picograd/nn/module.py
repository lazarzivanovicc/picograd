from picograd.tensor.tensor_base import Tensor


class Module:
    """
    Base class for each layer and neural network, inspired by pytorch's Module class.

    Simplest Modules like linear layer may contain only simple params of class Tensor.

    Modules can also be nested inside other Modules also
    """
    def parameters(self):
        params: list[Tensor] = []
        for element in self.__dict__.values():
            if isinstance(element, Tensor):
                params.append(element)
            elif isinstance(element, Module):
                params.extend(element.parameters())
        
        return params
    
    def register_buffer(self, name: str, tensor: Tensor | None) -> None:
        """
        Method that allows the registration of non-parameter values. To be implemented.
        """
        pass
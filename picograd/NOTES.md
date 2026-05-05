# Intro
Each neural newtork is a composit function.

Neural network can be represented as mathematical expression.

Mathematical expression can be represented as a graph.


Simple example in pseudocode could look like this:

a = new Node(data=3, parents=())
b = new Node(data=5, parents=())
c = a + b 

where c would be new Node(data=a.data + b.data, parents(a, b))

# Auto diff
Automatic differentiation is a method for computing derivatives and it comes from the field of numerical analysis.

It is different from symbolic differentiation or numerical differentiaion.

AutoDiff leverages computational graph and chain rule in order to precisely and effectively calculate gradients. There are 2 primary forms or modes of AutoDiff.

* **Forward mode:** It is used to calculate gradients (derivatives) during the forward pass through the graph. It is not effective when there is huge number of inputs and scalar output like in neural nets. I think it is better suited for the situations where we have limited amount of inputs and greater number of outputs.

* **Backward mode:** This is essentialy what is artificial neural networks. We first compute the forward pass and we save whatever we need to use during the backward pass computation. We go over the graph starting from loss all the way to each node calculating gradients.

Prior to autodiff people used to manualy write both forward and backward operations for each layer. Check /mangrad to see how OGs did it. Great resource to learn more about this can be found here [8].

Karpathy's micrograd forms Tensor-centric graph meaning that nodes in the graph are tensors, while tinygrad and PyTorch are Function or Operation centric.

# Design
Most of modern deep learning framerworks consist of: tensor library, automatic differentiation enginee and compiler.

## Tensor

## Autograd Enginee

## Compiler

# Resources
[1] Automatic differentiation: https://www.youtube.com/watch?v=wG_nF1awSSY
[2] AD: https://www.youtube.com/watch?v=jS-0aAamC64
[3] Forward AD: ttps://www.youtube.com/watch?v=QwFLA5TrviI
[4] Backprop is not just a chain rule: https://timvieira.github.io/blog/backprop-is-not-just-the-chain-rule/
[5] Autodiff Umass: https://people.cs.umass.edu/~domke/courses/sml2011/08autodiff_nnets.pdf
[6] VecDerivatives Stanford: https://cs231n.stanford.edu/vecDerivs.pdf
[7] Derivatives Stanford: https://cs231n.stanford.edu/handouts/derivatives.pdf 
[8] Lincoln Deep Learning Library: https://github.com/SethHWeidman/lincoln 
import numpy as np
import math
import random

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value:(data={self.data})"

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
    
    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward 

        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")
        def _backward():
            self.grad += 1/other.data * out.grad
            other.grad += -(self.data)/(other.data*other.data) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        out = Value(math.exp(self.data), (self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range (nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum(wi*xi for wi, xi in zip(self.w, x))
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    #a layer contains nout number of neurons. each neuron will take in nin inputs
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        #nin is the number of inputs
        #nouts is indicating the number of outputs per layer? size is simply indicating the nin nout of each layer (and implicitly indicating how many layers there are)
        sz = [nin] + nouts #size = size of number of inputs + size of number of outputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))] #creates layers with nin = sz[i] and output = sz[i+1], as outputs must match inputs for next layer
    
    def __call__(self, x):
        for layer in self.layers:
            x= layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def main():
    n = MLP(3, [4, 4, 1])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    epochs = 1000
    for epoch in range(epochs):
        for p in n.parameters():
            p.grad = 0.0
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
        loss.backward()
        
        for p in n.parameters():
            p.data += -0.01 * p.grad
        print(f"Epoch {epoch} loss: {loss}")

if __name__ == "__main__":
    main()
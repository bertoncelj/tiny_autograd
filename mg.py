import math
import numpy as np
# import mathplotlib.pylot as plt


class Value:
    def __init__(self,data, _children=(), _op='',label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"


    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self,other), "+")
        # out._backward = _backward()

        def _backward():
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self,other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other),"*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad 
        out._backward = _backward
        return out

    def __rmul__(self,other):
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")
        
        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x),(self, ), "exp")

        def _backward():
            self.grad += out.data * out.grad  
        out._backward = _backward

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

def neuron_model():
    #inputs x1,x2 
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    # weights w1, w2
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    # bais of the neouron
    b = Value(6.7, label="b")
    # x1w1 + x2w2 + b
    x1w1 = x1*w1; x1w1.label = "x1*w1"
    x2w2 = x2*w2; x2w2.label = "x2*w2"
    x1w1x2w2 = x1w1 + x2*w2; x1w1x2w2.label = "x1*w1 + x2*w2"
    # add baies
    n = x1w1x2w2  + b; n.label = "n"
    o = n.tanh(); o.label = "o"
    # gradient calculation

    print(o)
    o.backward()
    print(o)

def example1():
    a = Value(3)
    a.exp()
    b =  2 * a
    c =  a + 2
    print(b)
    print(c)
     
    

example1()
neuron_model()

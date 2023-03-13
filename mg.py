import math
import numpy as np
# import mathplotlib.pylot as plt


def f(x):
    return 3*x**2 -4*x + 5

h = 0.00001
x = 2/3

#inputs 
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
a += h
d2 = a*b + c

print("d1", d1)
print("d2", d2)
print("slope", (d2-d1)/h)


class Value:
    def __init__(self,data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self,other):
        out = Value(self.data + other.data)
        return out

    def __mul__(self,other):
        out = Value(self.data * other.data)
        return out

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

print(a*b + c)

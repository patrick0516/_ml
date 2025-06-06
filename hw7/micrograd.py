import math

class Value:
    def __init__(self, data, _children=(), op='', _backward=None):
        self.data = data 
        self._prev = set(_children)  
        self.op = op  
        self._backward = _backward 

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')

    def backward(self):
        if self._backward:
            self._backward()

def f(x, y, z):
    return x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

def gradient_descent(learning_rate=0.1, iterations=100):
    x = Value(5.0) 
    y = Value(5.0) 
    z = Value(5.0) 

    for i in range(iterations):
        f_value = f(x, y, z)
        
        fx = 2 * x.data - 2
        fy = 2 * y.data - 4
        fz = 2 * z.data - 6
        
        x.data -= learning_rate * fx
        y.data -= learning_rate * fy
        z.data -= learning_rate * fz
        
        print(f"Iteration {i+1}: x = {x.data}, y = {y.data}, z = {z.data}, f(x, y, z) = {f_value}")

gradient_descent(learning_rate=0.1, iterations=100)

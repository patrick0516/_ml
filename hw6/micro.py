import math

class Value:
    def __init__(self, data, _children=(), op='', _backward=None):
        self.data = data  
        self._prev = set(_children) 
        self.op = op 
        self._backward = _backward  

    def sigmoid(self):
        return Value(1 / (1 + math.exp(-self.data)), (self,), 'sigmoid', self._sigmoid_backward)

    def _sigmoid_backward(self):
        sigmoid_val = 1 / (1 + math.exp(-self.data))
        return sigmoid_val * (1 - sigmoid_val)

    def exp(self):
        return Value(math.exp(self.data), (self,), 'exp', self._exp_backward)

    def _exp_backward(self):
        return math.exp(self.data)

x = Value(0.5)
sigmoid_result = x.sigmoid()
exp_result = x.exp()

print("Sigmoid:", sigmoid_result.data)
print("Exp:", exp_result.data)

"""
>>> from chainer import Chain
>>> from chainer import links as L
>>> from chainer.optimizers import SGD
>>> import numpy as np

>>> class Model(Chain):
...    def __init__(self):
...        super(Model, self).__init__()
...        with self.init_scope():
...            self.some_link = L.Convolution2D(10, 10)
...            self.some_parameter = Parameter(np.array([1.]))
...
>>> model = Model()
>>> optimizer = SGD(lr=0.1)
>>> optimizer.setup(model)

>>> print(optimizer.lr)
{'some_link/W': 0.1, 'some_link/b': 0.1, 'some_parameter': 0.1}
>>> print(model.lr)
{'some_link/W': 0.1, 'some_link/b': 0.1, 'some_parameter': 0.1}
>>> print(model.some_link.lr)
{'W': 0.1, 'b': 0.1}

>>> optimizer.lr *= 0.1
>>> print(optimizer.lr)
{'some_link/W': 0.01, 'some_link/b': 0.01, 'some_parameter': 0.01}

>>> model.some_link.lr *= 5
>>> print(optimizer.lr)
{'some_link/W': 0.05, 'some_link/b': 0.05, 'some_parameter': 0.01}

>>> model.some_parameter.lr *= 0.1
>>> print(optimizer.lr)
{'some_link/W': 0.05, 'some_link/b': 0.05, 'some_parameter': 0.001}
"""
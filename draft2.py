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

>>> print(optimizer.hyperparam.lr)
{'some_link/W': 0.1, 'some_link/b': 0.1, 'some_parameter': 0.1}
>>> print(model.hyperparam.lr)
{'some_link/W': 0.1, 'some_link/b': 0.1, 'some_parameter': 0.1}
>>> print(model.some_link.hyperparam.lr)
{'W': 0.1, 'b': 0.1}

>>> optimizer.hyperparam.lr *= 0.1
>>> print(optimizer.hyperparam.lr)
{'some_link/W': 0.01, 'some_link/b': 0.01, 'some_parameter': 0.01}

>>> model.some_link.hyperparam.lr *= 5
>>> print(optimizer.hyperparam.lr)
{'some_link/W': 0.05, 'some_link/b': 0.05, 'some_parameter': 0.01}

>>> model.some_parameter.hyperparam.lr *= 0.1
>>> print(optimizer.hyperparam.lr)
{'some_link/W': 0.05, 'some_link/b': 0.05, 'some_parameter': 0.001}
"""
"""
利点
- 統一感がある
- hyperparamがプロキシを返すのはそれなりにわかりやすいような気がする

欠点
- 既存のコードのoptimizer.lr *= 2を修正する必要がある

- 実装面が複雑
-- optimizer.hyperparam.lrが__repr__や__imul__を備えたプロキシを返す関数である必要がある
--- プロキシは内部的にLink.params()の返り値を走査し，そのupdate_rules.hyperparamを書き換えて行く

-- draft1と同様，大量の修正が必要になる

- UIとしての落ち度がある
-- model内に，parameterを共有している2つのLinkがあったとき，\
   model.linkA.lr *= 0.1, model.linkB.lr *= 0.1すると，予期せず共有パラメータのlrが0.01倍されてしまう
"""
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

>>> print(optimizer.hyperparams())
{'some_link/W': {'lr': 0.1}, 'some_link/b': {'lr': 0.1}, 'some_parameter': {'lr': 0.1}}
>>> for param_name, hyperparam in model.hyperparams():
...     print(param_name, hyperparam)
some_link/W {'lr': 0.1}
some_link/b {'lr': 0.1}
some_parameter {'lr': 0.1}
>>> print(model.some_link.hyperparams())
{'W': {'lr': 0.1}, 'b': {'lr': 0.1}}

>>> for _, hyperparam in optimizer.hyperparams():
...     hyperparam.lr *= 0.1
...
>>> print(optimizer.hyperparams())
{'some_link/W': {'lr': 0.01}, 'some_link/b': {'lr': 0.01}, 'some_parameter': {'lr': 0.01}}

>>> for _, hyperparam in model.some_link.hyperparams():
...     hyperparam.lr *= 5
...
>>> print(optimizer.hyperparams())
{'some_link/W': {'lr': 0.05}, 'some_link/b': {'lr': 0.05}, 'some_parameter': {'lr': 0.01}}

>>> for _, hyperparam in model.some_parameter.hyperparams():
...     hyperparam.lr *= 0.1
...
>>> print(optimizer.hyperparams())
{'some_link/W': {'lr': 0.05}, 'some_link/b': {'lr': 0.05}, 'some_parameter': {'lr': 0.001}}
"""
"""
利点
- 統一感がある
-- 特に，params()と同じメソッドで呼び出せるのは良い
- 一切ラッパークラスが必要ない
- 実装が簡単
-- hyperparams()はparams()を呼ばず，似た動作を名前を保存させながら行えば良い

欠点
- 既存のコードのoptimizer.lr *= 2を修正する必要がある

- params()と似た関数なのに返り値型が違う
-- いっそparams()も修正してしまえば統一感は出る
--- 修正箇所は増える
-- ここをサボると，__repr__の出力が[{'lr': 0.05}, {'lr': 0.05}, {'lr': 0.05}]みたいになってわけがわからない

- UIとしての落ち度がある
-- model内に，parameterを共有している2つのLinkがあったとき，\
   model.linkA.lr *= 0.1, model.linkB.lr *= 0.1すると，予期せず共有パラメータのlrが0.01倍されてしまう
"""
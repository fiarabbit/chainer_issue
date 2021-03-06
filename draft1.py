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
"""
利点
- 直感的でわかりやすい
- 既存のコードのoptimizer.lr *= 2を修正しなくてよい
欠点
- 実装面が複雑
-- optimizer.lrが__repr__や__imul__を備えたプロキシを返す関数である必要がある
--- プロキシは内部的にLink.params()の返り値を走査し，そのupdate_rules.hyperparamを書き換えて行く

-- optimizerの属性アクセスの返り値型がわかりにくい
--- 存在しない属性アクセス値を全てHyperparameterの名称だと思って本当に良いのか疑問

-- Parameter, Link, Chain, ChainList, Optimizer, Hyperparameterを全部書き換える必要がある
--- Parameter
---- 現時点で属性アクセスでハイパラが取れない
--- Link
---- Link.params()がパラメータの名前を返す必要がある(Proxy.__repr__のため)
--- Chain, ChainList
---- Chain.params()がリンクの名前とパラメータの名前を返す必要がある(同上)
--- Optimizer
---- Proxyの実装
--- などなど，修正箇所は多岐に渡る

- UIとしての落ち度がある
-- model内に，parameterを共有している2つのLinkがあったとき，\
   model.linkA.lr *= 0.1, model.linkB.lr *= 0.1すると，予期せず共有パラメータのlrが0.01倍されてしまう
"""
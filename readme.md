- やりたいこと
    - `optimizer.target`の子パラメータの`UpdateRule`の`lr`を一括更新
    - `model.conv1`の子パラメータの`UpdateRule`の`lr`を一括更新
    - (子パラメータの名称とその`update_rule.hyperparam`の全表示)


- draft1.py
    - 最も直感的な操作を目指した
    - アクセス方法
        - `optimizer.lr`
        - `model.conv1.lr`
        - `model.conv1.W.lr`
    - `optimizer.totally_irrelevant_attribute`が何を返すべきか不明瞭

- draft2.py
    - draft1の問題を解決しようとおもった
    - アクセス方法
        - `optimizer.hyperparam.lr`
        - `model.conv1.hyperparam.lr`
        - `model.conv1.W.hyperparam.lr`
    - 実装が闇

- draft3.py
    - 闇な実装を排除した
    - アクセス方法
        - `[h.lr for h in optimizer.hyperparams()]`
        - `[h.lr for h in model.conv1.hyperparams()]`
        - `[h.lr for h in model.conv1.W.hyperparams()]`
    - 一括更新には対応していない
        - 手動でfor文を回してもらう
        - 現状とほとんど変わらない？
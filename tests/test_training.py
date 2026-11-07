import numpy as np
from layers import Dense, ReLU
from losses import SoftmaxCrossEntropy
from model import Sequential, SGD


def make_model():
    return Sequential([Dense(20, 32, seed=0), ReLU(), Dense(32, 3, seed=1)])


def test_params_collects_weights_and_grads():
    model = make_model()
    x = np.random.default_rng(0).normal(size=(4, 20)).astype(np.float32)
    out = model.forward(x)
    model.backward(np.ones_like(out))
    pairs = model.params()
    assert len(pairs) == 4                          # W, b for each Dense; ReLU adds none
    for p, g in pairs:
        assert p.shape == g.shape


def test_sgd_can_overfit_tiny_batch():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 20)).astype(np.float32)
    y = rng.integers(0, 3, size=16)
    model, loss_fn = make_model(), SoftmaxCrossEntropy()
    opt = SGD(model, lr=0.1)
    first = None
    for _ in range(200):
        loss = loss_fn.forward(model.forward(x), y)
        first = loss if first is None else first
        model.backward(loss_fn.backward())
        opt.step()
    assert loss < 0.5 * first
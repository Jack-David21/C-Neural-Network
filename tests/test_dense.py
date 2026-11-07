import numpy as np
from layers import Dense, ReLU
from losses import SoftmaxCrossEntropy
from tests.gradcheck import numerical_grad


def test_dense_shapes():
    layer = Dense(8, 5, seed=0)
    x = np.random.default_rng(0).normal(size=(4, 8)).astype(np.float32)
    out = layer.forward(x)
    dx = layer.backward(np.ones_like(out))
    assert out.shape == (4, 5)
    assert dx.shape == x.shape
    assert layer.dW.shape == layer.W.shape
    assert layer.db.shape == layer.b.shape


def test_dense_gradients():
    rng = np.random.default_rng(0)
    layer = Dense(8, 5, dtype=np.float64, seed=1)    # float64 for precision
    x = rng.normal(size=(4, 8))
    w = rng.normal(size=(4, 5))                      # random weights make a scalar loss

    def loss():
        return (layer.forward(x) * w).sum()

    loss()
    dx = layer.backward(w)
    assert np.abs(dx - numerical_grad(loss, x)).max() < 1e-7
    assert np.abs(layer.dW - numerical_grad(loss, layer.W)).max() < 1e-7
    assert np.abs(layer.db - numerical_grad(loss, layer.b)).max() < 1e-7


def test_dense_relu_dense_softmax_chain():
    rng = np.random.default_rng(2)
    d1 = Dense(8, 6, dtype=np.float64, seed=3)
    relu = ReLU()
    d2 = Dense(6, 3, dtype=np.float64, seed=4)
    loss_fn = SoftmaxCrossEntropy()
    x = rng.normal(size=(7, 8))
    y = rng.integers(0, 3, size=7)

    def loss():
        return loss_fn.forward(d2.forward(relu.forward(d1.forward(x))), y)

    loss()
    d1.backward(relu.backward(d2.backward(loss_fn.backward())))
    assert np.abs(d1.dW - numerical_grad(loss, d1.W)).max() < 1e-7
    assert np.abs(d2.dW - numerical_grad(loss, d2.W)).max() < 1e-7
    assert np.abs(d1.db - numerical_grad(loss, d1.b)).max() < 1e-7
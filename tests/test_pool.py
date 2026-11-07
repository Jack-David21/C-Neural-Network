import numpy as np
from layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU
from losses import SoftmaxCrossEntropy
from model import Sequential
from tests.gradcheck import numerical_grad


def test_pool_hand_example():
    x = np.array([[[[1, 3, 2, 0],
                    [4, 2, 1, 5],
                    [0, 1, 7, 2],
                    [3, 6, 4, 1]]]], dtype=np.float64)
    pool = MaxPool2D(2)
    out = pool.forward(x)
    assert np.array_equal(out[0, 0], [[4, 5], [6, 7]])
    dx = pool.backward(np.ones_like(out))
    expected = np.array([[0, 0, 0, 0],
                         [1, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0]], dtype=np.float64)
    assert np.array_equal(dx[0, 0], expected)       # gradient lands only on the maxima


def test_pool_shapes():
    pool = MaxPool2D(2)
    assert pool.forward(np.zeros((4, 16, 28, 28), dtype=np.float32)).shape == (4, 16, 14, 14)
    assert pool.forward(np.zeros((4, 32, 14, 14), dtype=np.float32)).shape == (4, 32, 7, 7)


def test_pool_gradient():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 6, 6))               # random values: no ties
    pool = MaxPool2D(2)
    w = rng.normal(size=(2, 3, 3, 3))               # random weights make a scalar loss

    def loss():
        return (pool.forward(x) * w).sum()

    loss()
    dx = pool.backward(w)
    assert np.abs(dx - numerical_grad(loss, x)).max() < 1e-7


def test_pool_tie_goes_to_one_position():
    x = np.ones((1, 1, 2, 2))
    pool = MaxPool2D(2)
    out = pool.forward(x)
    dx = pool.backward(np.ones_like(out))
    assert dx.sum() == 1                            # exactly one position gets the gradient


def test_flatten_roundtrip():
    x = np.random.default_rng(0).normal(size=(4, 32, 7, 7))
    fl = Flatten()
    out = fl.forward(x)
    assert out.shape == (4, 1568)
    assert np.array_equal(fl.backward(out), x)


def test_full_cnn_chain_gradients():
    rng = np.random.default_rng(0)
    c1 = Conv2D(1, 2, dtype=np.float64, seed=1)
    c2 = Conv2D(2, 3, dtype=np.float64, seed=2)
    head = Dense(12, 3, dtype=np.float64, seed=3)   # 3 maps x 2 x 2 after two pools on 8x8
    model = Sequential([c1, ReLU(), MaxPool2D(2), c2, ReLU(), MaxPool2D(2), Flatten(), head])
    loss_fn = SoftmaxCrossEntropy()
    x = rng.normal(size=(2, 1, 8, 8))
    y = rng.integers(0, 3, size=2)

    def loss():
        return loss_fn.forward(model.forward(x), y)

    loss()
    model.backward(loss_fn.backward())
    for p, g in [(c1.W, c1.dW), (c1.b, c1.db), (c2.W, c2.dW), (head.W, head.dW)]:
        assert np.abs(g - numerical_grad(loss, p)).max() < 1e-6
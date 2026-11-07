import numpy as np
from layers import ReLU
from losses import SoftmaxCrossEntropy
from tests.gradcheck import numerical_grad


def test_loss_of_zero_logits_is_ln10():
    loss = SoftmaxCrossEntropy().forward(np.zeros((4, 10)), np.array([0, 1, 2, 3]))
    assert abs(loss - np.log(10)) < 1e-12          # 2.3026...


def test_softmax_ce_gradient():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 10))              # float64 for precision
    y = rng.integers(0, 10, size=5)
    fn = SoftmaxCrossEntropy()
    fn.forward(logits, y)
    analytic = fn.backward()
    numeric = numerical_grad(lambda: fn.forward(logits, y), logits)
    assert np.abs(analytic - numeric).max() < 1e-7


def test_relu_gradient():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 6))
    w = rng.normal(size=(4, 6))                    # random weights make a scalar loss
    relu = ReLU()
    relu.forward(x)
    analytic = relu.backward(w)
    numeric = numerical_grad(lambda: (relu.forward(x) * w).sum(), x)
    assert np.abs(analytic - numeric).max() < 1e-7
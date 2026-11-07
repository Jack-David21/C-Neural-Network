import numpy as np
import pytest
from layers import Conv2D
from tests.gradcheck import numerical_grad


def naive_conv(x, W, b, stride, pad):
    """Slow reference: the sliding window written as plain loops."""
    B, C, H, Wd = x.shape
    Co, _, k, _ = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (Wd + 2 * pad - k) // stride + 1
    out = np.zeros((B, Co, Ho, Wo))
    for n in range(B):
        for f in range(Co):
            for i in range(Ho):
                for j in range(Wo):
                    patch = xp[n, :, i * stride:i * stride + k, j * stride:j * stride + k]
                    out[n, f, i, j] = (patch * W[f]).sum() + b[f]
    return out


def test_hand_example():
    """The 3x3 image and two 2x2 filters from the walkthrough."""
    layer = Conv2D(1, 2, k=2, stride=1, pad=0, dtype=np.float64)
    layer.W[:] = [[[[1, 0], [0, -1]]], [[[0, 1], [1, 0]]]]
    x = np.array([[[[1, 2, 0], [0, 1, 3], [2, 1, 0]]]], dtype=np.float64)
    out = layer.forward(x)
    assert np.array_equal(out[0, 0], [[0, -1], [-1, 1]])
    assert np.array_equal(out[0, 1], [[2, 1], [3, 4]])


def test_network_shapes():
    c1 = Conv2D(1, 16, seed=0)
    c2 = Conv2D(16, 32, seed=1)
    x = np.zeros((4, 1, 28, 28), dtype=np.float32)
    assert c1.forward(x).shape == (4, 16, 28, 28)
    assert c2.forward(np.zeros((4, 16, 14, 14), dtype=np.float32)).shape == (4, 32, 14, 14)


@pytest.mark.parametrize("stride,pad", [(1, 1), (2, 0), (2, 1)])
def test_forward_matches_naive(stride, pad):
    rng = np.random.default_rng(0)
    layer = Conv2D(3, 4, k=3, stride=stride, pad=pad, dtype=np.float64, seed=1)
    layer.b[:] = rng.normal(size=4)
    x = rng.normal(size=(2, 3, 6, 6))
    out = layer.forward(x)
    assert np.abs(out - naive_conv(x, layer.W, layer.b, stride, pad)).max() < 1e-10


@pytest.mark.parametrize("stride,pad", [(1, 1), (2, 0), (2, 1)])
def test_conv_gradients(stride, pad):
    rng = np.random.default_rng(0)
    layer = Conv2D(2, 3, k=3, stride=stride, pad=pad, dtype=np.float64, seed=1)
    layer.b[:] = rng.normal(size=3)
    x = rng.normal(size=(2, 2, 6, 6))
    w = rng.normal(size=layer.forward(x).shape)       # random weights make a scalar loss

    def loss():
        return (layer.forward(x) * w).sum()

    loss()
    dx = layer.backward(w)
    assert np.abs(dx - numerical_grad(loss, x)).max() < 1e-7
    assert np.abs(layer.dW - numerical_grad(loss, layer.W)).max() < 1e-7
    assert np.abs(layer.db - numerical_grad(loss, layer.b)).max() < 1e-7
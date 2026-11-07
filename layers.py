import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class ReLU:
    def forward(self, x):
        self.mask = x > 0              
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask 

    def params(self):
        return []       

class Dense:
    def __init__(self, d_in, d_out, dtype=np.float32, seed=None):
        rng = np.random.default_rng(seed)
        # He initialization: std = sqrt(2 / d_in)
        self.W = rng.normal(0.0, np.sqrt(2.0 / d_in), size=(d_in, d_out)).astype(dtype)
        self.b = np.zeros(d_out, dtype=dtype)
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x                       # needed by backward
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout        # (D_in, D_out)
        self.db = dout.sum(axis=0)       # (D_out,)
        return dout @ self.W.T           # dx, (N, D_in)
    
    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]

def im2col(x, k, stride, pad):
    """(B, C, H, W) -> (B*Ho*Wo, C*k*k): one row per patch."""
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::stride, ::stride]
    B, C, Ho, Wo = win.shape[:4]                      # win: (B, C, Ho, Wo, k, k)
    return win.transpose(0, 2, 3, 1, 4, 5).reshape(B * Ho * Wo, C * k * k)


def col2im(cols, x_shape, k, stride, pad):
    """Reverse of im2col for gradients: scatter rows back to pixels, adding overlaps."""
    B, C, H, W = x_shape
    Hp, Wp = H + 2 * pad, W + 2 * pad
    Ho = (Hp - k) // stride + 1
    Wo = (Wp - k) // stride + 1
    cols = cols.reshape(B, Ho, Wo, C, k, k).transpose(0, 3, 4, 5, 1, 2)  # (B, C, k, k, Ho, Wo)
    x = np.zeros((B, C, Hp, Wp), dtype=cols.dtype)
    for i in range(k):                                # only k*k slice-adds, no pixel loops
        for j in range(k):
            x[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride] += cols[:, :, i, j]
    return x[:, :, pad:Hp - pad, pad:Wp - pad] if pad else x

class Conv2D:
    def __init__(self, c_in, c_out, k=3, stride=1, pad=1, dtype=np.float32, seed=None):
        rng = np.random.default_rng(seed)
        fan_in = c_in * k * k                         # He init: d_in is the patch size
        self.W = rng.normal(0.0, np.sqrt(2.0 / fan_in),
                            size=(c_out, c_in, k, k)).astype(dtype)
        self.b = np.zeros(c_out, dtype=dtype)
        self.k, self.stride, self.pad = k, stride, pad
        self.dW = None
        self.db = None

    def forward(self, x):
        B, C, H, W = x.shape
        Ho = (H + 2 * self.pad - self.k) // self.stride + 1
        Wo = (W + 2 * self.pad - self.k) // self.stride + 1
        c_out = self.W.shape[0]
        self.x_shape = x.shape
        self.cols = im2col(x, self.k, self.stride, self.pad)      # kept for backward
        out = self.cols @ self.W.reshape(c_out, -1).T + self.b    # (B*Ho*Wo, c_out)
        out = out.reshape(B, Ho, Wo, c_out).transpose(0, 3, 1, 2)
        return np.ascontiguousarray(out)                          # (B, c_out, Ho, Wo)

    def backward(self, dout):
        c_out = self.W.shape[0]
        d2 = dout.transpose(0, 2, 3, 1).reshape(-1, c_out)        # (B*Ho*Wo, c_out)
        self.dW = (d2.T @ self.cols).reshape(self.W.shape)
        self.db = d2.sum(axis=0)
        dcols = d2 @ self.W.reshape(c_out, -1)                    # (B*Ho*Wo, C*k*k)
        return col2im(dcols, self.x_shape, self.k, self.stride, self.pad)

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]

class MaxPool2D:
    """k x k max pooling with stride k (non-overlapping windows)."""

    def __init__(self, k=2):
        self.k = k

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.k
        assert H % k == 0 and W % k == 0, "H and W must be divisible by k"
        Ho, Wo = H // k, W // k
        self.shape = x.shape
        # (B, C, Ho, k, Wo, k) -> (B, C, Ho, Wo, k*k): each window becomes one row of k*k values
        win = x.reshape(B, C, Ho, k, Wo, k).transpose(0, 1, 2, 4, 3, 5).reshape(B, C, Ho, Wo, k * k)
        self.idx = win.argmax(axis=-1)                       # winner position inside each window
        return np.take_along_axis(win, self.idx[..., None], axis=-1)[..., 0]

    def backward(self, dout):
        B, C, H, W = self.shape
        k = self.k
        Ho, Wo = H // k, W // k
        dwin = np.zeros((B, C, Ho, Wo, k * k), dtype=dout.dtype)
        np.put_along_axis(dwin, self.idx[..., None], dout[..., None], axis=-1)  # gradient only to the winner
        return dwin.reshape(B, C, Ho, Wo, k, k).transpose(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

    def params(self):
        return []                                            # nothing to learn


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)

    def params(self):
        return []
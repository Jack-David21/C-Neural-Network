import numpy as np

from layers import Conv2D, MaxPool2D, Flatten, Dense, ReLU

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):         
            dout = layer.backward(dout)
        return dout

    def params(self):
        return [pg for layer in self.layers for pg in layer.params()]

class SGD:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

    def step(self):
        for p, g in self.model.params():            
            p -= self.lr * g                        

def build_baseline(hidden=256, seed=0):
    return Sequential([
        Dense(784, hidden, seed=seed),
        ReLU(),
        Dense(hidden, 10, seed=seed + 1),
    ])

def build_cnn(head="A", seed=0):
    if head not in ("A", "B"):
        raise ValueError("head must be 'A' or 'B'")
    layers = [
        Conv2D(1, 16, k=3, stride=1, pad=1, seed=seed),
        ReLU(),
        MaxPool2D(k=2),
        Conv2D(16, 32, k=3, stride=1, pad=1, seed=seed + 1),
        ReLU(),
        MaxPool2D(k=2),
        Flatten(),
    ]

    if head == "A":
        layers.append(Dense(1568, 10, seed=seed + 2))
    else:
        layers += [
            Dense(1568, 128, seed=seed + 2),
            ReLU(),
            Dense(128, 10, seed=seed + 3),
        ]

    return Sequential(layers)

def build_cnn_stacked(seed=0):
    """Two 3x3 convs before each pool (28x28 -> 14x14 -> 7x7), then head B."""
    return Sequential([
        Conv2D(1, 16, seed=seed), ReLU(),
        Conv2D(16, 16, seed=seed + 1), ReLU(),
        MaxPool2D(k=2),
        Conv2D(16, 32, seed=seed + 2), ReLU(),
        Conv2D(32, 32, seed=seed + 3), ReLU(),
        MaxPool2D(k=2),
        Flatten(),
        Dense(1568, 128, seed=seed + 4), ReLU(),
        Dense(128, 10, seed=seed + 5),
    ])

def load_weights(model, path):
    """Copy the saved arrays (arr_0, arr_1, ...) into the model, in params() order."""
    with np.load(path) as f:
        saved = [f[f"arr_{i}"] for i in range(len(f.files))]
    params = model.params()
    assert len(saved) == len(params), "weight file does not match this model"
    for (p, _), w in zip(params, saved):
        assert p.shape == w.shape, "weight shape mismatch"
        p[...] = w
    return model
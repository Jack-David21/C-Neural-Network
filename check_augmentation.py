import numpy as np
import matplotlib.pyplot as plt

from train import DATA, augment_batch

X = np.load(DATA)["X_val"][::800][:8]          # 8 drawings from different classes
aug = augment_batch(X, np.random.default_rng(0))
fig, ax = plt.subplots(2, 8, figsize=(12, 3.2))
for j in range(8):
    ax[0, j].imshow(X[j], cmap="gray_r")
    ax[1, j].imshow(aug[j], cmap="gray_r")
    ax[0, j].axis("off")
    ax[1, j].axis("off")
fig.savefig("augment_check.png", dpi=120)
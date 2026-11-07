import numpy as np


class SoftmaxCrossEntropy:

    def forward(self, logits, y):
        n = logits.shape[0]
        z = logits - logits.max(axis=1, keepdims=True)    
        exp = np.exp(z)
        sum_exp = exp.sum(axis=1, keepdims=True)
        self.probs = exp / sum_exp
        self.y = y
        # log p_correct = z_correct - log(sum_exp), avoids log(0)
        log_p = z[np.arange(n), y] - np.log(sum_exp[:, 0])
        return -log_p.mean()

    def backward(self):
        n = self.probs.shape[0]
        dlogits = self.probs.copy()
        dlogits[np.arange(n), self.y] -= 1                
        return dlogits / n
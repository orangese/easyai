"""

"easyai._advanced._optimizers.py"

Advanced keras optimizers. Not for use by easyai users-- does not use easyai API.

"""

from time import time

import numpy as np

def gradient_ascent(x, fn, iters, lr, max_loss=np.float("inf"), verbose=True):
    for iteration in range(iters):
        start = time()
        loss, grad = fn([x])
        if loss > max_loss:
            if verbose:
                print("Loss exceeded max loss")
            break
        if verbose:
            print(" - {}s - iteration: {} - loss: {}".format(round(time() - start), iteration, round(loss, 2)))
        x += lr * np.array(grad)
    return x
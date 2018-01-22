import numpy as np
import matplotlib.pyplot as plt

lr = 5e-6
dr = 0.01 /25
ep = np.arange(0, 1500+1)
e2t = ep.copy()
e2t[:25] *= 100
e2t[25:] = e2t[24] + 5 * np.arange(1, ep.size-25+1)
it = np.arange(0, e2t[-1]+1)
learning = lambda t: lr * 1. / (1. + dr * t)
plt.subplot(121)
plt.xlabel("Iterations")
plt.plot(it, learning(it))
plt.subplot(122)
plt.xlabel("Epochs")
plt.plot(ep, learning(e2t))
plt.show()
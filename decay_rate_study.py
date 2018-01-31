import numpy as np
import matplotlib.pyplot as plt

lr = 5e-4
L = 85
dr = 1E-2 /L
ep = np.arange(0, 50)
#e2t = ep.copy()
#e2t[:25] *= 100
#e2t[25:] = e2t[24] + 5 * np.arange(1, ep.size-25+1)
#it = np.arange(0, e2t[-1]+1)
it = ep*L
learning = lambda t: lr * 1. / (1. + dr * t)
#plt.subplot(121)
#plt.xlabel("Iterations")
#plt.plot(it, learning(it))
#plt.subplot(122)
plt.xlabel("Epochs")
plt.stem(ep, learning(it))
#plt.plot(ep, learning(e2t))
plt.show()
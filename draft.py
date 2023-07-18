import numpy as np

idx = np.arange(10)
true_coef = (-1) ** idx * np.exp(-idx / 10)
print(true_coef)
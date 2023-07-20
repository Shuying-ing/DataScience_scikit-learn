import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.0, 2.0 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.stem(x, y)

plt.show()

"""
make_blobs() function generates a random n-class classification problem with specified parameters

"""

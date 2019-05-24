import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(start=-10, stop=10, num=1000, endpoint=True)
y1 = np.exp(-0.1 * x**2)
y2 = np.exp(-10 * x**2)

plt.plot(x, y1, color="red", label="small gamma")
plt.plot(x, y2, color="blue", ls="--", label="large gamma")
plt.legend()
plt.show()

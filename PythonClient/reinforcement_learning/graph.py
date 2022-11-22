import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv('out.csv')
first_column = df.iloc[:, 1]
plt.plot(first_column)


window =50
average_y = []
for ind in range(len(first_column) - window + 1):
    average_y.append(np.mean(first_column[ind:ind+window]))

plt.plot(average_y)
plt.show()
plt.savefig('foo.png')

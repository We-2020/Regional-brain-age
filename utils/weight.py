import pandas as pd
import numpy as np


matrix = [1107, 1214, 1696, 4041, 980, 348, 201, 225, 317, 380, 470, 604, 923, 1134, 887, 564, 296, 41, 9]

max_val = max(matrix)

for i in range(len(matrix)):
    # matrix[i] = 1 - matrix[i] / sta
    matrix[i] = min(3,float(max_val) / matrix[i])
print(matrix)

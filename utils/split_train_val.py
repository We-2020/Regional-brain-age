import pandas as pd
from tqdm import tqdm
import numpy as np


data_list = pd.read_csv("../config1w/disease/ADNI_disease_3.csv", header=None)

data = []
for index, row in tqdm(data_list.iterrows()):
    pid = str(row[0])
    label = float(row[2])
    data.append([pid, label])

np.random.shuffle(data)
length = len(data)
c1 = int(length * 0.2)
c2 = int(length * 0.4)
c3 = int(length * 0.6)
c4 = int(length * 0.8)
data_1 = data[0:c1]
data_2 = data[c1:c2]
data_3 = data[c2:c3]
data_4 = data[c3:c4]
data_5 = data[c4:length]
data_total = [data_1 ,data_2, data_3, data_4, data_5]

for i in range(5):
    val_data = data_total[i]
    train_data = []
    for j in range(5):
        if i != j:
            train_data += data_total[j]

    np.savetxt('../config1w/configDisease/train_ADNI3_' + str(i+1) + '.csv', train_data, '%s', delimiter=',')
    np.savetxt('../config1w/configDisease/val_ADNI3_' + str(i+1) + '.csv', val_data, '%s', delimiter=',')


# c1 = int(length * 0.9)
# data_val = data[c1:length]
# data_train = data[0:c1]
#
# np.savetxt('./config/train_6200.csv', data_train, '%s', delimiter=',')
# np.savetxt('./config/val_6200.csv', data_val, '%s', delimiter=',')
#
# print('total: {} | train: {} | val: {}'
#       .format(length, len(data_train), len(data_val)))
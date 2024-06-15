import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from torch.nn import functional as F
import glob
import torch
import nibabel as nib
from collections import defaultdict
import torchvision.transforms as transforms
from torch.utils.data import Dataset,ConcatDataset,DataLoader




def get_Dataframe_Data(test_size,path,name,random_state=257572):
    a = pd.read_csv(path,header=None)
    a = a[a[5] == name]
    data = a.iloc[:,2:4]
    train,val = train_test_split(data,test_size = test_size,random_state=random_state)
#     return train,val
    return train,val


class dataGenerator(Dataset):

    def __init__(self, data, env):
        self.data = data
        self.env = env
        self.transform = transforms.Compose([
            transforms.ToTensor()
                                             ])

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
#         id, age,sex = self.data.iloc[index] sex改动
        ids,age = self.data.iloc[index]
        with self.env.begin(write=False) as txn:
            buf = txn.get(ids.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        x = img_flat.copy().reshape(91, 109, 91)      # np格式
        img = (x-np.min(x))/(np.max(x)-np.min(x))


        img = self.transform(img)  # np -> torch  [91, 109, 91]
        img = img.unsqueeze(0)       # 增加一个维度，留给channel , 过3dcnn  [1, 91, 109, 91]
        label = torch.FloatTensor([float(age)]).squeeze()
        sample = img, label

        return sample
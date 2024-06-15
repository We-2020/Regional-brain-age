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
import random
from monai import transforms as tf


def gen_cord(imgsize=(128, 128, 128), scale=(0.08, 1.), ratio = (3 / 4, 4 / 3)):
    sc = random.random() * (scale[1] - scale[0]) + scale[0]
    ra1 = random.random() * (ratio[1] - ratio[0]) + ratio[0]
    ra2 = random.random() * (ratio[1] - ratio[0]) + ratio[0]

    x = (sc * imgsize[0] * imgsize[0] * imgsize[0] / (ra1 * ra2)) ** (1 / 3)
    y = ra1 * x
    z = ra2 * x
    return int(x), int(y), int(z)

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
        self.x, self.y, self.z = gen_cord(imgsize=(91, 91, 109))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
                                             ])

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
#         id, age,sex = self.data.iloc[index] sex改动
        ids,age = self.data.iloc[index]
        with self.env.begin(write=False) as txn:
            buf = txn.get(ids.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        x = img_flat.copy().reshape(1,91, 109, 91)      # np格式
        img = (x-np.min(x))/(np.max(x)-np.min(x))
        x, y, z = gen_cord(imgsize=(91, 109, 91))
        crop_op = tf.CenterSpatialCrop([x, y, z])
        resize_op = tf.Resize([128, 128, 128])
        img = resize_op(crop_op(img))
        
        label = torch.FloatTensor([float(age)]).squeeze()
        sample = img, label

        return sample
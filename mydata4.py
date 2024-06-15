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

# def get_Dataframe_Data(test_size,path):
#     data = pd.read_excel(path,header=None)
#     train,val = train_test_split(data,test_size = test_size,random_state=257572)
# #     return train,val
#     return train,val

# class dataGenerator(Dataset):

#     def __init__(self, data, env):
#         self.data = data
#         self.env = env
#         self.image_path1 = "ROI_MNI_V4.nii"
#         self.image_obj1 = nib.load(self.image_path1)
#         self.image_data1 = np.array(self.image_obj1.get_fdata())
#         self.transform = transforms.Compose([
#             transforms.ToTensor()
#                                              ])

#     def __len__(self):  # 返回整个数据集的大小
#         return len(self.data)

#     def __getitem__(self, index):  # 根据索引index返回dataset[index]
# #         id, age,sex = self.data.iloc[index] sex改动
#         id, age = self.data.iloc[index]

#         with self.env.begin(write=False) as txn:
#             buf = txn.get(id.encode())
#         img_flat = np.frombuffer(buf, dtype=np.float32)
#         x = img_flat.copy().reshape(91, 109, 91) # np格式
#         img = (x-np.min(x))/(np.max(x)-np.min(x))
# #         img[~(self.image_data1 == self.i)] = 0


#         img = self.transform(img)  # np -> torch  [91, 109, 91]
#         img = img.unsqueeze(0)       # 增加一个维度，留给channel , 过3dcnn  [1, 91, 109, 91]
#         label = torch.FloatTensor([float(age)]).squeeze()
#         sample = img, label

#         return sample

def get_Dataframe_Data(test_size,a,name):
    a = a[a[5] == name]
    data = a.iloc[:,2:4]
    if len(data) < 20:
        train = None
        val = data
    else:
        train,val = train_test_split(data,test_size = test_size,random_state=257572)
#     return train,val
    return train,val


class dataGenerator(Dataset):

    def __init__(self, data, env,i):
        self.data = data
        self.env = env
        self.i = i
        self.image_path1 = "ROI_MNI_V4.nii"
        self.image_obj1 = nib.load(self.image_path1)
        self.image_data1 = np.array(self.image_obj1.get_fdata())
        self.transform = transforms.Compose([
            transforms.ToTensor()
                                             ])

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
#         id, age,sex = self.data.iloc[index] sex改动
        id, age = self.data.iloc[index]

        with self.env.begin(write=False) as txn:
            buf = txn.get(id.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        x = img_flat.copy().reshape(91, 109, 91) # np格式
        img = (x-np.min(x))/(np.max(x)-np.min(x))
        img[~(self.image_data1 == self.i)] = 0
        


        img = self.transform(img)  # np -> torch  [91, 109, 91]
        img = img.unsqueeze(0)       # 增加一个维度，留给channel , 过3dcnn  [1, 91, 109, 91]
        label = torch.FloatTensor([float(age)]).squeeze()
        sample = img, label

        return sample
import numpy as np
import torch
from monai import transforms
import random
from imgaug import augmenters as iaa
from torch.utils.data import Dataset
import time
import random
import lmdb
# from .utils_aug import coordinateTransformWrapper

def gen_cord(imgsize=(128, 128, 128), scale=(0.08, 1.), ratio = (3 / 4, 4 / 3)):
    sc = random.random() * (scale[1] - scale[0]) + scale[0]
    ra1 = random.random() * (ratio[1] - ratio[0]) + ratio[0]
    ra2 = random.random() * (ratio[1] - ratio[0]) + ratio[0]

    x = (sc * imgsize[0] * imgsize[0] * imgsize[0] / (ra1 * ra2)) ** (1 / 3)
    y = ra1 * x
    z = ra2 * x
    return int(x), int(y), int(z)





class my_dataset(Dataset):

    def __init__(self, data_list_path, env, train=False, aug_mode=0):
        self.data_list = np.loadtxt(data_list_path, str, delimiter=",")
        self.env = env
        self.train = train
        self.aug_mode = aug_mode

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # dataset_id, site_id, pid, age, gender = self.data_list[index]
        pid, age = self.data_list[index]

        if not hasattr(self, 'txn'):
            self.txn = lmdb.open(self.env, readonly=True).begin(buffers=True)

        buf = self.txn.get(pid.encode())
        # with self.env.begin(write=False) as txn:
        #     buf = txn.get(pid.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        img = img_flat.copy().reshape(91, 109, 91)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)

        label = torch.FloatTensor([float(age)])
        sample = img, label

        return sample
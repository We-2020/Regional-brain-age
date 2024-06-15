import numpy as np
import torch
from monai import transforms
import random
from torch.utils.data import Dataset
import time
import random
# from .utils_aug import coordinateTransformWrapper
import nibabel as nib

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
        # self.data_list = np.loadtxt(data_list_path, str, delimiter=",")
        # 有表头用下面的
        self.data_list = np.genfromtxt(data_list_path, dtype=str, delimiter=',', skip_header=1) 
        self.env = env
        self.train = train
        self.aug_mode = aug_mode

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # dataset_id, site_id, pid, age, gender = self.data_list[index]
        pid, age, _ = self.data_list[index]
        # pid, age = self.data_list[index]
        



        with self.env.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        img = img_flat.copy().reshape(91, 109, 91)

        '''translate 10'''
        # if self.train:
        #     coordinateTransformWrapper(img, maxDeg=0, maxShift=10, mirror_prob=0.)

        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)



        '''3d，数据增强  random value '''
        if self.train:

            if self.aug_mode == 1:
                ''' CenterSpatialCrop and resize : scale=(0.08, 1.), ratio = (3 / 4, 4 / 3) '''
                x, y, z = gen_cord(imgsize=(91, 109, 91))
                crop_op = transforms.CenterSpatialCrop([x, y, z])
                img = crop_op(img)
            elif self.aug_mode == 2:
                ''' rotation : 45 45 45'''
                rotate_op = transforms.Rotate(angle=[random.randint(0, 45),random.randint(0, 45),random.randint(0, 45)])  # angle (Union[Sequence[float], float]) – Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
                img = rotate_op(img)
            elif self.aug_mode == 3:
                ''' flip : None 0'''
                flip_op = transforms.Flip(spatial_axis=None)  # spatial_axis (Union[Sequence[int], int, None]) – spatial axes along which to flip over.
                img = flip_op(img)
            elif self.aug_mode == 4:
                ''' random flip'''
                flip_op = transforms.RandFlip(prob=0.5,spatial_axis=random.randint(0, 2))  # spatial_axis (Union[Sequence[int], int, None]) – spatial axes along which to flip over.
                img = flip_op(img)
            elif self.aug_mode == 5:
                ''' random axis flip'''
                flip_op = transforms.RandAxisFlip(prob=0.5)  # spatial_axis (Union[Sequence[int], int, None]) – spatial axes along which to flip over.
                img = flip_op(img)
            elif self.aug_mode == 6:
                ''' gaussian noise : prob0.1 std0.1'''
                gaussian_op = transforms.RandGaussianNoise(std=0.02)
                img = gaussian_op(img)
            elif self.aug_mode == 7:
                ''' gaussian noise : prob0.1 std0.1'''
                gaussian_op = transforms.RandGaussianNoise(std=0.1)
                img = gaussian_op(img)
            elif self.aug_mode == 8:
                x, y, z = gen_cord(imgsize=(91, 109, 91))
                crop_op = transforms.CenterSpatialCrop([x, y, z])
                img = crop_op(img)
                flip_op = transforms.RandFlip(prob=0.5, spatial_axis=random.randint(0,2))
                img = flip_op(img)
            ''' translate '''
            # translate_op = transforms.utils.create_translate(spatial_dims, shift, device=None, backend=TransformBackends.NUMPY)[source]


            ''' no aug '''
            resize_op = transforms.Resize([128, 128, 128])
            img = resize_op(img)

        else:
            resize_op = transforms.Resize([128, 128, 128])
            img = resize_op(img)


        label = torch.FloatTensor([float(age)])
        sample = img, label

        return sample


class my_dataset_mask(Dataset):

    def __init__(self, data_list_path, env, train=False, aug_mode=0, mask_index=2002):
        self.data_list = np.loadtxt(data_list_path, str, delimiter=",")
        self.env = env
        self.train = train
        self.aug_mode = aug_mode
        self.mask = mask_index

        self.image_path1 = "ROI_MNI_V4.nii"
        self.image_obj1 = nib.load(self.image_path1)
        self.image_data1 = np.array(self.image_obj1.get_fdata())


    def __len__(self):  # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # dataset_id, site_id, pid, age, gender = self.data_list[index]
        pid, age = self.data_list[index]



        with self.env.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        img = img_flat.copy().reshape(91, 109, 91)

        '''translate 10'''
        # if self.train:
        #     coordinateTransformWrapper(img, maxDeg=0, maxShift=10, mirror_prob=0.)

        # 把非这个脑区的部分全部mask掉
        img[~(self.image_data1 == self.mask)] = 0

        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        rk = torch.distributed.get_rank()
        device_op = transforms.ToDevice('cuda:' + str(rk))
        img = device_op(img)



        '''3d，数据增强  random value '''
        if self.train:

            if self.aug_mode == 1:
                ''' CenterSpatialCrop and resize : scale=(0.08, 1.), ratio = (3 / 4, 4 / 3) '''
                x, y, z = gen_cord(imgsize=(91, 109, 91))
                crop_op = transforms.CenterSpatialCrop([x, y, z])
                img = crop_op(img)
            elif self.aug_mode == 2:
                ''' rotation : 45 45 45'''
                rotate_op = transforms.Rotate(angle=[random.randint(0, 45),random.randint(0, 45),random.randint(0, 45)])  # angle (Union[Sequence[float], float]) – Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
                img = rotate_op(img)
            elif self.aug_mode == 3:
                ''' flip : None 0'''
                flip_op = transforms.Flip(spatial_axis=None)  # spatial_axis (Union[Sequence[int], int, None]) – spatial axes along which to flip over.
                img = flip_op(img)
            elif self.aug_mode == 4:
                ''' random flip'''
                flip_op = transforms.RandFlip(prob=0.5,spatial_axis=random.randint(0, 2))  # spatial_axis (Union[Sequence[int], int, None]) – spatial axes along which to flip over.
                img = flip_op(img)
            elif self.aug_mode == 5:
                ''' random axis flip'''
                flip_op = transforms.RandAxisFlip(prob=0.5)  # spatial_axis (Union[Sequence[int], int, None]) – spatial axes along which to flip over.
                img = flip_op(img)
            elif self.aug_mode == 6:
                ''' gaussian noise : prob0.1 std0.1'''
                gaussian_op = transforms.RandGaussianNoise(std=0.02)
                img = gaussian_op(img)
            elif self.aug_mode == 7:
                ''' gaussian noise : prob0.1 std0.1'''
                gaussian_op = transforms.RandGaussianNoise(std=0.1)
                img = gaussian_op(img)
            elif self.aug_mode == 8:
                x, y, z = gen_cord(imgsize=(91, 109, 91))
                crop_op = transforms.CenterSpatialCrop([x, y, z])
                img = crop_op(img)
                flip_op = transforms.RandFlip(prob=0.5, spatial_axis=random.randint(0,2))
                img = flip_op(img)
            ''' translate '''
            # translate_op = transforms.utils.create_translate(spatial_dims, shift, device=None, backend=TransformBackends.NUMPY)[source]


            ''' no aug '''
            resize_op = transforms.Resize([128, 128, 128])
            img = resize_op(img)

        else:
            resize_op = transforms.Resize([128, 128, 128])
            img = resize_op(img)


        label = torch.FloatTensor([float(age)])
        sample = img, label

        return sample


class my_dataset_multichannel(Dataset):

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



        with self.env.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        img = img_flat.copy().reshape(3, 91, 109, 91)

        '''translate 10'''
        # if self.train:
        #     coordinateTransformWrapper(img, maxDeg=0, maxShift=10, mirror_prob=0.)

        img = torch.from_numpy(img)
        rk = torch.distributed.get_rank()
        device_op = transforms.ToDevice('cuda:' + str(rk))
        img = device_op(img)


        '''3d，数据增强  random value '''
        if self.train:

            if self.aug_mode == 1:
                ''' CenterSpatialCrop and resize : scale=(0.08, 1.), ratio = (3 / 4, 4 / 3) '''
                x, y, z = gen_cord(imgsize=(91, 109, 91))
                crop_op = transforms.CenterSpatialCrop([x, y, z])
                img = crop_op(img)
            elif self.aug_mode == 2:
                ''' rotation : 45 45 45'''
                rotate_op = transforms.Rotate(angle=[random.randint(0, 45),random.randint(0, 45),random.randint(0, 45)])  # angle (Union[Sequence[float], float]) – Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
                img = rotate_op(img)
            elif self.aug_mode == 3:
                ''' flip : None 0'''
                flip_op = transforms.Flip(spatial_axis=None)  # spatial_axis (Union[Sequence[int], int, None]) – spatial axes along which to flip over.
                img = flip_op(img)
            elif self.aug_mode == 4:
                ''' random flip'''
                flip_op = transforms.RandFlip(prob=0.5,spatial_axis=random.randint(0, 2))  # spatial_axis (Union[Sequence[int], int, None]) – spatial axes along which to flip over.
                img = flip_op(img)
            elif self.aug_mode == 5:
                ''' random axis flip'''
                flip_op = transforms.RandAxisFlip(prob=0.5)  # spatial_axis (Union[Sequence[int], int, None]) – spatial axes along which to flip over.
                img = flip_op(img)
            elif self.aug_mode == 6:
                ''' gaussian noise : prob0.1 std0.1'''
                gaussian_op = transforms.RandGaussianNoise(std=0.02)
                img = gaussian_op(img)
            elif self.aug_mode == 7:
                ''' gaussian noise : prob0.1 std0.1'''
                gaussian_op = transforms.RandGaussianNoise(std=0.1)
                img = gaussian_op(img)
            elif self.aug_mode == 8:
                x, y, z = gen_cord(imgsize=(91, 109, 91))
                crop_op = transforms.CenterSpatialCrop([x, y, z])
                img = crop_op(img)
                flip_op = transforms.RandFlip(prob=0.5, spatial_axis=random.randint(0,2))
                img = flip_op(img)
            ''' translate '''
            # translate_op = transforms.utils.create_translate(spatial_dims, shift, device=None, backend=TransformBackends.NUMPY)[source]


            ''' no aug '''
            resize_op = transforms.Resize([128, 128, 128])
            img = resize_op(img)

        else:
            resize_op = transforms.Resize([128, 128, 128])
            img = resize_op(img)


        label = torch.FloatTensor([float(age)])
        sample = img, label

        return sample
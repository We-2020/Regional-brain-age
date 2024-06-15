import os

import pandas as pd
from audtorch.metrics.functional import pearsonr
from mydata1 import dataGenerator,get_Dataframe_Data
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset,ConcatDataset,DataLoader
import warnings
import lmdb
import torch
from torch.optim.lr_scheduler import LambdaLR,CosineAnnealingLR
from model.resnet import resnet50
import logging  
from model.GlobalLocalTransformer import GlobalLocalBrainAge

import numpy as np
import torchvision.transforms as transforms




class my_dataset(Dataset):

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
        img = img[10:90:2,:,:]

        img = self.transform(img)  # np -> torch  [91, 109, 91]
        img = img.transpose(1,0)
        label = torch.FloatTensor([float(age)]).squeeze()
        sample = img, label

        return sample

# 配置日志输出格式和级别  
logging.basicConfig(filename='GLT.log', level=logging.INFO,   
                    format='%(asctime)s %(levelname)s: %(message)s')  

# 12 4 24 12 avg 步长  attention
batch_size  = 16
torch.manual_seed(3407)
# cpu/cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

#prepare data
env = lmdb.open("/home/huyixiao/DataCache/ba", readonly=True, lock=False, readahead=False,
                    meminit=False)
def calculate_r2(preds, labels):
    sq_diff = (preds - labels) ** 2
    mean_labels = labels.mean()
    var_labels = labels.var()
    ss_res = sq_diff.sum()
    ss_tot = torch.sum((labels - mean_labels) ** 2.0)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Du = []
for k in range(1,6):
    tr_path = f'5-fold/train_{k}.csv'
    v_path = f'5-fold/val_{k}.csv'
    a = pd.read_csv(tr_path,header=None)
    b = pd.read_csv(v_path,header=None)

    losses = []
    maess = []


    #     train_IXI,val_IXI = get_Dataframe_Data(0.2)
    train_IXI = a
    val_IXI = b
    # train = dataGenerator(train_IXI,env,i)
    # val = dataGenerator(val_IXI,env,i)
    # train_loader = DataLoader(train, batch_size  = batch_size,  shuffle = True)
    # val_loader = DataLoader(val, batch_size  = batch_size,  shuffle = False)

    # train_IXI,val_IXI = get_Dataframe_Data(0.2)
    # train = dataGenerator(train_IXI,env)
    # val = dataGenerator(val_IXI,env)
    # train_loader = DataLoader(train, batch_size  = batch_size,  shuffle = True)
    # val_loader = DataLoader(val, batch_size  = batch_size,  shuffle = True)
    # prepare model


    # optimizer = torch.optim.Adam(params=model.parameters(),lr = 5e-5)
    # # optimizer = torch.optim.SGD(params=model.parameters(),lr = 0.00001, momentum=0.7)
    # scheduler =  CosineAnnealingLR(optimizer, T_max=16, eta_min=7e-7)
    # # print('training...')
    # criterion = nn.MSELoss()
    # L1 = nn.L1Loss()

    train = my_dataset(train_IXI,env)
    val = my_dataset(val_IXI,env)
    train_loader = DataLoader(train, batch_size  = batch_size,  shuffle = True,num_workers=12)
    val_loader = DataLoader(val, batch_size  = batch_size,  shuffle = False,num_workers=12)


    # train_IXI,val_IXI = get_Dataframe_Data(0.2)
    # train = dataGenerator(train_IXI,env)
    # val = dataGenerator(val_IXI,env)
    # train_loader = DataLoader(train, batch_size  = batch_size,  shuffle = True)
    # val_loader = DataLoader(val, batch_size  = batch_size,  shuffle = True)
    # prepare model
    model = GlobalLocalBrainAge(40,patch_size=64,step=32,nblock=6,backbone='vgg8')
    #     model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f"{i}.pth").items()},strict=False)
    model = nn.DataParallel(model).to(device)


    optimizer = torch.optim.Adam(params=model.parameters(),lr = 1e-4)
    # optimizer = torch.optim.SGD(params=model.parameters(),lr = 0.00001, momentum=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=1, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08)
    # print('training...')
    criterion = nn.MSELoss()
    L1 = nn.L1Loss()
    epochs = 80
    best = 10000
    best_M = 20000

    for epoch in range(epochs):
        train_loss = 0
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        for imgs,age in tqdm(train_loader,ncols=50):
            imgs = imgs.float()
            imgs, age = imgs.to(device), age.to(device)
            optimizer.zero_grad()
            pred = model(imgs).float().squeeze()
            loss = criterion(pred,age.float().squeeze())
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.data
        train_loss = train_loss / len(train_loader)


        test_loss = 0

        print('val')
        mae = 0
        R = 0
        R2 = 0
        with torch.no_grad():
            model.eval()
            for imgs, age in val_loader:
                imgs, age = imgs.to(device), age.to(device)
                imgs = imgs.float()
                pred = model(imgs).float().squeeze()
                loss = criterion(pred, age.float())
                l1 = L1(pred, age.float())
                r2 = calculate_r2(pred, age.float())
                if pred.numel() == 1:
                    # pred = pred.unsqueeze(0)
                    continue
                r = pearsonr(pred, age.float())
                test_loss = test_loss + loss.data
                mae = mae + l1.data
                R = R + r.data
                R2 = R2 + r2.data
            MAE = mae / len(val_loader)
            R = R / len(val_loader)
            R2 = R2 / len(val_loader)
            test_loss = test_loss / len(val_loader)
            print(f'Epoch:{epoch+1}/{epochs}------train_loss:{train_loss}---test_loss:{test_loss}---R:{R}---R^2:{R2}--lr:{lr}---MAE:{MAE}')
            logging.info(f'Epoch:{epoch+1}/{epochs}------train_loss:{train_loss}---test_loss:{test_loss}---R:{R}---R^2:{R2}--lr:{lr}---MAE:{MAE}')
            if test_loss < best:
                bMSE = test_loss
                bMAE = MAE
                bR = R
                bR2 = R2
    #             torch.save(model.state_dict(),'net_params.pth.')
                torch.save(model.module.state_dict(), f"GLT{k}.pth")
        scheduler.step(test_loss)
    print('best,best_M',bMAE,bMSE)
    Du.append([k,bMAE,bMSE,bR,bR2])
print(Du)
with open('GLT.txt','w',encoding='utf8') as f:
    for kk in range(len(Du)):
        f.write(f"{str(float(Du[kk][0]))}Fold--MAE:{str(float(Du[kk][1]))}--MSE:{str(float(Du[kk][2]))}--R:{str(float(Du[kk][3]))}--R^2:{str(float(Du[kk][4]))}")
        
        
        
        

import os
import pandas as pd
from audtorch.metrics.functional import pearsonr
from mydata1D import dataGenerator,get_Dataframe_Data
from mydata1T import dataGenerator as dataG
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset,ConcatDataset,DataLoader
import warnings
import torch
from torch.optim.lr_scheduler import LambdaLR,CosineAnnealingLR
from model.au128 import UNetWithTransformerEncoder
import logging  
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from accelerate import Accelerator
from utils.utils_dataset import *
import lmdb
from noise import UNetn
import pytorch_warmup as warmup
from model.SQET import SQET



class NoiseModel(nn.Module):
    def __init__(
        self,
        util_model,
        learning_rate=1e-3,
        noise_coeff=15,
        min_scale=0,
        max_scale=1,
        batch_size=16,
        pretrained=None
    ):
        super().__init__()
        self.util_model = util_model
        # for layer in self.util_model.layers:
        #     layer.trainable = False
        for param in self.util_model.parameters():
            param.requires_grad = False

        self.noise_model = UNetn()


        self.normal = torch.distributions.normal.Normal(0, 1)
        self.learning_rate = learning_rate
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise_coeff = noise_coeff
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size

    def forward(self, x):

        B = torch.sigmoid(self.noise_model(x))

        # sample from normal  distribution
#         epsilon = self.normal.sample(B.shape).type_as(B)

        # reparametiation trick
        # print('B',torch.mean(B),torch.max(B),torch.min(B))
        # print('epsilon',torch.mean(epsilon),torch.max(epsilon),torch.min(epsilon))
        noise = B * 10
        age_pred = self.util_model((x + noise).float()).squeeze()

        return B,noise,age_pred

def train(model,criterion,L1,optimizer,train_loader,val_loader,epoch,epochs,lr,k,scheduler,warmup_scheduler,device):
    model.train()
        
    for imgs,age in train_loader:
        optimizer.zero_grad()
        imgs, age = imgs.to(device), age.to(device)
        B,noise,age_pred = model(imgs).float().squeeze()
        loss = self.criterion(age_pred.float(), age.float()) * 1.5 - torch.var(B) * 10 ** 2 - self.noise_coeff * torch.mean(
            B.log()
        )
        loss.backward()
        optimizer.step()

    Loss = 0
    BS = 0
    MSE = 0
    bloss = 10000
    with torch.no_grad():
        model.eval()
        for imgs, age in val_loader:
            imgs, age = imgs.to(device), age.to(device)
            imgs = imgs.float()
            B,noise,age_pred = model(imgs).float().squeeze()
            bs = torch.mean(B.log())
            mse = self.criterion(age_pred.float(), age.float())
            loss =  mse * 1.5 - torch.var(B) * 10 ** 2 - self.noise_coeff * bs
            Loss += loss.data
            BS += bs.data
            MSE += mse.data
        Loss = Loss / len(val_loader)
        BS = BS / len(val_loader)
        MSE = MSE / len(val_loader)
        
        if Loss < bloss:
#             torch.save(model.state_dict(),'net_params.pth.')
            torch.save(model.module.state_dict(), f"unoise{k}.pth")
    with warmup_scheduler.dampening():
        if epoch < 10:
            pass
        else:
            scheduler.step()



def main(dataframe_paths, env, batch_size, epochs,k):
    accelerator = Accelerator(split_batches=True)

    # 数据准备

    
    # 实例化dataGenerator
    # train_dataset = dataGenerator(train_IXI, lmdb_path)
    # val_dataset = dataGenerator(val_IXI, lmdb_path)
    train_dataset = my_dataset(dataframe_paths[0], env, True, 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,drop_last=True)    # nw 一般为0

    val_dataset = my_dataset(dataframe_paths[1], env, False, 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,drop_last=True)

    # train_dataset = dataGenerator(train_IXI, lmdb_path)
    # val_dataset = dataGenerator(val_IXI, lmdb_path)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,drop_last=True)
    # 模型配置
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = "./results_checkpoint/Unet3/checkpoint_best.tar"

    util_model = UNetWithTransformerEncoder(32, 8, 6, 128, 0.1).to(device)
    checkpoint1 = torch.load(model_path)
    util_model.load_state_dict(checkpoint1['model_state_dict'], strict=False)
    model = NoiseModel(util_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-1)
    
    # optimizer = torch.optim.SGD(params=model.parameters(),lr = 0.00001, momentum=0.5)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08)
    scheduler =  CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=7e-6)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
    #                                   weight_decay=args.weight_decay)
    #     # TODO 学习率
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=2e-5)
    #     # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=4)
    # accelerator.print('training...')
    criterion = nn.MSELoss()
    L1 = nn.L1Loss()
    
    # 训练循环
    for epoch in tqdm(range(epochs)):
        # ... 训练和验证过程 ...
        # 记录日志、保存模型等
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        train(model,criterion,L1,optimizer,train_loader,val_loader,epoch,epochs,lr,k,scheduler,warmup_scheduler,device)
        


            
            
if __name__ == '__main__':
#    
    k = 1
    dataframe_paths = ['csvs/MaleTrain4265.csv', 'csvs/MaleTest1070.csv']
    # dataframe_paths = ['5-fold/train_2.csv', '5-fold/val_2.csv']
    batch_size = 4
    epochs = 200
    env = lmdb.open("/home/caojiaxiang/ba", readonly=True, lock=False, readahead=False,
                    meminit=False)
    main(dataframe_paths, env, batch_size, epochs,k)

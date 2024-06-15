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
from model.resnet import I3Res50
import pytorch_warmup as warmup
from model.SQET import SQET




def calculate_r2(preds, labels):
    sq_diff = (preds - labels) ** 2
    mean_labels = labels.mean()
    var_labels = labels.var()
    ss_res = sq_diff.sum()
    ss_tot = torch.sum((labels - mean_labels) ** 2.0)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def train(accelerator,model,criterion,L1,optimizer,train_loader,val_loader,epoch,epochs,lr,k,scheduler,warmup_scheduler):
    model.train()
        
    for imgs,age in train_loader:
        optimizer.zero_grad()
        imgs, age = imgs, age
        pred = model(imgs).float().squeeze()
        loss = criterion(pred, age.float().squeeze())
        accelerator.backward(loss)
        optimizer.step()

    test_loss = 0  
    mae = 0
    R = 0
    R2 = 0
    best = 10000
    best_M = 20000
    bMSE = 0
    bMAE = 0
    bR = 0
    bR2 = 0
    with torch.no_grad():
        # 用于保存所有预测结果
        Loss = []
        MAE = []
        R2 = []
        R = []
        model.eval()
        for imgs, age in val_loader:
            imgs, all_age = imgs.float().to(accelerator.device), age.to(accelerator.device)
            all_pred = model(imgs).float()
#             all_pred,all_age = accelerator.gather_for_metrics((pred.squeeze(), age.squeeze()))

            
            accelerator.print('age',all_age)
            accelerator.print('all_pred',all_pred)
            loss = criterion(all_pred, all_age)
            l1 = L1(all_pred, all_age)
            r2 = calculate_r2(all_pred, all_age)
            # if pred.numel() == 1:
            #     # pred = pred.unsqueeze(0)
            #     continue
#             r = pearsonr(all_pred, all_age)
            Loss.append(loss.item())
            MAE.append(l1.item())
            R2.append(r2.item())
#             R.append(r.item())
            accelerator.print('l1',l1)
            
            
            
        
        
        if accelerator.is_main_process:
            MAE = sum(MAE) / len(val_loader)
#             R = sum(R) / len(val_loader)
            R2 = sum(R2) / len(val_loader)
            test_loss = sum(Loss) / len(val_loader)
            accelerator.print(f'Epoch:{epoch+1}/{epochs}-------test_loss:{test_loss}---R:{R}---R^2:{R2}--lr:{lr}---MAE:{MAE}')
            logging.info(f'Epoch:{epoch+1}/{epochs}--------test_loss:{test_loss}---R:{R}---R^2:{R2}--lr:{lr}---MAE:{MAE}')
            if MAE < best:
                bMSE = test_loss
                bMAE = MAE
                bR = R
                bR2 = R2
                best = MAE
    #             torch.save(model.state_dict(),'net_params.pth.')
                torch.save(model.module.state_dict(), f"transunet{k}.pth")
    with warmup_scheduler.dampening():
        if epoch < 10:
            pass
        else:
            scheduler.step()
    return (bMAE,bMSE,bR,bR2)


def main(dataframe_paths, env, batch_size, epochs,k):
    accelerator = Accelerator(split_batches=True)

    # 数据准备

    
    # 实例化dataGenerator
    # train_dataset = dataGenerator(train_IXI, lmdb_path)
    # val_dataset = dataGenerator(val_IXI, lmdb_path)
    train_dataset = my_dataset(dataframe_paths[0], env, True, 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,drop_last=True)    # nw 一般为0

    val_dataset = my_dataset(dataframe_paths[1], env, False, 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0,drop_last=True)

    # train_dataset = dataGenerator(train_IXI, lmdb_path)
    # val_dataset = dataGenerator(val_IXI, lmdb_path)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,drop_last=True)
    # 模型配置
    model = UNetWithTransformerEncoder(32, 8, 6, 128, 0.1)
#     model = I3Res50(num_classes=1, use_nl=False)
    


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-1)
    
    # optimizer = torch.optim.SGD(params=model.parameters(),lr = 0.00001, momentum=0.5)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=2, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-6, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=1e-5)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=7e-6)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
    #                                   weight_decay=args.weight_decay)
    #     # TODO 学习率
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=2e-5)
    #     # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=5)
    # accelerator.print('training...')
    criterion = nn.MSELoss()
    L1 = nn.L1Loss()
    
    train_loader, model, optimizer, scheduler,warmup_scheduler = accelerator.prepare(train_loader, model, optimizer,scheduler,warmup_scheduler)
    # 训练循环
    for epoch in tqdm(range(epochs), disable=not accelerator.is_local_main_process):
        # ... 训练和验证过程 ...
        # 记录日志、保存模型等
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        bMAE,bMSE,bR,bR2 = train(accelerator,model,criterion,L1,optimizer,train_loader,val_loader,epoch,epochs,lr,k,scheduler,warmup_scheduler)
        
    accelerator.print(bMAE,bMSE,bR,bR2)

            
            
if __name__ == '__main__':
#    
    k = 1
    dataframe_paths = ['csvs/MaleTrain4265.csv', 'csvs/MaleTest1070.csv']
    # dataframe_paths = ['5-fold/train_2.csv', '5-fold/val_2.csv']
    batch_size = 16
    epochs = 200
    env = lmdb.open("/home/caojiaxiang/ba", readonly=True, lock=False, readahead=False,
                    meminit=False)
    main(dataframe_paths, env, batch_size, epochs,k)

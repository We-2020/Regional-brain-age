import os
import pandas as pd
from mydata4 import dataGenerator,get_Dataframe_Data
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset,ConcatDataset,DataLoader
import warnings
import lmdb
import torch
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import LambdaLR,CosineAnnealingLR
from model.au128 import UNetWithTransformerEncoder


batch_size  = 12
# cpu/cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

#prepare data
env = lmdb.open("/home/huyixiao/DataCache/ba", readonly=True, lock=False, readahead=False,
                    meminit=False)



tr_path = '/home/huyixiao/ba-dataset/config1w/train_1.csv'
v_path = '/home/huyixiao/ba-dataset/config1w/val_1.csv'
a = pd.read_csv(tr_path,header=None)
b = pd.read_csv(v_path,header=None)

losses = []
maess = []
path = '/data/hgz_backup/huguozhen/brain_age/data_config/mixed/test_id.csv'

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
train = dataGenerator(train_IXI,env)
val = dataGenerator(val_IXI,env)
train_loader = DataLoader(train, batch_size  = batch_size,  shuffle = True)
val_loader = DataLoader(val, batch_size  = batch_size,  shuffle = False)
# prepare model
model = model = UNetWithTransformerEncoder(32, 8, 6, 128, 0.1)
#     model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(f"{i}.pth").items()},strict=False)
model = nn.DataParallel(model).to(device)

optimizer = torch.optim.Adam(params=model.parameters(),lr = 5e-5)
# optimizer = torch.optim.SGD(params=model.parameters(),lr = 0.00001, momentum=0.7)
scheduler =  CosineAnnealingLR(optimizer, T_max=16, eta_min=7e-7)
# print('training...')
criterion = nn.MSELoss()
L1 = nn.L1Loss()

# yi = [2001
# ,2101
# ,2301
# ,2331
# ,2401
# ,3001
# ,4201
# ,5001
# ,8121
# ,9011
# ,9062
# ,9140
# ,9150
# ,9160
# ,9170]
lis =  [2001., 2002., 2101., 2102., 2111., 2112., 2201., 2202., 2211., 2212., 2301.,
 2302., 2311., 2312., 2321., 2322., 2331., 2332., 2401., 2402., 2501., 2502., 2601.,
 2602., 2611., 2612., 2701., 2702., 3001., 3002., 4001., 4002., 4011., 4012., 4021.,
 4022., 4101., 4102., 4111., 4112., 4201., 4202., 5001., 5002., 5011., 5012., 5021.,
 5022., 5101., 5102., 5201., 5202., 5301., 5302., 5401., 5402., 6001., 6002., 6101.,
 6102., 6201., 6202., 6211., 6212., 6221., 6222., 6301., 6302., 6401., 6402., 7001.,
 7002., 7011., 7012., 7021., 7022., 7101., 7102., 8101., 8102., 8111., 8112., 8121.,
 8122., 8201., 8202., 8211., 8212., 8301., 8302., 9001., 9002., 9011., 9012., 9021.,
 9022., 9031., 9032., 9041., 9042., 9051., 9052., 9061., 9062., 9071., 9072., 9081.,
 9082., 9100., 9110., 9120., 9130., 9140., 9150., 9160., 9170.]
# lis = list(set(quan)-set(yi))
bestlist = []
for i in lis:
    train = dataGenerator(train_IXI,env,i)
    val = dataGenerator(val_IXI,env,i)
    train_loader = DataLoader(train, batch_size  = batch_size,  shuffle = True)
    val_loader = DataLoader(val, batch_size  = batch_size,  shuffle = False)
    
    epochs = 100
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
            loss = criterion(pred,age.float())
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.data
        train_loss = train_loss / len(train_loader)


        test_loss = 0

        print('val')
        mae = 0
        with torch.no_grad():
            model.eval()
            for imgs, age in val_loader:
                imgs, age = imgs.to(device), age.to(device)
                imgs = imgs.float()
                pred = model(imgs).float().squeeze()
                loss = criterion(pred, age.float())
                l1 = L1(pred, age.float())
                test_loss = test_loss + loss.data
                mae = mae + l1.data
            MAE = mae / len(val_loader)
            test_loss = test_loss / len(val_loader)
            print(f'Epoch:{epoch+1}/{epochs}------train_loss:{train_loss}---test_loss:{test_loss}---lr:{lr}---MAE:{MAE}')
            if test_loss < best:
                best = test_loss
                best_M = MAE
    #             torch.save(model.state_dict(),'net_params.pth.')
                torch.save(model.module.state_dict(), f"/data/cjx/pths/{int(i)}.pth")
    scheduler.step()
    bestlist.append({'seq':i,'best':best,'best_M':best_M})
    print('best,best_M',best,best_M)
print(bestlist)
    


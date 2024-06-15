import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import lmdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os
import time

from utils.utils_dataset import *
from model.SQET import SQET
from model.sfcn import SFCN
from model.DACNN import DACNN
from model.resnet import resnet18
from model.LocalLocalTransformer3d import LocalLocalTransformer3d as LLT


torch.backends.cudnn.benchmark = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


# loss_dict：  单个batch结果
# batch_loss： 所有batch  累加
# {'aux1','aux2','aux3','final','total_loss'}
def append_weighted_loss(batch_loss, loss_dict,args):
    tot_loss = 0.
    for k in loss_dict:
        if k == 'final':
            w_loss = 1. * loss_dict[k]
        else:   # k == axu012
            w_loss = args.aux_weight * loss_dict[k]         # TODO 系数设置
        tot_loss += w_loss

        if k in batch_loss:
            batch_loss[k] += w_loss.item()
        else:
            batch_loss[k] = w_loss.item()

    if 'total_loss' in batch_loss:
        batch_loss['total_loss'] += tot_loss.item()
    else:
        batch_loss['total_loss'] = tot_loss.item()

    return tot_loss


def get_loss_train(out, labels, type='l1'):
    if type == 'l1':
        criterion = torch.nn.L1Loss()
    else:
        assert False

    # if isinstance(out, dict):
    #     return {k: criterion(out[k], labels) for k in out}
    # else:
    #     return {'final': criterion(out, labels)}

    if isinstance(out, dict):
        return {k: weight_caculate_loss(out[k], labels) for k in out}
    else:
        return {'final': weight_caculate_loss(out, labels)}

matrix = [3.6504065040650406, 3.328665568369028, 2.3826650943396226, 1.0, 4.123469387755102, 5, 5, 5, 5, 5, 5, 5, 4.378114842903575, 3.5634920634920637, 4.555806087936866, 5, 5, 5, 5]

def weight(x):

    return matrix[int(x // 5)-1]

def weight_caculate_loss(x,y):
    criterion = torch.nn.L1Loss()
    n = x.shape[0]
    ret = 0.
    for i in range(n):
        ret += criterion(x[i], y[i]) * weight(y[i])
    return ret / n


# 计算一个batch的loss
# L1 loss , avg
def get_loss(out, labels, type='l1'):
    if type == 'l1':
        criterion = torch.nn.L1Loss()
    else:
        assert False

    if isinstance(out, dict):
        return {k: criterion(out[k], labels) for k in out}
    else:
        return {'final': criterion(out, labels)}


# 一个epoch
def train(model, train_loader, optimizer, args, rank):
    model.train()
    batch_loss = {}
    load_time = []
    train_time = []
    t0 = time.time()

    for batch_idx, batch_data in enumerate(train_loader):   # train_loader size = total / batch_total
        t1 = time.time()
        load_time.append(t1 - t0)
        t0 = t1

        img, target = batch_data
        img = img.float().to(rank)
        target = target.float().to(rank)

        optimizer.zero_grad()
        output = model(img)

        loss_dict = get_loss_train(output, target)                    # 得到 一个batch的Loss
        tot_loss = append_weighted_loss(batch_loss, loss_dict, args)  # 累加 一个batch的loss
        tot_loss.backward()
        # nn.utils.clip_grad_value_(model.parameters(), clip_value=1)     # TODO 梯度裁剪
        optimizer.step()

        t1 = time.time()
        train_time.append(t1 - t0)
        t0 = t1
     
    loss_avg = {}
    for k in batch_loss:
        loss_avg[k] = batch_loss[k] / len(train_loader)     # 反向传播聚合了tot_loss，并没有聚合batch_loss  loss是本地的数据loss
    
    load_time = sum(load_time) / len(load_time)
    train_time = sum(train_time) / len(train_time)
    return loss_avg, load_time, train_time



def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):

    sst = time.perf_counter()
    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    setup(rank, world_size, args.port)
    torch.cuda.set_device(rank)
    task_name = args.task_name
    BATCH_SIZE = args.batch // world_size   # 每一个设备的batch_size
    EPOCHS = args.epochs


    # TODO:实例化模型
    model = SFCN()

    torch.manual_seed(1)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # TODO 路径
    train_data_list_path = r'./config1w/train_' + str(args.cross_val) + '.csv'
    val_data_list_path = r'./config1w/val_' + str(args.cross_val) + '.csv'


    env = lmdb.open("/home/huyixiao/DataCache/ba", readonly=True, lock=False, readahead=False,
                    meminit=False)

    # 输入数据
    train_dataset = my_dataset(train_data_list_path, env, True, args.aug_method)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, sampler=train_sampler)    # nw 一般为0

    val_dataset = my_dataset(val_data_list_path, env, False)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=0, sampler=val_sampler,drop_last=True)


    res_path = './results_output/' + task_name + '/'
    checkpoint_path = './results_checkpoint/' + task_name + '/'
    checkpoint_file = checkpoint_path + 'checkpoint_best.tar'

    if rank == 0:
        tb_logger = SummaryWriter('./tensorboard_logs/' + args.task_name)
        print('Train: {} | Val: {} '.format(len(train_dataset), len(val_dataset)))

        if not os.path.exists(res_path):
            os.makedirs(res_path)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        fo = open(res_path + 'output_train.txt', 'w')
        print('--------args----------')
        for k in list(vars(args).keys()):
            print('%s: %s' % (k, vars(args)[k]))
            print('%s: %s' % (k, vars(args)[k]), file=fo)
        print(dict, file=fo)
        print('--------args----------\n')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    # TODO 调整
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)




    # 训练过程
    train_loss_mae = []
    val_loss_mae = []
    best_val_epoch = 0
    best_val_mae = 1000000000

    if rank == 0:
        loop_range = tqdm(range(0, EPOCHS))
    else:
        loop_range = range(0, EPOCHS)

    ######################################################
    for epoch in loop_range:
        train_loader.sampler.set_epoch(epoch)
        train_loss, load_time, train_time = train(model, train_loader, optimizer, args, rank)

        model.eval()
        with torch.no_grad():

            # 1. 得到本进程所有batch的loss
            batch_loss = {}
            for data, label in val_loader:
                data, label = data.to(rank), label.to(rank)
                predictions = model(data)
                loss_dict = get_loss(predictions, label)
                append_weighted_loss(batch_loss, loss_dict, args)


            # 2. 进行gather
            lis = list(batch_loss.keys())
            sum_t = [batch_loss[k] for k in lis]
            sum_t.append(len(val_loader))
            sum_t = torch.tensor(sum_t).to(rank)
            torch.distributed.all_reduce(sum_t)     # all_reduce = 求和

            val_loss = {}
            for ii in range(len(lis)):
                val_loss[lis[ii]] = (sum_t[ii] / sum_t[-1]).item()

            if rank == 0:
                train_loss_mae.append(train_loss['final'])
                val_loss_mae.append(val_loss['final'])
                for k in train_loss:
                    tb_logger.add_scalar('Train_loss/'+k, train_loss[k], epoch)
                    tb_logger.add_scalar('Validation_loss/'+k, val_loss[k], epoch)
                tb_logger.add_scalar('Learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
                tb_logger.add_scalar('Time/Load_time', load_time, epoch)
                tb_logger.add_scalar('Time/Train_time', train_time, epoch)

                print('Train | Epoch: {} | Loss: {:.6f}'.format(epoch, train_loss['final']))
                print('Train | Epoch: {} | Loss: {:.6f}'.format(epoch, train_loss['final']), file=fo)
                print('Val   | Epoch: {} |  MAE: {:.4f}'.format(epoch,  val_loss['final']))
                print('Val   | Epoch: {} |  MAE: {:.4f}'.format(epoch,  val_loss['final']), file=fo)
                print('----------------')

                if val_loss['final'] < best_val_mae:
                    best_val_mae = val_loss['final']
                    best_val_epoch = epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_mse_min': best_val_mae
                    }, checkpoint_file)

        # scheduler.step()

    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    eed = time.perf_counter()
    duration = int(eed - sst)

    cleanup()
    # 记录最优结果
    if rank == 0:
        print('best_val_epoch: ', best_val_epoch)
        print('best_val_epoch: ', best_val_epoch, file=fo)
        fo.write(f'Start: {st}  End: {ed}  Duration: {duration // 60} min {duration % 60} s\n')
        fo.close()

        # 画图
        x = np.arange(1, EPOCHS + 1)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(task_name)

        plt.plot(x, train_loss_mae[0:])
        plt.plot(x, val_loss_mae[0:])
        plt.legend(['train_loss_mae', 'val_loss_mae'])
        plt.savefig(res_path + 'loss_plot.png', format='png')  # 先save,再show
        plt.show()


if __name__ == '__main__':


    gpu_ids = "0,1,2,3"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    # TODO 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts', '--task_name',     default='SFCN-5xloss', type=str, required=False)
    parser.add_argument('-b',  '--batch',         default=64, type=int, required=False) # 96
    parser.add_argument('-e',  '--epochs',        default=500, type=int, required=False)
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, required=False)
    parser.add_argument('-wd', '--weight_decay',  default=5e-5, type=float, required=False)
    parser.add_argument('-cv', '--cross_val',     default='2k', type=str, required=False)
    parser.add_argument('-aux', '--aux_weight',   default=0, type=float, required=False)
    parser.add_argument('-p',   '--port',         default='12345', type=str, required=False)
    parser.add_argument('-aug', '--aug_method',   default=1, type=float, required=False)
    parser.add_argument('-ps', '--intro',
                        default="没有使用任何schedule", type=str, required=False)
    args = parser.parse_args()

    world_size = len(gpu_ids.split(','))
    mp.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size
    )

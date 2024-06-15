'''
    only for GLT
'''
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

from model.GlobalLocalTransformer import GlobalLocalBrainAge as GLT


import pytorch_warmup as warmup

torch.backends.cudnn.benchmark = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

class my_dataset(Dataset):

    def __init__(self, data_list_path, env, train=False):
        self.data_list = np.loadtxt(data_list_path, str, delimiter=",")
        self.env = env
        self.train = train

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data_list)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # dataset_id, site_id, pid, age, gender = self.data_list[index]
        pid, age = self.data_list[index]


        with self.env.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        img = img_flat.copy().reshape(91, 109, 91)

        img = img[10:90:2,:,:]
        # img = img[:, :, 10:90:2]
        # img = img.transpose(2,1,0)
        #img = torch.transpose(img, 0, 2)


        label = torch.FloatTensor([float(age)])
        sample = img, label

        return sample


# 一个epoch
def train(model, train_loader, optimizer, args, rank):
    model.train()
    batch_loss = []

    for batch_idx, batch_data in enumerate(train_loader):   # train_loader size = total / batch_total
        img, target = batch_data
        img = img.float().to(rank)
        target = target.float().to(rank)

        optimizer.zero_grad()
        output = model(img)

        # output = sum(output[1:]) / len(output[1:])
        output = torch.sum(output[0:], dim=0) / (output.shape[0])
        loss = F.l1_loss(output, target, reduction='mean')

        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=1)     # TODO 梯度裁剪
        optimizer.step()
        batch_loss.append(loss.item())

    loss_avg = sum(batch_loss) / len(batch_loss)
    return loss_avg



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12348'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):

    sst = time.perf_counter()
    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    setup(rank, world_size)
    torch.cuda.set_device(rank)
    task_name = args.task_name
    BATCH_SIZE = args.batch // world_size   # 每一个设备的batch_size
    EPOCHS = args.epochs

    # dict = {'dim': (64, 128, 256),
    #         'depth':(4, 8, 5),
    #         'global_window_size': (8, 4, 2),
    #         'use_se': True,
    #         'use_pse': False,
    #         'use_inception': True,
    #         'use_sequence_pooling': True,
    #         'input_shape': [128, 128, 128]
    #         }


    # TODO:实例化模型

    # model = cf_se(
    #     dim=dict['dim'],  # dimension at each stage
    #     depth=dict['depth'],  # depth of transformer at each stage
    #     global_window_size=dict['global_window_size'],  # global window sizes at each stage
    #     use_se=dict['use_se'],
    #     use_pse=dict['use_pse'],
    #     use_inception=dict['use_inception'],
    #     use_sequence_pooling=dict['use_sequence_pooling'],
    #     input_shape=dict['input_shape'],
    #     attn_dropout=args.dropout_att,
    #     ff_dropout=args.dropout_ff
    # )
    model = GLT(40,patch_size=64,step=32,nblock=6,backbone='vgg8')

    torch.manual_seed(1)
    model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # TODO 路径
    train_data_list_path = r'./config1w/train_' + str(args.cross_val) + '.csv'
    val_data_list_path = r'./config1w/val_' + str(args.cross_val) + '.csv'


    # TODO env-setting
    env = lmdb.open("/home/huyixiao/DataCache/ba", readonly=True, lock=False, readahead=False,
                    meminit=False)

    # 输入数据
    train_dataset = my_dataset(train_data_list_path, env, True)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, sampler=train_sampler)    # nw 一般为0

    val_dataset = my_dataset(val_data_list_path, env, False)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    # TODO TIPS batch_size 一定要能整除，不然头几张会多算几次
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
    # TODO 怎么用
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=2, eta_min=2e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=5)



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
        train_loss = train(model, train_loader, optimizer, args, rank)

        model.eval()
        with torch.no_grad():

            # 1. 得到本进程所有batch的loss
            batch_loss = 0
            for data, label in val_loader:
                data, label = data.to(rank), label.to(rank)
                output = model(data)
                output = torch.sum(output[0:], dim=0) / (output.shape[0])
                loss = F.l1_loss(output, label, reduction='mean')
                batch_loss += loss.item()


            # 2. 进行gather
            batch_loss = torch.tensor(batch_loss).to(rank)
            val_num = torch.tensor(len(val_loader)).to(rank)
            torch.distributed.all_reduce(batch_loss)     # all_reduce = 求和
            torch.distributed.all_reduce(val_num)

            val_loss = batch_loss / val_num


            if rank == 0:
                train_loss_mae.append(train_loss)
                val_loss_mae.append(val_loss)
                tb_logger.add_scalar('Train_loss/', train_loss, epoch)
                tb_logger.add_scalar('Validation_loss/', val_loss, epoch)
                tb_logger.add_scalar('Learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

                print('Train | Epoch: {} | Loss: {:.6f}'.format(epoch, train_loss))
                print('Train | Epoch: {} | Loss: {:.6f}'.format(epoch, train_loss), file=fo)
                print('Val   | Epoch: {} |  MAE: {:.4f}'.format(epoch,  val_loss))
                print('Val   | Epoch: {} |  MAE: {:.4f}'.format(epoch,  val_loss), file=fo)
                print('----------------')

                if val_loss < best_val_mae:
                    best_val_mae = val_loss
                    best_val_epoch = epoch
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_mse_min': best_val_mae
                    }, checkpoint_file)

        with warmup_scheduler.dampening():
            if epoch < 5:
                pass
            else:
                scheduler.step()

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

    # gpu_ids = "0,1,2,3,4,5,6,7"
    # gpu_ids = "0,1,2,3"
    gpu_ids = "0,1,2,3"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    # TODO 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('-ts', '--task_name',     default='F-glt-dim0-cv2', type=str, required=False)
    parser.add_argument('-b',  '--batch',         default=40, type=int, required=False)
    parser.add_argument('-e',  '--epochs',        default=200, type=int, required=False)
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, required=False)
    parser.add_argument('-wd', '--weight_decay',  default=0.1, type=float, required=False)
    parser.add_argument('-cv', '--cross_val',     default='2', type=int, required=False)
    parser.add_argument('-dp_t', '--dropout_att', default=0., type=float, required=False)
    parser.add_argument('-dp_f', '--dropout_ff',  default=0.5, type=float, required=False)
    parser.add_argument('-aux', '--aux_weight',   default=0.2, type=float, required=False)
    parser.add_argument('-ps', '--intro',
                        default="GLT 200 epoch,40 slice", type=str, required=False)
    # parser.add_argument('--t', default=0.8, type=float, required=True)
    args = parser.parse_args()

    world_size = len(gpu_ids.split(','))
    mp.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size
    )

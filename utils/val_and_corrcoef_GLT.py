import os
import argparse
import torch
import lmdb
from torch.utils.data import DataLoader, Dataset
import numpy as np
from monai import transforms

from model.SQET import CrossFormer as cf_se
from model.sfcn import SFCN
from model.DACNN import DACNN
from model.resnet import resnet18
from model.LocalLocalTransformer3d import LocalLocalTransformer3d as LLT
from model.GlobalLocalTransformer import GlobalLocalBrainAge as GLT
import torch.nn.functional as F

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

        # t0 = time.time()

        with self.env.begin(write=False) as txn:
            buf = txn.get(pid.encode())
        img_flat = np.frombuffer(buf, dtype=np.float32)
        img = img_flat.copy().reshape(91, 109, 91)

        img = img[40:60:5, :, :]

        label = torch.FloatTensor([float(age)])
        sample = img, label

        return sample


if __name__ == '__main__':
    gpu_ids = "6"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    # TODO 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('-cv', '--cross_val',     default=2, type=int, required=False)
    args = parser.parse_args()

    task_name = 'test-glt'
    checkpoint = torch.load(
        './results_checkpoint/' + task_name + '/checkpoint_best.tar')
    BATCH_SIZE = 16
    res_path = './results_output/' + task_name + '/'

    # 路径
    val_data_list_path = r'./config/val_' + str(args.cross_val) + '.csv'

    env = lmdb.open("/home/huyixiao/DataCache/lmdb/ba_6000", readonly=True, lock=False, readahead=False,
                    meminit=False)

    val_dataset = my_dataset(val_data_list_path, env, train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = GLT(4, patch_size=64, step=32, nblock=6, backbone='vgg8')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    model.requires_grad_(False)

    pred = []
    batch_loss = []
    for batch_idx, batch_data in enumerate(val_loader):
        img, target = batch_data
        img = img.float().cuda()
        target = target.float().cuda()

        output = model(img)
        output = torch.sum(output[0:], dim=0) / (output.shape[0])
        # predictions = predictions['final']

        loss = F.l1_loss(output, target, reduction='mean')
        batch_loss.append(loss.item())

        for i in range(output.shape[0]):
            pred.append([target[i].item(), output[i].item()])

    loss_avg = sum(batch_loss) / len(batch_loss)
    print('loss_avg:',loss_avg)

    pred = np.array(pred)
    y_true = pred[:, 0].squeeze()
    y_pred = pred[:, 1].squeeze()
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print('corr:',corr)


    cnt_total = 0
    cnt_cs = 0
    f_pred = open(res_path + 'pred_best.txt', 'w')
    for y_true, y_pred in pred:
        print('{:.6f}, {:.6f}'.format(y_true, y_pred), file=f_pred, flush=True)
        cnt_total += 1
        if abs(y_true - y_pred) < 5:
            cnt_cs += 1
    print('cs:', float(cnt_cs/cnt_total))

    f_pred.flush()
    f_pred.close()


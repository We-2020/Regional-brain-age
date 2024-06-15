import os
import argparse
import torch
import lmdb
from torch.utils.data import DataLoader, Dataset
import numpy as np
from monai import transforms
import numpy as np
import pandas as pd
import scipy.stats as stats

from model.sfcn import SFCN
from model.DACNN import DACNN
from model.resnet import resnet18
from model.LocalLocalTransformer3d import LocalLocalTransformer3d as LLT
from model.GlobalLocalTransformer import GlobalLocalBrainAge as GLT

def calculate_spearman_correlation(X, Y):
    return stats.spearmanr(X, Y)[0]
def calculate_spearman_correlation_p(X, Y):
    return stats.spearmanr(X, Y)[1]

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

        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        resize_op = transforms.Resize([128, 128, 128])
        img = resize_op(img)


        label = torch.FloatTensor([float(age)])
        sample = img, label

        return sample


if __name__ == '__main__':
    gpu_ids = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    parser = argparse.ArgumentParser()
    # TODO
    parser.add_argument('-cv', '--cross_val',     default=2, type=int, required=False)
    args = parser.parse_args()

    BATCH_SIZE = 16

    # 路径
    val_data_list_path = r'./config1w/val_' + str(args.cross_val) + '.csv'

    env = lmdb.open("/home/huyixiao/DataCache/ba", readonly=True, lock=False, readahead=False,
                    meminit=False)

    val_dataset = my_dataset(val_data_list_path, env, train=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # dict = {'dim': (64, 128, 256),
    #         'depth':(4, 8, 5),
    #         'global_window_size': (8, 4, 2),
    #         'use_se': True,
    #         'use_pse': False,
    #         'use_inception': True,
    #         'use_sequence_pooling': True,
    #         'input_shape': [128, 128, 128]
    #         }
    #
    # model = cf_se(
    #     dim=dict['dim'],  # dimension at each stage
    #     depth=dict['depth'],  # depth of transformer at each stage
    #     global_window_size=dict['global_window_size'],  # global window sizes at each stage
    #     use_se=dict['use_se'],
    #     use_pse=dict['use_pse'],
    #     use_inception=dict['use_inception'],
    #     use_sequence_pooling=dict['use_sequence_pooling'],
    #     input_shape=dict['input_shape'],
    # )

    # TODO
    task_name = 'F-SFCN-cv2'
    checkpoint = torch.load(
        './results_checkpoint/' + task_name + '/checkpoint_best.tar')
    res_path = './results_output/' + task_name + '/'

    # TODO
    # model = DACNN()
    # model = resnet18()
    model = SFCN()

    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    model.requires_grad_(False)

    pred = []
    for batch_idx, batch_data in enumerate(val_loader):
        img, target = batch_data
        img = img.float().cuda()
        target = target.float().cuda()

        predictions = model(img)
        predictions = predictions['final']

        for i in range(predictions.shape[0]):
            pred.append([target[i].item(), predictions[i].item()])

    pred = np.array(pred)
    y_true = pred[:, 0].squeeze()
    y_pred = pred[:, 1].squeeze()
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print('PCC:',corr)

    calculate_spearman_correlation_p(y_true, y_pred)
    print('SRCC:', corr)



    cnt_total = 0
    cnt_cs = 0
    cnt_3 = 0
    f_pred = open(res_path + 'pred_best.txt', 'w')
    for y_true, y_pred in pred:
        print('{:.6f}, {:.6f}'.format(y_true, y_pred), file=f_pred, flush=True)
        cnt_total += 1
        if abs(y_true - y_pred) < 5:
            cnt_cs += 1
        if abs(y_true - y_pred) < 3:
            cnt_3 += 1
    print('cs5:', float(cnt_cs/cnt_total))
    print('cs3:', float(cnt_3/cnt_total))




    f_pred.flush()
    f_pred.close()


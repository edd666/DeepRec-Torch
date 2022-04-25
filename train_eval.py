# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Date    : 2022/1/24
# @Contact : liaozhi_edo@163.com


"""
    Pytorch模型的常用训练评估代码
"""

# packages
import time
import torch
import datetime
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def get_time_dif(start_time):
    """
    获取时间间隔

    :param start_time: 起始时间
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time

    return datetime.timedelta(seconds=int(round(time_dif)))


class CustomDataset(Dataset):
    """
    【data_dict】
        Dense: float32
        Sparse: int64
        Varlen_Sparse: int64

    【label】
        label: float32
    """
    def __init__(self, df, dense_feature_columns, sparse_feature_columns, varlen_sparse_feature_columns=None):
        self.dense_feature_columns = dense_feature_columns
        self.sparse_feature_columns = sparse_feature_columns
        self.varlen_sparse_feature_columns = varlen_sparse_feature_columns
        self.data_len = len(df)
        self.data = {col: df[col].values for col in dense_feature_columns + sparse_feature_columns}
        if self.varlen_sparse_feature_columns:
            for col in self.varlen_sparse_feature_columns:
                self.data[col] = np.vstack(df[col].values)
        self.label = df['label'].values

    def __len__(self):

        return self.data_len

    def __getitem__(self, idx):
        data_dict = dict()
        for col in self.dense_feature_columns:
            data_dict[col] = torch.tensor(self.data[col][idx]).float()  # float32
        for col in self.sparse_feature_columns:
            data_dict[col] = torch.tensor(self.data[col][idx]).long()  # int64
        if self.varlen_sparse_feature_columns:
            for col in self.varlen_sparse_feature_columns:
                data_dict[col] = torch.tensor(self.data[col][idx, :]).long()  # int64

        return data_dict, torch.tensor(self.label[idx]).float()


def train(model, train_dataloader, valid_dataloader, loss_fn, optimizer, path, device='cpu', epochs=5):
    # 1,模型
    model = model.to(device)
    print(f'Train on: {device}\n')

    # 2,训练参数
    flag = False
    last_improve = 0
    interval = 1000
    require_improvement = 2 * interval
    best_auc = float('-inf')

    # 3,训练
    model.train()
    total_batch = 0
    start_time = time.time()
    for epoch in range(epochs):

        for x, y in train_dataloader:
            x, y = {n: v.to(device) for n, v in x.items()}, y.to(device)

            # Compute prediction and loss
            logit = model(x).reshape(-1)
            loss = loss_fn(logit, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每隔interval个batch评估下验证集
            if total_batch % interval == 0:

                # train
                pred = torch.sigmoid(logit.detach()).cpu().numpy()
                auc = roc_auc_score(y.detach().cpu().numpy(), pred)

                # valid
                valid_loss, valid_auc = evaluate(model, valid_dataloader, loss_fn, device)

                if valid_auc > best_auc:
                    best_auc = valid_auc
                    improve = '*'
                    last_improve = total_batch
                    torch.save(model.state_dict(), path)  # save model
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  ' \
                      'Train Loss: {1:>6.3f},  Train Auc: {2:>6.3f},  ' \
                      'Val Loss: {3:>6.3f},  Val Auc: {4:>6.3f},  ' \
                      'Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), auc, valid_loss, valid_auc, time_dif, improve))

                # train
                model.train()

            total_batch += 1

            # 如果超过require_improvement个batch没有提升,停止训练
            if total_batch - last_improve > require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            break

    return


def evaluate(model, dataloader, loss_fn, device):
    # 1,参数
    loss = 0
    total_batch = 0
    y_true = []
    y_pred = []

    # 2,模型评估
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = {n: v.to(device) for n, v in x.items()}, y.to(device)

            # loss
            logit = model(x).reshape(-1)
            loss += loss_fn(logit, y).item()

            # auc
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.sigmoid(logit).cpu().numpy())

            total_batch += 1

    loss /= total_batch
    auc = roc_auc_score(y_true, y_pred)

    return loss, auc


def predict(model, dataloader, device):
    y_pred = list()
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = {n: v.to(device) for n, v in x.items()}
            y_pred.extend(torch.sigmoid(model(x).reshape(-1)).cpu().numpy())

    return y_pred

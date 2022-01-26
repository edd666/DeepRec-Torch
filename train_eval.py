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
from torch import nn
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


def train(model, train_dataloader, valid_dataloader, loss_fn, optimizer, path, device='cpu', epochs=5):
    # 1,模型
    model = model.to(device)
    print(f'Train on: {device}\n')

    # 2,训练超参
    flag = False
    interval = 2000
    last_improve = 0
    require_improvement = 4000
    best_auc = float('-inf')

    # 3,训练
    model.train()
    total_batch = 0
    start_time = time.time()
    for epoch in range(epochs):

        for x, y in train_dataloader:
            x, y = {n: v.to(device) for n, v in x.items()}, y.to(device)

            # Compute prediction and loss
            logit, afn_logit, dnn_logit = model(x)
            logit, afn_logit, dnn_logit = logit.reshape(-1), afn_logit.reshape(-1), dnn_logit.reshape(-1)
            loss = loss_fn(logit, y.float())
            loss += loss_fn(afn_logit, y.float())
            loss += loss_fn(dnn_logit, y.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每隔interval个batch评估下验证集
            if total_batch % interval == 0:
                # train set
                prob = torch.sigmoid(logit.detach()).cpu().tolist()
                auc = roc_auc_score(y.detach().cpu().tolist(), prob)

                # valid set
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
    true_list = []
    prob_list = []

    # 2,模型评估
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = {n: v.to(device) for n, v in x.items()}, y.to(device)

            # loss
            logit, _, _ = model(x)
            logit = logit.reshape(-1)
            loss += loss_fn(logit, y.float()).item()

            # auc
            prob = torch.sigmoid(logit).cpu().tolist()
            true_list.extend(y.cpu().tolist())
            prob_list.extend(prob)

            total_batch += 1

    loss /= total_batch
    auc = roc_auc_score(true_list, prob_list)

    return loss, auc

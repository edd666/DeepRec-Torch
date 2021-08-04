# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021/7/30
# @Contact : liaozhi_edo@163.com


"""
    Model Train
"""

# packages
import torch
from tqdm import tqdm


def train(model, train_dataloader, valid_dataloader, loss_fn, optimizer, path, epochs=5):
    """
    Pytorch模型的训练代码

    :param model: torch.Module 模型
    :param train_dataloader: dataloader 训练数据集
    :param valid_dataloader: dataloader 验证数据集
    :param loss_fn: torch.Module 损失函数
    :param optimizer: torch.optim 优化器
    :param path: str 模型保存路径
    :param epochs: int 训练次数
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f'Train on {device}\n')

    # early stop
    best_loss = 1000.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        # train
        model.train()
        num_batches = len(train_dataloader)
        train_loss = 0.0
        for x, y in tqdm(train_dataloader, total=num_batches):
            x, y = x.to(device), y.to(device)

            # Compute prediction and loss
            u_v, i_v = model(x)
            logits = torch.sum(u_v * i_v, dim=1)
            loss = loss_fn(logits, y.float())
            train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= num_batches
        print(f"Train: \n loss: {train_loss:>8f}")

        # eval
        model.eval()
        num_batches = len(valid_dataloader)
        valid_loss = 0.0
        with torch.no_grad():
            for x, y in valid_dataloader:
                x, y = x.to(device), y.to(device)

                u_v, i_v = model(x)
                logits = torch.sum(u_v * i_v, dim=1)

                valid_loss += loss_fn(logits, y.float()).item()

        valid_loss /= num_batches
        print(f"Valid: \n loss: {valid_loss:>8f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), path)
            print(f"Model save to: {path}")
        else:
            print('Valid loss is not decrease in 1 epoch and break train')
            break

    print('Done!')

    return

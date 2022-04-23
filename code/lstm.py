import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df3 = pd.read_excel('./附件1：赛题A数据.xlsx')
df3['PDCP_total'] = df3['小区PDCP层所接收到的上行数据的总吞吐量比特'].add(
    df3['小区PDCP层所发送的下行数据的总吞吐量比特'])
print(df3)

import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ipt_size = 1 * 24 * 3   # 输入行数 = 1*小时*天
opt_size = 1 * 24 * 3   # 输出行数

cells = df3['小区编号'].unique()


def switchToCell(cellNum, df_raw):  # 小区号
    cells = df_raw['小区编号'].unique()
    TIME_STAMP = list(df_raw[df_raw['小区编号'] == cells[cellNum]].index)
    return TIME_STAMP

def make_dataset3(df, step=None, test=False):

    columns_3 = ['时间', '小区编号', '基站编号', '小区内的平均用户数', 'PDCP_total', '平均激活用户数']
    features = columns_3[-3:]

    df3_ = df.loc[:, columns_3]

    df3_[features] = df3_[features].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))   # 归一化

    x = np.empty([0,3])
    y = np.empty([0,3])

    c = 0
    pk = df3_[features].to_numpy()
    if test: 
        pk = pk[pk.shape[0]%opt_size:]
        edge = opt_size
    else:
        edge = ipt_size+opt_size
    while c+edge <= pk.shape[0]:
        x = np.append(x, pk[c:c+ipt_size, :], axis=0)
        y = np.append(y, pk[c+ipt_size:c+ipt_size+opt_size, :], axis=0)
        if step:
            c += step
        else:
            c += opt_size
        
    x = x.reshape(-1, ipt_size, 3)
    y = y.reshape(-1, opt_size*3)

    div = int(x.shape[0]*(1-0.3))   # 训练集比例
    x_train = torch.tensor(x[:div]).float().reshape(-1, ipt_size, 3).to(device)
    y_train = torch.tensor(y[:div]).float().reshape(-1, opt_size * 3).to(device)

    x_test = torch.tensor(x[div:]).float().reshape(-1, ipt_size, 3).to(device)
    y_test = torch.tensor(y[div:]).float().reshape(-1, opt_size * 3).to(device)

    return x_train, y_train, x_test, y_test, pk


# cells = df3['小区编号'].unique()
# for cell in cells:
#     df_ = df3[df3['小区编号']==cell]
#     x_train, y_train, x_test, y_test, pk = make_dataset3(df_, 1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.k = 12
        self.lstm1 = nn.LSTM(3, self.k, num_layers=2, batch_first=True)
        self.sigmod = nn.Sigmoid()
        self.linear2 = nn.Linear(self.k * ipt_size, 3 * opt_size)
    
    def forward(self, ipt):
        x, (h, c) = self.lstm1(ipt)
        x = self.sigmod(x)
        x = x.reshape(-1, self.k * ipt_size)
        x = self.linear2(x)

        return x

from tqdm import tqdm

torch.cuda.empty_cache()

model = Net().to(device)
# cost = torch.nn.MSELoss()
cost = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_cell(cell, df):
    losses = []

    x_train, y_train, x_test, y_test, _= make_dataset3(df, 1)

    model.train()
    bar = tqdm(range(8000))
    for i in bar:
        pred = model(x_train)
        loss = cost(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            test_loss = cost(model(x_test), y_test)
            bar.set_description('loss:{}  test_loss:{}'.format(loss.data.float().cpu(), test_loss.data.float().cpu()))
            losses.append([loss.data.float().cpu(), test_loss.data.float().cpu()])
            # print(i, loss, test_loss)


    x_train, y_train, x_test, y_test, pk= make_dataset3(df)

    model.eval()
    result = list(model(x_train).data.reshape(-1).cpu()) + \
        list(model(x_test).data.reshape(-1).cpu())
    result = np.array(result).reshape(-1, 3)

    fig1 = plt.figure(figsize=(20, 5))
    ax1 = fig1.add_subplot()
    ax1.plot(np.array(range(len(losses))).dot(20), losses)
    ax1.set_ylim(0, 0.004)
    ax1.set_title('batch: {}'.format(cell))

    plt.figure(figsize=(20, 3))
    plt.plot(range(result.shape[0]), result)
    plt.vlines(model(x_train).data.reshape(-1).cpu().reshape(-1, 3).shape[0], 0, 0.5, colors='red')

    plt.figure(figsize=(20, 3))
    plt.vlines(model(x_train).data.reshape(-1).cpu().reshape(-1, 3).shape[0], 0, 0.5, colors='red')
    pk = np.array(pk).reshape(-1, 3)
    plt.plot(range(pk.shape[0]), pk)

    plt.show()

    k = torch.tensor(pk[624:], dtype=torch.float).reshape(-1, 24, 3).to(device)
    ans = model(k).data.cpu()
    return ans.numpy()



cells = df3['小区编号'].unique()
df_ans = pd.DataFrame(columns=[1,2,3, 'xq'])
df_ans['xq'] = np.array([cells for i in range(24*3)]).flatten()
for i, cell in enumerate(cells):
    df_ = df3[df3['小区编号']==cell]
    ans = train_cell(cell, df_)
    ans = ans.reshape(-1, 3)

    for j, data in enumerate(ans):
        df_ans.iloc[i+58*j, :3] = data
print(df_ans)

df_ans.to_csv('./ans.csv')
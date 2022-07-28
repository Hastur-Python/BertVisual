from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
import torch
import torch.nn as nn
from matplotlib.font_manager import FontProperties
import random
import pandas as pd
from tokenization.tokenizer import Tokenizer



def draw():
    # 定义热图的横纵坐标
    xLabel = ['A', 'B', 'C', 'D', 'E']
    yLabel = ['1', '2', '3', '4', '5']

    # 准备数据阶段，利用random生成二维数据（5*5）
    data = []
    for i in range(5):
        temp = []
        for j in range(5):
            k = random.randint(0, 100)
            temp.append(k)
        data.append(temp)

    # 作图阶段
    fig = plt.figure()
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    # 定义横纵坐标的刻度
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    # 作图并选择热图的颜色填充风格，这里选择hot
    im = ax.imshow(data)
    # 增加右侧的颜色刻度条
    plt.colorbar(im)
    # 增加标题
    plt.title("This is a title")
    # show
    plt.show()


def draw_attn():
    df = pd.read_csv('data/data.csv')
    seq = df.iloc[13][0]
    tokenizer = Tokenizer('data/vocab.dict')
    ids, segments = tokenizer(seq)
    model = torch.load('checkpoint/model_9.pt')
    tokens = tokenizer.convert_ids_to_tokens(ids)
    xlabel = tokens
    ylabel = tokens
    logits, attn = model(ids.unsqueeze(0), segments.unsqueeze(0), return_attn=True)
    print(attn.size())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_yticks(range(len(ylabel)))
    ax.set_yticklabels(ylabel)
    ax.set_xticks(range(len(xlabel)))
    ax.set_xticklabels(xlabel)
    l = len(tokens)
    # 作图并选择热图的颜色填充风格，这里选择hot
    im = ax.imshow(attn[:, 2, :, :].reshape(l, l).detach().numpy())
    # 增加右侧的颜色刻度条
    plt.colorbar(im)
    # 增加标题
    plt.title("This is a title")
    # show
    plt.show()


def predict(i):
    df = pd.read_csv('data/data.csv')
    seq = df.iloc[i][0]
    tokenizer = Tokenizer('data/vocab.dict')
    ids, segments = tokenizer(seq)
    model = torch.load('checkpoint/model_9.pt')
    tokens = tokenizer.convert_ids_to_tokens(ids)
    xlabel = tokens
    ylabel = tokens
    logits = model(ids.unsqueeze(0), segments.unsqueeze(0), return_attn=False)
    pred = logits.argmax(dim=-1)
    print(pred)


draw_attn()
predict(13)
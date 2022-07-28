import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model.Bert import Bert
from tokenization.tokenizer import Tokenizer
import pandas as pd
import numpy as np


def eval_model(val_loader, model, tokenizer):
    acc = 0
    tot = 14
    for batch in val_loader:
        seq, label = batch
        ids, segments = tokenizer(seq)
        ids, segments = ids.unsqueeze(0), segments.unsqueeze(0)
        logits = model(ids, segments)
        pred = logits.argmax(dim=-1).reshape(-1).item()
        if pred == label:
            acc += 1
    return float(acc) / tot


def train(model, loss_fn, train_loader, val_loader, n_epochs, optimizer, tokenizer):
    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(train_loader):
            seq, label = batch
            label = torch.tensor(label, dtype=torch.long).unsqueeze(0)
            ids, segments = tokenizer(seq)
            ids, segments = ids.unsqueeze(0), segments.unsqueeze(0)
            logits = model(ids, segments)
            loss = loss_fn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"epoch={epoch}, batch_idx={batch_idx}, loss={loss}")
        torch.save(model, f'../checkpoint/model_{epoch}.pt')
        acc = eval_model(val_loader, model, tokenizer)
        print(f'epoch={epoch}, acc={acc}')


def train_loader(path):
    df = pd.read_csv(path)
    ls = []
    for i in range(df.index.size):
        ls.append((df.iloc[i][0], df.iloc[i][1]))
    return ls


if __name__ == '__main__':
    Config = dict()
    Config['n_layers'] = 6
    Config['d_model'] = 64
    Config['d_k'] = 64
    Config['d_v'] = 64
    Config['n_head'] = 4
    Config['d_hid'] = 64
    Config['vocab_size'] = 46
    Config['max_len'] = 100
    Config['n_classes'] = 4

    model = Bert(**Config)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=0.01)
    tokenizer = Tokenizer('../data/vocab.dict')
    loss_fn = nn.CrossEntropyLoss()
    train(model, loss_fn, train_loader('../data/data.csv'), train_loader('../data/data.csv'), 10,
          optimizer, tokenizer)
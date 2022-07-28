import numpy as np
import pandas as pd
import pickle

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    texts = df['text']
    vocab = dict()
    i = 1
    for text in texts:
        text = text.split(' ')
        for word in text:
            if vocab.get(word) is None:
                vocab[word] = len(vocab) + 1
    vocab['<cls>'] = len(vocab) + 1
    vocab['<sep>'] = len(vocab) + 1
    vocab["<pad>"] = len(vocab) + 1
    print(len(vocab))
    file = open('vocab.dict', 'wb')
    pickle.dump(vocab, file)
    file.close()

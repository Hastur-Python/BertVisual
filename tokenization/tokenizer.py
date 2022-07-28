import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class Tokenizer:
    def __init__(self, path):
        file = open(path, 'rb')
        vocab = pickle.load(file)
        self.vocab = vocab
        file.close()

    def _convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token))
        return ids

    def _get_tokens_and_segments(self, tokens_a, tokens_b=None):
        r"""
        :param tokens_a:分词后的输入
        :param tokens_b:
        :return:
        """
        tokens = []
        tokens += ['<cls>'] + tokens_a + ['<sep>']
        segments = [0] * (len(tokens_a) + 2)
        if tokens_b:
            tokens += tokens_b + ['<sep>']
            segments += [1] * (len(tokens_b) + 1)
        return tokens, segments

    def __call__(self, tokens_a, tokens_b=None):
        if isinstance(tokens_a, str):
            tokens_a = tokens_a.split(' ')

        if tokens_b:
            if isinstance(tokens_b, str):
                tokens_b = tokens_b.split(' ')

            tokens, segments = self._get_tokens_and_segments(tokens_a, tokens_b)
            ids = self._convert_tokens_to_ids(tokens)

            ids, segments = torch.tensor(ids, dtype=torch.long), torch.tensor(segments, dtype=torch.long)
            return ids, segments
        else:
            tokens, segments = self._get_tokens_and_segments(tokens_a)
            ids = self._convert_tokens_to_ids(tokens)
            ids, segments = torch.tensor(ids, dtype=torch.long), torch.tensor(segments, dtype=torch.long)
            return ids, segments

    def convert_ids_to_tokens(self, ids):
        tokens = []
        reverse_vocab = {v: k for k, v in self.vocab.items()}

        for id in ids:
            tokens.append(reverse_vocab[id.item()])
        return tokens

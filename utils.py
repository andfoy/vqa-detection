# -*- coding: utf-8 -*-

"""
Misc functions and class wrappers.
"""

import sys
import time
import torch
import codecs
from visdom import Visdom


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


class VisdomWrapper(Visdom):
    def __init__(self, *args, **kwargs):
        Visdom.__init__(self, *args, **kwargs)
        self.plots = {}

    def init_line_plot(self, name,
                       X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1, 3)).cpu(), **opts):
        self.plots[name] = self.line(X=X, Y=Y, opts=opts)

    def plot_line(self, name, **kwargs):
        self.line(win=self.plots[name], **kwargs)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def add_to_corpus(self, line):
        """Tokenizes a text line."""
        # Add words to the dictionary
        words = line.split() + ['<eos>']
        # tokens = len(words)
        for word in words:
            self.dictionary.add_word(word)

    def tokenize(self, line):
        # Tokenize line contents
        words = line.split() + ['<eos>']
        tokens = len(words)
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            if word not in self.dictionary.word2idx:
                word = '<unk>'
            ids[token] = self.dictionary.word2idx[word]
            token += 1

        return ids

    def tokenize_file(self, file_path):
        tokens = []
        with codecs.open(file_path, 'r', 'utf-8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word not in self.dictionary.word2idx:
                        word = '<unk>'
                    token = self.dictionary.word2idx[word]
                    tokens.append(token)
        return torch.LongTensor(tokens)

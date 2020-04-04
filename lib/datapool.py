import torch
import numpy as np


class DataPool():
    def __init__(self, pool_size, split=True):
        self.pool_size = pool_size
        self.split = split
        self.pool = []

    def add(self, data):
        if self.split:
            num = data.size(0)
            for i in range(num):
                if len(self.pool) < self.pool_size:
                    self.pool.append(data[i])
                elif np.random.random() > 0.5:
                    idx = np.random.randint(0, self.pool_size, 1)[0]
                    self.pool[idx] = data[i]
        else:
            if len(self.pool) < self.pool_size:
                self.pool.append(data)
            elif np.random.random() > 0.5:
                idx = np.random.randint(0, self.pool_size, 1)[0]
                self.pool[idx] = data

    def query(self, batch=1):
        if len(self.pool) <= batch:
            return self.pool.copy()
        else:
            idxes = np.random.randint(0, len(self.pool), batch)
            result = [self.pool[idx] for idx in idxes]
            return result


class MutilDomainDataPool():
    def __init__(self, pool_size, split=True):
        self.pool_size = pool_size
        self.split = split
        self.pool = {}

    def add(self, label, data):
        if label not in self.pool.keys():
            self.pool[label] = []
        if self.split:
            num = data.size(0)
            for i in range(num):
                if len(self.pool[label]) < self.pool_size:
                    self.pool[label].append(data[i])
                elif np.random.random() > 0.5:
                    idx = np.random.randint(0, self.pool_size, 1)[0]
                    self.pool[label][idx] = data[i]
        else:
            if len(self.pool[label]) < self.pool_size:
                self.pool[label].append(data)
            elif np.random.random() > 0.5:
                idx = np.random.randint(0, self.pool_size, 1)[0]
                self.pool[label][idx] = data

    def query(self, label, batch=1):
        if len(self.pool[label]) <= batch:
            return self.pool[label].copy()
        else:
            idxes = np.random.randint(0, len(self.pool[label]), batch)
            result = [self.pool[label][idx] for idx in idxes]
            return result

    def size(self, label=-1):
        if label == -1:
            num = 0
            for i in self.pool:
                num += len(self.pool[i])
            return num
        elif label not in self.pool:
            return 0
        else:
            return len(self.pool[label])

    def labels(self):
        return list(self.pool.keys())

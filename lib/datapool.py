import torch
import numpy as np


class DataPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.pool = []

    def add(self, data):
        num = data.size(0)
        for i in range(num):
            if len(self.pool) < self.pool_size:
                self.pool.append(data[i])
            elif np.random.random() > 0.5:
                idx = np.random.randint(0, self.pool_size, 1)[0]
                self.pool[idx] = data[i]

    def query(self, batch):
        if len(self.pool) < batch:
            return torch.stack(self.pool)
        else:
            idxes = np.random.randint(0, len(self.pool), batch)
            result = [self.pool[idx] for idx in idxes]
            return torch.stack(result)

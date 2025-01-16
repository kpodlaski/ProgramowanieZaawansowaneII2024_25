import math
import torch
import numpy as np
from random import randint
from torch.utils.data import TensorDataset, DataLoader

def generate_dataset_element(k, window_size):
    x = 2 * math.pi * k / 1000
    ys = []
    for i in range(1, window_size + 1):
        y = math.sin(x - i * 2 * math.pi / 1000)
        ys.append(y)
    return np.array(ys), np.array([math.sin(x)])

def generate_sinus_dataset(size, window_size=4, batch_size=64):
    patterns = []
    values = []
    for k in range(size):
        k = randint(0,1000)
        ys, xs = generate_dataset_element(k, window_size)
        patterns.append(ys)
        values.append(xs)
    tensor_patterns = torch.Tensor(patterns)
    tensor_values = torch.Tensor(values)
    my_dataset = TensorDataset( tensor_patterns, tensor_values)
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size)
    return my_dataset, my_dataloader



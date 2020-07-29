import pickle
from copy import copy

import torch.optim as optim
import torch
import torchvision
from torchvision.transforms import transforms
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from memory import Memory
from model import Net
import matplotlib.pyplot as plt
def cossim(mem,q):
    query = F.normalize(torch.matmul(mem.query_proj.weight, q) + mem.query_proj.bias, dim=0)
    cs = torch.matmul(query, torch.t(mem.keys))
    return cs

if __name__ == "__main__":

    for mit in range(64):
        net=torch.load("net_{}".format(mit))
        mem=net.memory
        with torch.no_grad():
            for cls in range(10):
                idxs,_=torch.where(mem.values==cls)
                keys=mem.keys[idxs]
                centroid=torch.mean(keys,dim=0)
                cs=cossim(mem,centroid)
                mcs=torch.mean(cs)
                plt.plot(cs)
                # for idx,v in enumerate(mem.values):
                    # if v==cls:
                    #     print(cls,idx)
                    #     query = F.normalize(torch.matmul(mem.query_proj.weight,mem.keys[idx])+mem.query_proj.bias,dim=0)
                    #     cossim = torch.matmul(query, torch.t(mem.keys))
                    #     print(cossim)
                    #     plt.scatter(range(mem.values.size()[0]),cossim.numpy(),c=mem.values)
                    #     break
                break
    plt.show()

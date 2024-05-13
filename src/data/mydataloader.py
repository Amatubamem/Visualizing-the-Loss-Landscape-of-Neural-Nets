import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.backends import mps
from matplotlib.colors import LogNorm

# import pandas as pd
import os
import glob
# from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import src.model.models as md
# from plotter import Plotter, NNPlotter
import src.model.normalizer as nz
from src.model.training import train_united_model, test_united_model

# ロバストな設定項目 頻繁には変更しない
# device setting
device = torch.device('mps' if mps.is_available() else "cpu") 
dtype = torch.float32

# settings
delays = 250 # embedding dimension
hidden_size = 200 # hidden layer size
dt = 0.01 # time step
comAidx = 14
indices = [14, 22, 19, 18, 21]
fileIDs = ['O-12139', 'O-14076', 'O-12841', 'O-12593', 'O-14063']
use_files = 5

learning_rate = 1e-3 #hyperparameter
layers_size = 5 

use_dst = True
U0 = md.DST_basis(delays) if use_dst else np.eye(delays, dtype = np.float32)
norm_func_tag = 'linear01'
dmddir = f'50As_Hsb{"DST" if use_dst else "TD"}{norm_func_tag}_{delays}'
normalizer = nz.Normalizer(nz.LinearNormalization(0.1))

# Dataloaderの作成
datasets = []
for index in indices:
    t, x3d, u3d, fileID = md.load_earthquake_data(index, 'trg', 'src/data/data_insuff/')
    xhorg = md.holizontal(x3d)
    xhdmd = np.load(f'src/data/{dmddir}/xrc_{delays}_{fileID}.npy')
    # xhdmd = np.load(f'{dmddir}/xrcbyA{comAidx}_{delays}_{fileID}.npy')
    uh = md.holizontal(u3d) 
    xnormorg = normalizer.norm(xhorg)
    xnormdmd = normalizer.norm(xhdmd)
    unorm = normalizer.norm(uh)
    datasets.append((t, xnormorg, xnormdmd, unorm, fileID))

Admds = {}
Xts1 = []; Xts2 = []
#! ORG or DMD
for i, (t, xnp, _, unp, fileID) in enumerate(datasets):
    L = min(len(t)*2//3, maxlen:=6000)
    Xnp = md.earthquake_to_hankel(xnp[:L], unp[:L], delays, use_dst)    
    Xtens = md.to_tensor(Xnp, dtype, device)
    Admds[indices[i]] = md.to_tensor(np.load(f'src/data/{dmddir}/{delays}_{fileID}.npy'), dtype, device)
    label = torch.ones(Xtens.shape[1], dtype=torch.float32, device=device) * indices[i]
    XtensLabel = torch.cat([Xtens, label.unsqueeze(0)], dim=0)
    Xts1.append(XtensLabel[:, :-1])
    Xts2.append(XtensLabel[:, 1:])
    # break
Xt1 = torch.cat(Xts1, dim=1); Xt2 = torch.cat(Xts2, dim=1)
train_dataloader, test_dataloader = md.train_test_split(Xt1.T, Xt2.T)

def dataloader():
    return train_dataloader, test_dataloader
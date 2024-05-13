import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.data.dataloader import dataloader
from src.model.training import train_united_model
import src.model.models as md

NUM_EPOCHS = 40

def prepare_trained_model(model):

    if os.path.isfile("./trained_model"):
        print("no need to train.")
        model.load_state_dict(torch.load("./trained_model"))
        return model
    else:
        train_(model)
        return model

def train(model):
    train_loader, test_loader = dataloader()


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            train_acc += (outputs.max(1)[1] == labels).sum().item()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.max(1)[1] == labels).sum().item()

        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)

        print('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'
              .format(epoch + 1, NUM_EPOCHS, i + 1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

    if False:
        # plt.figure()
        # plt.plot(range(NUM_EPOCHS), train_loss_list, color='blue', linestyle='-', label='train_loss')
        # plt.plot(range(NUM_EPOCHS), val_loss_list, color='green', linestyle='--', label='val_loss')
        # plt.legend()
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.title('Training and validation loss')
        # plt.grid()
        #
        # plt.figure()
        # plt.plot(range(NUM_EPOCHS), train_acc_list, color='blue', linestyle='-', label='train_acc')
        # plt.plot(range(NUM_EPOCHS), val_acc_list, color='green', linestyle='--', label='val_acc')
        # plt.legend()
        # plt.xlabel('epoch')
        # plt.ylabel('acc')
        # plt.title('Training and validation accuracy')
        # plt.grid()
        pass

    torch.save(model.state_dict(), './trained_model')

    return model

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
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import src.model.models as md
# from plotter import Plotter, NNPlotter
import src.model.normalizer as nz
from src.model.training import train_united_model, test_united_model

def train_(model):
    # ロバストな設定項目 頻繁には変更しない
    # device setting
    device = torch.device('mps' if mps.is_available() else "cpu") 
    model.to(device)
    dtype = torch.float32

    # settings
    delays = 250 # embedding dimension
    indices = [14, 22, 19, 18, 21]

    learning_rate = 1e-3 #hyperparameter

    use_dst = True
    norm_func_tag = 'linear01'
    dmddir = f'50As_Hsb{"DST" if use_dst else "TD"}{norm_func_tag}_{delays}'
    normalizer = nz.Normalizer(nz.LinearNormalization(0.1))
    # Dataloaderの作成
    datasets = []
    for index in indices:
        t, x3d, u3d, fileID = md.load_earthquake_data(index, 'trg', 'src/data/data_insuff/')
        xhorg = md.holizontal(x3d)
        xhdmd = np.load(f'src/data/{dmddir}/xrc_{delays}_{fileID}.npy')
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
    
    # training settings
    epochs = 30  #!
    

    # * training routine
    if epochs != 0:
        # 損失関数，最適化手法の設定
        loss_fn = md.UnitedDMDLoss(model, Admds, Admds[14], loss_ratio=[0.15, 1., 0.]) #!
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        pre_epochs = len(model.testlosses)
        for i in range(epochs):
            print(f"Epoch {pre_epochs+i+1}\n-------------------------------")
            train_united_model(train_dataloader, model, loss_fn, optimizer)
            test_loss = test_united_model(test_dataloader, model, loss_fn)
            model.testlosses.append(test_loss)
        print("Done!")

        ## 保存
    torch.save(model.state_dict(), "./trained_model")
        # model.to(device)
    return model.cpu()
        
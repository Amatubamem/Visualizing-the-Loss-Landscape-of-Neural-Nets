import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from src.data.mydataloader import dataloader


def eval_loss(model):
    _, test_loader = dataloader()
    criterion = nn.CrossEntropyLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    correct = 0
    total_loss = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            batch_size = inputs.size(0)
            total += batch_size
            inputs = Variable(inputs)
            targets = Variable(targets)
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets).sum().item()

    return total_loss / total, 100. * correct / total


import numpy as np
import torch
import torch.nn as nn
from torch.backends import mps

import src.model.models as md
import src.model.normalizer as nz
from src.model.training import train_united_model, test_united_model

def eval_loss_(model):
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
    
    loss_fn = md.UnitedDMDLoss(model, Admds, Admds[14], loss_ratio=[0.15, 1., 0.]) #!
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pre_epochs = len(model.testlosses)
    test_loss, _, _, _ = test_united_model(test_dataloader, model, loss_fn)
    return test_loss, 0
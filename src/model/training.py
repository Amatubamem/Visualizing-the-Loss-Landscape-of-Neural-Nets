"""
training and testing function for nn.Module networks
"""

import torch
from torch import nn, Tensor

from src.model.models import *
# from networks import *
# from lossfunctions import *

#-------------- for single models -----------------------------------
def train_single_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        def closure():
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(closure)
        #logging
        if batch % (size // len(X) // 5) == (size // len(X) // 5) - 1:
            loss = closure().item()
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_single_model(dataloader, model, loss_fn):
    # test function for united model (cutting label)
    size = len(dataloader.dataset)
    model.eval()
    test_losses = []
    mselosses = []
    dmdlosses = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            mseloss, dmdloss, dmdloss1 = (x.item() for x in loss_fn.get_each(pred, y))
            mselosses.append(mseloss)
            dmdlosses.append(dmdloss)
            test_losses.append(mseloss + dmdloss + dmdloss1)
    testloss = sum(test_losses) / len(test_losses)
    mseloss = sum(mselosses) / len(mselosses)
    dmdloss = sum(dmdlosses) / len(dmdlosses)
    print(f"Test Avg loss: {testloss:>8f} mse:{mseloss:>8f} dmd:{dmdloss:>8f}\n")
    return testloss, mseloss, dmdloss
        

#-------------- for united models -----------------------------------
def train_united_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        def closure():
            pred = model(X[:, :-1])
            loss = loss_fn(pred, y[:, :-1], int(X[-1, -1]))
            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(closure)
        #logging
        if batch % (size // len(X) // 5) == (size // len(X) // 5) - 1:
            loss = closure().item()
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_united_model(dataloader, model, loss_fn):
    # test function for united model (cutting label)
    size = len(dataloader.dataset)
    model.eval()
    test_losses = []
    mselosses = []
    dmdlosses = []
    dmdlosses1 = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X[:, :-1])
            mseloss, dmdloss, dmdloss1 = (x.item() for x in loss_fn.get_each(pred, y[:, :-1], int(X[-1, -1])))
            mselosses.append(mseloss)
            dmdlosses.append(dmdloss)
            dmdlosses1.append(dmdloss1)
            test_losses.append(mseloss + dmdloss + dmdloss1)
    testloss = sum(test_losses) / len(test_losses)
    mseloss = sum(mselosses) / len(mselosses)
    dmdloss = sum(dmdlosses) / len(dmdlosses)
    dmdloss1 = sum(dmdlosses1) / len(dmdlosses1)
    print(f"Test Avg loss: {testloss:>8f} mse:{mseloss:>8f} dmd:{dmdloss:>8f} dmd1:{dmdloss1:>8f}\n")
    return testloss, mseloss, dmdloss, dmdloss1
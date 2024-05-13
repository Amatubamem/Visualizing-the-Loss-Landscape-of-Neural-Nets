"""
ネットワークや損失関数，データローダーなどの定義
変更した場合参照するnotebookを再実行する必要がある
"""

import torch
from torch import nn, Tensor
import numpy as np
from numpy import ndarray  
from itertools import islice
from torch.utils.data import Dataset, DataLoader
import sklearn

from typing import TypeVar, Union
import glob
import os

import time
import warnings
import tqdm 

# ndarryaまたはTensorを受け取る型ヒント
NdarrayTensor = TypeVar('NdarrayTensor', ndarray, Tensor)
ModuleType = TypeVar('ModuleType', bound=nn.Module)

warnings.simplefilter('always', UserWarning)
# warnings.filterwarnings('error', category=UserWarning)

'''
888b     d888  .d88888b.  8888888b.  8888888888 888      
8888b   d8888 d88P" "Y88b 888  "Y88b 888        888      
88888b.d88888 888     888 888    888 888        888      
888Y88888P888 888     888 888    888 8888888    888      
888 Y888P 888 888     888 888    888 888        888      
888  Y8P  888 888     888 888    888 888        888      
888   "   888 Y88b. .d88P 888  .d88P 888        888      
888       888  "Y88888P"  8888888P"  8888888888 88888888 
'''
class NetworkModule(nn.Module):
    def __init__(self):
        super(NetworkModule, self).__init__()

NetworkModuleType = TypeVar('NetworkModuleType', bound=NetworkModule)
    
class DeeptimedelayNN(NetworkModule):
    '''
    multidimentional-time-delay neural network
    n : input&output dimension
    m : hidden layer dimension
    '''
    def __init__(self, input_size: int, hidden_size: int, layers_size: int = 5) -> None:
        super(DeeptimedelayNN, self).__init__()
        self.testlosses = []
        self.relu = nn.ReLU()
        layers = [nn.Linear(input_size, hidden_size)]
        layers.extend([nn.Linear(hidden_size, hidden_size) for _ in range(layers_size-2)])
        layers.append(nn.Linear(hidden_size, input_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        assert x.device == self.layers[0].weight.device
        assert x.shape[-1] == self.layers[0].in_features
        
        for layer in self.layers:
            if layer is self.layers[-1]:
                x = layer(x)
            else:
                x = self.relu(layer(x))
        return x

    def forward_with_intermediates(self, x: Tensor) -> list[Tensor]:
        zs = []
        for layer in self.layers:
            x = layer(x)
            zs.append(x.clone())
            x = self.relu(x)
        return zs
    
class DeeptimedelayNN_(NetworkModule):
    '''
    multidimentional-time-delay neural network
    n : input&output dimension
    m : hidden layer dimension
    '''
    def __init__(self, input_size: int, hidden_size: int, layers_size: int = 5) -> None:
        super(DeeptimedelayNN_, self).__init__()
        self.testlosses = []
        self.relu = nn.ReLU()
        layers = [nn.Linear(input_size, hidden_size, bias=False)]
        layers.extend([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(layers_size-2)])
        layers.append(nn.Linear(hidden_size, input_size, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.layers[0].in_features:
            raise ValueError(f'input size must be {self.layers[0].in_features}, but got {x.shape[1]}')
        for layer in self.layers:
            if layer is self.layers[-1]:
                x = layer(x)
            else:
                x = self.relu(layer(x))
        return x

    def forward_with_intermediates(self, x: Tensor) -> list[Tensor]:
        zs = []
        for layer in self.layers:
            x = layer(x)
            zs.append(x.clone())
            x = self.relu(x)
        return zs


'''
888       .d88888b.   .d8888b.   .d8888b.  
888      d88P" "Y88b d88P  Y88b d88P  Y88b 
888      888     888 Y88b.      Y88b.      
888      888     888  "Y888b.    "Y888b.   
888      888     888     "Y88b.     "Y88b. 
888      888     888       "888       "888 
888      Y88b. .d88P Y88b  d88P Y88b  d88P 
88888888  "Y88888P"   "Y8888P"   "Y8888P"  
'''

class LossFunctionModule(nn.Module):
    pass

LossFunctionModuleType = TypeVar('LossFunctionModuleType', bound=LossFunctionModule)

class UnitedDMDLoss(nn.Module):
    def __init__(self, model, dmd_data, commonA, loss_ratio = [1., 1., 1.]) -> None:
        super(UnitedDMDLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.model = model
        self.dmd_data = dmd_data
        self.commonA = commonA
        self.loss_ratio = loss_ratio
        self.exp = BiasedExpantion(self.model)

    def dmdloss(self):
        diff = self.exp.F0 - self.commonA
        dmdloss = torch.sqrt(torch.sum(diff * diff))
        return dmdloss
    
    def dmdloss1(self, target, label):
        diff = self.exp.Fsigma(target[-1])+self.exp.F0 - self.dmd_data[label]
        dmdloss = torch.sqrt(torch.sum(diff * diff))
        return dmdloss
    
    def get_each(self, predicted, target, label) -> tuple[list[float], list[float], list[float]]:
        self.exp = BiasedExpantion(self.model)
        loss = self.mse_loss(predicted, target)
        dmdloss = self.dmdloss()
        dmdloss1 = self.dmdloss1(target, label)
        return loss * self.loss_ratio[0], dmdloss * self.loss_ratio[1], dmdloss1 * self.loss_ratio[2]
        
    def forward(self, predicted, target, label):
        loss, dmdloss, dmdloss1 = self.get_each(predicted, target, label)
        return loss + dmdloss + dmdloss1

class DMDLoss(nn.Module):
    def __init__(self, model, dmd_data, loss_ratio = [1., 1.]) -> None:
        super(DMDLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.model = model
        self.dmd_data = dmd_data
        self.loss_ratio = loss_ratio
        self.exp = BiasedExpantion(self.model)

    def dmdloss(self):
        diff = self.exp.F0 - self.dmd_data
        dmdloss = torch.sqrt(torch.sum(diff * diff))
        return dmdloss
    
    def get_each(self, predicted, target) -> tuple[list[float], list[float]]:
        self.exp = BiasedExpantion(self.model)
        loss = self.mse_loss(predicted, target)
        dmdloss = self.dmdloss()
        return loss * self.loss_ratio[0], dmdloss * self.loss_ratio[1]
        
    def forward(self, predicted, target):
        loss, dmdloss = self.get_each(predicted, target)
        return loss + dmdloss

'''
88888888888 8888888b.         d8888 888b    888  .d8888b.  8888888888  .d88888b.  8888888b.  888b     d888 
    888     888   Y88b       d88888 8888b   888 d88P  Y88b 888        d88P" "Y88b 888   Y88b 8888b   d8888 
    888     888    888      d88P888 88888b  888 Y88b.      888        888     888 888    888 88888b.d88888 
    888     888   d88P     d88P 888 888Y88b 888  "Y888b.   8888888    888     888 888   d88P 888Y88888P888 
    888     8888888P"     d88P  888 888 Y88b888     "Y88b. 888        888     888 8888888P"  888 Y888P 888 
    888     888 T88b     d88P   888 888  Y88888       "888 888        888     888 888 T88b   888  Y8P  888 
    888     888  T88b   d8888888888 888   Y8888 Y88b  d88P 888        Y88b. .d88P 888  T88b  888   "   888 
    888     888   T88b d88P     888 888    Y888  "Y8888P"  888         "Y88888P"  888   T88b 888       888 
'''
def to_hankel(x: NdarrayTensor, d: int) -> NdarrayTensor:
    """
    make a Hankel matrix from a waveform
    >>> wave2hankel(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), 3)
    np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9]])
    """    
    q = x.shape[0] - d + 1
    if isinstance(x, ndarray):
        idx = np.arange(d)[:, None] + np.arange(q)
        H = x[idx]
        return H
    elif isinstance(x, Tensor):
        idx = torch.arange(d)[:, None] + torch.arange(q)
        H = x[idx]
        return H
    else:
        raise TypeError('waveform must be ndarray or Tensor')

def from_hankel(H: NdarrayTensor, d: int | None = None) -> NdarrayTensor:
    """
    >>> hankel2wave(np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9]]), 3)
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    if type(H) != type(H[0]):
        raise TypeError('types of Hankel matrix and its elements must be same')
    
    if d is None:
        d = H.shape[0]
    q = H.shape[1]
    N = q + d - 1
    
    if isinstance(H, ndarray):
        X_emp = np.full([d, N], np.nan)
        i = np.arange(d)[:, None]
        j = np.arange(q) + i
        X_emp[i, j] = H
        x_mrc = np.nanmean(X_emp, axis=0)
        return x_mrc
    
    elif isinstance(H, Tensor):
        X_emp = torch.full((d, N), float('nan'), dtype=H.dtype, device=H.device)
        i = torch.arange(d)[:, None]
        j = torch.arange(q) + i
        X_emp[i, j] = H
        x_mrc = torch.nanmean(X_emp, dim=0)
        return x_mrc
    
    else:
        raise TypeError('Hankel matrix must be ndarray or Tensor')
    
def to_hankel_3d(x3d: Tensor, d: int, DST: Tensor | None = None) -> Tensor:
    dtype = x3d.dtype
    device = x3d.device

    H = torch.zeros([3*d, x3d.shape[1]-d+1], dtype=dtype, device=device)
    if DST is None:
        invDST = torch.eye(d, dtype=dtype, device=device)
    else:
        invDST = torch.linalg.inv(DST.cpu()).to(dtype=dtype, device=device)

    H[0::3, :] = invDST @ to_hankel(x3d[0], d)
    H[1::3, :] = invDST @ to_hankel(x3d[1], d)
    H[2::3, :] = invDST @ to_hankel(x3d[2], d)
    return H

def from_hankel_3d(H: Tensor, d: int | None = None, DST: Tensor | None = None) -> Tensor:
    if d is None:
        d = int(H.shape[0]/3)
    if DST is None:
        DST = torch.eye(d, dtype=H.dtype, device=H.device)

    x_ns = from_hankel(DST @ H[0::3, :], d)
    x_ew = from_hankel(DST @ H[1::3, :], d)
    x_ud = from_hankel(DST @ H[2::3, :], d)
    x3d = torch.vstack((x_ns, x_ew, x_ud))
    return x3d

def holizontal(x3d: ndarray) -> ndarray:
    x_ns, x_ew, x_ud = x3d

    if isinstance(x3d, ndarray):
        x_nsew = np.vstack([x_ns, x_ew]).T
        
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2)
    pca.fit(x_nsew)

    # 固有ベクトルを使用してABを変換
    x_holizontal = pca.transform(x_nsew)[:,0]
    return x_holizontal

def earthquake_to_hankel(x: ndarray, u: ndarray, delays: int, use_DST: bool = False) -> ndarray:
    if use_DST:
        inv_dst = np.linalg.inv(DST_basis(delays))
    else:
        inv_dst = np.eye(delays)

    # Hankel行列に変換, DST変換
    Wx = inv_dst @ to_hankel(x, delays)
    Wu = inv_dst @ to_hankel(u, delays)
    X = np.concatenate((Wx, Wu), axis=0)
    return X

def earthquake_to_hankel_3d(x3d: Tensor, u3d: Tensor, delays: int, use_DST: bool = False) -> Tensor:
    if use_DST:
        dst=torch.from_numpy(DST_basis(delays)).to(dtype=x3d.dtype)
    else:
        dst=None
    Wx = to_hankel_3d(x3d, delays, DST=dst)
    Wu = to_hankel_3d(u3d, delays, DST=dst)
    X = torch.cat((Wx, Wu), dim=0)
    return X


def DST_basis(delays: int) -> ndarray:
    # DST-basis
    j_indices, i_indices = np.meshgrid(np.arange(delays), np.arange(delays), indexing='ij')
    Phi_ = np.sqrt(2/delays) * np.sin((2*i_indices+1)*(j_indices+1)*np.pi/(2*delays))
    return Phi_.T

def to_tensor(x: ndarray, dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')) -> Tensor:
    return torch.from_numpy(x).clone().to(dtype=dtype, device=device)

def to_numpy(x: Tensor) -> ndarray:
    return x.detach().clone().cpu().numpy()


'''
888      8888888888        d8888 8888888b.  888b    888 
888      888              d88888 888   Y88b 8888b   888 
888      888             d88P888 888    888 88888b  888 
888      8888888        d88P 888 888   d88P 888Y88b 888 
888      888           d88P  888 8888888P"  888 Y88b888 
888      888          d88P   888 888 T88b   888  Y88888 
888      888         d8888888888 888  T88b  888   Y8888 
88888888 8888888888 d88P     888 888   T88b 888    Y888 
'''

class CustomDataset(Dataset):
    def __init__(self, X: Tensor, Y: Tensor):
        self.X = X
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

def train_test_split(X1: torch.Tensor, X2: torch.Tensor, train_ratio: float = 0.8, batch_size: int = 8) -> tuple[DataLoader, DataLoader]:
    dataset = CustomDataset(X1, X2)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



'''
8888888888 Y88b   d88P 8888888b.         d8888 888b    888 88888888888 8888888  .d88888b.  888b    888 
888         Y88b d88P  888   Y88b       d88888 8888b   888     888       888   d88P" "Y88b 8888b   888 
888          Y88o88P   888    888      d88P888 88888b  888     888       888   888     888 88888b  888 
8888888       Y888P    888   d88P     d88P 888 888Y88b 888     888       888   888     888 888Y88b 888 
888           d888b    8888888P"     d88P  888 888 Y88b888     888       888   888     888 888 Y88b888 
888          d88888b   888          d88P   888 888  Y88888     888       888   888     888 888  Y88888 
888         d88P Y88b  888         d8888888888 888   Y8888     888       888   Y88b. .d88P 888   Y8888 
8888888888 d88P   Y88b 888        d88P     888 888    Y888     888     8888888  "Y88888P"  888    Y888                                                                                               
'''

def find_edges_of_ones(b: int, L: int) -> tuple[int, int]:
    """
    与えられた自然数bのバイナリ表現において、両端の1の位置を返す関数。
    ビット位置は右端を0としてカウントする。

    :param b: バイナリ表現を見たい自然数
    :return: 右端と左端の1の位置（タプル形式）
    """
    if b == 0:
        return -1, -1  # bが0の場合、1のビットは存在しない

    right_one = 0
    temp = b
    while (temp & 1) == 0 and right_one < L:
        temp >>= 1  # 右シフトして最下位ビットをチェック
        right_one += 1

    left_one = L - 1
    while (b & (1 << left_one)) == 0 and left_one >= 0:
        left_one -= 1

    return (left_one, right_one)

def householder_v_batch(z: Tensor) -> Tensor:
    if z.dtype == torch.complex64:
        raise TypeError('z must be real tensor')

    # 1次元テンソル（単一のベクトル）の場合、次元を増やす
    if z.ndim == 1:
        z = z.unsqueeze(0)

    abs_z = torch.abs(z)
    diff = abs_z - z
    dot_products = torch.sum(z * diff, dim=1, keepdim=True)

    conditions = torch.all(torch.abs(z) == z, dim=1)
    conditions_complex = torch.squeeze(dot_products <= 0)

    valid_indices = ~conditions & ~conditions_complex
    valid_diff = diff[valid_indices]
    valid_dot = dot_products[valid_indices]

    result = torch.zeros_like(z)
    result[valid_indices] = valid_diff / torch.sqrt(2 * valid_dot)

    # 元の次元に戻す
    return result.squeeze(0) if z.shape[0] == 1 else result

def householder_batch(W: Tensor, z: Tensor) -> Tensor:
    vz = householder_v_batch(z)

    # 1次元テンソルの場合、次元を増やす
    if vz.ndim == 1:
        vz = vz.unsqueeze(0)

    # 行列演算を適用
    M = -torch.bmm(vz.unsqueeze(2), vz.unsqueeze(1)) @ W

    return M.squeeze(0) if z.ndim == 1 else M


def householder_v(z:Tensor) -> Tensor:
        if z.dtype == torch.complex64:
            raise TypeError('z must be real tensor')
        if torch.all(torch.abs(z) == z): # zが正
            return torch.zeros_like(z)
        elif torch.dot(z, torch.abs(z) - z) <= 0: # 分母が虚数
            return (torch.abs(z) - z) / torch.sqrt(torch.abs(2 *  torch.dot(z, torch.abs(z) - z)))
        else: 
            return (torch.abs(z) - z) / torch.sqrt(2 *  torch.dot(z, torch.abs(z) - z))

def householder(W:Tensor, z:Tensor) -> Tensor:
    """
    W: 2d tensor
    """
    vz = householder_v(z.detach())
    try:
        M = - torch.outer(vz, vz) @ W
    except:
        print(z.shape, vz.shape)
        M = torch.zeros_like(W)

    return M

def householder_C(W:Tensor, z:Tensor) -> Tensor:
    """
    W: 2d tensor
    """
    warnings.warn('models.householder_C is deprecated. use models.householder instead. That is faster than this.', UserWarning)
    def v(z:Tensor) -> Tensor:
        if z.dtype != torch.complex64:
            raise TypeError('z must be complex tensor')
        if torch.all(torch.abs(z) == z):
            return torch.zeros_like(z)
        # elif torch.dot(z, torch.abs(z) - z) <= 0:
        #     # print('分母を絶対値に')
        #     return (torch.abs(z) - z) / torch.sqrt(torch.abs(2 *  torch.dot(z, torch.abs(z) - z)))
        else:
            return (torch.abs(z) - z) / torch.sqrt(2 *  torch.dot(z, torch.abs(z) - z))

    vz = v(z.detach().clone().cpu().to(torch.complex64))
    M = - torch.outer(vz, vz) @ W.detach().clone().cpu().to(dtype=torch.complex64)
    return M # torch.real(M).to(W.device)

def Fzero(model: NetworkModule) -> Tensor:
    # F = torch.eye(model.fc1.weight.shape[1]).to(next(model.parameters()).device)
    # print(F.device)
    params = model.named_parameters()
    F = next(params)[1]
    for name, param in params:
        if 'weight' in name:
            F = param @ F
    return F

class BiasedExpantion():
    def __init__(self, model) -> None:
        self.model = model
        
        self.Wlist = [param for name, param in model.named_parameters() if 'weight' in name]
        self.blist = [param for name, param in model.named_parameters() if 'bias' in name]
        self.Alist = [self._make_matrix_A(W, b) for W, b in zip(self.Wlist, self.blist)]
        self.F0 = self._make_matrix_fzero()
        
        self.L = len(self.Wlist)
        
    @staticmethod
    def _make_matrix_A(W, b):
        wb = torch.hstack([W, b.unsqueeze(1)])
        zeroone = torch.cat([torch.zeros(W.shape[1]), torch.ones(1)]).to(wb.device)
        A = torch.vstack([wb, 
                          zeroone])
        return A

    def _make_matrix_fzero(self):
        result = self.Alist[-1] # A_L
        for A in self.Alist[-2::-1]: # A_L-1, ..., A_1
            result = torch.matmul(result, A)
        return result

    def _make_matrix_M_list(self, x):
        zlist = self.model.forward_with_intermediates(x)
        Mlist = []
        for i, (A, z) in enumerate(zip(self.Alist, zlist)):
            z1 = torch.cat([z, torch.ones(1, device=z.device)])
            M = householder(A, z1) if i < self.L-1 else torch.zeros_like(A)
            Mlist.append(M)
        return Mlist
    
    def _make_matrix_F(self, x) -> torch.Tensor:
        Mlist = self._make_matrix_M_list(x)
        F = self.Alist[0] + Mlist[0]
        for A, M in zip(self.Alist[1:], Mlist[1:]):
            F = torch.matmul((A+M), F) # AM_{k+1} @ AM_{k} @ ... @ AM_{1} @ A_{0}
        assert not torch.isnan(F).any()
        assert not torch.isinf(F).any()
        return F
    
    def Fsigma(self, x) -> torch.Tensor:
        F = self._make_matrix_F(x)
        return F - self.F0
    
    def forward(self, x) -> torch.Tensor:
        F = self._make_matrix_F(x)
        return (F @ x).squeeze()
    
class Expantion():
    '''
    this expantion is only for fully-connected network
    if you want to use this for CNN, you should rewrite this class
    '''
    def __init__(self, model: DeeptimedelayNN | DeeptimedelayNN_):
        self.model = model
        self.F0 = Fzero(model)
        self.W_mat = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.W_mat.append(param)
        self.L = len(self.W_mat)
        self.ML = torch.zeros_like(self.W_mat[-1])
        self.device = self.F0.device
        self.dtype = self.F0.dtype
        self.input_size = self.F0.shape[1]
        # self.output_size = self.F0.shape[0]

    def M_matrix(self, x: Tensor) -> list[Tensor]:
        zlist = self.model.forward_with_intermediates(x)
        M_mat = []
        for i, (W, z) in enumerate(zip(self.W_mat, zlist)):
            M = householder(W, z) if i < self.L-1 else self.ML
            M_mat.append(M)
        return M_mat

    def Fsigma(self, x: Tensor) -> Tensor:
        M_mat = self.M_matrix(x)
        F = torch.eye(self.input_size, dtype=self.dtype, device=self.device)
        for W, M in zip(self.W_mat, M_mat):
            F = (W+M) @ F
        
        if torch.isnan(F).any() or torch.isinf(F).any():
            raise ValueError('F contains non-finite values')
        
        return F - self.F0
    
    def Fsigma_(self, x: Tensor) -> tuple[list[Tensor], list[Tensor], Tensor]:
        zlist = self.model.forward_with_intermediates(x)
        vlist = [householder_v(z) for z in zlist[:-1]] 
        # wlist = [-v.T @ W for v, W in zip(vlist, self.W_mat[:-1])]
        M_mat = [householder(W, z) if i < self.L-1 else self.ML for i, (W, z) in enumerate(zip(self.W_mat, zlist))]

        Phi = []
        Psi = []
        C_ll = torch.zeros((self.L-1, self.L-1))
        for i in range(self.L-1):
            if i == 0:
                Phi_l = self.W_mat[self.L-1-i]
                Psi_r = self.W_mat[i]
            else:
                Phi_l =  torch.linalg.multi_dot(self.W_mat[:self.L-2-i:-1])
                Psi_r = torch.linalg.multi_dot(self.W_mat[i::-1])
            Phi.append(Phi_l @ vlist[i])
            Psi.append(vlist[i] @ Psi_r)

        for b in range(1, 1 << (self.L-1)):
            lmax, lmin = find_edges_of_ones(b, self.L-1)
            if lmax == lmin:
                C_ll[lmax, lmin] += 1.
                continue
            elif lmax == lmin + 1:
                C_ll[lmax, lmin] += (vlist[lmax] @ self.W_mat[lmax] @ vlist[lmin]).item()
                continue
            b_binary =  (((1 << np.arange(self.L-1)) & b) > 0).astype(int)

            B = [M if b else W for b, W, M in zip(b_binary, self.W_mat[:-1], M_mat[:-1])]
            C_ll[lmax, lmin] += torch.linalg.multi_dot([vlist[lmax], self.W_mat[lmax], *B[lmax-1:lmin:-1], vlist[lmin]]).item()


        return Phi, Psi, C_ll

    
    def Fsigma_C(self, x: Tensor) -> Tensor:
        warnings.warn('models.Expantion.Fsigma_C is not accurate. Use models.Expantion.Fsigma instead.', UserWarning)
        zlist = self.model.forward_with_intermediates(x)
        M_mat = []
        for i,(W, z) in enumerate(zip(self.W_mat, zlist)):
            if i < self.L-1:
                M = householder_C(W, z)
                M_mat.append(M)
            else:
                M_mat.append(self.ML.detach().clone().cpu().to(torch.complex64)) #empty matrix

        F = torch.eye(self.input_size, dtype=torch.complex64).cpu()
        for W, M in zip(self.W_mat, M_mat):
            F = (W.detach().clone().cpu().to(M.dtype) + M) @ F
        
        if torch.isnan(F).any().item():
            raise ValueError('Fsigma has nan')
        return torch.real(F).to(self.device) - self.F0


def parzen_window(b: float, df: float) -> tuple[np.ndarray, np.ndarray]:
    u = 280 / 151 / b
    n = int((1 / u) / df) + 1
    g = np.linspace(-1 / u, 1 / u, 2 * n + 1)

    th = np.pi * u * g / 2
    Wf = 3 / 4 * u * np.sinc(th / np.pi)**4  # ベクトル化
    Wf[g == 0] = 3 / 4 * u  # ゼロ点の処理

    return g, Wf

def compute_fft(dt: float, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = next(2**i for i in range(20) if 2**i >= len(acc))

    x = np.zeros(N)
    x[:len(acc)] = acc
    D_f = np.fft.fft(x)
    f = np.fft.fftfreq(n=N, d=dt)

    Amp_norm = np.abs(D_f / (N / 2))

    half_N = N // 2
    return f[1:half_N], Amp_norm[1:half_N]

def smooth_spectrum(freq: np.ndarray, spec: np.ndarray, b: float) -> np.ndarray:
    df = freq[1] - freq[0]
    g, Wf = parzen_window(b, df)

    Pow_norm = spec**2
    Pow_smooth = np.convolve(Pow_norm, Wf * df, mode='same')

    Amp_smooth = np.sqrt(Pow_smooth)
    return Amp_smooth



def fourier(x: Tensor, dt: float) -> tuple[Tensor, Tensor]:
    """
    x: 1d tensor
    """
    n = x.shape[0]
    freq = torch.fft.fftfreq(n, dt)
    spectrum = torch.fft.fft(x)
    amplitude = torch.sqrt(spectrum.real**2 + spectrum.imag**2)
    return freq, amplitude

'''
8888888b.         d8888 88888888888        d8888 888       .d88888b.         d8888 8888888b.  8888888888 8888888b.  
888  "Y88b       d88888     888           d88888 888      d88P" "Y88b       d88888 888  "Y88b 888        888   Y88b 
888    888      d88P888     888          d88P888 888      888     888      d88P888 888    888 888        888    888 
888    888     d88P 888     888         d88P 888 888      888     888     d88P 888 888    888 8888888    888   d88P 
888    888    d88P  888     888        d88P  888 888      888     888    d88P  888 888    888 888        8888888P"  
888    888   d88P   888     888       d88P   888 888      888     888   d88P   888 888    888 888        888 T88b   
888  .d88P  d8888888888     888      d8888888888 888      Y88b. .d88P  d8888888888 888  .d88P 888        888  T88b  
8888888P"  d88P     888     888     d88P     888 88888888  "Y88888P"  d88P     888 8888888P"  8888888888 888   T88b 
'''

def load_earthquake_data(file_no: int, trg_or_prd: str = 'trg', dir: str = 'data/') -> tuple[ndarray, ndarray, ndarray, str]:
    """
    load the earthquake data from .cor file
    file_no; 
    Number of file/case
    
    ## Args
    file_no, trg_or_prd, dir

    file_no; Number of file/case  
    trg_or_prd;
    trg: Data for training
    prd: Data for prediction  
    dir; Directory name

    ## Returns
    t, x3d, u3d, fileID
    
    NS; [:, 0]
    EW; [:, 1]
    UD; [:, 2]
    (Gal)

    x3d.shape == (N, 3)
    """
    dt = 0.01
    path = dir + trg_or_prd
    # path = '../data/' + trg_or_prd
    

    # 初期微動観測（トリガーの開始）から10s以前（1000ステップ）が記録に含まれる
    # 計器の常時微動はデータに含まないようにする
    st_ = 1000

    # @Ground surface
    file_name_1 = sorted(glob.glob(path + '/acc_x/*.cor'))
    x3d = np.loadtxt(file_name_1[file_no], skiprows=1, delimiter=',')[st_:].T
    # x_ns = x3d[0]; x_ew = x3d[1]; x_ud = x3d[2]
    
    # @Engineering bedrock
    file_name_2 = sorted(glob.glob(path + '/acc_u/*.cor'))
    u3d = np.loadtxt(file_name_2[file_no], skiprows=1, delimiter=',')[st_:].T
    # u_ns = u3d[0]; u_ew = u3d[1]; u_ud = u3d[2]
    
    fileID = os.path.basename(file_name_1[file_no]).split('.')[0] # O-01234
    
    N = x3d.shape[1]
    t = np.linspace(0, (N-1)*dt, N)
    return t, x3d, u3d, fileID
    # return t, x_ns, x_ew, x_ud, u_ns, u_ew, u_ud, fileID
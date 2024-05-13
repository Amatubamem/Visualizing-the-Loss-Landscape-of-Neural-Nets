from distutils.fancy_getopt import longopt_re
from distutils.log import Log
from re import L
from turtle import color
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
import numpy as np
from typing import Any

import models as md

import time
import tqdm


class Plotter():
    def __init__(self, n: int):
        '''
        Plotter class for plotting waves.
        ### Parameters
        n: int
            Number of waves to plot.
            n must be same as len(waves) and len(labels).
        '''
        self.n = n
        if n > 1:
            self.fig, self.axs = plt.subplots(n, figsize=(16, 3.5*n))
        elif n == 1:
            self.fig, ax = plt.subplots(1, figsize=(16, 3.5))
            self.axs = [ax]
        else:
            raise ValueError('number of waves must be greater than 0.')
        

    def plot_waves(self, 
                   waves: list[np.ndarray], 
                   labels: list[str],
                   ticks: list[np.ndarray] | np.ndarray,
                   dash: bool = False) -> None:
        '''
        Plot waves with labels and ticks in one figure.
        '''
        for obj in [waves, labels]:
            assert self.n == len(obj)
        if isinstance(ticks, list):
            assert self.n == len(ticks)
        else:
            ticks = [ticks for _ in range(len(waves))] 

        for ax, tick, target, label in zip(self.axs, ticks, waves, labels):
            ax.plot(tick, target, lw=1.5, c='k', ls='--' if dash else '-', label=label, alpha=1.0)
            ax.legend(frameon=False)
    
    def plot_overlay(self,
                     waves: list[np.ndarray],
                     labels: list[str],
                     ticks: list[np.ndarray] | np.ndarray,
                     colors: list[str] | None = None) -> None:
        '''
        Plot waves with labels and ticks in one figure.
        '''
        if colors is None:
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors = [color_cycle[i % len(color_cycle)] for i in range(self.n)]
        for obj in [waves, labels, colors]:
            assert self.n == len(obj)
        if isinstance(ticks, list):
            assert self.n == len(ticks)
        else:
            ticks = [ticks for _ in range(len(waves))] 

        for ax, tick, target, color, label in zip(self.axs, ticks, waves, colors, labels):
            ax.plot(tick, target, lw=1.5, c=color, ls='-', label=label, alpha=0.8)
            ax.legend(frameon=False)

    def set_xylavels(self):
        for i, ax in enumerate(self.axs):
            if i == self.n - 1:
                ax.set_xlabel('Tims [s]')
            ax.set_ylabel('Acc. [cm/s$^2$]')

    def set_xylim(self, xlim: tuple[float, float], ylim: tuple[float, float]):
        for ax in self.axs:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            
    def set_xlim(self, *args):
        for ax in self.axs:
            ax.set_xlim(*args)
            
    def set_ylim(self, *args):
        for ax in self.axs:
            ax.set_ylim(*args)

    def set_xylim_closeup(self, t, x):
        left = max(0, int(np.argmax(np.abs(x))) - int(len(x)*1.1/15)) # 最大振幅点の1.1/15前を左端に
        right = min(len(x), left+int(3.6*len(x)/15)) # 左端から3.6/15を右端に 
        gmax = np.max(np.abs(x))*1.25
        self.set_xylim((t[left], t[right]), (-1*gmax, gmax))


class NNPlotter():
    def __init__(self, savedir: str, fileID: str, t: np.ndarray, x: np.ndarray, u: np.ndarray):
        self.savedir = savedir
        self.fileID = fileID
        self.t = t
        self.x = x
        self.u = u
        self.dt = 0.01

    def plot_original(self):
        plotter = Plotter(2)
        plotter.plot_waves([self.x, self.u], ['surface', 'bedrock'], self.t)
        plotter.fig.savefig(f'{self.savedir}/original_{self.fileID}.png')
        plt.close()

    def plot_error(self, losses: list):
        fig, ax = plt.subplots(1, figsize=(16, 3))
        ax.plot(losses, lw=1.5, c='r', ls='-', label='error', alpha=1.0)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Error')
        fig.savefig(f'{self.savedir}/error_{self.fileID}.png')
        plt.close()

    def plot_AF0(self, Admd: np.ndarray, F0: np.ndarray, scale: str = 'log'):
        fig, axs = plt.subplots(1,2, figsize=(16, 7))

        match scale:
            case 'log':
                plt.set_cmap('jet')
                axs[0].imshow(np.abs(Admd), norm=LogNorm())
                axs[1].imshow(np.abs(F0), norm=LogNorm())
            case 'linear':
                plt.set_cmap('bwr')
                axs[0].imshow(Admd)
                axs[1].imshow(F0)
            case _:
                raise ValueError(f'Invalid scale: {scale}')
            
        axs[0].title.set_text('A')
        axs[1].title.set_text('F0')
        fig.savefig(f'{self.savedir}/AF0_{self.fileID}.png')
        plt.close()

    def plot_recon_closeup(self, xrc: np.ndarray, x0rc: np.ndarray, xdmdrc: np.ndarray):
        bwaves = [self.x for _ in range(3)]
        blabels = ['Observation' for _ in range(len(bwaves))]

        twaves = [xrc, x0rc, xdmdrc]
        tlabels = ['Reconstruction', 'F0 Reconstruction', 'DMD Reconstruction']

        plotter = Plotter(3)
        plotter.plot_waves(bwaves, blabels, self.t, dash=True)
        plotter.plot_overlay(twaves, tlabels, self.t, colors=['tab:red', 'tab:cyan', 'tab:cyan'])
        plotter.set_xylim_closeup(self.t, self.x)

        plotter.fig.tight_layout()
        plotter.fig.savefig(f'{self.savedir}/ReconCloseUp_{self.fileID}.png')
        plt.close()

    def plot_recon_all(self, xrc: np.ndarray, x0rc: np.ndarray, xdmdrc: np.ndarray):
        waves = [self.x, xrc, x0rc, xdmdrc, self.u]
        labels = ['surface', 'reconstraction', 'F0 reconstruction', 'DMD reconstruction', 'bedrock']

        plotter = Plotter(5)
        plotter.plot_waves(waves, labels, self.t)
        plotter.fig.savefig(f'{self.savedir}/ReconWide_{self.fileID}.png')
        plt.close()
    
    def plot_freq(self, xrc: np.ndarray, x0rc: np.ndarray, xdmdrc: np.ndarray):
        plt.figure(figsize=(20,4))

        waves = [self.x, xrc, x0rc, xdmdrc]
        labels = ['original', 'reconstraction', 'F0 reconstruction', 'DMD reconstruction']
        cms = ['k-', 'r-', 'b-', 'y--']
        alphas = [1.0, 0.6, 0.4, 0.3]

        for wave, label, cm, alpha in zip(waves, labels, cms, alphas):
            f, amp = md.compute_fft(self.dt, wave)
            amp_pzn = md.smooth_spectrum(f, amp, 0.1)
            plt.plot(1/f, amp_pzn, cm, lw=.75, alpha=alpha, label=label)
        
        plt.xlim(.02, 20)
        plt.xscale('log'); plt.yscale('log')
        plt.xlabel('Period (s)'); plt.ylabel('Amplitude (cm/s$^2$)')
        plt.legend()
        plt.savefig(f'{self.savedir}/freq_{self.fileID}.png')
        plt.close()

    def plot_rank(self, F_ranks: np.ndarray, Fsig_ranks: np.ndarray, Fsig_tops: np.ndarray, skip: int = 20):
        Fmin = np.min(F_ranks)-5; Fmax = np.max(F_ranks)+2

        fig, ax1 = plt.subplots(figsize=(12,4))
        xax = np.linspace(0, skip*(len(Fsig_ranks)-1), len(Fsig_ranks))
        color = 'tab:red'
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('ranks', color=color)
        ax1.plot(xax, Fsig_ranks + Fmin*np.ones_like(Fsig_ranks), '.', color='b', markersize=6, label='rank(Fsig)')
        ax1.plot(xax, F_ranks   , '. ', color='r', markersize=6, label='rank(F)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yticks(range(Fmin, Fmax))

        # 2つ目のy軸のプロット
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Singular Values (%)', color=color)
        ax2.plot(xax, Fsig_tops[:, 0], 'g-', markersize=4, label='1st SV (Fsig)', alpha=0.5)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        fig.savefig(f'{self.savedir}/ranks_{self.fileID}.png')
        plt.close()

    def plot_unstability(self, unstability: np.ndarray, skip: int = 20):
        fig, ax = plt.subplots(figsize=(12,4))
        xax = np.linspace(0, skip*(len(unstability)-1), len(unstability))
        ax.plot(xax, unstability, 'b-', markersize=4, label='unstability', alpha=1.)
        ax.set_xlabel('step')
        ax.set_ylabel('||Fsig||$_2$/||F||$_2$')
        fig.tight_layout()
        fig.savefig(f'{self.savedir}/stability_{self.fileID}.png')
        plt.close()

    def plot_eig(self, eig_list: np.ndarray):
        raise NotImplementedError
        r_evlx_st = self.t[0] + (np.arange(0, cal_size, Dn))*self.dt
        r_evlx_ed = r_evlx_st + (b+d+1)*self.dt
        r_evlx = (r_evlx_st + r_evlx_ed)/2
        r_evly = np.count_nonzero(~np.isnan(lm_all[:, :]), axis=1)
        pick_list = np.arange(0, int(cal_size/Dn), int(cal_size/Dn/6))


        def circle(r):
            t = np.linspace(0, 2*np.pi, 360)
            x = r*np.cos(t)
            y = r*np.sin(t)
            return x, y
        uc_x, uc_y = circle(1)
        fig = plt.figure(figsize=[12, 6], dpi=120)

        ax1 = fig.add_subplot(1, 1, 1); #ax1.grid()
        ax2 = ax1.twinx(); ax2.grid()

        ax1.plot(self.t, self.x, lw=1/2, alpha=0.7, label='Ground surface')
        ax1.plot(self.t, self.u, lw=1/2, alpha=0.7, label='Engineering bedrock')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acc. (cm/s$^2$)')
        #ax1.set_title('index: '+str(idx_sample))
        ax1.set_xlim(self.t[0]-self.dt, self.t[-1]+self.dt)
        ax1.set_ylim(-10, 10)

        ax2.scatter(r_evlx_st, r_evly, marker='o', s=5, alpha=0.3, color='tab:grey')
        ax2.scatter(r_evlx_ed, r_evly, marker='s', s=5, alpha=0.3, color='tab:grey')
        for j in range(int(cal_size/Dn)):
            ax2.plot([r_evlx_st[j],r_evlx_ed[j]], [r_evly[j],r_evly[j]], c='tab:grey', lw=1, ls='--', alpha=0.3)
        ax2.plot(r_evlx, r_evly, c='black', zorder=3)
        #ax2.set_ylim(0, 40)
        ax2.set_ylabel('Rank(${\Phi}^{(j)}$)')


        #'''
        for i, j in enumerate(pick_list):
            ax2.scatter((r_evlx_st[j]+r_evlx_ed[j])/2, r_evly[j], s=100, zorder=4, marker='*', ec='white', lw=1/2, color=cmap(i))
            inset_axes = plt.axes([(i+1)/len(pick_list)*0.85-0.01, 0.75, 0.05*1.25, 0.1*1.25]) 
            inset_axes.scatter([0.925], [0.925], s=100, zorder=3, marker='*', ec='white', lw=1/2, color=cmap(i))
            inset_axes.scatter(lm_all[j, :].real, lm_all[j, :].imag, color='red', marker='o', ec='tab:grey', s=5, lw=1/4, alpha=0.8, zorder=3)
            inset_axes.plot(uc_x, uc_y, ls='--', c='tab:grey', lw=1)
            inset_axes.axvline(0, c='tab:grey', lw=1, ls='--')
            inset_axes.axhline(0, c='tab:grey', lw=1, ls='--')
            inset_axes.set_xlim(-1.05, 1.05)
            inset_axes.set_ylim(-1.05, 1.05)
            inset_axes.tick_params(labelsize=8)
        #'''

        plt.tight_layout()
        plt.show()
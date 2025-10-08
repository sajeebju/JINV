
from __future__ import annotations
import torch
import torch.nn as nn
from .utils import soft_argmax

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=7, p=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_out, c_out, k, padding=p),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.down1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool1d(2)
        self.down2 = ConvBlock(base, base*2)
        self.pool2 = nn.MaxPool1d(2)
        self.bott  = ConvBlock(base*2, base*4)
        self.up2   = nn.ConvTranspose1d(base*4, base*2, 2, stride=2)
        self.dec2  = ConvBlock(base*4, base*2)
        self.up1   = nn.ConvTranspose1d(base*2, base, 2, stride=2)
        self.dec1  = ConvBlock(base*2, base)
        self.out   = nn.Conv1d(base, 1, 1)
    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        b  = self.bott(p2)
        u2 = self.up2(b)
        if u2.size(-1) != d2.size(-1):
            L = min(u2.size(-1), d2.size(-1)); u2 = u2[..., :L]; d2 = d2[..., :L]
        d2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(d2)
        if u1.size(-1) != d1.size(-1):
            L = min(u1.size(-1), d1.size(-1)); u1 = u1[..., :L]; d1 = d1[..., :L]
        d1 = self.dec1(torch.cat([u1, d1], dim=1))
        return self.out(d1).squeeze(1)

class PhysicsLoss(nn.Module):
    def __init__(self, w_pick=0.1, pick_win_s=2.0, dt=0.1):
        super().__init__()
        self.w_pick = w_pick
        self.pick_win = int(round(pick_win_s / dt))
        self.l1 = nn.L1Loss()
    def forward(self, pred, target):
        loss = self.l1(pred, target)
        T = target.shape[-1]
        abs_t = torch.abs(target); center = torch.argmax(abs_t, dim=-1)
        picks_pred, picks_true = [], []
        for b in range(target.shape[0]):
            c = int(center[b].item())
            s = max(0, c - self.pick_win); e = min(T, c + self.pick_win)
            pp = soft_argmax(torch.abs(pred[b, s:e]), beta=25.0) + s
            pt = soft_argmax(torch.abs(target[b, s:e]), beta=25.0) + s
            picks_pred.append(pp); picks_true.append(pt)
        picks_pred = torch.stack(picks_pred); picks_true = torch.stack(picks_true)
        pick_loss = self.l1(picks_pred, picks_true) / T
        return loss + self.w_pick * pick_loss

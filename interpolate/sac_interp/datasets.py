from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from obspy import read
from .metadata import TraceRecord
from .config import WindowConfig
from .utils import robust_scale, slice_window, bp_zero, trace_time_axis, interp_to_axis

class WindowDataset(Dataset):
    """Triplets over windowed signals aligned around S (per component)."""
    def __init__(self, recs: List[TraceRecord], cfg: WindowConfig, max_gap: int = 6):
        self.recs = recs; self.cfg = cfg; self.max_gap = max_gap
        self.cache: Dict[str, np.ndarray] = {}
        idx = list(range(len(recs))); idx.sort(key=lambda i: recs[i].offset_km)
        self.triplets: List[Tuple[int,int,int]] = []
        N = len(idx)
        for a in range(N):
            for c in range(a+2, min(N, a+2+max_gap)):
                for b in range(a+1, c):
                    i,j,k = idx[a], idx[b], idx[c]
                    self.triplets.append((i,j,k))

    def __len__(self): return len(self.triplets)

    def _get(self, i: int) -> np.ndarray:
        key = self.recs[i].path
        if key in self.cache: return self.cache[key]
        st = read(key); tr = st[0]
        fs = 1.0 / float(tr.stats.delta)
        x = tr.data.astype(np.float32)
        x = x - np.mean(x)
        # simple cosine taper
        n_taper = max(8, int(self.cfg.taper_frac * x.size))
        if n_taper > 0 and 2 * n_taper < x.size:
            win = np.hanning(2 * n_taper)
            taper = np.concatenate([win[:n_taper], np.ones(x.size - 2 * n_taper, dtype=np.float32), win[n_taper:]])
            x = x * taper
        x = bp_zero(x, fs, self.cfg.bp_low, self.cfg.bp_high, self.cfg.corners)
        x = robust_scale(x)
        sac = getattr(tr.stats, 'sac', None)
        B = float(getattr(sac, 'b', 0.0) if sac is not None else 0.0)
        t = B + np.arange(x.size, dtype=np.float32) / fs
        S_time = self.recs[i].S_time
        if S_time is None:
            peak_idx = int(np.argmax(np.abs(x)))
            S_time = float(t[peak_idx])
        y, _ = slice_window(x, t, float(S_time), self.cfg.tmin, self.cfg.tmax)
        L = (y.shape[0] // 4) * 4
        if L >= 8 and L < y.shape[0]:
            y = y[:L]
        self.cache[key] = y.astype(np.float32)
        return self.cache[key]

    def __getitem__(self, n: int):
        i,j,k = self.triplets[n]
        wi, wj, wk = self._get(i), self._get(j), self._get(k)
        L = min(wi.size, wj.size, wk.size); wi, wj, wk = wi[:L], wj[:L], wk[:L]
        ri, rj, rk = self.recs[i], self.recs[j], self.recs[k]
        denom = (rk.offset_km - ri.offset_km)
        alpha = (rj.offset_km - ri.offset_km) / denom if denom != 0 else 0.5
        alpha_vec = np.full((L,), float(alpha), dtype=np.float32)
        x = np.stack([wi, wk, alpha_vec], axis=0)
        y = wj.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), (i,j,k)

class FullSeriesDataset(Dataset):
    """Triplets over full-length signals. Align left/right onto TARGET time axis."""
    def __init__(self, recs: List[TraceRecord], max_gap: int = 4, robust: bool = True):
        self.recs = recs; self.max_gap = max_gap; self.robust = robust
        self.traces = {i: read(r.path)[0] for i,r in enumerate(recs)}
        idxs = list(range(len(recs))); idxs.sort(key=lambda i: recs[i].offset_km)
        self.triplets: List[Tuple[int,int,int]] = []
        N = len(idxs)
        for a in range(N):
            for c in range(a+2, min(N, a+2+max_gap)):
                for b in range(a+1, c):
                    i,j,k = idxs[a], idxs[b], idxs[c]
                    self.triplets.append((i,j,k))

    def __len__(self): return len(self.triplets)

    def __getitem__(self, n: int):
        i,j,k = self.triplets[n]
        tri, trj, trk = self.traces[i], self.traces[j], self.traces[k]
        t_axis = trace_time_axis(trj)
        wi = interp_to_axis(t_axis, tri)
        wk = interp_to_axis(t_axis, trk)
        wj = trj.data.astype(np.float32)
        if self.robust:
            wi = robust_scale(wi); wj = robust_scale(wj); wk = robust_scale(wk)
        L = min(wi.size, wj.size, wk.size)
        L4 = (L // 4) * 4
        wi, wj, wk = wi[:L4], wj[:L4], wk[:L4]
        ri, rj, rk = self.recs[i], self.recs[j], self.recs[k]
        denom = (rk.offset_km - ri.offset_km)
        alpha = (rj.offset_km - ri.offset_km) / denom if denom != 0 else 0.5
        alpha_vec = np.full((L4,), float(alpha), dtype=np.float32)
        x = np.stack([wi, wk, alpha_vec], axis=0)
        y = wj.astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), (i,j,k)


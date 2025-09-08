from __future__ import annotations
import math, os, re
from typing import Optional, Tuple
import numpy as np
from obspy import read
from obspy.core.trace import Trace
from obspy.signal.filter import bandpass
import torch
from .config import SAC_UNDEF

STATION_RE = re.compile(r"([A-Za-z]+)(\d+)")
COMP_RE = re.compile(r"\.([rztRZT])$")

def natural_station_key(path: str) -> Tuple[int, str]:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    m = STATION_RE.search(name)
    if m:
        prefix, num = m.group(1), int(m.group(2))
        return (num, prefix)
    return (10**9, name)

def detect_component(path: str, tr: Optional[Trace] = None) -> str:
    m = COMP_RE.search(path)
    if m:
        c = m.group(1).lower()
        if c in ("r","z"): return c
    if tr is not None:
        k = getattr(getattr(tr.stats, "sac", {}), "kcmpnm", None)
        if isinstance(k, str) and k.upper() in ("R","Z"):
            return k.lower()
    return "r"

def bp_zero(x: np.ndarray, fs: float, fmin: float, fmax: float, corners: int = 4) -> np.ndarray:
    if x.size < 32 or fmin <= 0 or fmax >= fs / 2: return x.copy()
    return bandpass(x, fmin, fmax, df=fs, corners=corners, zerophase=True)

def robust_scale(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = np.median(np.abs(x)) * 1.4826
    m = m if m > eps else (np.max(np.abs(x)) + eps)
    return x / (m + eps)

def soft_argmax(x: torch.Tensor, beta: float = 25.0) -> torch.Tensor:
    T = x.shape[-1]
    idx = torch.arange(T, device=x.device, dtype=x.dtype)
    w = torch.softmax(beta * x, dim=-1)
    return (w * idx).sum(dim=-1)

def slice_window(tr: np.ndarray, t: np.ndarray, t_center: float, tmin: float, tmax: float):
    start = t_center + tmin
    end   = t_center + tmax
    dt    = t[1] - t[0]
    n     = int(round((end - start) / dt))
    if n <= 0:
        return np.zeros(1, dtype=np.float32), np.array([0.0], dtype=np.float32)
    ti = np.arange(n, dtype=np.float32) * dt + start
    y  = np.interp(ti, t, tr, left=0.0, right=0.0).astype(np.float32)
    return y, ti

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p = math.radians
    dlat = p(lat2 - lat1); dlon = p(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p(lat1))*math.cos(p(lat2))*math.sin(dlon/2)**2
    return 2 * r * math.asin(math.sqrt(a))

def trace_time_axis(tr: Trace) -> np.ndarray:
    sr = float(tr.stats.sampling_rate)
    n  = int(tr.stats.npts)
    sac = getattr(tr.stats, "sac", None)
    b = float(getattr(sac, "b", 0.0) if sac is not None else 0.0)
    return b + np.arange(n, dtype=np.float32) / sr

def interp_to_axis(ref_axis: np.ndarray, other: Trace) -> np.ndarray:
    so = float(other.stats.sampling_rate)
    n  = int(other.stats.npts)
    sac_o = getattr(other.stats, "sac", None)
    b_o = float(getattr(sac_o, "b", 0.0) if sac_o is not None else 0.0)
    t_oth = b_o + np.arange(n, dtype=np.float32) / so
    return np.interp(ref_axis, t_oth, other.data.astype(np.float32), left=0.0, right=0.0).astype(np.float32)

def lerp(a: Optional[float], b: Optional[float], alpha: float) -> Optional[float]:
    if a is None and b is None: return None
    if a is None: return float(b)
    if b is None: return float(a)
    return float((1 - alpha) * a + alpha * b)


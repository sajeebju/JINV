from __future__ import annotations
import math, os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from obspy import read
from obspy.core.trace import Trace
from .config import WindowConfig
from .metadata import TraceRecord
from .datasets import WindowDataset, FullSeriesDataset
from .model import UNet1D, PhysicsLoss
from .utils import trace_time_axis, interp_to_axis, robust_scale, robust_amp, lerp

# ---------- training ----------
def train_model(recs: List[TraceRecord], train_mode: str, window_cfg: WindowConfig,
                epochs: int, batch_size: int, lr: float, device: str, model_out: str):
    if train_mode == 'window':
        dataset = WindowDataset(recs, window_cfg, max_gap=6)
    elif train_mode == 'full':
        dataset = FullSeriesDataset(recs, max_gap=4, robust=True)
    else:
        raise SystemExit("train_mode must be 'window' or 'full'.")
    if len(dataset) < 8:
        raise SystemExit("Not enough triplets to train. Check data/xmap/filtering.")
    from torch.utils.data import DataLoader, random_split
    N = len(dataset); n_val = int(round(0.2 * N)); n_train = max(1, N - n_val)
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet1D(in_ch=3, base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = PhysicsLoss(w_pick=0.1, pick_win_s=2.0, dt=window_cfg.dt)
    best_val = float('inf'); patience, bad = 8, 0
    for ep in range(1, epochs + 1):
        model.train(); tr_loss = 0.0
        for x,y,_ in tr_loader:
            x = x.to(device, dtype=torch.float32); y = y.to(device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            yhat = model(x); loss = loss_fn(yhat, y)
            loss.backward(); opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= max(1, len(tr_loader.dataset))
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for x,y,_ in va_loader:
                x = x.to(device, dtype=torch.float32); y = y.to(device, dtype=torch.float32)
                yhat = model(x); loss = loss_fn(yhat, y)
                va_loss += loss.item() * x.size(0)
        va_loss /= max(1, len(va_loader.dataset))
        print(f"Epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        if va_loss < best_val - 1e-4:
            best_val = va_loss; bad = 0
            torch.save({'model': model.state_dict(),
                        'train_mode': train_mode,
                        'window_cfg': window_cfg.__dict__}, model_out)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping."); break
    print(f"Best val loss: {best_val:.4f}. Saved to {model_out}")

# ---------- write SAC ----------
def write_windowed_like(ref_path: str, out_path: str, y: np.ndarray, cfg: WindowConfig,
                        S_time: Optional[float], rayp: Optional[float], comp: str):
    st = read(ref_path); tr = st[0]
    fs = 1.0 / tr.stats.delta
    new = Trace(data=y.astype(np.float32))
    new.stats = tr.stats.copy()
    new.stats.npts = y.size
    new.stats.delta = 1.0 / fs
    if getattr(new.stats, 'sac', None) is None:
        new.stats.sac = {}
    sac = new.stats.sac
    b = float((S_time or 0.0) + cfg.tmin)
    sac['b'] = b
    sac['e'] = b + (y.size - 1) * new.stats.delta
    if S_time is not None: sac['t8'] = float(S_time)
    sac['t2'] = -12345.0
    if rayp is not None: sac['user0'] = float(rayp)
    sac['kcmpnm'] = 'R' if comp.lower() == 'r' else 'Z'
    new.write(out_path, format='SAC')

def write_full_like(template_tr: Trace, out_path: str, data: np.ndarray,
                    S_time: Optional[float], rayp: Optional[float], comp: str):
    new = Trace(data=data.astype(np.float32))
    new.stats = template_tr.stats.copy()
    new.stats.npts = int(template_tr.stats.npts)
    new.stats.delta = float(template_tr.stats.delta)
    if getattr(new.stats, "sac", None) is None: new.stats.sac = {}
    sac = new.stats.sac
    if S_time is not None: sac["t8"] = float(S_time)
    sac["t2"] = -12345.0
    if rayp is not None: sac["user0"] = float(rayp)
    sac["kcmpnm"] = 'R' if comp.lower() == 'r' else 'Z'
    b = float(getattr(sac, "b", 0.0))
    sac["e"] = b + (new.stats.npts - 1) * new.stats.delta
    new.write(out_path, format="SAC")

# ---------- planning over X ----------
def compute_global_plan_x(recs: List[TraceRecord], target_dx: float,
                          min_between: int, max_between: int) -> List[Tuple[str,str,int]]:
    # choose one per station (prefer .z)
    choice: Dict[str, TraceRecord] = {}
    for r in recs:
        cur = choice.get(r.station)
        if (cur is None) or (cur.comp != 'z' and r.comp == 'z'):
            choice[r.station] = r
    ordered = list(choice.values()); ordered.sort(key=lambda rr: rr.x_dist)
    out: List[Tuple[str,str,int]] = []
    for a,b in zip(ordered[:-1], ordered[1:]):
        dist = abs(b.x_dist - a.x_dist)
        n_between = int(max(0, math.floor(dist / max(1e-6, target_dx))))
        n_between = max(min_between, min(max_between, n_between))
        out.append((a.station, b.station, n_between))
    return out

# ---------- inference ----------
def infer_virtual(recs: List[TraceRecord], out_dir: str, model_path: str,
                  target_dx: float, min_between: int, max_between: int,
                  infer_mode: str, window_cfg: WindowConfig, device: str):
    os.makedirs(out_dir, exist_ok=True)
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)
    train_mode = ckpt.get('train_mode', 'window')
    if infer_mode == 'auto':
        infer_mode = train_mode
    elif infer_mode not in ('window','full'):
        raise SystemExit("--infer_mode must be 'auto', 'window', or 'full'.")

    model = UNet1D(in_ch=3, base=32).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    plan_global = compute_global_plan_x(recs, target_dx, min_between, max_between)

    # group by component, run independently (so r and z are separate runs)
    comps = sorted(set(r.comp for r in recs if r.comp in ('r','z')))
    for comp in comps:
        rc = [r for r in recs if r.comp == comp]
        if len(rc) < 2: continue
        rec_by_sta = {r.station: r for r in rc}

        if infer_mode == 'window':
            # ---- WINDOWED INFERENCE with proper de-normalization ----
            # cache normalized windows and their robust scales
            cache_n: Dict[str, np.ndarray] = {}
            cache_s: Dict[str, float] = {}

            for r in rc:
                st = read(r.path); tr = st[0]
                fs = 1.0 / float(tr.stats.delta)
                x = tr.data.astype(np.float32)
                x = x - np.mean(x)  # DC remove
                sac = getattr(tr.stats, 'sac', None)
                B = float(getattr(sac, 'b', 0.0) if sac is not None else 0.0)
                t = B + np.arange(x.size, dtype=np.float32) / fs
                S_time = r.S_time
                if S_time is None:
                    peak_idx = int(np.argmax(np.abs(x)))
                    S_time = float(t[peak_idx])

                # slice window (mirrors training dataset)
                # NOTE: slice_window is assumed to be available in your codebase
                y_raw, _ = slice_window(tr=x, t=t, t_center=float(S_time),
                                        tmin=window_cfg.tmin, tmax=window_cfg.tmax)

                # enforce multiple of 4 for UNet stride compatibility
                L = (y_raw.shape[0] // 4) * 4
                if L >= 8 and L < y_raw.shape[0]:
                    y_raw = y_raw[:L]

                # robust scale & normalize (same logic as training)
                s = robust_amp(y_raw)
                y_n = (y_raw / s).astype(np.float32)

                cache_n[r.path] = y_n
                cache_s[r.path] = float(s)

            # align by shortest window
            L = min(w.size for w in cache_n.values())

            for left_sta, right_sta, n_between in plan_global:
                if n_between <= 0: continue
                if left_sta not in rec_by_sta or right_sta not in rec_by_sta:
                    continue
                left = rec_by_sta[left_sta]; right = rec_by_sta[right_sta]
                wi_n = cache_n[left.path][:L]; wk_n = cache_n[right.path][:L]
                sL = cache_s[left.path];       sR  = cache_s[right.path]

                for m in range(1, n_between+1):
                    alpha = m / (n_between + 1)
                    alpha_vec = np.full((L,), alpha, dtype=np.float32)
                    x = np.stack([wi_n, wk_n, alpha_vec], axis=0)[None, ...]
                    with torch.no_grad():
                        y_n = model(torch.from_numpy(x).to(device)).cpu().numpy()[0]

                    # de-normalize prediction using interpolated robust scale
                    sV = (1.0 - alpha) * sL + alpha * sR
                    y = (y_n * sV).astype(np.float32)

                    S_v = lerp(left.S_time, right.S_time, alpha)
                    rayp_v = lerp(left.rayp, right.rayp, alpha)
                    out_name = f"{left.station}_V{m:02d}_to_{right.station}.{comp}"
                    out_path = os.path.join(out_dir, out_name)
                    write_windowed_like(left.path, out_path, y, window_cfg, S_v, rayp_v, comp)
                    print(f"Wrote {out_path} (windowed, .{comp})")

        else:  # ---- FULL-SERIES INFERENCE with de-normalization ----
            for left_sta, right_sta, n_between in plan_global:
                if n_between <= 0: continue
                if left_sta not in rec_by_sta or right_sta not in rec_by_sta:
                    continue
                left = rec_by_sta[left_sta]; right = rec_by_sta[right_sta]

                trL = read(left.path)[0]; trR = read(right.path)[0]
                tL  = trace_time_axis(trL)
                wL  = trL.data.astype(np.float32)
                wR_on_L = interp_to_axis(tL, trR)

                # inputs: robust-normalized (as in training with robust=True)
                wL_n = robust_scale(wL)
                wR_n = robust_scale(wR_on_L)

                # length alignment / stride compatibility
                Lsig = min(wL_n.size, wR_n.size)
                L4   = (Lsig // 4) * 4
                wL_n, wR_n = wL_n[:L4], wR_n[:L4]

                # robust amplitudes from UNNORMALIZED signals for de-normalization
                mL = robust_amp(wL[:L4] if wL.size >= L4 else wL)
                mR = robust_amp(wR_on_L[:L4] if wR_on_L.size >= L4 else wR_on_L)

                for m in range(1, n_between+1):
                    alpha = m / (n_between + 1)
                    alpha_vec = np.full((L4,), alpha, dtype=np.float32)
                    x = np.stack([wL_n, wR_n, alpha_vec], axis=0)[None, ...]
                    with torch.no_grad():
                        y_n = model(torch.from_numpy(x).to(device)).cpu().numpy()[0]

                    # de-normalize prediction
                    mV = (1.0 - alpha) * mL + alpha * mR
                    y  = (y_n * mV).astype(np.float32)

                    # pad back to full length if needed
                    if L4 < trL.stats.npts:
                        pad = trL.stats.npts - L4
                        y = np.pad(y, (0, pad), mode='constant', constant_values=0.0)

                    S_v = lerp(left.S_time, right.S_time, alpha)
                    rayp_v = lerp(left.rayp, right.rayp, alpha)
                    out_name = f"{left.station}_V{m:02d}_to_{right.station}.{comp}"
                    out_path = os.path.join(out_dir, out_name)
                    write_full_like(trL, out_path, y, S_v, rayp_v, comp)
                    print(f"Wrote {out_path} (full-series, .{comp})")


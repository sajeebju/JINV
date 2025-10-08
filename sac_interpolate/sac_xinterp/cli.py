# 1D-X port of your legacy CLI: same commands, plus windowed targeted inference
# and robust S-time handling for targeted-X (no snapping; strict bracketing + alpha).
from pathlib import Path
import argparse
import glob
import os
import sys

import numpy as np
import torch
from obspy import read  # needed for S-time resolution and local reads

from .metadata import read_sac_metadata_x, parse_xmap
from .engine import train_model, infer_virtual, write_full_like, write_windowed_like
from .config import WindowConfig, PhysicsKeys
from .model import UNet1D
from .utils import (
    trace_time_axis,
    interp_to_axis,
    robust_scale,
    lerp,
    order_by_x,
    nearest_segment_and_alpha_x,  # still available, but we use our own bracketing below
    robust_amp,
    slice_window,
)

def _gather_paths(data_dir, pattern):
    paths = glob.glob(os.path.join(data_dir, pattern))
    paths.sort()
    return paths

def _filter_by_comp(recs, comp):
    if comp in ("r", "z"):
        return [r for r in recs if r.comp == comp]
    return recs

def _save_meta_csv(recs, out_path):
    hdr = "path,station,comp,delta,S_time,rayp,X\n"
    with open(out_path, "w") as f:
        f.write(hdr)
        for r in recs:
            s_time = "" if r.S_time is None else ("%0.6f" % float(r.S_time))
            rayp = "" if r.rayp is None else ("%0.6f" % float(r.rayp))
            line = "%s,%s,%s,%0.6f,%s,%s,%0.3f\n" % (
                r.path, r.station, r.comp, float(r.delta),
                s_time, rayp, float(r.x_dist)
            )
            f.write(line)
    print("Wrote metadata CSV to %s" % out_path)

def _load_targets_x_file(path):
    # Accept lines:  name X   OR   X
    out = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if (not s) or s.startswith("#"):
                continue
            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]
            if len(parts) >= 2:
                try:
                    nm = parts[0]
                    xv = float(parts[1])
                    out.append((nm, xv))
                except Exception:
                    continue
            elif len(parts) == 1:
                try:
                    xv = float(parts[0])
                    out.append(xv)
                except Exception:
                    continue
    return out


def _maxamp_t_rel(tr):
    """Return time (in seconds) RELATIVE TO B of max |amp|: idx*dt (not b+idx*dt)."""
    dt = float(tr.stats.delta)
    x  = tr.data.astype(np.float32, copy=False)
    if x.size == 0:
        return 0.0
    peak_idx = int(np.argmax(np.abs(x)))
    return float(peak_idx * dt)
# ---------- Robust S-time tools ----------
def _maxamp_time_from_trace(tr):
    sac = getattr(tr.stats, 'sac', None)
    dt = float(tr.stats.delta)
    b  = float(getattr(tr.stats.sac, 'b', 0.0) if sac is not None else 0.0)
    x  = tr.data.astype(np.float32, copy=False)
    if x.size == 0:
        return b
    peak_idx = int(np.argmax(np.abs(x)))
    return float(b + peak_idx * dt)

def _t8_or_none(tr):
    sac = getattr(tr.stats, 'sac', None)
    if sac is None:
        return None
    try:
        t8 = getattr(sac, "t8")
        if t8 is None:
            return None
        t8 = float(t8)
        if t8 == -12345.0:
            return None
        return t8
    except Exception:
        return None

def _resolve_S_time(rec, tr=None):
    """
    Resolve S_time (RELATIVE TO B) with a strict priority:
      1) rec.S_time if provided by metadata (already relative to B)
      2) SAC t8 pick (if set and not -12345) (relative to B)
      3) Fallback: RELATIVE time of max |amp| (idx*dt)
    """
    if getattr(rec, "S_time", None) is not None:
        return float(rec.S_time)
    if tr is None:
        tr = read(rec.path)[0]
    t8 = _t8_or_none(tr)
    if t8 is not None:
        return float(t8)
    return _maxamp_t_rel(tr)

def _resolve_S_pair(left, right, trL, trR, tol=1e-6):
    """
    Resolve (S_left, S_right) both RELATIVE-TO-B (so they can be written directly to t8):
      1) Prefer rec.S_time if both present and differ > tol
      2) Else prefer SAC t8 if both present and differ > tol
      3) Else fall back to RELATIVE max-abs times for BOTH
    """
    SL_meta = getattr(left,  "S_time", None)
    SR_meta = getattr(right, "S_time", None)
    if SL_meta is not None and SR_meta is not None:
        SL_meta = float(SL_meta); SR_meta = float(SR_meta)
        if abs(SL_meta - SR_meta) > tol:
            return SL_meta, SR_meta

    SL_t8 = _t8_or_none(trL)
    SR_t8 = _t8_or_none(trR)
    if SL_t8 is not None and SR_t8 is not None and abs(SL_t8 - SR_t8) > tol:
        return SL_t8, SR_t8

    # fallback: RELATIVE max-abs time for BOTH
    SL_ma = _maxamp_t_rel(trL)
    SR_ma = _maxamp_t_rel(trR)
    return SL_ma, SR_ma

# ---------- Bracketing without snapping ----------
def _bracket_and_alpha_by_x(ordered_recs, tx):
    """Return (left_rec, right_rec, alpha) for target X value tx.
    ordered_recs must be sorted by x_dist ascending. No snapping; strict interpolation.
    """
    xs = [float(r.x_dist) for r in ordered_recs]
    n = len(xs)
    if n < 2:
        return None, None, None
    for i in range(n - 1):
        xl, xr = xs[i], xs[i+1]
        # works for ascending or descending segments
        if (xl <= tx <= xr) or (xr <= tx <= xl):
            denom = (xr - xl)
            if abs(denom) < 1e-12:
                return ordered_recs[i], ordered_recs[i+1], 0.5
            alpha = (tx - xl) / denom
            # clamp numerically
            if alpha < 0.0: alpha = 0.0
            if alpha > 1.0: alpha = 1.0
            return ordered_recs[i], ordered_recs[i+1], float(alpha)
    # out of range → choose nearest edge segment
    if tx < xs[0]:
        return ordered_recs[0], ordered_recs[1], 0.0
    if tx > xs[-1]:
        return ordered_recs[-2], ordered_recs[-1], 1.0
    return None, None, None

# ---------- FULL: targeted inference ----------
def _infer_at_x(recs, targets, out_dir, model_path, device="cpu",
                comp=None, name_prefix="V"):
    """Targeted inference at specific X positions (FULL-TRACE workflow)."""

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)
    model = UNet1D(in_ch=3, base=32).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # which components?
    if comp in ("r", "z"):
        comps = [comp]
    else:
        s = set([r.comp for r in recs if r.comp in ("r", "z")])
        comps = sorted(list(s))

    norm_targets = []
    for t in targets:
        if isinstance(t, tuple) and len(t) == 2:
            nm, xv = t[0], float(t[1])
        else:
            nm, xv = None, float(t)
        norm_targets.append((nm, xv))

    for c in comps:
        rc = _filter_by_comp(recs, c)
        if len(rc) < 2:
            continue
        ordered = order_by_x(rc)
        tr_cache = {}

        def _get_trace(path):
            if path in tr_cache:
                return tr_cache[path]
            tr_cache[path] = read(path)[0]
            return tr_cache[path]

        for t_idx, (tname, tx) in enumerate(norm_targets):
            left, right, alpha = _bracket_and_alpha_by_x(ordered, float(tx))
            if left is None or right is None:
                continue

            if tname is None:
                tname = "%s_%02d" % (name_prefix, t_idx + 1)

            trL = _get_trace(left.path)
            trR = _get_trace(right.path)

            # Align right trace onto left time axis
            tL = trace_time_axis(trL)
            wL = trL.data.astype(np.float32)
            wR_on_L = interp_to_axis(tL, trR)

            # Robust-normalize inputs for network
            wL_n = robust_scale(wL)
            wR_n = robust_scale(wR_on_L)

            # UNet stride alignment
            Lsig = min(wL_n.size, wR_n.size)
            L4   = (Lsig // 4) * 4
            wL_n = wL_n[:L4]
            wR_n = wR_n[:L4]

            # robust amplitudes from UN-NORMALIZED signals for de-normalization
            wL_raw      = wL[:L4] if wL.size >= L4 else wL
            wR_on_L_raw = wR_on_L[:L4] if wR_on_L.size >= L4 else wR_on_L
            mL = robust_amp(wL_raw)
            mR = robust_amp(wR_on_L_raw)

            alpha_vec = np.full((L4,), float(alpha), dtype=np.float32)
            x = np.stack([wL_n, wR_n, alpha_vec], axis=0)[None, ...]

            with torch.no_grad():
                y_n = model(torch.from_numpy(x).to(device)).cpu().numpy()[0]

            # de-normalize network output with blended robust amplitude
            mV = (1.0 - float(alpha)) * mL + float(alpha) * mR
            y  = (y_n * mV).astype(np.float32)

            # Pad back to original length if needed
            if L4 < int(trL.stats.npts):
                pad = int(trL.stats.npts) - L4
                y = np.pad(y, (0, pad), mode='constant', constant_values=0.0)

            # Header interpolation using robust S-time resolution (no snap)
            S_left, S_right = _resolve_S_pair(left, right, trL, trR)
            S_v = S_left + float(alpha) * (S_right - S_left)
            rayp_v = lerp(left.rayp, right.rayp, alpha)

            out_name = "%s.%s" % (tname, c)
            out_path = os.path.join(out_dir, out_name)
            write_full_like(trL, out_path, y, S_v, rayp_v, c)
            print("Wrote %s (targeted-X FULL, .%s) using %s--%s alpha=%.6f  S_v=%.6f" %
                  (out_path, c, left.station, right.station, alpha, S_v))

# ---------- WINDOWED: targeted inference ----------
def _infer_at_x_windowed(recs, targets, out_dir, model_path, device="cpu",
                         comp=None, name_prefix="V", window_cfg=None):
    """
    Targeted inference at specific X positions (WINDOWED workflow):
      - slice left/right around S-time using window_cfg (tmin/tmax)
      - robust-normalize windows
      - UNet on [left, right, alpha]
      - de-normalize with blended robust scales
      - write windowed SAC with correct absolute window timing
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # load ckpt + model
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)

    # window config: prefer checkpoint, else provided fallback
    ckpt_cfg = ckpt.get('window_cfg', None)
    if ckpt_cfg and isinstance(ckpt_cfg, dict):
        wc = WindowConfig(**ckpt_cfg)
    else:
        wc = window_cfg if window_cfg is not None else WindowConfig()

    model = UNet1D(in_ch=3, base=32).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # choose components
    if comp in ("r", "z"):
        comps = [comp]
    else:
        s = set([r.comp for r in recs if r.comp in ("r", "z")])
        comps = sorted(list(s))

    # normalize target list to (name, x)
    norm_targets = []
    for t in targets:
        if isinstance(t, tuple) and len(t) == 2:
            nm, xv = t[0], float(t[1])
        else:
            nm, xv = None, float(t)
        norm_targets.append((nm, xv))

    for c in comps:
        rc = _filter_by_comp(recs, c)
        if len(rc) < 2:
            continue

        ordered = order_by_x(rc)

        # per-station cache: normalized window + robust scale
        cache_n = {}
        cache_s = {}
        cache_L = {}

        for r in rc:
            st = read(r.path); tr = st[0]
            fs = 1.0 / float(tr.stats.delta)
            x = tr.data.astype(np.float32)
            x = x - np.mean(x)
            sac = getattr(tr.stats, 'sac', None)
            B = float(getattr(sac, 'b', 0.0) if sac is not None else 0.0)
            t = B + np.arange(x.size, dtype=np.float32) / fs

            # Robust S-time resolution for slicing
            S_time = _resolve_S_time(r, tr)

            # slice window around S_time
            y_raw, _ = slice_window(tr=x, t=t, t_center=float(S_time),
                                    tmin=wc.tmin, tmax=wc.tmax)

            # enforce stride multiple
            L = (y_raw.shape[0] // 4) * 4
            if L >= 8 and L < y_raw.shape[0]:
                y_raw = y_raw[:L]

            # robust scale & normalize
            s = robust_amp(y_raw)
            y_n = (y_raw / max(s, 1e-12)).astype(np.float32)

            cache_n[r.path] = y_n
            cache_s[r.path] = float(s)
            cache_L[r.path] = int(y_n.shape[0])

        # now do targeted-X inference
        for t_idx, (tname, tx) in enumerate(norm_targets):
            left, right, alpha = _bracket_and_alpha_by_x(ordered, float(tx))
            if left is None or right is None:
                continue
            if tname is None:
                tname = "%s_%02d" % (name_prefix, t_idx + 1)

            wi_n = cache_n[left.path]
            wk_n = cache_n[right.path]
            sL   = cache_s[left.path]
            sR   = cache_s[right.path]

            # align lengths for this pair
            Lpair = min(cache_L[left.path], cache_L[right.path])
            if Lpair < 8:
                continue
            L4 = (Lpair // 4) * 4
            wi_n = wi_n[:L4]
            wk_n = wk_n[:L4]

            alpha_vec = np.full((L4,), float(alpha), dtype=np.float32)
            xnet = np.stack([wi_n, wk_n, alpha_vec], axis=0)[None, ...]

            with torch.no_grad():
                y_n = model(torch.from_numpy(xnet).to(device)).cpu().numpy()[0]

            # de-normalize with blended robust scale
            sV = (1.0 - float(alpha)) * sL + float(alpha) * sR
            y  = (y_n * sV).astype(np.float32)

            # header interpolation fields with robust S-time resolution
            trL = read(left.path)[0]
            trR = read(right.path)[0]
            S_left, S_right = _resolve_S_pair(left, right, trL, trR)
            S_v = S_left + float(alpha) * (S_right - S_left)
            rayp_v = lerp(left.rayp,   right.rayp,   alpha)

            out_name = "%s.%s" % (tname, c)
            out_path = os.path.join(out_dir, out_name)

            # write as WINDOWED trace with absolute slice timing
            write_windowed_like(left.path, out_path, y, wc, S_v, rayp_v, c)
            print("Wrote %s (targeted-X WINDOWED .%s) using %s--%s alpha=%.6f  S_v=%.6f"
                  % (out_path, c, left.station, right.station, alpha, S_v))

def main():
    ap = argparse.ArgumentParser(description="sac_xinterp CLI (X-only port)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # prepare
    ap_prep = sub.add_parser("prepare", help="Scan files and export metadata CSV.")
    ap_prep.add_argument("--data_dir", default=".")
    ap_prep.add_argument("--pattern", required=True)
    ap_prep.add_argument("--xmap", required=True, help="Text map: <station> <X>")
    ap_prep.add_argument("--out", required=True)

    # train
    ap_tr = sub.add_parser("train", help="Train model.")
    ap_tr.add_argument("--data_dir", default=".")
    ap_tr.add_argument("--pattern", required=True)
    ap_tr.add_argument("--xmap", required=True)
    ap_tr.add_argument("--comp", default=None)
    ap_tr.add_argument("--train_mode", default="full", choices=["full","window"])
    ap_tr.add_argument("--epochs", type=int, default=30)
    ap_tr.add_argument("--batch_size", type=int, default=16)
    ap_tr.add_argument("--lr", type=float, default=1e-3)
    ap_tr.add_argument("--device", default="cpu")
    ap_tr.add_argument("--model_out", default="model.ckpt")
    ap_tr.add_argument("--dt", type=float, default=0.1)

    # infer (distance-based placement along X)
    ap_inf = sub.add_parser("infer", help="Infer evenly spaced virtuals by X distance.")
    ap_inf.add_argument("--data_dir", default=".")
    ap_inf.add_argument("--pattern", required=True)
    ap_inf.add_argument("--xmap", required=True)
    ap_inf.add_argument("--comp", default=None)
    ap_inf.add_argument("--model", required=True)
    ap_inf.add_argument("--infer_mode", default="full", choices=["auto","full","window"])
    ap_inf.add_argument("--target_dx", type=float, default=5.0)
    ap_inf.add_argument("--min_between", type=int, default=0)
    ap_inf.add_argument("--max_between", type=int, default=6)
    ap_inf.add_argument("--out_dir", required=True)
    ap_inf.add_argument("--device", default="cpu")

    # infer-at-x (targeted X positions)
    ap_ia = sub.add_parser("infer-at-x", help="Infer virtuals at target X positions.")
    ap_ia.add_argument("--data_dir", default=".")
    ap_ia.add_argument("--pattern", required=True)
    ap_ia.add_argument("--xmap", required=True)
    ap_ia.add_argument("--comp", default=None)
    ap_ia.add_argument("--model", required=True)
    ap_ia.add_argument("--out-dir", required=True)
    ap_ia.add_argument("--device", default="cpu")
    ap_ia.add_argument("--targets", required=True, help="File with lines: 'name X' or 'X'")
    ap_ia.add_argument("--name-prefix", default="V")
    ap_ia.add_argument("--infer_mode", default="full", choices=["auto","full","window"])
    # fallbacks if ckpt has no window_cfg:
    ap_ia.add_argument("--dt",   type=float, default=0.1)
    ap_ia.add_argument("--tmin", type=float, default=-5.0)
    ap_ia.add_argument("--tmax", type=float, default=20.0)

    args = ap.parse_args()

    if args.cmd == "prepare":
        paths = _gather_paths(args.data_dir, args.pattern)
        if not paths:
            raise SystemExit("No files matched pattern.")
        xmap = parse_xmap(args.xmap)
        physics = PhysicsKeys()
        recs = read_sac_metadata_x(paths, physics, xmap)
        _save_meta_csv(recs, args.out)

    elif args.cmd == "train":
        paths = _gather_paths(args.data_dir, args.pattern)
        if not paths:
            raise SystemExit("No files matched pattern.")
        xmap = parse_xmap(args.xmap)
        physics = PhysicsKeys()
        recs = read_sac_metadata_x(paths, physics, xmap)
        if args.comp:
            recs = _filter_by_comp(recs, args.comp)
        cfg = WindowConfig(dt=args.dt)
        train_model(recs, args.train_mode, cfg,
                    args.epochs, args.batch_size, args.lr,
                    args.device, args.model_out)

    elif args.cmd == "infer":
        paths = _gather_paths(args.data_dir, args.pattern)
        if not paths:
            raise SystemExit("No files matched pattern.")
        xmap = parse_xmap(args.xmap)
        physics = PhysicsKeys()
        recs = read_sac_metadata_x(paths, physics, xmap)
        if args.comp:
            recs = _filter_by_comp(recs, args.comp)
        cfg = WindowConfig(dt=0.1)  # not used in full mode but required by engine API
        infer_virtual(recs, args.out_dir, args.model,
                      args.target_dx, args.min_between, args.max_between,
                      args.infer_mode, cfg, args.device)

    elif args.cmd == "infer-at-x":
        paths = _gather_paths(args.data_dir, args.pattern)
        if not paths:
            raise SystemExit("No files matched pattern in --data_dir.")
        xmap = parse_xmap(args.xmap)
        physics = PhysicsKeys()
        recs = read_sac_metadata_x(paths, physics, xmap)
        if args.comp:
            recs = _filter_by_comp(recs, args.comp)
        targets = _load_targets_x_file(args.targets)
        if not targets:
            raise SystemExit("No valid targets found in --targets file.")

        # decide mode (if 'auto', prefer window if the checkpoint says so)
        infer_mode = args.infer_mode
        if infer_mode == "auto":
            try:
                ckpt_probe = torch.load(args.model, map_location="cpu")
                train_mode = ckpt_probe.get("train_mode", "full")
            except Exception:
                train_mode = "full"
            infer_mode = "window" if train_mode == "window" else "full"

        if infer_mode == "window":
            wc = WindowConfig(dt=args.dt, tmin=args.tmin, tmax=args.tmax)
            _infer_at_x_windowed(
                recs, targets, args.out_dir,
                args.model, args.device, args.comp,
                args.name_prefix, window_cfg=wc
            )
        else:
            _infer_at_x(
                recs, targets, args.out_dir,
                args.model, args.device, args.comp,
                args.name_prefix
            )

    else:
        raise SystemExit("Unknown command: %s" % args.cmd)


if __name__ == "__main__":
    main()

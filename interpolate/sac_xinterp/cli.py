
# 1D-X port of your legacy CLI: same commands, but no lat/lon. Uses an X map.
from pathlib import Path
import argparse
import glob
import os
import sys

import numpy as np
import torch

from .metadata import read_sac_metadata_x, parse_xmap
from .engine import train_model, infer_virtual, write_full_like
from .config import WindowConfig, PhysicsKeys
from .model import UNet1D
from .utils import trace_time_axis, interp_to_axis, robust_scale, lerp, order_by_x, nearest_segment_and_alpha_x

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

def _infer_at_x(recs, targets, out_dir, model_path, device="cpu",
                comp=None, name_prefix="V"):
    """Targeted inference at specific X positions (no coords)."""
    from obspy import read

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
            left, right, alpha = nearest_segment_and_alpha_x(ordered, tx)
            if left is None or right is None:
                continue

            if tname is None:
                tname = "%s_%02d" % (name_prefix, t_idx + 1)

            trL = _get_trace(left.path)
            trR = _get_trace(right.path)

            tL = trace_time_axis(trL)
            wL = trL.data.astype(np.float32)
            wR_on_L = interp_to_axis(tL, trR)

            wL_n = robust_scale(wL)
            wR_n = robust_scale(wR_on_L)

            Lsig = min(wL_n.size, wR_n.size)
            L4 = (Lsig // 4) * 4
            wL_n = wL_n[:L4]
            wR_n = wR_n[:L4]

            alpha_vec = np.full((L4,), float(alpha), dtype=np.float32)
            x = np.stack([wL_n, wR_n, alpha_vec], axis=0)[None, ...]

            with torch.no_grad():
                y_n = model(torch.from_numpy(x).to(device)).cpu().numpy()[0]
            y = y_n.astype(np.float32)

            if L4 < int(trL.stats.npts):
                pad = int(trL.stats.npts) - L4
                y = np.pad(y, (0, pad), mode='constant', constant_values=0.0)

            S_v = lerp(left.S_time, right.S_time, alpha)
            rayp_v = lerp(left.rayp, right.rayp, alpha)

            out_name = "%s.%s" % (tname, c)
            out_path = os.path.join(out_dir, out_name)
            write_full_like(trL, out_path, y, S_v, rayp_v, c)
            print("Wrote %s (targeted-X, .%s) using %s--%s alpha=%.3f" %
                  (out_path, c, left.station, right.station, alpha))

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
        _infer_at_x(recs, targets, args.out_dir,
                    args.model, args.device, args.comp,
                    args.name_prefix)

    else:
        raise SystemExit("Unknown command: %s" % args.cmd)


if __name__ == "__main__":
    main()

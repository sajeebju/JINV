from __future__ import annotations
import argparse, glob, os, re, torch, pandas as pd
from typing import List
from .config import WindowConfig, PhysicsKeys
from .metadata import parse_refpoints, read_sac_metadata, export_t8_dat, TraceRecord
from .engine import train_model, infer_virtual

def _collect_paths(data_dir: str, pattern: str) -> List[str]:
    paths = [os.path.join(data_dir, p) for p in sorted(os.listdir(data_dir))
             if re.fullmatch(pattern.replace('*','.*'), p)]
    if not paths:
        paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    return paths

def cmd_prepare(args):
    phys = PhysicsKeys()
    ref = parse_refpoints(args.refpoints)
    paths = _collect_paths(args.data_dir, args.pattern)
    if not paths: raise SystemExit("No files matched.")
    recs = read_sac_metadata(paths, phys, refpoints=ref)
    if args.comp in ('r','z'):
        recs = [r for r in recs if r.comp == args.comp]
    rows = []
    for r in recs:
        rows.append({'station': r.station, 'comp': r.comp, 'path': r.path,
                     'delta': r.delta, 'offset_km': r.offset_km,
                     'lat': r.lat, 'lon': r.lon,
                     'S_time(t8)': r.S_time, 'rayp(user0)': r.rayp})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote metadata: {args.out} ({len(rows)} rows)")

def cmd_train(args):
    phys = PhysicsKeys()
    ref = parse_refpoints(args.refpoints)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    # train per component
    targets = ['r','z'] if args.comp == 'both' else [args.comp]
    for comp in targets:
        patt = args.pattern
        paths = _collect_paths(args.data_dir, patt)
        if not paths: raise SystemExit(f"No files matched: {patt}")
        recs = read_sac_metadata(paths, phys, refpoints=ref)
        recs = [r for r in recs if r.comp == comp]
        if len(recs) < 3:
            print(f"Skip {comp}: need >=3 traces, got {len(recs)}")
            continue
        wcfg = WindowConfig(dt=args.dt, tmin=args.tmin, tmax=args.tmax,
                            bp_low=args.bp_low, bp_high=args.bp_high, corners=args.corners)
        model_out = args.model_out if args.comp != 'both' else os.path.splitext(args.model_out)[0] + f"_{comp}.ckpt"
        print(f"Training {comp}-component -> {model_out}")
        train_model(recs, train_mode=args.train_mode, window_cfg=wcfg,
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    device=device, model_out=model_out)

def cmd_infer(args):
    phys = PhysicsKeys()
    ref = parse_refpoints(args.refpoints)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    # infer per component, requiring model per component
    if args.comp == 'both':
        runs = [('r', args.model_r), ('z', args.model_z)]
    else:
        runs = [(args.comp, args.model)]
    for comp, ckpt in runs:
        if ckpt is None:
            raise SystemExit(f"--comp {comp} requires --model_{comp} (or use --model with --comp r|z)")
        paths = _collect_paths(args.data_dir, args.pattern)
        if not paths: raise SystemExit(f"No files matched: {args.pattern}")
        recs_all = read_sac_metadata(paths, phys, refpoints=ref)
        recs = [r for r in recs_all if r.comp == comp]
        if len(recs) < 2:
            print(f"Skip {comp}: need >=2 traces, got {len(recs)}")
            continue
        wcfg = WindowConfig(dt=args.dt, tmin=args.tmin, tmax=args.tmax,
                            bp_low=args.bp_low, bp_high=args.bp_high, corners=args.corners)
        print(f"Inferencing {comp}-component with {ckpt}")
        infer_virtual(recs, out_dir=args.out_dir, model_path=ckpt,
                      target_km=args.target_km, min_between=args.min_between, max_between=args.max_between,
                      infer_mode=args.infer_mode, window_cfg=wcfg, device=device,
                      update_coords=args.update_coords, coord_base=args.refpoints, coord_out=args.coord_out)

def cmd_export_t8(args):
    phys = PhysicsKeys()
    ref = parse_refpoints(args.refpoints)
    paths = _collect_paths(args.data_dir, args.pattern)
    if not paths: raise SystemExit("No files matched.")
    recs = read_sac_metadata(paths, phys, refpoints=ref)
    export_t8_dat(recs, args.out)

def make_argparser():
    p = argparse.ArgumentParser(description="SAC profile interpolator (ML-only, split by component)")
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('prepare', help='Scan SAC headers (+refpoints) to CSV')
    sp.add_argument('--data_dir', required=True)
    sp.add_argument('--pattern', default='WB*.[rz]')
    sp.add_argument('--refpoints', default=None)
    sp.add_argument('--comp', choices=['r','z','both'], default='both')
    sp.add_argument('--out', default='meta.csv')
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser('train', help='Train U-Net per component (window/full)')
    sp.add_argument('--data_dir', required=True)
    sp.add_argument('--pattern', default='WB*.[rz]')
    sp.add_argument('--refpoints', default=None)
    sp.add_argument('--comp', choices=['r','z','both'], default='both')
    sp.add_argument('--train_mode', choices=['window','full'], default='window')
    sp.add_argument('--epochs', type=int, default=30)
    sp.add_argument('--batch_size', type=int, default=16)
    sp.add_argument('--lr', type=float, default=1e-3)
    # window cfg
    sp.add_argument('--dt', type=float, default=0.1)
    sp.add_argument('--tmin', type=float, default=-5.0)
    sp.add_argument('--tmax', type=float, default=20.0)
    sp.add_argument('--bp_low', type=float, default=0.08)
    sp.add_argument('--bp_high', type=float, default=1.0)
    sp.add_argument('--corners', type=int, default=4)
    sp.add_argument('--model_out', default='interp.ckpt')
    sp.add_argument('--cpu', action='store_true')
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser('infer', help='Generate virtual traces (+ optional coord update)')
    sp.add_argument('--data_dir', required=True)
    sp.add_argument('--pattern', default='WB*.[rz]')
    sp.add_argument('--refpoints', default=None, help='Base coord file to update (for --update_coords)')
    sp.add_argument('--comp', choices=['r','z','both'], default='both')
    sp.add_argument('--model', default=None, help='Checkpoint for single-comp runs (use with --comp r|z)')
    sp.add_argument('--model_r', default=None, help='Checkpoint for R if --comp both')
    sp.add_argument('--model_z', default=None, help='Checkpoint for Z if --comp both')
    sp.add_argument('--infer_mode', choices=['auto','window','full'], default='auto')
    sp.add_argument('--out_dir', default='virtual_out')
    sp.add_argument('--target_km', type=float, default=5.0)
    sp.add_argument('--min_between', type=int, default=0)
    sp.add_argument('--max_between', type=int, default=6)
    sp.add_argument('--dt', type=float, default=0.1)
    sp.add_argument('--tmin', type=float, default=-5.0)
    sp.add_argument('--tmax', type=float, default=20.0)
    sp.add_argument('--bp_low', type=float, default=0.08)
    sp.add_argument('--bp_high', type=float, default=1.0)
    sp.add_argument('--corners', type=int, default=4)
    sp.add_argument('--cpu', action='store_true')
    sp.add_argument('--update_coords', action='store_true')
    sp.add_argument('--coord_out', default='coord_updated.xy')
    sp.set_defaults(func=cmd_infer)

    sp = sub.add_parser('export_t8', help='Export t8.dat (per station, prefer .z)')
    sp.add_argument('--data_dir', required=True)
    sp.add_argument('--pattern', default='WB*.[rz]')
    sp.add_argument('--refpoints', default=None)
    sp.add_argument('--out', default='t8.dat')
    sp.set_defaults(func=cmd_export_t8)
    return p

def main():
    p = make_argparser()
    args = p.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()


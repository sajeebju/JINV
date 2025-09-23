
from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from obspy import read
from .config import PhysicsKeys, SAC_UNDEF
from .utils import natural_station_key, detect_component

@dataclass
class TraceRecord:
    path: str
    station: str
    comp: str
    delta: float
    S_time: Optional[float]
    rayp: Optional[float]
    x_dist: float  # <--- 1D distance coordinate
    # lat/lon removed

def parse_xmap(path: str) -> Dict[str, float]:
    """Text file with lines '<station> <X>'.
    Station can be file stem (e.g., WB02) or full filename (WB02.r).
    """
    m: Dict[str, float] = {}
    if not path or not os.path.isfile(path):
        raise SystemExit("Missing --xmap or file not found.")
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]
            if len(parts) < 2: continue
            key, val = parts[0], parts[1]
            try:
                m[key] = float(val)
            except Exception:
                continue
    return m

def _get_float_safe(sac, key: str) -> Optional[float]:
    if sac is None: return None
    val = getattr(sac, key, None)
    if val is None: return None
    try:
        v = float(val)
    except Exception:
        return None
    from math import isclose
    if isclose(v, SAC_UNDEF): return None
    return v

def read_sac_metadata_x(paths: List[str],
                        physics: PhysicsKeys,
                        xmap: Dict[str, float]) -> List[TraceRecord]:
    recs: List[TraceRecord] = []
    for p in paths:
        st = read(p)
        tr = st[0]
        sac = getattr(tr.stats, 'sac', None)
        sta = tr.stats.station or os.path.splitext(os.path.basename(p))[0]
        # strip trailing .r/.z from filename for matching
        sta_stem = re.sub(r"\.[rztRZT]$", "", sta)
        comp = detect_component(p, tr)
        S    = _get_float_safe(sac, physics.key_S)
        rayp = _get_float_safe(sac, physics.key_rayp)

        # match X from xmap using several keys
        X = None
        for k in (sta_stem, os.path.basename(p), os.path.splitext(os.path.basename(p))[0],
                  sta_stem.upper(), sta_stem.lower()):
            if k in xmap:
                X = float(xmap[k]); break
        if X is None:
            raise SystemExit(f"No X found in xmap for station/file '{sta_stem}' ({p}).")

        recs.append(TraceRecord(
            path=p, station=sta_stem, comp=comp, delta=float(tr.stats.delta),
            S_time=S, rayp=rayp, x_dist=X
        ))

    # make sure they are ordered stably by X
    recs.sort(key=lambda r: (r.x_dist, natural_station_key(r.path)))
    return recs

def export_t8_dat_x(recs: List[TraceRecord], out_path: str):
    # prefer .z if both exist
    choice: Dict[str, TraceRecord] = {}
    for r in recs:
        if r.station not in choice: choice[r.station] = r
        elif r.comp == 'z' and choice[r.station].comp != 'z':
            choice[r.station] = r
    with open(out_path, 'w') as f:
        for sta, r in choice.items():
            S   = r.S_time if r.S_time is not None else 0.0
            x   = r.x_dist
            rayp = r.rayp if r.rayp is not None else 0.0
            f.write(f"{sta}.z {S:12.6f} {x:10.3f} {rayp:12.6f}\n")

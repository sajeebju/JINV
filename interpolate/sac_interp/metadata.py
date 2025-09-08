from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from obspy import read
from .config import PhysicsKeys, SAC_UNDEF
from .utils import natural_station_key, detect_component, haversine_km

@dataclass
class TraceRecord:
    path: str
    station: str
    comp: str
    delta: float
    S_time: Optional[float]
    rayp: Optional[float]
    offset_km: float
    lat: Optional[float] = None
    lon: Optional[float] = None

def parse_refpoints(path: Optional[str]) -> Dict[str, Tuple[float, float]]:
    ref: Dict[str, Tuple[float, float]] = {}
    if not path or not os.path.isfile(path): return ref
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            s = s.replace(",", " ")
            parts = [p for p in s.split() if p]
            if len(parts) < 3: continue
            sta, a, b = parts[0], parts[1], parts[2]
            try:
                lat = float(a); lon = float(b)
            except Exception:
                continue
            ref[re.sub(r"\.[rzt]$", "", sta)] = (lat, lon)
    return ref

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

def read_sac_metadata(paths: List[str],
                      physics: PhysicsKeys,
                      refpoints: Optional[Dict[str, Tuple[float, float]]] = None) -> List[TraceRecord]:
    recs: List[TraceRecord] = []
    for p in paths:
        st = read(p)
        tr = st[0]
        sac = getattr(tr.stats, 'sac', None)
        sta = tr.stats.station or os.path.splitext(os.path.basename(p))[0]
        sta = re.sub(r"\.[rzt]$", "", sta)
        comp = detect_component(p, tr)
        S    = _get_float_safe(sac, physics.key_S)
        rayp = _get_float_safe(sac, physics.key_rayp)
        lat = lon = None
        if refpoints and sta in refpoints:
            lat, lon = refpoints[sta]
        recs.append(TraceRecord(
            path=p, station=sta, comp=comp, delta=float(tr.stats.delta),
            S_time=S, rayp=rayp, offset_km=np.nan, lat=lat, lon=lon
        ))

    # assign along-profile offsets
    have_geo = sum(1 for r in recs if r.lat is not None and r.lon is not None) >= max(2, int(0.7 * len(recs)))
    if have_geo:
        cum = 0.0; prev_lat = prev_lon = None
        for r in recs:
            if prev_lat is not None and r.lat is not None and r.lon is not None:
                d = haversine_km(prev_lat, prev_lon, r.lat, r.lon)
                cum += float(max(d, 0.0))
            r.offset_km = cum
            if r.lat is not None and r.lon is not None:
                prev_lat, prev_lon = r.lat, r.lon
        base = min([rr.offset_km for rr in recs if np.isfinite(rr.offset_km)], default=0.0)
        for r in recs:
            if not np.isfinite(r.offset_km): r.offset_km = base
    else:
        recs.sort(key=lambda r: natural_station_key(r.path))
        for i, r in enumerate(recs):
            r.offset_km = float(i)
    return recs

def export_t8_dat(recs: List[TraceRecord], out_path: str):
    # prefer .z if both exist
    choice: Dict[str, TraceRecord] = {}
    for r in recs:
        if r.station not in choice: choice[r.station] = r
        elif r.comp == 'z' and choice[r.station].comp != 'z':
            choice[r.station] = r
    with open(out_path, 'w') as f:
        for sta, r in choice.items():
            S   = r.S_time if r.S_time is not None else 0.0
            lat = r.lat if r.lat is not None else 0.0
            lon = r.lon if r.lon is not None else 0.0
            rayp = r.rayp if r.rayp is not None else 0.0
            f.write(f"{sta}.z {S:12.6f} {lat:8.3f} {lon:8.3f} {rayp:12.6f}\n")


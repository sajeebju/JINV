
from __future__ import annotations
from dataclasses import dataclass

SAC_UNDEF = -12345.0

@dataclass
class WindowConfig:
    dt: float = 0.1
    tmin: float = -5.0
    tmax: float = 20.0
    bp_low: float = 0.08
    bp_high: float = 1.0
    corners: int = 4
    taper_frac: float = 0.05

@dataclass
class PhysicsKeys:
    key_S: str = "t8"       # S-arrival header key
    key_rayp: str = "user0" # optional ray parameter

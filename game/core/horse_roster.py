from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

from panda3d.core import Vec4


class Rarity(IntEnum):
    COMMON = 1
    RARE = 2
    LEGENDARY = 3


SPAWN_WEIGHTS = {
    Rarity.COMMON: 4,
    Rarity.RARE: 2,
    Rarity.LEGENDARY: 1,
}


@dataclass(frozen=True)
class HorseSpecies:
    id: str
    name: str
    map_id: str
    rarity: Rarity
    body: Vec4
    head: Vec4
    legs: Vec4
    mane: Vec4


ROSTER: List[HorseSpecies] = [
    # --- Meadow (oval) ---
    HorseSpecies(
        id="meadow_chestnut",
        name="CHESTNUT",
        map_id="meadow",
        rarity=Rarity.COMMON,
        body=Vec4(0.55, 0.30, 0.15, 1.0),
        head=Vec4(0.55, 0.30, 0.15, 1.0),
        legs=Vec4(0.30, 0.18, 0.10, 1.0),
        mane=Vec4(0.10, 0.07, 0.05, 1.0),
    ),
    HorseSpecies(
        id="meadow_bay",
        name="BAY",
        map_id="meadow",
        rarity=Rarity.COMMON,
        body=Vec4(0.70, 0.50, 0.28, 1.0),
        head=Vec4(0.55, 0.36, 0.20, 1.0),
        legs=Vec4(0.20, 0.13, 0.08, 1.0),
        mane=Vec4(0.15, 0.10, 0.07, 1.0),
    ),
    HorseSpecies(
        id="meadow_dapple",
        name="DAPPLE",
        map_id="meadow",
        rarity=Rarity.COMMON,
        body=Vec4(0.62, 0.62, 0.65, 1.0),
        head=Vec4(0.78, 0.78, 0.80, 1.0),
        legs=Vec4(0.30, 0.30, 0.32, 1.0),
        mane=Vec4(0.85, 0.85, 0.88, 1.0),
    ),
    HorseSpecies(
        id="meadow_painted",
        name="PAINT",
        map_id="meadow",
        rarity=Rarity.RARE,
        body=Vec4(0.92, 0.90, 0.85, 1.0),
        head=Vec4(0.45, 0.25, 0.15, 1.0),
        legs=Vec4(0.92, 0.90, 0.85, 1.0),
        mane=Vec4(0.45, 0.25, 0.15, 1.0),
    ),
    HorseSpecies(
        id="meadow_sunset",
        name="SUNSET",
        map_id="meadow",
        rarity=Rarity.RARE,
        body=Vec4(0.92, 0.55, 0.25, 1.0),
        head=Vec4(0.95, 0.78, 0.55, 1.0),
        legs=Vec4(0.55, 0.25, 0.10, 1.0),
        mane=Vec4(0.98, 0.92, 0.78, 1.0),
    ),
    HorseSpecies(
        id="meadow_golden_pegasus",
        name="PEGASUS",
        map_id="meadow",
        rarity=Rarity.LEGENDARY,
        body=Vec4(0.95, 0.82, 0.30, 1.0),
        head=Vec4(0.98, 0.92, 0.62, 1.0),
        legs=Vec4(0.95, 0.82, 0.30, 1.0),
        mane=Vec4(1.00, 0.98, 0.85, 1.0),
    ),
    # --- Highlands (winding) ---
    HorseSpecies(
        id="highland_pony",
        name="HIGHLAND",
        map_id="highlands",
        rarity=Rarity.COMMON,
        body=Vec4(0.32, 0.20, 0.12, 1.0),
        head=Vec4(0.10, 0.08, 0.06, 1.0),
        legs=Vec4(0.10, 0.08, 0.06, 1.0),
        mane=Vec4(0.08, 0.06, 0.05, 1.0),
    ),
    HorseSpecies(
        id="highland_roan",
        name="ROAN",
        map_id="highlands",
        rarity=Rarity.COMMON,
        body=Vec4(0.45, 0.50, 0.55, 1.0),
        head=Vec4(0.55, 0.58, 0.62, 1.0),
        legs=Vec4(0.20, 0.22, 0.25, 1.0),
        mane=Vec4(0.10, 0.12, 0.15, 1.0),
    ),
    HorseSpecies(
        id="highland_cloud",
        name="DRIFTER",
        map_id="highlands",
        rarity=Rarity.COMMON,
        body=Vec4(0.85, 0.86, 0.90, 1.0),
        head=Vec4(0.95, 0.95, 0.97, 1.0),
        legs=Vec4(0.78, 0.80, 0.85, 1.0),
        mane=Vec4(0.98, 0.98, 1.00, 1.0),
    ),
    HorseSpecies(
        id="highland_storm",
        name="STORM",
        map_id="highlands",
        rarity=Rarity.RARE,
        body=Vec4(0.18, 0.22, 0.32, 1.0),
        head=Vec4(0.06, 0.07, 0.10, 1.0),
        legs=Vec4(0.06, 0.07, 0.10, 1.0),
        mane=Vec4(0.78, 0.82, 0.92, 1.0),
    ),
    HorseSpecies(
        id="highland_ember",
        name="EMBER",
        map_id="highlands",
        rarity=Rarity.RARE,
        body=Vec4(0.55, 0.10, 0.08, 1.0),
        head=Vec4(0.10, 0.06, 0.05, 1.0),
        legs=Vec4(0.08, 0.05, 0.04, 1.0),
        mane=Vec4(0.95, 0.50, 0.15, 1.0),
    ),
    HorseSpecies(
        id="highland_frostmane",
        name="FROSTMANE",
        map_id="highlands",
        rarity=Rarity.LEGENDARY,
        body=Vec4(0.92, 0.96, 0.98, 1.0),
        head=Vec4(0.96, 0.98, 1.00, 1.0),
        legs=Vec4(0.85, 0.92, 0.96, 1.0),
        mane=Vec4(0.55, 0.85, 0.95, 1.0),
    ),
]


def roster_for_map(map_id: str) -> List[HorseSpecies]:
    return [s for s in ROSTER if s.map_id == map_id]


def species_by_id(species_id: str) -> Optional[HorseSpecies]:
    for s in ROSTER:
        if s.id == species_id:
            return s
    return None


def pick_species(rng: random.Random, map_id: str) -> HorseSpecies:
    pool = roster_for_map(map_id)
    if not pool:
        raise ValueError(f"no species for map_id={map_id!r}")
    weights = [SPAWN_WEIGHTS[s.rarity] for s in pool]
    return rng.choices(pool, weights=weights, k=1)[0]

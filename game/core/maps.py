from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from panda3d.core import Vec4

from game.core.track import OvalTrack
from game.core.winding_track import WindingTrack


@dataclass(frozen=True)
class MapDef:
    id: str
    title: str
    description: str
    swatch_color: Vec4
    make_track: Callable[[], object]


MAPS: List[MapDef] = [
    MapDef(
        id="meadow",
        title="MEADOW LOOP",
        description="A scenic oval ride through grassy fields. Cosy.",
        swatch_color=Vec4(0.30, 0.65, 0.30, 1.0),
        make_track=lambda: OvalTrack(a=30.0, b=18.0),
    ),
    MapDef(
        id="highlands",
        title="HIGHLANDS",
        description="A longer, winding route through high country.",
        swatch_color=Vec4(0.35, 0.45, 0.62, 1.0),
        make_track=lambda: WindingTrack(),
    ),
]


def get_map(map_id: str) -> MapDef:
    for m in MAPS:
        if m.id == map_id:
            return m
    raise KeyError(f"unknown map_id={map_id!r}")


DEFAULT_MAP_ID = MAPS[0].id

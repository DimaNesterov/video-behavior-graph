"""Typed graph schema (spec section 9). Closed node/edge vocabulary."""
from pydantic import BaseModel


class ObjectNode(BaseModel):
    id: str
    type: str                 # from configs/ontology.yaml
    bbox_px: list[float]
    pos_m: list[float]        # ground-plane position (foot point of bbox)
    confidence: float
    source: str


class ZoneNode(BaseModel):
    id: str
    type: str                 # walkable_area for v0
    area_m2: float
    cell_m: float


class EntryExitNode(BaseModel):
    id: str
    type: str                 # entry_exit
    subtype: str = "boundary_exit"   # boundary_exit | building_entry
    linked_object: str | None = None
    location_m: list[float]
    usage_frequency: float


class Edge(BaseModel):
    source: str
    target: str
    type: str                 # near | adjacent_to | connected_to
    distance_m: float | None = None
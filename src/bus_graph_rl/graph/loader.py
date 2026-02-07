from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import osmnx as ox
import networkx as nx

@dataclass(frozen=True)
class OSMGraphSpec:
    area_name: Optional[str] = "Toulouse"
    area_point: Optional[Tuple[float, float]] = None  # (lat, lon)
    area_distance: Optional[int] = None               # meters
    network_type: str = "drive"
    simplify: bool = True

def load_osm_graph(spec: OSMGraphSpec) -> nx.MultiDiGraph:
    """Load an OSM street network graph.

    Mirrors the exploratory notebook logic:
    - graph_from_place OR graph_from_point
    - directed
    - add edge speeds + travel times
    """
    ox.config(log_console=False, use_cache=True)

    if spec.area_point is not None and spec.area_distance is not None:
        G = ox.graph_from_point(spec.area_point, dist=spec.area_distance, network_type=spec.network_type, simplify=spec.simplify)
    else:
        if not spec.area_name:
            raise ValueError("OSMGraphSpec.area_name must be provided when area_point is None")
        G = ox.graph_from_place(spec.area_name, network_type=spec.network_type, simplify=spec.simplify)

    G = G.to_directed()
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G

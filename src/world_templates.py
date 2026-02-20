from __future__ import annotations

from typing import Any, Dict


ROOM_BASIC_DIMENSIONS = {
    "width": 8.0,
    "length": 8.0,
    "height": 3.0,
}


def build_room_basic_template(
    dimensions: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    dims = dict(ROOM_BASIC_DIMENSIONS)
    if dimensions:
        for key in ("width", "length", "height"):
            value = dimensions.get(key)
            if isinstance(value, (int, float)) and value > 0:
                dims[key] = float(value)

    half_width = dims["width"] / 2.0
    half_length = dims["length"] / 2.0
    half_height = dims["height"] / 2.0

    nodes = [
        {
            "id": "floor",
            "kind": "plane",
            "size": [dims["width"], dims["length"]],
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "collider": True,
            "teleportable": True,
        },
        {
            "id": "ceiling",
            "kind": "plane",
            "size": [dims["width"], dims["length"]],
            "position": [0.0, dims["height"], 0.0],
            "rotation": [180.0, 0.0, 0.0],
            "collider": True,
            "teleportable": False,
        },
        {
            "id": "wall_north",
            "kind": "box",
            "size": [dims["width"], dims["height"], 0.2],
            "position": [0.0, half_height, -half_length],
            "rotation": [0.0, 0.0, 0.0],
            "collider": True,
            "teleportable": False,
        },
        {
            "id": "wall_south",
            "kind": "box",
            "size": [dims["width"], dims["height"], 0.2],
            "position": [0.0, half_height, half_length],
            "rotation": [0.0, 0.0, 0.0],
            "collider": True,
            "teleportable": False,
        },
        {
            "id": "wall_east",
            "kind": "box",
            "size": [0.2, dims["height"], dims["length"]],
            "position": [half_width, half_height, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "collider": True,
            "teleportable": False,
        },
        {
            "id": "wall_west",
            "kind": "box",
            "size": [0.2, dims["height"], dims["length"]],
            "position": [-half_width, half_height, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "collider": True,
            "teleportable": False,
        },
    ]

    return {
        "template_id": "room_basic",
        "dimensions": dims,
        "nodes": nodes,
    }


def build_template_geometry(template_id: str) -> Dict[str, Any]:
    if template_id == "room_basic":
        return build_room_basic_template()
    raise ValueError(f"Unsupported template_id '{template_id}'")

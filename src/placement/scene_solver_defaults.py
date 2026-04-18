from __future__ import annotations

SEATING_ROLES = {"chair", "bench", "sofa"}
EDGE_BIASED_ROLES = {"bed", "sofa", "cabinet", "appliance", "table"}
GROUP_ZONE_DEFAULTS = {
    "dining_set": "center",
    "lounge_cluster": "edge",
    "reading_corner": "corner",
    "bedside_cluster": "back",
    "workstation": "back",
}
GROUP_CONSTRAINT_DEFAULTS = {
    "dining_set": "floor",
    "lounge_cluster": "against_wall",
    "reading_corner": "against_wall",
    "bedside_cluster": "against_wall",
    "workstation": "against_wall",
}

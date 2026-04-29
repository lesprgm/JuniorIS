from __future__ import annotations

from src.planning.scene_program_policy import policy_dict, policy_set

SEATING_ROLES = policy_set("seating_roles")
EDGE_BIASED_ROLES = policy_set("edge_biased_roles")
GROUP_ZONE_DEFAULTS = policy_dict("group_zone_defaults")
GROUP_CONSTRAINT_DEFAULTS = policy_dict("group_constraint_defaults")

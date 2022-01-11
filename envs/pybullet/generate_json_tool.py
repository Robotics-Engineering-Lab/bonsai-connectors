keys = ("gripper_x", "gripper_y", "gripper_z", 
                "gripper_vel_x", "gripper_vel_y", "gripper_vel_z", 
                "target_x", "target_y", "target_z",
                "object_x", "object_y", "object_z", 
                "object_vel_x", "object_vel_y", "object_vel_z",
                "dst_ot_x", "dst_ot_y", "dst_ot_z",
                "dst_og_x", "dst_og_y", "dst_og_z"
                )

import json
state = dict()
state["category"] = "Struct"
state["fields"] = []
for i in keys:
    field = ({"name" : i,
            "type" : {"category": "Number.Float32"}}
    )
    state["fields"].append(field)
json_obj = json.dumps(state, indent = 2)
print(json_obj)
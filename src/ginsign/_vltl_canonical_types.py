"""Canonical VLTL-Bench type-constants dictionary.

Hand-curated by the dataset authors; extracted verbatim from
eval_pipeline.ipynb (cell at line ~1897). The `lane` list for
traffic_light was abbreviated in the notebook (`all_roads`); we
reconstruct it from the same dirs/nums/roads product.

This dictionary is the source of truth for which sort each constant
belongs to in VLTL-Bench. signature_builder cross-references the
corpus against this map.
"""

_DIRS = ["north", "south", "east", "west", "northwest", "northeast", "southwest", "southeast"]
_NUMS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
_ROADS = ["street", "avenue"]
_ALL_ROADS = [f"{d}_{n}_{r}" for d in _DIRS for n in _NUMS for r in _ROADS]


TYPE_CONSTANTS = {
    "search_and_rescue": {
        "person": [
            "safe_civilian", "safe_hostile", "safe_person", "safe_rescuer", "safe_victim",
            "unsafe_civilian", "unsafe_person", "unsafe_rescuer", "unsafe_victim",
            "injured_civilian", "injured_hostile", "injured_person", "injured_rescuer",
            "injured_victim",
        ],
        "threat": [
            "active_debris", "active_fire_source", "active_flood", "active_gas_leak",
            "active_unstable_beam", "debris", "fire_source", "flood", "gas_leak",
            "impending_debris", "impending_fire_source", "impending_flood",
            "impending_gas_leak", "impending_unstable_beam", "inactive_debris",
            "inactive_fire_source", "inactive_flood", "inactive_gas_leak",
            "inactive_unstable_beam", "unstable_beam", "nearest_fire_source",
            "nearest_flood", "nearest_gas_leak", "nearest_unstable_beam",
            "probable_debris", "probable_fire_source", "probable_flood",
            "probable_gas_leak", "probable_unstable_beam", "nearest_debris",
        ],
    },
    "traffic_light": {
        "light": ["light_east", "light_north", "light_south", "light_west"],
        "color": ["green", "red", "yellow"],
        "lane": list(_ALL_ROADS),
        "traffic_target": [
            "car", "collision", "cyclist", "motorcycle", "pedestrian", "person",
            "vehicle", "jaywalker",
        ],
    },
    "warehouse": {
        "item": [
            "aeroplane", "apple", "backpack", "banana", "baseball_bat", "baseball_glove",
            "bear", "bed", "bench", "bicycle", "bird", "boat", "book", "bottle", "bowl",
            "broccoli", "bus", "cake", "car", "carrot", "cat", "cell_phone", "chair",
            "clock", "cow", "cup", "dining_table", "dog", "donut", "elephant",
            "fire_hydrant", "fork", "frisbee", "giraffe", "hair-drier", "handbag",
            "horse", "hot_dog", "keyboard", "kite", "knife", "laptop", "microwave",
            "motorbike", "mouse", "orange", "oven", "parking_meter", "person",
            "pizza", "potted_plant", "refrigerator", "remote", "sandwich", "scissors",
            "sheep", "sink", "skateboard", "skis", "snowboard", "sofa", "spoon",
            "sports_ball", "stop_sign", "suitcase", "surfboard", "teddy-bear",
            "tennis_racket", "tie", "toaster", "toilet", "toothbrush", "traffic_light",
            "train", "truck", "tv_monitor", "umbrella", "vase", "wine_glass", "zebra",
        ],
        "location": ["loading_dock", "shelf"],
    },
}

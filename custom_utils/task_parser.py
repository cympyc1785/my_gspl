import os
import csv


# A task string names one scene, and for the segment-based task types one segment
# inside it. These types carry a trailing segment index and read their caption from
# data_root_path; "gt" is the other family and reads its text out of the scene itself.
SEG_TASK_TYPES = ("pred", "tartanair", "scannet")
GT_TASK_TYPE = "gt"

# Splits whose name itself contains '_', so task.split("_") cannot recover them.
# Order matters: the first prefix that matches wins.
SEG_SPLIT_PREFIXES = ("scannet_output_sampled", "Easy_left", "Easy_right", "Hard_left", "Hard_right")
GT_SPLIT_PREFIXES = ("youtube_vis", "SAV", "VOST", "dynamic_replica", "uvo")

# Where each split family keeps its scenes.
SCENE_DATA_ROOT = "/data1/cympyc1785/SceneData"
SAMPLED_DATA_ROOT = "/data3/cympyc1785"

DL3DV_SPLITS = tuple(f"{i}K" for i in range(1, 8))
DYNAMICVERSE_SPLITS = ("DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "dynamic_replica",
                       "spring", "youtube_vis", "uvo")
DYNPOSE_SPLITS = tuple(f"dynpose-{i:04d}" for i in range(0, 90))
TARTANAIR_SPLITS = ("Easy_left", "Easy_right", "Hard_left", "Hard_right")
SCANNET_SPLITS = ("scannet_output_sampled",)

# (splits, scene root template, point cloud file, pcd_type) - first match wins
SCENE_ROOT_RULES = (
    (DL3DV_SPLITS,        SCENE_DATA_ROOT + "/DL3DV/scenes/{split}/{scene}",                  "scene.pt",  "torch"),
    (DYNAMICVERSE_SPLITS, SCENE_DATA_ROOT + "/DynamicVerse/scenes/{split}/{scene}",           "scene.ply", "ply"),
    (DYNPOSE_SPLITS,      SCENE_DATA_ROOT + "/DynamicVerse/scenes/dynpose-100k/{split}/{scene}", "scene.ply", "ply"),
    (SCANNET_SPLITS,      SAMPLED_DATA_ROOT + "/scannet_output_sampled/{scene}",              "scene.ply", "ply"),
    (TARTANAIR_SPLITS,    SAMPLED_DATA_ROOT + "/tartanair_output_sampled/{split}/{scene}",    "scene.ply", "ply"),
)


def is_seg_task(task_type):
    """True for the task types that address a single segment of a scene."""
    return task_type in SEG_TASK_TYPES


def parse_task(task, task_type):
    """Split a task string into (split, scene_name, seg_idx_str).

    seg_idx_str is None for the "gt" task type, which has no segment index.
        scannet_output_sampled_scene0000_00_00_0 -> ("scannet_output_sampled", "scene0000_00_00", "0")
        Easy_left_abandonedfactory_P000_00_0     -> ("Easy_left", "abandonedfactory_P000_00", "0")
        1K_<hash>_3                              -> ("1K", "<hash>", "3")
        youtube_vis_<scene>                      -> ("youtube_vis", "<scene>", None)
    """
    if is_seg_task(task_type):
        for prefix in SEG_SPLIT_PREFIXES:
            if task.startswith(prefix):
                scene_name, seg_idx_str = task[len(prefix) + 1:].rsplit("_", 1)
                return prefix, scene_name, seg_idx_str
        # splits without '_' in the name: <split>_<scene>_<seg>
        split, scene_name, seg_idx_str = task.split("_")
        return split, scene_name, seg_idx_str

    if task_type == GT_TASK_TYPE:
        for prefix in GT_SPLIT_PREFIXES:
            if task.startswith(prefix):
                return prefix, task[len(prefix) + 1:], None
        # splits without '_' in the name: <split>_<scene>
        split, scene_name = task.split("_")
        return split, scene_name, None

    raise ValueError("Invalid task type", task_type)


def resolve_scene(split, scene_name):
    """Return (scene_root_path, pcd_path, pcd_type) for a split/scene pair."""
    for splits, template, pcd_name, pcd_type in SCENE_ROOT_RULES:
        if split in splits:
            scene_root_path = template.format(split=split, scene=scene_name)
            return scene_root_path, os.path.join(scene_root_path, pcd_name), pcd_type

    raise ValueError("No scene root known for split", split)


def load_csv_tasks(csv_path):
    """Read (task_strings, orig_indices) from a distance-band CSV.

    Columns: orig_index, scene_path, avg_dist_nearest_k. Each scene_path
    (e.g. /.../DL3DV/scenes/1K/<hash>/5) becomes the '<split>_<scene>_<seg>'
    task string parse_task() expects, i.e. its last 3 components joined by '_'.
    Raises FileNotFoundError when csv_path is missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    tasks = []
    orig_idxs = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row.get("scene_path", "").strip().rstrip("/")
            if not sp:
                continue
            tasks.append("_".join(sp.split("/")[-3:]))
            try:
                orig_idxs.append(int(row.get("orig_index", -1)))
            except (TypeError, ValueError):
                orig_idxs.append(-1)

    return tasks, orig_idxs


def resolve_task_idx(task_str, orig_index, task_list, index_map=None):
    """Map a CSV-derived task string to its index in the .txt-based task_list.

    index_map is an optional prebuilt {task_string: index} for repeated lookups.
    Returns None when the task is not in task_list.
    """
    if index_map is None:
        index_map = {t: i for i, t in enumerate(task_list)}
    if task_str in index_map:
        return index_map[task_str]

    # fallback: same-order lists (e.g. valid_task_list.txt matches test_data_final)
    if 0 <= orig_index < len(task_list) and task_list[orig_index] == task_str:
        return orig_index

    return None

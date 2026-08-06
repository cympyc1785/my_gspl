import os
import torch
import viser
import trimesh
import argparse
import numpy as np

from custom_utils.custom_panel import CustomPanel

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", type=str, default="/data2/ckd248/SCVideo/camera_generation/evaluation")
parser.add_argument("--ply_path", type=str, default=None)
parser.add_argument("--torch", action="store_true")
parser.add_argument("--mesh", action="store_true")
parser.add_argument("--task_type", type=str, default="pred")
parser.add_argument("--task_offset", type=int, default=0)
parser.add_argument("--pcd", action="store_true")
parser.add_argument("--custom_camera_data_path", type=str, default="./camera_data.json")
parser.add_argument("--data_root_path", type=str, default="/data2/ckd248/SCVideo/camera_generation/evaluation")
parser.add_argument("--task_list_path", type=str, default=None)
args = parser.parse_args()

root_path = args.root_path

# =========================
# Start Viser server
# =========================
server = viser.ViserServer(port=9009)

if args.torch:
    pcd_type = "torch"
elif args.mesh:
    pcd_type = "mesh"
elif args.pcd:
    pcd_type = "pcd"
else:
    pcd_type=None

panel = CustomPanel(None, server, root_path,
                    custom_camera_data_path=args.custom_camera_data_path,
                    ply_path=args.ply_path,
                    pcd_type=pcd_type,
                    task_list_path=args.task_list_path,
                    data_root_path=args.data_root_path,
                    task_type=args.task_type,
                    task_offset=args.task_offset,
                    )

import time
while True:
    time.sleep(1)
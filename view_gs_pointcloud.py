import os
import torch
import viser
import trimesh
import argparse
import numpy as np

from custom_utils.custom_panel import CustomPanel

parser = argparse.ArgumentParser()
# parser.add_argument("root_path", type=str, default=None)
parser.add_argument("ply_path", type=str)
args = parser.parse_args()

# ply_path = args.ply_path
# if args.root_path.endswith(".ply"):
#     ply_path = args.root_path
# else:
#     ply_path = os.path.join(args.root_path, "scene.ply")

points = torch.load(args.ply_path)

xyz = points[:, :3].numpy()
rgb = points[:, 3:6].numpy()

# =========================
# Start Viser server
# =========================
server = viser.ViserServer()

# =========================
# Visualize sparse points
# =========================
server.scene.add_point_cloud(
    name=f"/pcd",
    points=xyz,
    colors=rgb,
    point_size=0.01,
)

root_path = os.path.dirname(args.ply_path)

panel = CustomPanel(None, server, root_path, points=points.numpy())

import time
while True:
    time.sleep(1)
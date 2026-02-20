import os

import viser
import trimesh
import argparse
import numpy as np

from custom_utils.custom_panel import CustomPanel

parser = argparse.ArgumentParser()
# parser.add_argument("ply_path", type=str)
parser.add_argument("root_path", type=str)
args = parser.parse_args()

# ply_path = args.ply_path
ply_path = os.path.join(args.root_path, "scene.ply")

g = trimesh.load(ply_path, process=False)

if isinstance(g, trimesh.PointCloud):
    pts = np.asarray(g.vertices, dtype=np.float32)
    cols = getattr(g, "colors", None)
    if cols is not None:
        cols = np.asarray(cols, dtype=np.uint8)[:, :3]
else:
    pts = np.asarray(g.vertices, dtype=np.float32)
    cols = None

# =========================
# Start Viser server
# =========================
server = viser.ViserServer()

# =========================
# Visualize sparse points
# =========================

server.scene.add_point_cloud(
    name="/pcd",
    points=pts,
    colors=cols,
    point_size=0.01,
)

panel = CustomPanel(None, server, args.root_path)

import time
while True:
    time.sleep(1)
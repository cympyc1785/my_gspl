import os

import viser
import trimesh
import argparse
import numpy as np

from custom_utils.custom_panel import CustomPanel

parser = argparse.ArgumentParser()
# parser.add_argument("root_path", type=str, default=None)
parser.add_argument("ply_paths", nargs="+", type=str)
args = parser.parse_args()

# ply_path = args.ply_path
# if args.root_path.endswith(".ply"):
#     ply_path = args.root_path
# else:
#     ply_path = os.path.join(args.root_path, "scene.ply")

pts_batch = []
cols_batch = []
for i, ply_path in enumerate(args.ply_paths):
    g = trimesh.load(ply_path, process=False)

    # if i == 0:
    #     g.apply_scale(0.1)

    if isinstance(g, trimesh.PointCloud):
        pts = np.asarray(g.vertices, dtype=np.float32)
        cols = getattr(g, "colors", None)
        if cols is not None:
            cols = np.asarray(cols, dtype=np.uint8)[:, :3]
    else:
        pts = np.asarray(g.vertices, dtype=np.float32)
        cols = None
    
    pts_batch.append(pts)
    cols_batch.append(cols)

# =========================
# Start Viser server
# =========================
server = viser.ViserServer()

# =========================
# Visualize sparse points
# =========================
for i, (pts, cols) in enumerate(zip(pts_batch, cols_batch)):
    server.scene.add_point_cloud(
        name=f"/pcd_{i}",
        points=pts,
        colors=cols,
        point_size=0.01,
    )

root_path = os.path.dirname(args.ply_paths[0])

panel = CustomPanel(None, server, root_path)

import time
while True:
    time.sleep(1)
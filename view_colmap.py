import os

import pycolmap
import numpy as np
import viser
import argparse

from custom_utils.custom_panel import CustomPanel
from scipy.spatial.transform import Rotation

# =========================
# Load COLMAP reconstruction
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("sparse_path", default="pavilion_1")
args = parser.parse_args()

recon = pycolmap.Reconstruction(args.sparse_path)

# =========================
# Start Viser server
# =========================
server = viser.ViserServer()

# =========================
# 1) Visualize sparse points
# =========================
points = []
colors = []

for p in recon.points3D.values():
    points.append(p.xyz)
    colors.append(p.color / 255.0)

points = np.asarray(points)
colors = np.asarray(colors)

server.scene.add_point_cloud(
    name="colmap_points",
    points=points,
    colors=colors,
    point_size=0.01,
)

# self.viewer_renderer = ViewerRenderer(
#     model,
#     renderer,
#     torch.tensor(background_color, dtype=torch.float, device=self.device),
#     difix=difix,
# )

# =========================
# Keep server alive
# =========================

panel = CustomPanel(None, server, os.path.dirname(args.sparse_path), recon=recon)

import time
while True:
    time.sleep(1)

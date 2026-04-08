import os

import viser
import trimesh
import argparse
import numpy as np

from custom_utils.custom_panel import CustomPanel

parser = argparse.ArgumentParser()
parser.add_argument("root_path", type=str, default=None)
args = parser.parse_args()

# =========================
# Start Viser server
# =========================
server = viser.ViserServer()

# server.scene.add_point_cloud(
#     name=f"/pelvis",
#     points=np.array([[0.0, 0.0, 0.0]]),
#     colors=np.array([[1.0, 0.0, 0.0]]),
#     point_size=0.1,
# )

panel = CustomPanel(None, server, args.root_path)

import time
while True:
    time.sleep(1)
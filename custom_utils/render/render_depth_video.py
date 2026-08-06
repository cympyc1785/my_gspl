#!/usr/bin/env python3
"""Render a DEPTH video from a scene point cloud, one frame per camera.

Same projection/z-buffer as render_scene_video.py, but writes the nearest
camera-space z per pixel (depth) instead of color, then colorizes it with a
colormap for visualization. A single global depth range (robust percentiles
over sampled frames) is used so colors are consistent across the video.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cameras import build_cameras
from check_cut_depth import load_ply_xyz, project_depth  # reuse depth projection


def build_cam(c):
    ext = np.eye(4); ext[:3, :3] = np.asarray(c["rotation"]); ext[:3, 3] = np.asarray(c["position"])
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = c["fx"]; K[1, 1] = c["fy"]; K[0, 2] = c["cx"]; K[1, 2] = c["cy"]
    return build_cameras(ext[None].astype(np.float32), K[None])[0]


def colorize(depth, vmin, vmax, cmap, bg=255):
    """depth [H,W] (0=empty) -> BGR uint8 with colormap; empty pixels = bg."""
    valid = depth > 0
    norm = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    col = cv2.applyColorMap(u8, cmap)          # BGR
    col[~valid] = (bg, bg, bg)
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--cmap", default="turbo", choices=["turbo", "inferno", "magma", "jet", "viridis"])
    ap.add_argument("--bg", type=int, default=255, help="empty-pixel color (0=black,255=white)")
    ap.add_argument("--pmin", type=float, default=2.0)
    ap.add_argument("--pmax", type=float, default=98.0)
    args = ap.parse_args()

    CMAP = {"turbo": cv2.COLORMAP_TURBO, "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA, "jet": cv2.COLORMAP_JET,
            "viridis": cv2.COLORMAP_VIRIDIS}[args.cmap]

    xyz = torch.as_tensor(load_ply_xyz(os.path.join(args.scene_dir, "scene.ply")),
                          device="cuda", dtype=torch.float32)
    print(f"points: {xyz.shape[0]}")
    cams = json.load(open(os.path.join(args.scene_dir, "cameras.json")))[::args.stride]

    # pass 1: render all depths (keep on cpu), collect valid depths for global range
    depths = []
    pool = []
    for i, c in enumerate(cams):
        d = project_depth(xyz, build_cam(c), args.radius).cpu().numpy()
        depths.append(d)
        v = d[d > 0]
        if v.size:
            pool.append(v[::37])  # subsample for percentile pooling
        if (i + 1) % 50 == 0:
            print(f"  depth {i+1}/{len(cams)}", flush=True)
    pooled = np.concatenate(pool)
    vmin, vmax = np.percentile(pooled, args.pmin), np.percentile(pooled, args.pmax)
    print(f"depth range (p{args.pmin}-p{args.pmax}): {vmin:.2f} ~ {vmax:.2f}  (near=파랑, far=빨강/turbo)")

    # pass 2: colorize + write
    H, W = depths[0].shape
    out = args.out or os.path.join(args.scene_dir, "scene_depth_render.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for d in depths:
        vw.write(colorize(d, vmin, vmax, CMAP, args.bg))
    vw.release()
    print(f"{len(depths)} frames -> {out}  ({W}x{H})")


if __name__ == "__main__":
    main()

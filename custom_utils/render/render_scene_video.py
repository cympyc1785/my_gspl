#!/usr/bin/env python3
"""Render a scene point cloud from all its cameras and save an mp4.

Point splatting with z-buffer (nearest point per pixel), same projection
convention as render_pcd.py / render_pointcloud_multi_cam. Colors from the PLY.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from cameras import build_cameras

DEVICE = "cuda"


def load_ply_xyzrgb(path):
    NP = {"char": "i1", "uchar": "u1", "uint8": "u1", "int8": "i1",
          "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
          "float": "f4", "float32": "f4", "double": "f8"}
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        f.readline()
        n = None; fields = []; inv = False
        while True:
            t = f.readline().split()
            if not t: continue
            if t[0] == b"element":
                inv = t[1] == b"vertex"; n = int(t[2]) if inv else n
            elif t[0] == b"property" and inv:
                fields.append((t[2].decode(), NP[t[1].decode()]))
            elif t[0] == b"end_header":
                break
        dt = np.dtype([(a, b) for a, b in fields])
        data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float32)
    rgb = np.stack([data["red"], data["green"], data["blue"]], 1).astype(np.uint8)
    return xyz, rgb


@torch.no_grad()
def render(xyz, rgb, camera, radius=2, bg=0):
    pts = torch.as_tensor(xyz, device=DEVICE)
    col = torch.as_tensor(rgb, device=DEVICE, dtype=torch.int16)
    M = pts.shape[0]
    ph = torch.cat([pts, torch.ones(M, 1, device=DEVICE)], 1)
    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    H, W = int(2 * cy), int(2 * cx)
    K = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                     device=DEVICE, dtype=torch.float32)
    E = camera.world_to_camera.to(DEVICE)

    cam = ph @ E
    z = cam[:, 2]
    m = z > 1e-5
    cam, z, pidx = cam[m], z[m], torch.nonzero(m, as_tuple=False).squeeze(1)
    pix = (K @ cam.T).T
    u = torch.round(pix[:, 0] / pix[:, 2]).long()
    v = torch.round(pix[:, 1] / pix[:, 2]).long()

    r = radius
    off = torch.stack(torch.meshgrid(torch.arange(-r, r + 1, device=DEVICE),
                                     torch.arange(-r, r + 1, device=DEVICE), indexing="ij"), -1).reshape(-1, 2)
    uu = (u[:, None] + off[:, 1][None]).reshape(-1)
    vv = (v[:, None] + off[:, 0][None]).reshape(-1)
    zz = z[:, None].expand(-1, off.shape[0]).reshape(-1)
    pp = pidx[:, None].expand(-1, off.shape[0]).reshape(-1)
    ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    uu, vv, zz, pp = uu[ok], vv[ok], zz[ok], pp[ok]

    lin = vv * W + uu
    order = torch.argsort(lin.to(torch.float64) * 1e10 + zz.to(torch.float64))
    lin, pp = lin[order], pp[order]
    keep = torch.ones_like(lin, dtype=torch.bool); keep[1:] = lin[1:] != lin[:-1]
    lin, pp = lin[keep], pp[keep]

    img = torch.full((H * W, 3), bg, device=DEVICE, dtype=torch.int16)
    img[lin] = col[pp]
    return img.view(H, W, 3).clamp(0, 255).to(torch.uint8).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth camera")
    ap.add_argument("--bg", type=int, default=0)
    args = ap.parse_args()

    xyz, rgb = load_ply_xyzrgb(os.path.join(args.scene_dir, "scene.ply"))
    print(f"points: {xyz.shape[0]}")
    cams = json.load(open(os.path.join(args.scene_dir, "cameras.json")))
    cams = cams[::args.stride]

    frames = []
    for i, c in enumerate(cams):
        ext = np.eye(4); ext[:3, :3] = np.asarray(c["rotation"]); ext[:3, 3] = np.asarray(c["position"])
        K = np.eye(3, dtype=np.float32); K[0, 0] = c["fx"]; K[1, 1] = c["fy"]; K[0, 2] = c["cx"]; K[1, 2] = c["cy"]
        cam = build_cameras(ext[None].astype(np.float32), K[None])[0]
        frames.append(render(xyz, rgb, cam, args.radius, args.bg))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cams)}", flush=True)

    out = args.out or os.path.join(args.scene_dir, "scene_render.mp4")
    H, W = frames[0].shape[:2]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"{len(frames)} frames -> {out}  ({W}x{H})")


if __name__ == "__main__":
    main()

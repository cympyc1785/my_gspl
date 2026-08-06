#!/usr/bin/env python3
"""Sanity-check: does per-segment cutting of a scene change the rendered depth?

For a few sampled segments we take that segment's first camera and project two
point sets into it (nearest-z per pixel), then compare the two depth maps:
  * CUT   : /data3/.../<scene>/seg_XXXX.ply   (segment point cloud)
  * UNCUT : /data1/.../<scene>/scene.ply       (full-scene mesh vertices)

Camera intrinsics/extrinsics are identical for both, so any depth difference
comes purely from the point sets. Projection math mirrors
custom_utils/render_pcd.py::render_pointcloud_multi_cam.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from cameras import build_cameras

UNCUT = "/data1/ckd248/data/scannet_ours_sample"
CUT = "/data3/ckd248/data/scannet_ours_sample"
DEVICE = "cuda"


def load_ply_xyz(path):
    """xyz float32 [N,3] from a binary_little_endian PLY (vertex element only)."""
    NP = {"char": "i1", "uchar": "u1", "uint8": "u1", "int8": "i1",
          "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
          "int32": "i4", "uint32": "u4", "float": "f4", "float32": "f4",
          "double": "f8", "float64": "f8"}
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        assert f.readline().split()[1] == b"binary_little_endian"
        n = None
        fields = []
        in_vertex = False
        while True:
            toks = f.readline().split()
            if not toks:
                continue
            if toks[0] == b"element":
                in_vertex = toks[1] == b"vertex"
                if in_vertex:
                    n = int(toks[2])
            elif toks[0] == b"property" and in_vertex:
                fields.append((toks[2].decode(), NP[toks[1].decode()]))
            elif toks[0] == b"end_header":
                break
        dt = np.dtype([(a, b) for a, b in fields])
        data = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)
    return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)


def build_camera(cam_entry):
    R = np.asarray(cam_entry["rotation"], dtype=np.float64)   # w2c
    t = np.asarray(cam_entry["position"], dtype=np.float64)
    ext = np.eye(4); ext[:3, :3] = R; ext[:3, 3] = t
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = cam_entry["fx"]; K[1, 1] = cam_entry["fy"]
    K[0, 2] = cam_entry["cx"]; K[1, 2] = cam_entry["cy"]
    return build_cameras(ext[None].astype(np.float32), K[None])[0]


@torch.no_grad()
def project_depth(xyz, camera, point_radius=2, out_hw=None):
    """Return depth map [H,W] (0 = empty), nearest camera-space z per pixel.

    xyz may be a numpy array or an already-on-GPU torch tensor.
    out_hw=(H,W) overrides the output size (default = 2cy x 2cx).
    """
    device = DEVICE
    pts = xyz if torch.is_tensor(xyz) else torch.as_tensor(xyz, device=device, dtype=torch.float32)
    M = pts.shape[0]
    pts_h = torch.cat([pts, torch.ones(M, 1, device=device)], dim=1)  # (M,4)

    fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
    H, W = out_hw if out_hw is not None else (int(2 * cy), int(2 * cx))
    K = torch.tensor([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                     device=device, dtype=torch.float32)
    E = camera.world_to_camera.to(device)  # (4,4)

    cam_pts = pts_h @ E
    z = cam_pts[:, 2]
    m = z > 1e-5
    cam_pts, z = cam_pts[m], z[m]
    if cam_pts.shape[0] == 0:
        return torch.zeros(H, W, device=device)

    pix = (K @ cam_pts.T).T
    u = torch.round(pix[:, 0] / pix[:, 2]).long()
    v = torch.round(pix[:, 1] / pix[:, 2]).long()
    m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[m], v[m], z[m]
    if u.numel() == 0:
        return torch.zeros(H, W, device=device)

    r = point_radius
    off = torch.stack(torch.meshgrid(
        torch.arange(-r, r + 1, device=device),
        torch.arange(-r, r + 1, device=device), indexing="ij"), dim=-1).reshape(-1, 2)
    dy, dx, Ksz = off[:, 0], off[:, 1], off.shape[0]
    uu = (u[:, None] + dx[None, :]).reshape(-1)
    vv = (v[:, None] + dy[None, :]).reshape(-1)
    zz = z[:, None].expand(-1, Ksz).reshape(-1)
    ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    uu, vv, zz = uu[ok], vv[ok], zz[ok]

    lin = vv * W + uu
    # nearest z wins per pixel
    order = torch.argsort(lin.to(torch.float64) * 1e10 + zz.to(torch.float64))
    lin, zz = lin[order], zz[order]
    keep = torch.ones_like(lin, dtype=torch.bool)
    keep[1:] = lin[1:] != lin[:-1]
    lin, zz = lin[keep], zz[keep]

    depth = torch.zeros(H * W, device=device)
    depth[lin] = zz
    return depth.view(H, W)


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "scene0000_00"
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    radius = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    cams = json.load(open(os.path.join(UNCUT, scene, "cameras.json")))
    cams_by_idx = {c["idx"]: c for c in cams}
    prompts = json.load(open(os.path.join(UNCUT, scene, "prompts.json")))
    seg_keys = sorted(prompts.keys(), key=int)

    # sample n_sample segments evenly
    picks = np.linspace(0, len(seg_keys) - 1, min(n_sample, len(seg_keys))).round().astype(int)
    picks = sorted(set(picks.tolist()))

    print(f"scene={scene}  segments={len(seg_keys)}  sampling {len(picks)}  radius={radius}")
    print(f"uncut scene.ply loading...", flush=True)
    uncut_xyz = torch.as_tensor(load_ply_xyz(os.path.join(UNCUT, scene, "scene.ply")),
                                device=DEVICE, dtype=torch.float32)
    print(f"  uncut points: {uncut_xyz.shape[0]}", flush=True)

    # Aggregate over ALL cameras in each segment's frame range.
    hdr = f"{'seg':>4} {'ncam':>5} {'cut_pts':>9} {'cover%':>7} {'ovl_px(tot)':>12} " \
          f"{'meanΔ':>9} {'maxΔ':>9} {'medΔ':>8} {'<1%z':>7} {'<5%z':>7}"
    print(hdr)
    print("-" * len(hdr))

    for p in picks:
        seg = seg_keys[p]
        s, e = prompts[seg]["frame_idx"]
        cut_xyz = torch.as_tensor(load_ply_xyz(os.path.join(CUT, scene, f"seg_{int(seg):04d}.ply")),
                                  device=DEVICE, dtype=torch.float32)

        tot_ovl = 0
        sum_diff = 0.0
        max_diff = 0.0
        n_lt1 = 0
        n_lt5 = 0
        cover_num = 0
        cover_den = 0
        med_list = []
        ncam = 0
        for fidx in range(s, e):
            if fidx not in cams_by_idx:
                continue
            cam = build_camera(cams_by_idx[fidx])
            d_cut = project_depth(cut_xyz, cam, radius)
            d_unc = project_depth(uncut_xyz, cam, radius)
            vcut, vunc = d_cut > 0, d_unc > 0
            ovl = vcut & vunc
            n_ovl = int(ovl.sum())
            cover_num += int(vcut.sum()); cover_den += int(vunc.sum())
            ncam += 1
            if n_ovl == 0:
                continue
            diff = (d_cut[ovl] - d_unc[ovl]).abs()
            rel = diff / d_unc[ovl].clamp(min=1e-6)
            tot_ovl += n_ovl
            sum_diff += float(diff.sum())
            max_diff = max(max_diff, float(diff.max()))
            n_lt1 += int((rel < 0.01).sum())
            n_lt5 += int((rel < 0.05).sum())
            med_list.append(float(diff.median()))

        mean_d = sum_diff / tot_ovl if tot_ovl else 0.0
        med_d = float(np.median(med_list)) if med_list else 0.0
        cover = cover_num / cover_den * 100 if cover_den else 0.0
        print(f"{seg:>4} {ncam:>5} {cut_xyz.shape[0]:>9} {cover:>6.1f}% {tot_ovl:>12} "
              f"{mean_d:>9.4f} {max_diff:>9.4f} {med_d:>8.4f} "
              f"{n_lt1/tot_ovl*100:>6.1f}% {n_lt5/tot_ovl*100:>6.1f}%", flush=True)


if __name__ == "__main__":
    main()

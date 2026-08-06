#!/usr/bin/env python3
"""Compare original GT sensor depth vs rendered depth, for each sampled segment's
FIRST camera:

  GT      : /data1/.../<scene>/depths/<frame:05d>.npy   (uint16 mm, 0 = invalid)
  render1 : project CUT   seg_XXXX.ply at that camera   (cut point cloud)
  render2 : project UNCUT scene.ply    at that camera   (full-scene mesh vertices)

Renders are done at the GT resolution. Comparison is on pixels valid in both
GT (>0) and the render (>0). Also reports median(render/GT) to expose any global
scale between the point-cloud coordinate units and GT meters.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from check_cut_depth import UNCUT, CUT, load_ply_xyz, build_camera, project_depth


def cmp(render, gt_m, tag):
    v = (render > 0) & (gt_m > 0)
    n = int(v.sum())
    if n == 0:
        return f"{tag}: no overlap"
    r = render[v]; g = gt_m[v]
    diff = (r - g).abs()
    ratio = (r / g.clamp(min=1e-6)).median()
    rel = diff / g.clamp(min=1e-6)
    return (f"{tag}: ovl={n:>8} med(r/gt)={float(ratio):.3f} "
            f"meanΔ={float(diff.mean()):.3f} medΔ={float(diff.median()):.3f} "
            f"<10cm={float((diff<0.10).float().mean())*100:5.1f}% "
            f"<5%={float((rel<0.05).float().mean())*100:5.1f}%")


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "scene0000_00"
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    radius = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    cams = json.load(open(os.path.join(UNCUT, scene, "cameras.json")))
    cbi = {c["idx"]: c for c in cams}
    prompts = json.load(open(os.path.join(UNCUT, scene, "prompts.json")))
    segk = sorted(prompts.keys(), key=int)
    picks = sorted(set(np.linspace(0, len(segk) - 1, min(n_sample, len(segk))).round().astype(int).tolist()))

    print(f"scene={scene} segments={len(segk)} sampling {len(picks)} radius={radius}")
    print("loading uncut scene.ply ...", flush=True)
    uncut = torch.as_tensor(load_ply_xyz(os.path.join(UNCUT, scene, "scene.ply")),
                            device="cuda", dtype=torch.float32)

    for p in picks:
        seg = segk[p]
        ff = prompts[seg]["frame_idx"][0]
        cam = build_camera(cbi[ff])
        gt = np.load(os.path.join(UNCUT, scene, "depths", f"{ff:05d}.npy")).astype(np.float32) / 1000.0
        H, W = gt.shape
        gt_t = torch.as_tensor(gt, device="cuda")

        cut = torch.as_tensor(load_ply_xyz(os.path.join(CUT, scene, f"seg_{int(seg):04d}.ply")),
                              device="cuda", dtype=torch.float32)
        r1 = project_depth(cut, cam, radius, out_hw=(H, W))
        r2 = project_depth(uncut, cam, radius, out_hw=(H, W))

        print(f"\n[seg {seg} | cam {ff} | GT {W}x{H} valid={int((gt_t>0).sum())}]")
        print("  GT vs render1(cut)  ", cmp(r1, gt_t, "R1"))
        print("  GT vs render2(uncut)", cmp(r2, gt_t, "R2"))


if __name__ == "__main__":
    main()

import add_pypath
import os
import argparse
import numpy as np
import cv2
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from internal.utils.colmap import read_model, qvec2rotmat
from internal.dataparsers.estimated_depth_colmap_dataparser import EstimatedDepthColmapDataParser

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir")
parser.add_argument("--depth_dir", type=str, default=None)
parser.add_argument("--output", "-o", type=str, default=None)
parser.add_argument("--point-max-error", type=float, default=1.5)
args = parser.parse_args()

if args.depth_dir is None:
    args.depth_dir = os.path.join(args.dataset_dir, "estimated_depths")
if args.output is None:
    args.output = os.path.join(args.dataset_dir, "estimated_depth_scales.json")

sparse_model_dir = os.path.join(args.dataset_dir, "sparse")
if os.path.exists(os.path.join(sparse_model_dir, "images.bin")) is False:
    sparse_model_dir = os.path.join(sparse_model_dir, "0")

cameras, images, points3d = read_model(sparse_model_dir)

# copied from https://github.com/graphdeco-inria/hierarchical-3d-gaussians/blob/main/preprocess/make_depth_scale.py

pts_indices = np.array([points3d[key].id for key in points3d])
pts_xyzs = np.array([points3d[key].xyz for key in points3d])
pts_errors = np.array([points3d[key].error for key in points3d])
points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
points3d_error_ordered = np.zeros([pts_indices.max() + 1, ])
points3d_ordered[pts_indices] = pts_xyzs
points3d_error_ordered[pts_indices] = pts_errors


def get_scales_old(key, cameras, images, points3d_ordered, points3d_error_ordered, args):
    image_meta = images[key]

    depth_file_path = os.path.join(args.depth_dir, "{}.npy".format(image_meta.name))

    if os.path.exists(depth_file_path) is False:
        depth_file_path = os.path.join(args.depth_dir, "{}.uint16.png".format(image_meta.name))
        if os.path.exists(depth_file_path) is False:
            return None

    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images[key].point3D_ids.astype(np.int64)

    # filter out invalid 3D points
    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    # get valid 3D point indices and 2D point xy
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    # reduce outliers
    pts_errors = points3d_error_ordered[pts_idx]
    valid_errors = pts_errors < args.point_max_error
    pts_idx = pts_idx[valid_errors]
    valid_xys = valid_xys[valid_errors]

    if len(pts_idx) > 0:
        # get 3D point xyz
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    # transform from world to camera
    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2]
    invmonodepthmap = EstimatedDepthColmapDataParser.load_depth_file(depth_file_path)  # already normalized

    if invmonodepthmap is None:
        return None

    # if invmonodepthmap.ndim != 2:
    #     invmonodepthmap = invmonodepthmap[..., 0]

    # invmonodepthmap = invmonodepthmap.astype(np.float32)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    # xys inside image
    maps = (valid_xys * s).astype(np.float32)
    valid = (
            (maps[..., 0] >= 0) *
            (maps[..., 1] >= 0) *
            (maps[..., 0] < cam_intrinsic.width * s) *
            (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))

    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        # depth values from colmap
        invcolmapdepth = invcolmapdepth[valid]
        # get depth values of these 2D points from the depth map

        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]

        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name, "scale": scale, "offset": offset}

def get_scales(key, cameras, images, points3d_ordered, points3d_error_ordered, args):
    image_meta = images[key]

    depth_file_path = os.path.join(args.depth_dir, f"{image_meta.name}.npy")
    if not os.path.exists(depth_file_path):
        depth_file_path = os.path.join(args.depth_dir, f"{image_meta.name}.uint16.png")
        if not os.path.exists(depth_file_path):
            return None

    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images[key].point3D_ids.astype(np.int64)

    # filter out invalid 3D points
    mask = (pts_idx >= 0) & (pts_idx < len(points3d_ordered))
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    # reduce outliers
    pts_errors = points3d_error_ordered[pts_idx]
    valid_errors = pts_errors < args.point_max_error
    pts_idx = pts_idx[valid_errors]
    valid_xys = valid_xys[valid_errors]

    if len(pts_idx) == 0:
        return {"image_name": image_meta.name, "scale": 0.0, "offset": 0.0}

    # world -> camera
    R = qvec2rotmat(image_meta.qvec)
    pts = points3d_ordered[pts_idx]
    pts = np.dot(pts, R.T) + image_meta.tvec

    z = pts[..., 2]
    good_z = z > 0
    if good_z.sum() <= 10:
        return {"image_name": image_meta.name, "scale": 0.0, "offset": 0.0}

    pts = pts[good_z]
    valid_xys = valid_xys[good_z]
    invcolmapdepth = 1.0 / pts[..., 2]

    invmonodepthmap = EstimatedDepthColmapDataParser.load_depth_file(depth_file_path)  # already normalized
    if invmonodepthmap is None:
        return None

    # ensure 2D float32
    if invmonodepthmap.ndim == 3:
        invmonodepthmap = invmonodepthmap[..., 0]
    invmonodepthmap = invmonodepthmap.astype(np.float32)

    Hm, Wm = invmonodepthmap.shape[:2]

    # anisotropic scale from COLMAP image coords -> depth-map coords
    sx = Wm / float(cam_intrinsic.width)
    sy = Hm / float(cam_intrinsic.height)

    maps = valid_xys.astype(np.float32).copy()
    maps[:, 0] *= sx
    maps[:, 1] *= sy

    # keep points inside depth map bounds AND positive depth
    inside = (
        (maps[:, 0] >= 0) & (maps[:, 1] >= 0) &
        (maps[:, 0] < Wm) & (maps[:, 1] < Hm)
    )

    if inside.sum() <= 10 or (invcolmapdepth.max() - invcolmapdepth.min()) <= 1e-3:
        return {"image_name": image_meta.name, "scale": 0.0, "offset": 0.0}

    maps = maps[inside]
    invcolmapdepth = invcolmapdepth[inside]

   # ---- cv2.remap with point-list: chunk to keep dst.rows < SHRT_MAX ----
    x = np.clip(maps[:, 0], 0, Wm - 1 - 1e-6).astype(np.float32)
    y = np.clip(maps[:, 1], 0, Hm - 1 - 1e-6).astype(np.float32)

    MAX_ROWS = 32000  # < 32767 (safe margin)

    invmono_list = []
    for start in range(0, x.shape[0], MAX_ROWS):
        end = min(start + MAX_ROWS, x.shape[0])

        mapx = x[start:end].reshape(-1, 1)  # (chunk,1)
        mapy = y[start:end].reshape(-1, 1)  # (chunk,1)

        chunk_vals = cv2.remap(
            invmonodepthmap, mapx, mapy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1)

        invmono_list.append(chunk_vals)

    invmonodepth = np.concatenate(invmono_list, axis=0)  # (N,)

    # robust scale/offset
    t_colmap = np.median(invcolmapdepth)
    s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

    t_mono = np.median(invmonodepth)
    s_mono = np.mean(np.abs(invmonodepth - t_mono))

    if s_mono <= 1e-12:
        return {"image_name": image_meta.name, "scale": 0.0, "offset": 0.0}

    scale = float(s_colmap / s_mono)
    offset = float(t_colmap - t_mono * scale)

    return {"image_name": image_meta.name, "scale": scale, "offset": offset}

depth_param_list = []
with ThreadPoolExecutor() as tpe:
    futures = []
    for image_idx in images:
        futures.append(tpe.submit(get_scales, image_idx, cameras, images, points3d_ordered, points3d_error_ordered, args))
    for i in tqdm(as_completed(futures), total=len(futures)):
        depth_param_list.append(i.result())

# depth_param_list = []
# for image_idx in images:
#     scale = get_scales(image_idx, cameras, images, points3d_ordered, points3d_error_ordered, args)
#     depth_param_list.append(scale)

depth_params = {
    depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
    for depth_param in depth_param_list if depth_param != None
}

with open(args.output, "w") as f:
    json.dump(depth_params, f, indent=4, ensure_ascii=False)

print("Saved to `{}`".format(args.output))

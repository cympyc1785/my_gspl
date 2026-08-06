import pycolmap
import argparse
import os
import json

import numpy as np

from scipy.spatial.transform import Rotation

parser = argparse.ArgumentParser()
parser.add_argument("--video-name", default="pavilion_1")
args = parser.parse_args()

root_path = f"/data1/cympyc1785/SceneData/{args.video_name}/scene_recon"
colmap_path = f"{root_path}/colmap"
sparse_path = f"{colmap_path}/sparse"

recon = pycolmap.Reconstruction(sparse_path)

def get_colmap_camera_params(recon: pycolmap.Reconstruction):
    """
    Output:
        extrinsics: [N, 4, 4] numpy array (w2c)
        intrinsics: [N, 3, 3] numpy array (w2c)
    """
    sorted_images = sorted(
        recon.images.values(),
        key=lambda img: img.name
    )

    extrinsic_list = []
    intrinsic_list = []
    for image in sorted_images:
        extrinsic_dict = image.cam_from_world().todict()

        xyzw_w2c, t_w2c = extrinsic_dict['rotation']['quat'], extrinsic_dict['translation']
        R_w2c = Rotation.from_quat(xyzw_w2c).as_matrix()

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        fx, fy, cx, cy = image.camera.params
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        extrinsic_list.append(extrinsic)
        intrinsic_list.append(intrinsic)
    extrinsics = np.stack(extrinsic_list)
    intrinsics = np.stack(intrinsic_list)

    return extrinsics, intrinsics

def get_camera_params_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    sorted_params = sorted(
        data,
        key=lambda x: x['img_name']
    )

    extrinsic_list = []
    intrinsic_list = []
    for params in sorted_params:
        R_w2c, t_w2c = params['rotation'], params['position']

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        fx, fy, cx, cy = params['fx'], params['fy'], params['cx'], params['cy']
        intrinsic = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        extrinsic_list.append(extrinsic)
        intrinsic_list.append(intrinsic)
    extrinsics = np.stack(extrinsic_list)
    intrinsics = np.stack(intrinsic_list)

    return extrinsics, intrinsics

def get_cameras_from_npz():
        cam_params = np.load(f"{root_path}/camera_params.npz", allow_pickle=True)
        extrinsics = cam_params['extrinsics']
        intrinsics = cam_params['intrinsics']

        return extrinsics, intrinsics

extrinsics, intrinsics = get_camera_params_from_json(os.path.join(root_path, "GS", "cameras.json"))

save_path = os.path.join(root_path, f"camera_params.npz")
np.savez(save_path, 
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )

print("Saved camera params at:", save_path)
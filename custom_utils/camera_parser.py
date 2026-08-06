import os
import json

import pycolmap
import numpy as np
import torch

from enum import Enum, auto
from scipy.spatial.transform import Rotation

from .cameras import build_cameras, make_list_to_cameras, convert_coordinate
from .camera_interpolate_utils import pose_normalize


class CAMERA_FILE_TYPES(Enum):
    COLMAP=auto()
    OURS=auto()
    GS=auto()
    NPZ=auto()
    CUSTOM=auto()
    MONST3R=auto()
    TRANSFORMS=auto()

    @classmethod
    def from_any(cls, x):
        if isinstance(x, cls):
            return x

        if isinstance(x, str):
            x = x.upper()
            if x in cls.__members__:
                return cls[x]

        raise ValueError(f"Invalid CAMERA_FILE_TYPES: {x}")

def get_cameras_with_path(camera_path, camera_file_type, recon=None, ref_camera_path=None, frame_idx=0):
    """Load cameras of `camera_file_type` from `camera_path`.

    recon            : pycolmap.Reconstruction, required by COLMAP (camera_path is ignored)
    ref_camera_path  : cameras.json to normalize against, required by MONST3R
    frame_idx        : reference frame inside ref_camera_path, used by MONST3R
    """
    camera_file_type = CAMERA_FILE_TYPES.from_any(camera_file_type)

    if camera_file_type == CAMERA_FILE_TYPES.COLMAP:
        return get_cameras_from_colmap(recon)

    if not os.path.exists(camera_path):
        raise ValueError(f"Camera file doesn't exists: {camera_path}")

    if camera_file_type == CAMERA_FILE_TYPES.OURS:
        return get_cameras_from_json(camera_path)
    elif camera_file_type == CAMERA_FILE_TYPES.GS:
        return get_cameras_from_GS_json(camera_path)
    elif camera_file_type == CAMERA_FILE_TYPES.NPZ:
        return get_cameras_from_npz(camera_path)
    elif camera_file_type == CAMERA_FILE_TYPES.CUSTOM:
        return get_cameras_from_custom(camera_path)
    elif camera_file_type == CAMERA_FILE_TYPES.MONST3R:
        return get_cameras_from_monst3r(camera_path, ref_camera_path, frame_idx)
    elif camera_file_type == CAMERA_FILE_TYPES.TRANSFORMS:
        return get_cameras_from_transforms(camera_path)
    else:
        raise ValueError(f"Invalid CAMERA_FILE_TYPES: {camera_file_type}")


def get_cameras_by_type(camera_type, root_camera_path=None, scene_root_path=None, root_path=None,
                        recon=None, custom_camera_path=None,
                        use_custom_camera=False, custom_camera_data_path=None,
                        data_root_path=None, task_list=None, current_task_idx=None,
                        valid_camera_pred_types=(), task_type="pred", frame_idx=0,
                        scale_factor=1.0, sample_range=None, sample_interval=1,
                        interpolate_num=None):
    """Resolve the camera file for `camera_type`, load it, and post-process into a Cameras.

    Path resolution follows the panel: root_camera_path, overridden by
    <scene_root_path>/cameras.json, then by the task's *_transforms_pred.json when
    camera_type is one of valid_camera_pred_types, then by custom_camera_data_path
    when use_custom_camera is set.

    sample_range     : (start, end) to subsample with sample_interval, None to keep all
    interpolate_num  : output length for pose_normalize, None to skip interpolation
    """
    camera_path = root_camera_path

    if scene_root_path is not None:
        camera_path = os.path.join(scene_root_path, "cameras.json")

    print("camera type:", camera_type)

    # get pred, gt camera
    if current_task_idx is not None and camera_type in valid_camera_pred_types:
        camera_path = os.path.join(data_root_path, camera_type, "test", f"{task_list[current_task_idx]}_transforms_pred.json")

    if use_custom_camera:
        if custom_camera_data_path is None:
            print("custom camera doesn't exists", custom_camera_data_path)
            return

        with open(custom_camera_data_path, "r") as f:
            camera_data = json.load(f)

        camera_path = camera_data["camera_path"]

    ref_camera_path = os.path.join(scene_root_path, "cameras.json") if scene_root_path is not None else None

    cameras = None
    if camera_type == "colmap":
        cameras = get_cameras_from_colmap(recon)
    elif camera_type == "ours":
        cameras = get_cameras_from_json(camera_path)
    elif camera_type == "GS":
        cameras = get_cameras_from_GS_json(camera_path)
    elif camera_type == "npz":
        cameras = get_cameras_from_npz(f"{root_path}/camera_params.npz")
    elif camera_type == "custom":
        cameras = get_cameras_from_custom(custom_camera_path)
    elif camera_type == "monst3r":
        cameras = get_cameras_from_monst3r(camera_path, ref_camera_path, frame_idx)
    elif camera_type == "transforms":
        tj = os.path.join(scene_root_path, "transforms.json") if scene_root_path is not None \
            else os.path.join(root_path, "transforms.json")
        cameras = get_cameras_from_transforms(tj)
    elif camera_type == "GT":
        cameras = get_cameras_from_json(camera_path)
        if task_type in ["pred", "tartanair", "scannet"]:
            cam_list = []
            for idx in range(frame_idx, frame_idx+49):
                cam_list.append(cameras[idx])
            cameras = cam_list
    elif camera_type in valid_camera_pred_types:
        cameras = get_cameras_from_monst3r(camera_path, ref_camera_path, frame_idx)
    else:
        raise ValueError("Camera type invalid.", camera_type)

    if sample_range is not None:
        s, e = int(sample_range[0]), int(sample_range[1])
        s = max(s, 0)
        e = min(e, len(cameras))
        cameras = [cameras[idx] for idx in range(s, e, sample_interval)]
        print("After Sample", len(cameras))

    camera_list = []
    for camera in cameras:
        camera.T /= scale_factor
        camera_list.append(camera)
    cameras = make_list_to_cameras(camera_list)

    if interpolate_num is not None:
        c2w = cameras.camera_to_world
        intrinsics = cameras.intrinsics
        interpolated_c2w, interpolated_intrins = pose_normalize(c2w, intrinsics,
                                                                camera_out_seq_len=interpolate_num)
        interpolated_w2c = convert_coordinate(interpolated_c2w)
        cameras = build_cameras(interpolated_w2c, interpolated_intrins)

    return cameras


def get_cameras_from_colmap(recon: pycolmap.Reconstruction):
    if recon is None:
        print("colmap camera doesn't exists")
        return None
    extrinsics, intrinsics = get_colmap_camera_params(recon)
    cameras = build_cameras(extrinsics, intrinsics)
    return cameras

def get_cameras_from_GS_json(camera_path):
    w2c_ext, intrinsics = get_camera_params_from_json(camera_path)
    w2c_ext = convert_coordinate(w2c_ext)
    cameras = build_cameras(w2c_ext, intrinsics)
    return cameras

def get_cameras_from_json(camera_path):
    w2c_ext, intrinsics = get_camera_params_from_json(camera_path)
    cameras = build_cameras(w2c_ext, intrinsics)
    return cameras

def get_cameras_from_npz(camera_path):
    cam_params = np.load(camera_path, allow_pickle=True)
    if 'extrinsics' in cam_params.keys():
        ext = cam_params['extrinsics']
    elif 'poses' in cam_params.keys():
        ext = cam_params['poses']
    else:
        raise KeyError("No extrinsic key", cam_params)
    intrinsics = cam_params['intrinsics']

    # w2c = convert_coordinate(ext)
    w2c = ext

    cameras = build_cameras(w2c, intrinsics)
    return cameras

def get_cameras_from_custom(camera_path):
    data = np.load(camera_path, allow_pickle=True)
    if "poses" in data.keys():
        w2c_ext = data["poses"]
    elif "extrinsics" in data.keys():
        w2c_ext = data["extrinsics"]
    intrinsics = data["intrinsics"]

    cameras = build_cameras(w2c_ext, intrinsics)
    return cameras

def get_cameras_from_monst3r(camera_path, ref_camera_path, frame_idx=0):
    """Load monst3r `transform_matrix` frames and normalize them so that the first
    frame lands on frame `frame_idx` of `ref_camera_path` (the scene's cameras.json)."""
    ref_w2cs, intrinsics = get_camera_params_from_json(ref_camera_path)

    with open(camera_path, "r") as f:
        frames = json.load(f)["frames"]

    c2ws = []
    for frame in frames:
        c2w = frame["transform_matrix"]
        c2ws.append(c2w)
    c2ws = torch.tensor(c2ws)

    # S = torch.diag(torch.tensor([1., -1., -1., 1.], device=c2ws.device))
    # c2ws = S @ c2ws

    # Convert camera convention
    # OpenCV -> OpenGL (NeRF)
    c2ws[:, :3, 1:3] *= -1

    R4 = torch.tensor([
        [-1.,  0.,  0., 0.],
        [ 0., -1.,  0., 0.],
        [ 0.,  0.,  1., 0.],
        [ 0.,  0.,  0., 1.]
    ])

    c2ws = R4 @ c2ws

    # c2ws[:, :3, :3] = Rx @ c2ws[:, :3, :3]
    # c2ws[:, :3, 3:4] = Rx @ c2ws[:, :3, 3:4]

    # Normalize
    ref_c2w = torch.inverse(torch.from_numpy(ref_w2cs[frame_idx:frame_idx+1]).float())
    T = ref_c2w @ torch.inverse(c2ws[:1])
    c2ws = T @ c2ws

    w2c_ext = convert_coordinate(c2ws)

    w2c_ext = w2c_ext.numpy()
    N = w2c_ext.shape[0]
    intrinsics = intrinsics[:N]
    cameras = build_cameras(w2c_ext, intrinsics)
    return cameras

def get_cameras_from_transforms(camera_path):
    """Load cameras from a nerfstudio transforms.json in the SAME frame as cameras.json
    (the 'ours' type), which is aligned with the point cloud. cameras.json was produced as
        W2C = inv( applied_transform · c2w_opengl · diag(1,-1,-1,1) )
    so we reproduce exactly that: OpenGL c2w -> OpenCV (flip y,z axes) -> left-multiply
    the world `applied_transform` (the swap/flip nerfstudio recorded) -> invert to w2c.
    (Skipping applied_transform leaves the cameras in the un-reoriented frame -> misaligned.)"""
    with open(camera_path, "r") as f:
        data = json.load(f)
    fx, fy = float(data["fl_x"]), float(data["fl_y"])
    cx, cy = float(data["cx"]), float(data["cy"])
    frames = sorted(data["frames"], key=lambda fr: fr["file_path"])
    c2ws = torch.tensor([fr["transform_matrix"] for fr in frames], dtype=torch.float32)
    # OpenGL/Blender c2w (nerfstudio) -> OpenCV c2w (cam looks +Z, y down)
    c2ws[:, :3, 1:3] *= -1
    # world reorientation nerfstudio applied (default identity) -> cameras.json / pc frame
    at = data.get("applied_transform", None)
    if at is not None:
        AT4 = torch.eye(4, dtype=torch.float32)
        AT4[:3, :4] = torch.tensor(at, dtype=torch.float32)
        c2ws = AT4 @ c2ws
    w2c_ext = convert_coordinate(c2ws).numpy()          # c2w -> w2c
    N = w2c_ext.shape[0]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    intrinsics = np.repeat(K[None], N, axis=0)
    cameras = build_cameras(w2c_ext, intrinsics)
    return cameras


# Helpers

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

        fx, fy, cx, cy = image.camera.params[:4]
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
    if not os.path.exists(json_path):
        raise ValueError("no json camera", json_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    extrinsic_list = []
    intrinsic_list = []
    for params in data:
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

def get_camera_params_from_json_deprecated(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    sorted_params = sorted(
        data,
        key=lambda x: x['img_name']
    )

    extrinsic_list = []
    intrinsic_list = []
    for params in sorted_params:
        R_c2w, t_c2w = params['rotation'], params['position']

        R_w2c = np.array(R_c2w).T
        t_w2c = (- R_w2c @ np.array(t_c2w).reshape(-1, 1)).squeeze(-1)

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        fx, fy = params['fx'], params['fy']

        if 'cx' not in params.keys():
            scale = 4
            fx /= scale
            fy /= scale
            width, height = params['width'], params['height']
            width /= scale
            height /= scale
            cx, cy = width/2.0, height/2.0
        else:
            cx, cy = params['cx'], params['cy']

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

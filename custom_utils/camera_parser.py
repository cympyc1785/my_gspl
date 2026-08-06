import numpy as np
import torch
import pycolmap
from enum import Enum, auto

from .cameras import Camera, Cameras, build_cameras, make_list_to_cameras, convert_coordinate, camera_to_fov_quat_position

class CAMERA_FILE_TYPES(Enum):
    GS=auto()
    JSON=auto()
    NPZ=auto()
    CUSTOM=auto()
    MONST3R=auto()

    @classmethod
    def from_any(cls, x):
        if isinstance(x, cls):
            return x

        if isinstance(x, str):
            x = x.upper()
            if x in cls.__members__:
                return cls[x]

        raise ValueError(f"Invalid CAMERA_FILE_TYPES: {x}")

def get_cameras_with_path(camera_path, camera_file_type: CAMERA_FILE_TYPES, camera_intrinsic_path=None):
    if not os.path.exists(camera_path):
        raise ValueError(f"Camera file doesn't exists: {camera_path}")

    if camera_file_type == CAMERA_FILE_TYPES.GS:
        return get_cameras_from_GS_json(camera_path)
    elif camera_file_type == CAMERA_FILE_TYPES.JSON:
        return get_cameras_from_json(camera_path)
    elif camera_file_type == CAMERA_FILE_TYPES.NPZ:
        return get_cameras_from_npz(camera_path)
    elif camera_file_type == CAMERA_FILE_TYPES.CUSTOM:
        return get_cameras_from_custom(camera_path)
    elif camera_file_type == CAMERA_FILE_TYPES.MONST3R:
        return get_cameras_from_monst3r(camera_path)
    else:
        raise ValueError(f"Invalid CAMERA_FILE_TYPES: {camera_file_type}")


def get_cameras_from_colmap(recon: pycolmap.Reconstruction):
    if recon is None:
        print("colmap camera doesn't exists")
        return None
    extrinsics, intrinsics = get_colmap_camera_params(self.recon)
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

def get_cameras_from_monst3r(camera_path):
    ref_w2cs, intrinsics = get_camera_params_from_json(camera_path)

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
    ref_c2w = torch.inverse(torch.from_numpy(ref_w2cs[self.frame_idx:self.frame_idx+1]).float())
    T = ref_c2w @ torch.inverse(c2ws[:1])
    c2ws = T @ c2ws

    w2c_ext = convert_coordinate(c2ws)

    w2c_ext = w2c_ext.numpy()
    N = w2c_ext.shape[0]
    intrinsics = intrinsics[:N]
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
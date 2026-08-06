import os

import numpy as np
import torch
import trimesh
import open3d as o3d

from enum import Enum, auto


class PCD_FILE_TYPES(Enum):
    COLMAP=auto()
    PLY=auto()
    NPY=auto()
    TORCH=auto()
    MESH=auto()
    PCD=auto()
    TARTANAIR=auto()

    @classmethod
    def from_any(cls, x):
        if isinstance(x, cls):
            return x

        if isinstance(x, str):
            x = x.upper()
            if x in cls.__members__:
                return cls[x]

        raise ValueError(f"Invalid PCD_FILE_TYPES: {x}")

def get_pcd_with_path(pcd_path, pcd_file_type, recon=None):
    """Load (points, colors) of `pcd_file_type` from `pcd_path`.

    recon : pycolmap.Reconstruction, required by COLMAP (pcd_path is ignored)
    """
    pcd_file_type = PCD_FILE_TYPES.from_any(pcd_file_type)

    if pcd_file_type == PCD_FILE_TYPES.COLMAP:
        return get_pcd_from_colmap(recon)

    if not os.path.exists(pcd_path):
        raise ValueError(f"Point cloud file doesn't exists: {pcd_path}")

    if pcd_file_type == PCD_FILE_TYPES.PLY:
        return get_pcd_from_ply(pcd_path)
    elif pcd_file_type == PCD_FILE_TYPES.NPY:
        return get_pcd_from_npy(pcd_path)
    elif pcd_file_type == PCD_FILE_TYPES.TORCH:
        return get_pcd_from_torch(pcd_path)
    elif pcd_file_type == PCD_FILE_TYPES.MESH:
        return get_pcd_from_mesh_obj(pcd_path)
    elif pcd_file_type == PCD_FILE_TYPES.PCD:
        return get_pcd_from_pcd(pcd_path)
    elif pcd_file_type == PCD_FILE_TYPES.TARTANAIR:
        return get_pcd_from_tartanair(pcd_path)
    else:
        raise ValueError(f"Invalid PCD_FILE_TYPES: {pcd_file_type}")


def get_pcd_by_type(pcd_type, root_path=None, pcd_path=None, recon=None, sample_num=None):
    """Resolve the point cloud file for `pcd_type`, load it, and optionally subsample.

    pcd_path falls back to <root_path>/scene.ply when not given.
    sample_num : number of points to randomly keep, None to keep all
    """
    default_pcd_path = os.path.join(root_path, "scene.ply") if root_path is not None else None
    if pcd_path is None:
        pcd_path = default_pcd_path

    print("loading", pcd_path)

    if pcd_type == "colmap":
        points, colors = get_pcd_from_colmap(recon)
    elif pcd_type == "ply":
        points, colors = get_pcd_from_ply(pcd_path)
    elif pcd_type == "npy":
        points, colors = get_pcd_from_npy(pcd_path)
    elif pcd_type == "torch":
        points, colors = get_pcd_from_torch(pcd_path)
    elif pcd_type == "mesh":
        points, colors = get_pcd_from_mesh_obj(pcd_path)
    elif pcd_type == "pcd":
        points, colors = get_pcd_from_pcd(pcd_path)
    elif pcd_type == "tartanair":
        points, colors = get_pcd_from_tartanair(pcd_path)
    else:
        raise ValueError("Invalid pcd type", pcd_type)

    if sample_num is not None:
        idx = torch.randint(0, points.shape[0], (sample_num,))
        points = points[idx]
        colors = colors[idx]

    return points, colors


def get_pcd_from_colmap(recon):
    points = []
    colors = []
    for p in recon.points3D.values():
        points.append(p.xyz)
        colors.append(p.color / 255.0)
    points = np.asarray(points)
    colors = np.asarray(colors)

    return points, colors

def get_pcd_from_ply(pcd_path):
    g = trimesh.load(pcd_path, process=False)
    if isinstance(g, trimesh.PointCloud):
        points = np.asarray(g.vertices, dtype=np.float32)
        colors = getattr(g, "colors", None)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.uint8)[:, :3]
    else:
        points = np.asarray(g.vertices, dtype=np.float32)
        colors = None

    return points, colors

def get_pcd_from_npy(pcd_path):
    points = np.load(pcd_path)
    colors = np.repeat([[0., 0., 1.0]], len(points), axis=0)

    return points, colors

def get_pcd_from_torch(pcd_path):
    points = torch.load(pcd_path)
    pts = points[:, :3].numpy()
    cols = points[:, 3:6].numpy()

    return pts, cols

def get_pcd_from_mesh_obj(pcd_path):
    mesh = trimesh.load(pcd_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    points, face_idx = trimesh.sample.sample_surface(mesh, 100000)
    colors = mesh.visual.face_colors[face_idx][:, :3] # remove alpha

    print(type(points))
    print(points.shape)
    print(colors.shape)

    return points, colors

def get_pcd_from_pcd(pcd_path):
    pcd_o3d = o3d.io.read_point_cloud(pcd_path)

    points = np.asarray(pcd_o3d.points)
    colors = np.asarray(pcd_o3d.colors) # 0~1 사이 값

    # g = trimesh.points.PointCloud(vertices=points, colors=colors)
    return points, colors

def get_pcd_from_tartanair(pcd_path):
    pcd_o3d = o3d.io.read_point_cloud(pcd_path)

    points = np.asarray(pcd_o3d.points)
    colors = np.asarray(pcd_o3d.colors) # 0~1 사이 값

    # Rotation for tartan air
    P = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float64)
    points = (P @ points.T).T

    # g = trimesh.points.PointCloud(vertices=points, colors=colors)
    return points, colors

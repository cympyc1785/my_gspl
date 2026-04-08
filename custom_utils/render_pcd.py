import os
import torch
import json
import numpy as np
import open3d as o3d
import imageio
from tqdm import tqdm

from cameras import Cameras, Camera, build_cameras, make_list_to_cameras

def save_video_imageio(images, save_path, fps=30):
    """
    images: (N, H, W, 3), torch.Tensor or np.ndarray
    save_path: e.g. 'output.mp4'
    """

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    # float [0,1] -> uint8 [0,255]
    if images.dtype != np.uint8:
        images = np.clip(images, 0.0, 1.0)
        images = (images * 255).astype(np.uint8)

    writer = imageio.get_writer(
        save_path,
        fps=fps,
        codec="libx264",
        quality=8
    )

    for i in range(images.shape[0]):
        writer.append_data(images[i])

    writer.close()

def render_pointcloud_multi_cam(points, cameras: Cameras, point_radius=2):
    if points is None:
        print("no points")
        return

    points_torch = torch.as_tensor(points, device="cuda", dtype=torch.float32)
    points_xyz = points_torch[:, :3]       # (M, 3)
    points_rgb = points_torch[:, 3:]       # (M, 3)
    device = points_xyz.device
    M = points_xyz.shape[0]

    points_h = torch.cat(
        [points_xyz, torch.ones(M, 1, device=device, dtype=torch.float32)],
        dim=1
    )  # (M, 4)

    extrinsics = cameras.world_to_camera.to(device)   # (C, 4, 4)
    num_cams = len(cameras)

    images = []

    for cam_idx, camera in tqdm(enumerate(cameras)):
        fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy

        H = int(2 * cy)
        W = int(2 * cx)

        K = torch.tensor([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0,  0,  1, 0],
            [0,  0,  0, 1],
        ], device=device, dtype=torch.float32)

        E = extrinsics[cam_idx]  # (4, 4)

        # world -> camera
        cam_pts = points_h @ E   # (M, 4)

        z = cam_pts[:, 2]
        valid = z > 1e-5
        cam_pts = cam_pts[valid]
        z = z[valid]
        pidx = torch.nonzero(valid, as_tuple=False).squeeze(1)

        if cam_pts.shape[0] == 0:
            images.append(torch.zeros(H, W, 3, device=device, dtype=torch.float32))
            continue

        # project
        pix = (K @ cam_pts.T).T
        u = torch.round(pix[:, 0] / pix[:, 2]).long()   # (N,)
        v = torch.round(pix[:, 1] / pix[:, 2]).long()   # (N,)

        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[valid]
        v = v[valid]
        z = z[valid]
        pidx = pidx[valid]

        if u.numel() == 0:
            images.append(torch.zeros(H, W, 3, device=device, dtype=torch.float32))
            continue

        # -----------------------------------------
        # batchified square splat
        # -----------------------------------------
        r = point_radius
        offsets = torch.stack(
            torch.meshgrid(
                torch.arange(-r, r + 1, device=device),
                torch.arange(-r, r + 1, device=device),
                indexing="ij",
            ),
            dim=-1
        ).reshape(-1, 2)   # (K, 2), [:,0]=dy, [:,1]=dx

        dy = offsets[:, 0]   # (K,)
        dx = offsets[:, 1]   # (K,)
        Ksz = offsets.shape[0]

        # (N, K)
        uu = u[:, None] + dx[None, :]
        vv = v[:, None] + dy[None, :]

        valid_splat = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)

        uu = uu.reshape(-1)
        vv = vv.reshape(-1)
        valid_splat = valid_splat.reshape(-1)

        # repeat point attrs for each offset
        z_rep = z[:, None].expand(-1, Ksz).reshape(-1)
        pidx_rep = pidx[:, None].expand(-1, Ksz).reshape(-1)

        uu = uu[valid_splat]
        vv = vv[valid_splat]
        z_rep = z_rep[valid_splat]
        pidx_rep = pidx_rep[valid_splat]

        lin_idx = vv * W + uu   # (Ns,)

        # same pixel -> nearest depth only
        sort_key = lin_idx.to(torch.float64) * 1e10 + z_rep.to(torch.float64)
        order = torch.argsort(sort_key)

        lin_idx = lin_idx[order]
        z_rep = z_rep[order]
        pidx_rep = pidx_rep[order]

        keep = torch.ones_like(lin_idx, dtype=torch.bool)
        keep[1:] = lin_idx[1:] != lin_idx[:-1]

        lin_idx = lin_idx[keep]
        pidx_rep = pidx_rep[keep]

        img = torch.zeros((H * W, 3), device=device, dtype=points_rgb.dtype)
        img[lin_idx] = points_rgb[pidx_rep].to(img.dtype)

        img = img.view(H, W, 3)
        images.append(img)
        # images.append(img.detach().cpu().numpy())

    renders = torch.stack(images).detach().cpu().numpy()
    # renders = images
    print("Rendering Done")
    return renders

def get_pcd_from_pcd(pcd_path):
    pcd_o3d = o3d.io.read_point_cloud(pcd_path)
    
    points = np.asarray(pcd_o3d.points)
    colors = np.asarray(pcd_o3d.colors) # 0~1 사이 값

    P = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float64)
    points = (P @ points.T).T
    
    # g = trimesh.points.PointCloud(vertices=points, colors=colors)
    return points, colors

def get_cameras_from_json(camera_path):
    w2c_ext, intrinsics = get_camera_params_from_json(camera_path)
    # w2c_ext = convert_coordinate(w2c_ext)
    cameras = build_cameras(w2c_ext, intrinsics)
    return cameras

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

if __name__ == "__main__":

    points, colors = get_pcd_from_pcd("/data1/cympyc1785/SceneData/tartanair/tartanair-v2/source_scenes/AbandonedFactory/AbandonedFactory_rgb.pcd")

    points = torch.from_numpy(np.concatenate([points, colors], axis=1)).to("cuda")

    cameras = get_cameras_from_json("/data1/cympyc1785/gaussian-splatting-lightning/SceneData/tartanair/tartanair-v2/w2c.json")

    cameras = [cameras[i] for i in range(0, 1000, 1)]
    cameras = make_list_to_cameras(cameras)

    renders = render_pointcloud_multi_cam(points, cameras)

    save_video_imageio(renders, "tartan.mp4")
    # imageio.mimsave(renders, "tartan.mp4", fps=10)

import os

import pycolmap
import numpy as np
import torch
import viser
import imageio
import json
import trimesh

from tqdm import tqdm
from .cameras import Camera, Cameras, build_cameras, make_list_to_cameras
from scipy.spatial.transform import Rotation

from .render_frustum import add_frustum_spline

class CustomPanel:
    def __init__(self, viewer, server, root_path, renderer=None, recon=None):
        self.server = server
        self.root_path = root_path
        self.renderer = renderer
        self.recon = recon
        self.scale_factor = 1.0
        if viewer is not None:
            self.scale_factor = viewer.scale_factor
            self.custom_camera_path = viewer.custom_camera_path

        self.valid_camera_types = ["GS", "colmap", "npz", "vae", "custom"]

        self.is_frustum_visualized = {}
        for cam_type in self.valid_camera_types:
            self.is_frustum_visualized[cam_type] = False
        self.frustum_handles = {}
        for cam_type in self.valid_camera_types:
            self.frustum_handles[cam_type] = []

        self.pcd = None

        self._setup_vis_folder()
        self.server.on_client_connect(self._handle_new_client)
    
    def _setup_vis_folder(self):
        with self.server.gui.add_folder("Viz"):
            self.camera_type = self.server.gui.add_dropdown(
                "Camera Type",
                self.valid_camera_types,
                initial_value="GS",
            )

            # self.w2c_coord_checkbox = self.server.gui.add_checkbox(
            #     "w2c Camera Coordinate",
            #     initial_value=True,
            # )

            self.fps = self.server.gui.add_number(
                "FPS",
                min=1,
                max=60,
                step=1,
                initial_value=10,
            )

            render_by_cam_button = self.server.gui.add_button(
                    "Render by Cam",
                    color="purple",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Save smplx scale",
                )
            @render_by_cam_button.on_click
            def _(event: viser.GuiEvent) -> None:
                try:
                    cameras = self.get_cameras_by_type()
                    self.render(cameras, fps=self.fps.value)

                except Exception as e:
                    print("Error:", e)

            render_by_client_button = self.server.gui.add_button(
                    "Render by Client",
                    color="Brown",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Save smplx scale",
                )
            @render_by_client_button.on_click
            def _(event: viser.GuiEvent) -> None:
                try:
                    cameras = self.get_cameras_by_type()
                    self.render_with_client(event.client, cameras, self.root_path)
                except Exception as e:
                    print("Error:", e)

            self.frustum_range = self.server.gui.add_vector2(
                    "Frustum Range",
                    min=(0, 1),
                    max=(1000, 1000),
                    step=1,
                    initial_value=(0, 50),
                )

            self.frustum_interval = self.server.gui.add_number(
                    "Frustum Interval",
                    min=1,
                    max=50,
                    step=1,
                    initial_value=5,
                )

            self.frustum_scale = self.server.gui.add_number(
                    "Frustum Scale",
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    initial_value=0.1,
                )

            self.frustum_thickness = self.server.gui.add_number(
                    "Frustum Thickness",
                    min=0.1,
                    max=10.0,
                    step=0.1,
                    initial_value=3,
                )

            self.frustum_start_color = self.server.gui.add_vector3(
                    "Frustum Start Color",
                    min=(0, 0, 0),
                    max=(1, 1, 1),
                    step=0.01,
                    initial_value=(1.0, 0.0, 0.0),
                )

            self.frustum_end_color = self.server.gui.add_vector3(
                    "Frustum End Color",
                    min=(0, 0, 0),
                    max=(1, 1, 1),
                    step=0.01,
                    initial_value=(0.0, 0.0, 1.0),
                )

            vis_camera_frustum_button = self.server.gui.add_button(
                    "Visualize Camera Frustum",
                    color="Green",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Visualize camera trajectory as point clouds.",
                )
            @vis_camera_frustum_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                camera_type = self.camera_type.value

                self.is_frustum_visualized[camera_type] = not self.is_frustum_visualized[camera_type]                
                cameras = self.get_cameras_by_type()

                fov, quat, position = camera_to_fov_quat_position(cameras[0])

                for handle_dict in self.frustum_handles[camera_type]:
                    for key, handle in handle_dict.items():
                        handle.remove()
                self.frustum_handles[camera_type].clear()

                if not self.is_frustum_visualized[camera_type]:
                    return

                f_min, f_max = self.frustum_range.value
                f_min = int(f_min)
                f_max = int(f_max)
                cam_min = f_min
                cam_max = min(f_max, len(cameras))

                start_color = np.array(self.frustum_start_color.value)
                end_color = np.array(self.frustum_end_color.value)
                t = np.linspace(0, 1, (cam_max - cam_min))[:, None]  # (N, 1)
                colors = (1 - t) * start_color + t * end_color  # (N, 3)

                for i in range(cam_min, cam_max, self.frustum_interval.value):
                    camera = cameras[i]
                    fov, quat, position = camera_to_fov_quat_position(camera)

                    handle = add_frustum_spline(
                        scene=client.scene,
                        name=f"camera_{i}_frustum_{camera_type}",
                        fov_y=fov,
                        aspect=camera.cx / camera.cy,
                        wxyz=quat,
                        position=position,
                        far=1.0,
                        scale=self.frustum_scale.value,
                        thickness=self.frustum_thickness.value,
                        color=colors[i],
                    )

                    self.frustum_handles[camera_type].append(handle)

            self.pcd_type = self.server.gui.add_dropdown(
                "Point Cloud Type",
                ["colmap", "ply", "npy"],
                initial_value="colmap",
            )

            vis_point_cloud_button = self.server.gui.add_button(
                "Visualize Point Cloud",
                color="Red",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Visualize Colmap.",
            )
            @vis_point_cloud_button.on_click
            def _(event: viser.GuiEvent) -> None:
                if self.pcd is not None:
                    self.pcd.remove()
                    self.pcd = None
                    return
                
                if self.pcd_type.value == "colmap":
                    points, colors = self.get_pcd_from_colmap()
                elif self.pcd_type.value == "ply":
                    points, colors = self.get_pcd_from_ply()
                elif self.pcd_type.value == "npy":
                    points, colors = self.get_pcd_from_npy()

                self.pcd = self.server.scene.add_point_cloud(
                    name="/point_cloud",
                    points=points,
                    colors=colors,
                    point_size=0.01,
                )

    def get_cameras_by_type(self):
        camera_type = self.camera_type.value

        cameras = None
        if camera_type == "colmap":
            cameras = self.get_cameras_from_colmap()
        elif camera_type == "GS":
            cameras = self.get_cameras_from_json()
        elif camera_type == "npz":
            cameras = self.get_cameras_from_npz()
        elif camera_type == "vae":
            cameras = self.get_cameras_from_vae()
        elif camera_type == "custom":
            cameras = self.get_cameras_from_custom()
        else:
            raise ValueError("Camera type invalid.", camera_type)
        
        camera_list = []
        for camera in cameras:
            camera.T /= self.scale_factor
            camera_list.append(camera)
        cameras = make_list_to_cameras(camera_list)

        return cameras

    def get_cameras_from_colmap(self):
        if self.recon is None:
            print("colmap camera doesn't exists")
            return None
        extrinsics, intrinsics = get_colmap_camera_params(self.recon)
        cameras = build_cameras(extrinsics, intrinsics)
        return cameras

    def get_cameras_from_json(self):
        w2c_ext, intrinsics = get_camera_params_from_json(os.path.join(self.root_path, "cameras.json"))
        # w2c_ext = convert_coordinate(w2c_ext)
        cameras = build_cameras(w2c_ext, intrinsics)
        return cameras

    def get_cameras_from_npz(self):
        cam_params = np.load(f"{self.root_path}/camera_params.npz", allow_pickle=True)
        if 'extrinsics' in cam_params.keys():
            ext = cam_params['extrinsics']
        elif 'poses' in cam_params.keys():
            ext = cam_params['poses']
        else:
            raise KeyError("No extrinsic key", cam_params)
        intrinsics = cam_params['intrinsics']

        w2c = convert_coordinate(ext)
        # w2c = ext

        cameras = build_cameras(w2c, intrinsics)
        return cameras
    
    def get_cameras_from_vae(self):
        _, intrinsics = get_camera_params_from_json(os.path.join(self.root_path, "cameras.json"))
        w2c_ext = np.load(f"{self.root_path}/vae4.npy", allow_pickle=True)
        N = w2c_ext.shape[0]
        intrinsics = intrinsics[:N]
        cameras = build_cameras(w2c_ext, intrinsics)
        return cameras
    
    def get_cameras_from_custom(self):
        if self.custom_camera_path is None:
            print("custom camera doesn't exists", self.custom_camera_path)
            return
        
        data = np.load(self.custom_camera_path, allow_pickle=True)
        if "poses" in data.keys():
            w2c_ext = data["poses"]
        elif "extrinsics" in data.keys():
            w2c_ext = data["extrinsics"]
        intrinsics = data["intrinsics"]

        cameras = build_cameras(w2c_ext, intrinsics)
        return cameras
    
    def get_cameras_from_monst3r(self):
        _, intrinsics = get_camera_params_from_json(os.path.join(self.root_path, "cameras.json"))

        with open("/data1/cympyc1785/caption/GenDoP/dataset/DATA/1_0000/shot_0070_transforms_cleaning.json", "r") as f:
            frames = json.load(f)["frames"]
        
        c2ws = []
        for frame in frames:
            c2w = frame["transform_matrix"]
            c2ws.append(c2w)
        c2ws = torch.tensor(c2ws)

        # S = torch.diag(torch.tensor([1., -1., -1., 1.], device=c2ws.device))
        # c2ws = S @ c2ws

        # Convert camera convention
        c2ws[:, :3, 1:3] *= -1

        # # Normalize
        # ref_w2c = torch.inverse(c2ws[:1])
        # c2ws = ref_w2c.repeat(c2ws.shape[0], 1, 1) @ c2ws

        w2c_ext = torch.linalg.inv(c2ws)

        w2c_ext = w2c_ext.numpy()
        N = w2c_ext.shape[0]
        intrinsics = intrinsics[:N]
        cameras = build_cameras(w2c_ext, intrinsics)
        return cameras

    def get_pcd_from_colmap(self):
        points = []
        colors = []
        for p in self.recon.points3D.values():
            points.append(p.xyz)
            colors.append(p.color / 255.0)
        points = np.asarray(points)
        colors = np.asarray(colors)

        return points, colors

    def get_pcd_from_ply(self):
        ply_path = os.path.join(self.root_path, "scene.ply")
        g = trimesh.load(ply_path, process=False)
        if isinstance(g, trimesh.PointCloud):
            points = np.asarray(g.vertices, dtype=np.float32)
            colors = getattr(g, "colors", None)
            if colors is not None:
                colors = np.asarray(colors, dtype=np.uint8)[:, :3]
        else:
            points = np.asarray(g.vertices, dtype=np.float32)
            colors = None
        
        return points, colors

    def get_pcd_from_npy(self):
        point_cloud_path = os.path.join(self.root_path, "point_cloud.npy")
        points = np.load(point_cloud_path)
        colors = np.repeat([[0., 0., 1.0]], len(points), axis=0)

        return points, colors

    def render_with_client(self, client, cameras: Cameras, out_dir):
        images = []
        print("Rendering...")
        for camera in tqdm(cameras):
            fov, quat, t = camera_to_fov_quat_position(camera)
            img = client.get_render(
                height=camera.height.item(),
                width=camera.width.item(),
                wxyz=quat,
                position=t,
                fov=fov,
            )
            images.append(img)

        save_path = os.path.join(out_dir, "colmap_scene_viz.mp4")
        imageio.mimsave(save_path, images, fps=10)
        print("Rendering done. Saved to", save_path)
    
    def render(self, cameras: Cameras, render_mode="rgb", optimizable=False, fps=10):
        print("Setting Render")
        print(self.renderer.renderer)
        available_outputs = self.renderer.renderer.get_available_outputs()

        output_type_info = available_outputs.get(render_mode, None)
        if output_type_info is None:
            return
        if render_mode == "exp_depth" or "n_contribs" or "alpha":
            output_type_info.visualizer = self.renderer.no_processing
        
        self.renderer._set_output_type(render_mode, output_type_info)
        renders = []
        print("Rendering...")
        for camera in tqdm(cameras):
            if render_mode == "rgb":
                if optimizable:
                    render = self.renderer.get_outputs(camera.to_device("cuda")).permute(1, 2, 0) # (H, W, 3)
                     # Clipping
                    render = torch.clamp(render, 0.0, 1.0)
                else:
                    render = self.renderer.get_outputs(camera.to_device("cuda")).detach().cpu().numpy()
                    # Clipping
                    render = np.clip(render, 0.0, 1.0)
                    render = (render * 255).astype(np.uint8).transpose(1, 2, 0)
            elif render_mode == "exp_depth":
                if optimizable:
                    render = self.renderer.get_outputs(camera.to_device("cuda")).permute(1, 2, 0)[:, :, 0] # (H, W)
                else:
                    render = self.renderer.get_outputs(camera.to_device("cuda")).detach().cpu().numpy()
                    depth = render.transpose(1, 2, 0) # (H, W, 3)
                    depth_map = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                    depth_map = cv2.applyColorMap((depth_map.clip(0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    render = np.concatenate([depth[:, :, 0:1], depth_map], axis=-1) # (H, W, 4)
            elif render_mode == "n_contribs" or render_mode == "alpha":
                if optimizable:
                    render = self.renderer.get_outputs(camera.to_device("cuda"))[0, :, :] # (H, W)
                else:
                    render = self.renderer.get_outputs(camera.to_device("cuda"))[0, :, :].detach().cpu().numpy()  # (H, W)
            renders.append(render)

        save_path = os.path.join(self.root_path, "scene_viz.mp4")
        imageio.mimsave(save_path, renders, fps=fps)

        if optimizable is False:
            renders = np.array(renders)
        else:
            renders = torch.stack(renders)
        
        print("Rendering done. Saved to", save_path)

        return renders

    def _handle_new_client(self, client):
        if True:
            return
        camera_type = 'vae'

        self.is_frustum_visualized[camera_type] = not self.is_frustum_visualized[camera_type]                
        cameras = self.get_cameras_from_monst3r()
    
        if cameras is None:
            raise ValueError("Camera Not Found")

        fov, quat, position = camera_to_fov_quat_position(cameras[0])

        for handle_dict in self.frustum_handles[camera_type]:
            for key, handle in handle_dict.items():
                handle.remove()
        self.frustum_handles[camera_type].clear()

        if not self.is_frustum_visualized[camera_type]:
            return

        f_min, f_max = self.frustum_range.value
        f_min = int(f_min)
        f_max = int(f_max)
        cam_min = f_min
        cam_max = min(f_max, len(cameras))

        start_color = np.array(self.frustum_start_color.value)
        end_color = np.array(self.frustum_end_color.value)
        t = np.linspace(0, 1, (cam_max - cam_min))[:, None]  # (N, 1)
        colors = (1 - t) * start_color + t * end_color  # (N, 3)

        for i in range(cam_min, cam_max, self.frustum_interval.value):
            camera = cameras[i]
            fov, quat, position = camera_to_fov_quat_position(camera)

            handle = add_frustum_spline(
                scene=client.scene,
                name=f"camera_{i}_frustum_{camera_type}",
                fov_y=fov,
                aspect=camera.cx / camera.cy,
                wxyz=quat,
                position=position,
                far=1.0,
                scale=0.01,
                thickness=self.frustum_thickness.value,
                color=colors[i],
            )

            # if i == 0:
            #     self.server.add_frame(
            #         name="first_cam_coord",
            #         position=position,
            #         wxyz=quat,
            #         axes_length=0.1,
            #         axes_radius=0.01,
            #         scale=0.01
            #     )

            self.frustum_handles[camera_type].append(handle)


def camera_to_fov_quat_position(camera: Camera):
    c2w = torch.linalg.inv(camera.world_to_camera.transpose(-1, -2)).detach().cpu().numpy()
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    focal = [camera.fx, camera.fy]
    princpt = [camera.cx, camera.cy]
    r = Rotation.from_matrix(R.tolist())
    quat = r.as_quat() # (x, y, z, w)
    quat = np.array([quat[3], quat[0], quat[1], quat[2]]) # (w, x, y, z)
    fov_radians = 2 * np.arctan(2 * princpt[1].cpu() / (2 * focal[1].cpu()))
    return fov_radians, quat, t

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

def convert_coordinate(extrinsic):
    if isinstance(extrinsic, np.ndarray):
        device = 'cpu'
    elif isinstance(extrinsic, torch.Tensor):
        device = extrinsic.device
    
    ext_tensor = torch.tensor(extrinsic)
    R = ext_tensor[..., :3, :3]
    t = ext_tensor[..., :3, 3]

    R_inv = R.transpose(-1, -2)
    t_inv = (-R_inv @ t.unsqueeze(-1)).squeeze(-1)

    ext_inv = torch.zeros_like(ext_tensor, device=device)
    ext_inv[..., :3, :3] = R_inv
    ext_inv[..., :3, 3] = t_inv
    ext_inv[..., 3, 3] = 1

    if isinstance(extrinsic, np.ndarray):
        ext_inv = ext_inv.numpy()

    return ext_inv


import traceback
import datetime
import os.path

import torch
import numpy as np
import viser
import viser.transforms as vtf
import re

import open3d as o3d
from .cameras import Camera, Cameras
from scipy.spatial.transform import Rotation
import imageio
import os
import time
from tqdm import tqdm
import threading
import json
import pycolmap
import cv2

from .render_frustum import add_frustum_spline

class EditPanel:
    def __init__(
            self,
            server: viser.ViserServer,
            viewer,
            tab,
    ):
        self.server = server
        self.viewer = viewer
        self.tab = tab

        if hasattr(viewer, "root_path"):
            self.root_path = viewer.root_path

        self.renderer = self.viewer.viewer_renderer

        self.is_frustum_visualized = {"colmap": False, "GS": False}
        self.frustum_handles = {"colmap": [], "GS": []}

        self.is_colmap_visualized = False
        self.is_point_cloud_visualized = False

        self._setup_point_cloud_folder()
        self._setup_gaussian_edit_folder()
        self._setup_save_gaussian_folder()
        self._setup_custom_folder()

    def _setup_point_cloud_folder(self):
        server = self.server
        with self.server.gui.add_folder("Point Cloud"):
            self.show_point_cloud_checkbox = server.gui.add_checkbox(
                "Show Point Cloud",
                initial_value=False,
            )
            self.point_cloud_color = server.gui.add_vector3(
                "Point Color",
                min=(0, 0, 0),
                max=(255, 255, 255),
                step=1,
                initial_value=(0, 255, 255),
            )
            self.point_size = server.gui.add_number(
                "Point Size",
                min=0.,
                initial_value=0.01,
                step=0.001,
            )
            self.point_sparsify = server.gui.add_number(
                "Point Sparsify",
                min=1,
                initial_value=10,
            )

            self.pcd = None

            @self.show_point_cloud_checkbox.on_update
            @self.point_cloud_color.on_update
            @self.point_size.on_update
            @self.point_sparsify.on_update
            def _(event: viser.GuiEvent):
                with self.server.atomic():
                    self._update_pcd()

    def _resize_grid(self, idx):
        exist_grid = self.grids[idx][0]
        exist_grid.remove()
        self.grids[idx][0] = self.server.add_grid(
            "/grid/{}".format(idx),
            width=self.grids[idx][2].value[0],
            height=self.grids[idx][2].value[1],
            wxyz=self.grids[idx][1].wxyz,
            position=self.grids[idx][1].position,
        )
        self._update_scene()

    def _setup_gaussian_edit_folder(self):
        server = self.server

        self.edit_histories = []

        with server.gui.add_folder("Edit"):
            # initialize a list to store panel(grid)'s information
            self.grids: dict[int, list[
                viser.MeshHandle,
                viser.TransformControlsHandle,
                viser.GuiInputHandle,
            ]] = {}
            self.grid_idx = 0

            add_grid_button = server.gui.add_button("Add Panel")
            self.delete_gaussians_button = server.gui.add_button(
                "Delete Gaussians",
                color="red",
            )

        self.grid_folders = {}

        # create panel(grid)
        def new_grid(idx, event):
            with self.server.gui.add_folder("Grid {}".format(idx)) as folder:
                self.grid_folders[idx] = folder

                # TODO: add height
                grid_size = server.gui.add_vector2("Size", initial_value=(10., 10.), min=(0., 0.), step=0.01)

                grid = server.add_grid(
                    "/grid/{}".format(idx),
                    height=grid_size.value[0],
                    width=grid_size.value[1],
                    position=event.client.camera.look_at,
                )
                grid_transform = server.add_transform_controls(
                    "/grid_transform_control/{}".format(idx),
                    wxyz=grid.wxyz,
                    position=grid.position,
                )

                # resize panel on size value changed
                @grid_size.on_update
                def _(event: viser.GuiEvent):
                    with event.client.atomic():
                        self._resize_grid(idx)

                # handle panel deletion
                grid_delete_button = server.gui.add_button("Delete")

                @grid_delete_button.on_click
                def _(_):
                    with server.atomic():
                        try:
                            self.grids[idx][0].remove()
                            self.grids[idx][1].remove()
                            self.grids[idx][2].remove()
                            self.grid_folders[idx].remove()  # bug
                        except Exception as e:
                            traceback.print_exc()
                        finally:
                            del self.grids[idx]
                            del self.grid_folders[idx]

                    self._update_scene()

            # update the pose of panel(grid) when grid_transform updated
            @grid_transform.on_update
            def _(_):
                self.grids[idx][0].wxyz = grid_transform.wxyz
                self.grids[idx][0].position = grid_transform.position
                self._update_scene()

            self.grids[self.grid_idx] = [grid, grid_transform, grid_size]
            self._update_scene()

        # setup callbacks

        @add_grid_button.on_click
        def _(event):
            with server.atomic():
                new_grid(self.grid_idx, event)
                self.grid_idx += 1

        @self.delete_gaussians_button.on_click
        def _(_):
            with server.atomic():
                gaussian_to_be_deleted, pose_and_size_list = self._get_selected_gaussians_mask(return_pose_and_size_list=True)
                self.edit_histories.append(pose_and_size_list)
                self.viewer.gaussian_model.delete_gaussians(gaussian_to_be_deleted)
                self._update_pcd()
            self.viewer.rerender_for_all_client()

    def _setup_save_gaussian_folder(self):
        with self.server.gui.add_folder("Save"):
            name_text = self.server.gui.add_text(
                "Name",
                initial_value=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
            save_button = self.server.gui.add_button("Save")

            @save_button.on_click
            def _(event: viser.GuiEvent):
                # skip if not triggered by client
                if event.client is None:
                    return
                try:
                    save_button.disabled = True

                    with self.server.atomic():
                        try:
                            # check whether is a valid name
                            name = name_text.value
                            match = re.search(r"^[a-zA-Z0-9_\-]+$", name)
                            if match:
                                output_directory = "edited"
                                os.makedirs(output_directory, exist_ok=True)
                                try:
                                    if len(self.edit_histories) > 0:
                                        torch.save(self.edit_histories, os.path.join(output_directory, f"{name}-edit_histories.ckpt"))
                                except:
                                    traceback.print_exc()

                                if self.viewer.checkpoint is None:
                                    from internal.utils.gaussian_utils import GaussianPlyUtils
                                    # save ply
                                    ply_save_path = os.path.join(output_directory, "{}.ply".format(name))
                                    GaussianPlyUtils.load_from_model_properties(self.viewer.gaussian_model.get_non_pre_activated_properties(), self.viewer.gaussian_model.max_sh_degree).to_ply_format().save_to_ply(ply_save_path)
                                    message_text = "Saved to {}".format(ply_save_path)
                                else:
                                    # save as a checkpoint if viewer started from a checkpoint
                                    checkpoint_save_path = os.path.join(output_directory, "{}.ckpt".format(name))
                                    checkpoint = self.viewer.checkpoint
                                    # update state dict of the checkpoint
                                    properties = self.viewer.gaussian_model.get_non_pre_activated_properties()
                                    for name, value in properties.items():
                                        key = "gaussian_model.gaussians.{}".format(name)
                                        checkpoint["state_dict"][key] = properties[name].to(device=checkpoint["state_dict"][key].device)
                                    # TODO: density controller and optimizer states need to be pruned too
                                    # save
                                    torch.save(checkpoint, checkpoint_save_path)
                                    message_text = "Saved to {}".format(checkpoint_save_path)
                            else:
                                message_text = "Invalid name"
                        except:
                            traceback.print_exc()

                    # show message
                    with event.client.gui.add_modal("Message") as modal:
                        event.client.gui.add_markdown(message_text)
                        close_button = event.client.gui.add_button("Close")

                        @close_button.on_click
                        def _(_) -> None:
                            modal.close()

                finally:
                    save_button.disabled = False

    def _setup_custom_folder(self):
        server = self.server
        with server.gui.add_folder("Viz"):
            self.camera_type = self.server.gui.add_dropdown(
                "Camera Type",
                ["colmap", "GS", "npz"],
                initial_value="colmap",
            )

            self.fps = self.server.gui.add_number(
                "FPS",
                min=1,
                max=60,
                step=1,
                initial_value=5,
            )

            render_by_cam_button = server.gui.add_button(
                "Render by Cam",
                color="purple",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Save smplx scale",
            )
            @render_by_cam_button.on_click
            def _(event: viser.GuiEvent) -> None:
                try:
                    camera_type = self.camera_type.value
                    if camera_type == "colmap":
                        cameras = self.get_cameras_from_colmap()
                    elif camera_type == "GS":
                        cameras = self.get_cameras_from_json()
                    elif camera_type == "npz":
                        cameras = self.get_cameras_from_npz()
                    else:
                        raise ValueError("Camera type invalid.", camera_type)
                    self.render(cameras, fps=self.fps.value)
                    # render_with_client(event.client, cameras, self.root_path, self.fps.value)
                except Exception as e:
                    print("Error:", e)
            
            render_by_pred_cam_button = server.gui.add_button(
                        "Render by Pred Cam",
                        color="Blue",
                        icon=viser.Icon.PLAYER_PLAY,
                        hint="Save smplx scale",
            )
            @render_by_pred_cam_button.on_click
            def _(event: viser.GuiEvent) -> None:
                try:
                    cam_params = np.load(f"{self.root_path}/pred.npz", allow_pickle=True)
                    w2c_ext = cam_params['extrinsics']
                    intrinsics = cam_params['intrinsics']
                    cameras = build_cameras(w2c_ext, intrinsics)
                    self.render(cameras, fps=self.fps.value)
                    # render_with_client(event.client, cameras, settings['render_output_path'])
                except Exception as e:
                    print("Error:", e)

            self.frustum_range = self.server.gui.add_vector2(
                "Frustum Range",
                min=(0, 1),
                max=(1000, 1000),
                step=1,
                initial_value=(0, 100),
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

                self.is_frustum_visualized[camera_type] = not self.is_frustum_visualized[camera_type]
                
                camera_type = self.camera_type.value
                if camera_type == "colmap":
                    cameras = self.get_cameras_from_colmap()
                elif camera_type == "GS":
                    cameras = self.get_cameras_from_json()
                elif camera_type == "npz":
                    cameras = self.get_cameras_from_npz()
                else:
                    raise ValueError("Camera type invalid.", camera_type)
            

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

            vis_colmap_button = self.server.gui.add_button(
                "Visualize Colmap",
                color="Red",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Visualize Colmap.",
            )
            @vis_colmap_button.on_click
            def _(event: viser.GuiEvent) -> None:
                self.get_custom_render_settings()
                sparse_path = f"{self.root_path}/colmap/sparse"
                recon = pycolmap.Reconstruction(sparse_path)

                points = []
                colors = []

                for p in recon.points3D.values():
                    points.append(p.xyz)
                    colors.append(p.color / 255.0)

                points = np.asarray(points)
                colors = np.asarray(colors)

                self.is_colmap_visualized = not self.is_colmap_visualized
                server.scene.add_point_cloud(
                    name="colmap_points",
                    points=points,
                    colors=colors,
                    point_size=0.01,
                    visible=self.is_colmap_visualized,
                )
            
            vis_point_cloud_button = self.server.gui.add_button(
                "Visualize Point Cloud",
                color="Red",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Visualize Colmap.",
            )
            @vis_point_cloud_button.on_click
            def _(event: viser.GuiEvent) -> None:
                point_cloud_path = f"{self.root_path}/point_cloud.npy"

                points = np.load(point_cloud_path)
                colors = np.repeat([[0., 0., 1.0]], len(points), axis=0)

                self.is_point_cloud_visualized = not self.is_point_cloud_visualized
                server.scene.add_point_cloud(
                    name="/point_cloud",
                    points=points,
                    colors=colors,
                    point_size=0.01,
                    visible=self.is_point_cloud_visualized,
                )

    def _get_selected_gaussians_mask(self, return_pose_and_size_list: bool = False):
        xyz = self.viewer.gaussian_model.get_xyz

        # if no grid exists, do not delete any gaussians
        if len(self.grids) == 0:
            return torch.zeros(xyz.shape[0], device=xyz.device, dtype=torch.bool)

        pose_and_size_list = []
        # initialize mask with True
        is_gaussian_selected = torch.ones(xyz.shape[0], device=xyz.device, dtype=torch.bool)
        for i in self.grids:
            # get the pose of grid, and build world-to-grid transform matrix
            grid = self.grids[i][0]
            se3 = torch.linalg.inv(torch.tensor(vtf.SE3.from_rotation_and_translation(
                vtf.SO3(grid.wxyz),
                grid.position,
            ).as_matrix()).to(xyz))
            # transform xyz from world to grid
            new_xyz = torch.matmul(xyz, se3[:3, :3].T) + se3[:3, 3]
            # find the gaussians to be deleted based on the new_xyz
            grid_size = self.grids[i][2].value
            x_mask = torch.abs(new_xyz[:, 0]) < grid_size[0] / 2
            y_mask = torch.abs(new_xyz[:, 1]) < grid_size[1] / 2
            z_mask = new_xyz[:, 2] > 0
            # update mask
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, x_mask)
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, y_mask)
            is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, z_mask)

            # add to history
            pose_and_size_list.append((se3.cpu(), grid_size))

        if return_pose_and_size_list is True:
            return is_gaussian_selected, pose_and_size_list
        return is_gaussian_selected

    def _get_selected_gaussians_indices(self):
        """
        get the index of the gaussians which in the range of grids
        :return:
        """
        selected_gaussian = torch.where(self._get_selected_gaussians_mask())
        return selected_gaussian

    @torch.no_grad()
    def _update_pcd(self, selected_gaussians_indices=None):
        self.remove_point_cloud()
        if self.show_point_cloud_checkbox.value is False:
            return
        xyz = self.viewer.gaussian_model.get_xyz
        colors = torch.tensor([self.point_cloud_color.value], dtype=torch.uint8, device=xyz.device).repeat(xyz.shape[0], 1)
        if selected_gaussians_indices is None:
            selected_gaussians_indices = self._get_selected_gaussians_indices()
        colors[selected_gaussians_indices] = 255 - torch.tensor(self.point_cloud_color.value).to(colors)

        point_sparsify = int(self.point_sparsify.value)
        self.show_point_cloud(xyz[::point_sparsify].cpu().numpy(), colors[::point_sparsify].cpu().numpy())

    def remove_point_cloud(self):
        if self.pcd is not None:
            self.pcd.remove()
            self.pcd = None

    def show_point_cloud(self, xyz, colors):
        self.pcd = self.server.add_point_cloud(
            "/pcd",
            points=xyz,
            colors=colors,
            point_size=self.point_size.value,
        )

    def _update_scene(self):
        selected_gaussians_indices = self._get_selected_gaussians_mask()
        self.viewer.gaussian_model.select(selected_gaussians_indices)
        self._update_pcd(selected_gaussians_indices)

        self.viewer.rerender_for_all_client()
        
    def get_cameras_from_colmap(self):
        sparse_path = f"{self.root_path}/colmap/sparse"
        recon = pycolmap.Reconstruction(sparse_path)
        extrinsics, intrinsics = get_colmap_camera_params(recon)
        cameras = build_cameras(extrinsics, intrinsics)
        return cameras


    def get_cameras_from_npz(self):
        w2c_ext, intrinsics = get_camera_params_from_json(os.path.join(self.root_path, "GS", "cameras.json"))
        w2c_ext = np.load(f"{self.root_path}/vae_camera_params.npy", allow_pickle=True)
        N = w2c_ext.shape[0]
        intrinsics = intrinsics[:N]
        # w2c_ext = cam_params['extrinsics']
        # intrinsics = cam_params['intrinsics']
        cameras = build_cameras(w2c_ext, intrinsics)
        return cameras

    def get_cameras_from_json(self):
        w2c_ext, intrinsics = get_camera_params_from_json(os.path.join(self.root_path, "GS", "cameras.json"))
        cameras = build_cameras(w2c_ext, intrinsics)
        return cameras

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

def build_cameras(extrinsics, intrinsics):
    """
    Input:
        extrinsics: [N, 4, 4] numpy array
        intrinsics: [N, 3, 3] numpy array
    """
    N = extrinsics.shape[0]
    R_w2c = extrinsics[:, :3, :3]
    T_w2c = extrinsics[:, :3, 3]

    # R_c2w = np.transpose(R_w2c, (0, 2, 1))
    # T_c2w = (- R_c2w @ T_w2c).squeeze(-1)

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    width = intrinsics[:, 0, 2] * 2
    height = intrinsics[:, 1, 2] * 2
    # width = np.array([432] * N)
    # height = np.array([240] * N)
    appearance_id = torch.zeros((N), dtype=torch.int)
    normalized_appearance_id = torch.zeros((N), dtype=torch.int)
    distortion_params = torch.zeros((N, 4), dtype=torch.int)
    camera_type = torch.zeros((N), dtype=torch.int)

    cameras = Cameras(
        R=torch.from_numpy(R_w2c).float(),
        T=torch.from_numpy(T_w2c).float(),
        fx=torch.from_numpy(fx).float(),
        fy=torch.from_numpy(fy).float(),
        cx=torch.from_numpy(cx).float(),
        cy=torch.from_numpy(cy).float(),
        width=torch.from_numpy(width).int(),
        height=torch.from_numpy(height).int(),
        appearance_id=appearance_id,
        normalized_appearance_id=normalized_appearance_id,
        distortion_params=distortion_params,
        camera_type=camera_type,
    )
    return cameras

def render_with_client(client, cameras: Cameras, out_dir, fps):
    images = []
    print("Rendering...")
    for camera in tqdm(cameras):
        fov, quat, t = camera_to_fov_quat_position(camera)
        # img = client.get_render(
        #     height=camera.height.item(),
        #     width=camera.width.item(),
        #     wxyz=np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]]),
        #     position=camera.T.numpy(),
        #     fov=camera.fov_y.item(),
        # )
        client.camera.wxyz = quat
        client.camera.position = t
        client.camera.fov = fov
        img = client.camera.get_render(
            height = camera.height.item(),
            width = camera.width.item(),
        )
        images.append(img)
    save_path = os.path.join(out_dir, "scene_viz.mp4")
    imageio.mimsave(save_path, images, fps=fps)
    print("Rendering done. Saved to", save_path)

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
        R_c2w, t_c2w = params['rotation'], params['position']

        R_w2c = np.array(R_c2w).T
        t_w2c = (- R_w2c @ np.array(t_c2w).reshape(-1, 1)).squeeze(-1)

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c

        fx, fy = params['fx'], params['fy']
        scale = 4
        fx /= scale
        fy /= scale
        # cx, cy = params['cx'], params['cy']
        width, height = params['width'], params['height']
        width /= scale
        height /= scale
        cx, cy = width/2.0, height/2.0
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



import os

import numpy as np
import torch
import viser
import imageio
import json
import time
import cv2
import shutil

from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from .cameras import Cameras, camera_to_fov_quat_position
from . import camera_parser, pointcloud_parser, task_parser
from .render_3dgs import get_renders

from .render_frustum import add_frustum_spline

from .color_points import color_points_by_attention


class PanelMode(Enum):
    """Which of the three ways the panel gets used. Controls that only make sense
    for one of them are not built in the others."""
    TASK = auto()   # walk a task list, loading each task's scene and cameras
    GS = auto()     # a 3DGS scene under root_path, rendered through the viewer
    PCD = auto()    # a point cloud scene loaded from ply_path

    @classmethod
    def resolve(cls, ply_path, pcd_type, task_list):
        """Pick the mode the arguments imply, in the order _handle_new_client used
        to branch: an explicit point cloud wins, then a task list, else the scene
        sitting under root_path."""
        if ply_path is not None and pcd_type is not None:
            return cls.PCD
        if task_list:
            return cls.TASK
        return cls.GS


class CustomPanel:
    def __init__(self, viewer, server, root_path, renderer=None, recon=None, custom_camera_data_path=None, pcd=None, ply_path=None,
    data_root_path=None, pcd_type=None, task_list_path=None, task_type="pred", task_offset=0, mode=None):
        self.server = server
        self.root_path = root_path
        self.scene_root_path = None
        self.root_camera_path = os.path.join(self.root_path, "cameras.json")
        self.renderer = renderer
        self.recon = recon
        self.scale_factor = 1.0
        self.points = None
        self.custom_camera_data_path = custom_camera_data_path
        self.pcd = pcd
        self.ply_path = ply_path
        self.input_pcd_type = pcd_type
        self.data_root_path = data_root_path
        self.output_dir = "results"
        self.current_task_idx = None
        self.pcd_path = None
        self.frame_idx = 0
        self.task_offset=task_offset

        os.makedirs(self.output_dir, exist_ok=True)

        self.task_type = task_type
        self.task_list = []
        if task_list_path is not None and os.path.exists(task_list_path):
            with open(task_list_path, "r") as f:
                tasks = f.readlines()
            for task in tasks:
                self.task_list.append(task.strip().replace("\n", ""))

        self.mode = mode if mode is not None else PanelMode.resolve(ply_path, pcd_type, self.task_list)
        print("panel mode:", self.mode.name)

        # CSV-based navigation (separate from the .txt-based task list)
        self.csv_task_list = []      # task strings converted from CSV scene_path
        self.csv_orig_index = []     # orig_index column from the CSV
        self._task_index_map = None  # {task_string: task_list index}, built lazily

        if viewer is not None:
            self.scale_factor = viewer.scale_factor
            self.custom_camera_path = viewer.custom_camera_path

        self.valid_camera_types = ["ours", "GS", "colmap", "npz", "custom", "monst3r", "transforms", "GT"]

        list_for_worldtraj = [
                            "director3d",
                            "gendop_text_rgbd",
                            "worlddirector_final",
                        ]
        list_for_others = [
                            "director3d",
                            "gendop_text_rgbd_origin",
                            "worlddirector"
                        ]

        list_for_cfg = [
            "ours_mild10_s20",
            "ours_mild10_unguided"
        ]

        if self.task_offset == 0:
            type_preset = list_for_worldtraj
        elif self.task_offset == 1:
            type_preset = list_for_others
        else:
            type_preset = list_for_cfg

        self.valid_camera_pred_types = type_preset
        self.render_targets = type_preset
        self.retrieve_targets = type_preset

        self.auto_frustum_targets = type_preset + ["GT"]
        
        if self.task_offset == 0 or self.task_offset == 1:
            color_preset = [
                [1.0, 1.0, 1.0],
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
            ]
        else:
            color_preset = [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]

        self.preset_frustum_colors = {
            "GT": [1.0, 0.0, 0.0]
        }

        for cam_type, color in zip(type_preset, color_preset):
            self.preset_frustum_colors[cam_type] = color

        self.is_frustum_visualized = {}
        self.frustum_handles = {}

        for cam_type in self.valid_camera_types:
            self.is_frustum_visualized[cam_type] = False
        for cam_type in self.valid_camera_types:
            self.frustum_handles[cam_type] = []
        
        for cam_type in self.valid_camera_pred_types:
            self.is_frustum_visualized[cam_type] = False
        for cam_type in self.valid_camera_pred_types:
            self.frustum_handles[cam_type] = []

        

        self.client_camera_save_path = os.path.join(self.root_path, "saved_client_camera.json")

        # attention_path = f"{self.root_path}/ray_sims_T_P.pt"
        attention_path = f"{self.root_path}/ray_sims_T_P_not_softmax.pt"

        self.attention = None
        if os.path.exists(attention_path):
            print("Attention Found", attention_path)
            attention = torch.load(attention_path, map_location="cuda") # (T, P)
            attention = torch.softmax(attention, dim=1)
            self.attention = attention / (attention.max(dim=1).values - attention.min(dim=1).values).unsqueeze(-1)
        else:
            print("Attention Not Found", attention_path)

        self._setup_vis_folder()
        self.server.on_client_connect(self._handle_new_client)

        if self.mode is PanelMode.PCD:
            self.pcd_type.value = self.input_pcd_type
            self.pcd_path = self.ply_path
            points, colors = self.get_pcd_by_type()

            self.points = torch.from_numpy(np.concatenate([points, colors], axis=1)).to("cuda")

            
    
    def _setup_vis_folder(self):
        with self.server.gui.add_folder("Viz"):
            if self.mode is PanelMode.TASK:
                self._setup_task_controls()
            self._setup_render_controls()
            self._setup_frustum_controls()
            self._setup_pcd_controls()
            if self.attention is not None:
                self._setup_attention_controls()

    def _setup_task_controls(self):
        """Task Idx / CSV Idx navigation."""
        self.task_slider = self.server.gui.add_slider(
            "Task Idx Slider",
            min=0,
            max=max(len(self.task_list) - 1, 0),
            initial_value = 0,
            step=1,
        )

        get_task_button = self.server.gui.add_button(
                "Get Task",
                color="purple",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Get Task",
            )
        @get_task_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self.load_task(self.task_slider.value, event.client)

        next_task_button = self.server.gui.add_button(
                "Next Task",
                color="Blue",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Get Next Task",
            )
        @next_task_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self.load_task(self.task_slider.value + 1, event.client)

        # --- CSV-based navigation (separate CSV Idx; syncs Task Idx on move) ---
        self.csv_task_path = self.server.gui.add_text(
            "CSV Task Path",
            initial_value="/data1/cympyc1785/SceneData/nearest_dist_test_data_final_k100_pct1to2.csv",
        )
        load_csv_button = self.server.gui.add_button(
                "Load CSV Tasks",
                color="green",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Load scene_path list from a CSV for CSV-idx navigation",
            )
        @load_csv_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self.load_tasks_from_csv(self.csv_task_path.value)

        self.csv_slider = self.server.gui.add_slider(
            "CSV Idx Slider",
            min=0,
            max=0,
            initial_value=0,
            step=1,
        )

        get_csv_button = self.server.gui.add_button(
                "Get CSV Task",
                color="green",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Load the CSV task at CSV Idx (also syncs Task Idx)",
            )
        @get_csv_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self.get_csv_task(self.csv_slider.value, event.client)

        next_csv_button = self.server.gui.add_button(
                "Next CSV Task",
                color="green",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Load the next CSV task (also syncs Task Idx)",
            )
        @next_csv_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self.get_csv_task(self.csv_slider.value + 1, event.client)

    def _setup_render_controls(self):
        """Camera type, fps, and every render/save button."""
        self.custom_camera_checkbox = self.server.gui.add_checkbox(
            "Viz Custom Camera",
            initial_value=False,
        )

        self.camera_type = self.server.gui.add_dropdown(
            "Camera Type",
            self.valid_camera_types + self.valid_camera_pred_types,
            initial_value="ours",
        )

        self.fps = self.server.gui.add_number(
            "FPS",
            min=1,
            max=60,
            step=1,
            initial_value=10,
        )

        render_by_cam_button = self.server.gui.add_button(
                "Render GS by Cam",
                color="purple",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Save smplx scale",
            )
        @render_by_cam_button.on_click
        def _(event: viser.GuiEvent) -> None:
            print("Rendering")
            cameras = self.get_cameras_by_type()
            # renders = self.render(cameras, fps=self.fps.value)
            scene_path = os.path.join(self.scene_root_path, "scene.ply")
            renders = get_renders(scene_path, cameras)
            print("Rendering Done.")

            output_dir = self.task_output_dir()

            os.makedirs(output_dir, exist_ok=True)

            timestamp = int(time.time())
            save_path = os.path.join(output_dir, f"{self.camera_type.value}_scene_viz_{timestamp}.mp4")
            imageio.mimsave(save_path, renders, fps=self.fps.value)

            print("Saving video done. Saved to", save_path)

        render_by_client_button = self.server.gui.add_button(
                "Render by Client",
                color="Brown",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Save smplx scale",
            )
        @render_by_client_button.on_click
        def _(event: viser.GuiEvent) -> None:
            cameras = self.get_cameras_by_type(sample=True)
            renders = self.render_with_client(event.client, cameras)

            output_dir = self.task_output_dir()

            os.makedirs(output_dir, exist_ok=True)

            timestamp = int(time.time())
            save_path = os.path.join(output_dir, f"{self.camera_type.value}_pointcloud_viz_{timestamp}.mp4")
            imageio.mimsave(save_path, renders, fps=self.fps.value)
            print("Saved to", save_path)

        if self.mode is PanelMode.TASK:   # render every prediction model through the client
            render_all_by_client_button = self.server.gui.add_button(
                    "Render All by client",
                    color="Brown",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Save smplx scale",
                )
            @render_all_by_client_button.on_click
            def _(event: viser.GuiEvent) -> None:
                init_cam_type = self.camera_type.value

                self.clean_all_camera_frustum()

                for cam_type in self.render_targets:
                    self.camera_type.value = cam_type
                    cameras = self.get_cameras_by_type()
                    renders = self.render_with_client(event.client, cameras)

                    output_dir = self.task_output_dir()

                    os.makedirs(output_dir, exist_ok=True)

                    timestamp = int(time.time())
                    save_path = os.path.join(output_dir, f"{self.camera_type.value}_pointcloud_viz_{timestamp}.mp4")
                    imageio.mimsave(save_path, renders, fps=self.fps.value)

                    frame_indices = {
                        "first": 0,
                        "middle": len(renders) // 2,
                        "last": len(renders) - 1
                    }

                    output_dir = os.path.join(output_dir, cam_type)
                    os.makedirs(output_dir, exist_ok=True)

                    for name, idx in frame_indices.items():
                        save_path = os.path.join(output_dir, f"{name}.png")
                        cv2.imwrite(save_path, renders[idx])

                self.camera_type.value = init_cam_type
                print("Saved all videos")

        render_by_point_renderer_button = self.server.gui.add_button(
                "Render by Point Renderer",
                color="Brown",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Save smplx scale",
            )
        @render_by_point_renderer_button.on_click
        def _(event: viser.GuiEvent) -> None:
            cameras = self.get_cameras_by_type(sample=True)
            renders = self.render_pointcloud_multi_cam(cameras)

            output_dir = self.task_output_dir()

            os.makedirs(output_dir, exist_ok=True)

            timestamp = int(time.time())
            save_path = os.path.join(output_dir, f"{self.camera_type.value}_point_scene_render_{timestamp}.mp4")
            print(len(renders))
            imageio.mimsave(save_path, renders, fps=self.fps.value)
            # save_video_imageio(renders, save_path, fps=self.fps.value)
            print("Saved to", save_path)

        render_gs_images_samples_button = self.server.gui.add_button(
                "Render GS Image Samples",
                color="purple",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Save smplx scale",
            )
        @render_gs_images_samples_button.on_click
        def _(event: viser.GuiEvent) -> None:
            cameras = self.get_cameras_by_type(sample=True)
            scene_path = os.path.join(self.scene_root_path, "scene.ply")
            renders = get_renders(scene_path, cameras)
            # self.render(cameras, fps=self.fps.value)

            output_dir = self.task_output_dir(self.camera_type.value)

            os.makedirs(output_dir, exist_ok=True)

            timestamp = int(time.time())
            for i, render in enumerate(renders):
                save_path = os.path.join(output_dir, f"scene_viz_{timestamp}_{i:03d}.png")
                imageio.imsave(save_path, render)

            print("Rendering done. Saved to", save_path)

        screenshot_button = self.server.gui.add_button(
            "Screenshot",
            color="Green",
            icon=viser.Icon.PLAYER_PLAY,
            hint="Screenshot",
        )
        @screenshot_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            image = client.camera.get_render(width=1280, height=720)

            output_dir = self.task_output_dir()

            os.makedirs(output_dir, exist_ok=True)

            timestamp = int(time.time())
            save_path = os.path.join(output_dir, f"{self.current_task_idx}_{self.camera_type.value}_screenshot_{timestamp}.png")
            imageio.imwrite(save_path, image)

            print(f"Image Saved: {save_path}")
            print(f"Image Shape: {image.shape}")

        save_masked_image = self.server.gui.add_button(
            "Save Masked Image",
            color="Blue",
            icon=viser.Icon.PLAYER_PLAY,
            hint="Save Masked Image",
        )
        @save_masked_image.on_click
        def _(event: viser.GuiEvent) -> None:

            alpha = 0.6

            inpainted_video_path = os.path.join(self.scene_root_path, "inpaint_result.mp4")
            if not os.path.exists(inpainted_video_path):
                print("no inpainted video")
                return

            cap = cv2.VideoCapture(inpainted_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            first_idx = 0
            mid_idx = total_frames // 2
            last_idx = total_frames - 1

            images_dir = os.path.join(self.scene_root_path, "images")
            img_list = os.listdir(images_dir)

            last_idx = min(last_idx, len(img_list) - 2)

            indices = {
                "first": first_idx,
                "middle": mid_idx,
                "last": last_idx
            }

            timestamp = int(time.time())

            output_dir = self.task_output_dir()

            os.makedirs(output_dir, exist_ok=True)

            for name, idx in indices.items():

                # video frame 읽기
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, inpainted_frame = cap.read()
                if not ret:
                    continue
                inpainted_frame = inpainted_frame.astype(np.float32)

                # mask path (frame index 기준)
                frame_path = os.path.join(self.scene_root_path, "images", f"{idx+1:05d}.png")
                mask_path = os.path.join(self.scene_root_path, "mask", f"{idx+1:05d}.png")

                if not os.path.exists(frame_path):
                    frame_path = os.path.join(self.scene_root_path, "images", f"{idx+1:05d}.jpg")
                    if not os.path.exists(frame_path):
                        print("no img", frame_path)
                        return
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(self.scene_root_path, "mask", f"{idx+1:05d}.jpg")
                    if not os.path.exists(mask_path):
                        print("no mask", mask_path)
                        return

                frame = imageio.imread(frame_path).astype(np.float32)
                mask = imageio.imread(mask_path).astype(np.float32)

                # frame = imageio.imread(frame_path)
                # mask = imageio.imread(mask_path)

                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]

                overlay = frame.copy()
                mask_region = mask.sum(axis=2) > 0

                overlay[mask_region] = (
                    frame[mask_region] * (1 - alpha) +
                    mask[mask_region] * alpha
                )

                overlay = overlay.astype(np.uint8)

                save_path = os.path.join(output_dir, f"{self.current_task_idx}_{self.camera_type.value}_masked_{name}_frame_{timestamp}.png")
                imageio.imwrite(save_path, overlay)

                # save_path = os.path.join(output_dir, f"{self.current_task_idx}_{self.camera_type.value}_frame_{name}_frame_{timestamp}.png")
                # imageio.imwrite(save_path, frame)

                # save_path = os.path.join(output_dir, f"{self.current_task_idx}_{self.camera_type.value}_mask_{name}_frame_{timestamp}.png")
                # imageio.imwrite(save_path, mask)

                save_path = os.path.join(output_dir, f"{self.current_task_idx}_{self.camera_type.value}_inpainted_{name}_frame_{timestamp}.png")
                cv2.imwrite(save_path, inpainted_frame)

            cap.release()
            print("Inpainted frames with mask saved")

        if self.mode is PanelMode.TASK:   # batch render and retrieve over every prediction model
            render_all_camera_preds = self.server.gui.add_button(
                    "Render All Camera Preds",
                    color="purple",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Save smplx scale",
                )
            @render_all_camera_preds.on_click
            def _(event: viser.GuiEvent) -> None:
                print("Rendering")
                init_cam_type = self.camera_type.value
                init_frustum_interval = self.frustum_interval.value
                init_interp_checkbox = self.interpolate_cam_checkbox.value

                scene_path = os.path.join(self.scene_root_path, "scene.ply")

                output_dir = self.task_output_dir()

                os.makedirs(output_dir, exist_ok=True)

                cam_dict = {}
                for cam_type in self.render_targets:
                    self.camera_type.value = cam_type
                    self.frustum_interval.value = 1
                    self.interpolate_cam_checkbox.value = True

                    cameras = self.get_cameras_by_type()
                    cam_dict[cam_type] = cameras

                def render_and_save_worker(scene_path, cameras, output_dir, cam_type, fps):
                    renders = get_renders(scene_path, cameras)
                    timestamp = int(time.time())
                    save_path = os.path.join(output_dir, f"{cam_type}_scene_viz_{timestamp}.mp4")
                    imageio.mimsave(save_path, renders, fps=self.fps.value)

                with ThreadPoolExecutor(max_workers=6) as ex:
                    futures = [ex.submit(render_and_save_worker, scene_path, cameras,
                                         output_dir, cam_type, self.fps.value) for cam_type, cameras in cam_dict.items()]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                        fut.result()

                self.camera_type.value = init_cam_type
                self.frustum_interval.value = init_frustum_interval
                self.interpolate_cam_checkbox.value = init_interp_checkbox

                print("Saving video done.")

            retrieve_all_camera_preds = self.server.gui.add_button(
                    "Retrieve All Camera Video",
                    color="purple",
                    icon=viser.Icon.PLAYER_PLAY,
                    hint="Save smplx scale",
                )
            @retrieve_all_camera_preds.on_click
            def _(event: viser.GuiEvent) -> None:
                print("Retrieving videos...")

                for cam_type in self.render_targets:
                    output_dir = self.task_output_dir()
                    os.makedirs(output_dir, exist_ok=True)

                    video_src_path = os.path.join(self.data_root_path, cam_type, "test", f"{self.task_list[self.current_task_idx]}_render.mp4")
                    video_dst_path = os.path.join(output_dir, f"{cam_type}_render.mp4")

                    if not os.path.exists(video_src_path):
                        print("no vid", video_src_path)
                        continue

                    shutil.copy(video_src_path, video_dst_path)

                    # capture 3 frames
                    cap = cv2.VideoCapture(video_dst_path)
                    if not cap.isOpened():
                        print(f"Cannot open video: {video_dst_path}")
                        continue

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_indices = {
                        "first": 0,
                        "middle": total_frames // 2,
                        "last": total_frames - 1
                    }

                    output_dir = os.path.join(output_dir, cam_type)
                    os.makedirs(output_dir, exist_ok=True)

                    for name, idx in frame_indices.items():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()

                        if not ret:
                            print(f"Failed to read {name} frame at index {idx}")
                            continue

                        save_path = os.path.join(output_dir, f"{name}.png")
                        cv2.imwrite(save_path, frame)

                    cap.release()

                print("Saving video done.")

    def _setup_frustum_controls(self):
        """Frustum appearance and the frustum visualisation buttons."""
        self.vis_frustum_idx = self.server.gui.add_number(
                "Viz Frustum Index",
                min=0,
                max=1000,
                step=1,
                initial_value=0,
            )

        self.vis_single_frustum = self.server.gui.add_checkbox(
            "Viz single frustum",
            initial_value=False,
        )

        self.frustum_range = self.server.gui.add_vector2(
                "Frustum Range",
                min=(0, 1),
                max=(1000, 1000),
                step=1,
                initial_value=(0, 150),
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
                initial_value=0.25,
            )

        self.frustum_thickness = self.server.gui.add_number(
                "Frustum Thickness",
                min=0.1,
                max=10.0,
                step=0.1,
                initial_value=8,
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

        self.interpolate_cam_num = self.server.gui.add_number(
                "Interpolation Cam Number",
                min=1,
                max=200,
                step=1,
                initial_value=120,
            )

        self.interpolate_cam_checkbox = self.server.gui.add_checkbox(
            "Interpolate Camera",
            initial_value=False,
        )

        vis_camera_frustum_button = self.server.gui.add_button(
                "Visualize Camera Frustum",
                color="Green",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Visualize camera trajectory.",
            )
        @vis_camera_frustum_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None

            self.visualize_camera_frustum(client)

        clear_all_camera_frustum_button = self.server.gui.add_button(
                "Clear All Camera Frustum",
                color="Red",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Clear all frustum",
            )
        @clear_all_camera_frustum_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self.clean_all_camera_frustum()

        if self.mode is PanelMode.TASK:   # frustum screenshot / screen record over every prediction model
            screenshot_all_camera_frustum_button = self.server.gui.add_button(
                "Screenshot all cam",
                color="Green",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Screenshot",
            )
            @screenshot_all_camera_frustum_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                init_cam_type = self.camera_type.value

                for cam_type in self.render_targets:
                    self.camera_type.value = cam_type

                    self.clean_all_camera_frustum()

                    self.visualize_camera_frustum(client)

                    image = client.camera.get_render(width=1280, height=720)

                    output_dir = self.task_output_dir("screenshot")

                    os.makedirs(output_dir, exist_ok=True)

                    timestamp = int(time.time())
                    save_path = os.path.join(output_dir, f"{self.current_task_idx}_{self.camera_type.value}_screenshot_{timestamp}.png")
                    imageio.imwrite(save_path, image)

                    print(f"Screenshot Saved: {save_path}")
                    # print(f"Image Shape: {image.shape}")

                self.camera_type.value = init_cam_type

            all_screen_record_button = self.server.gui.add_button(
                "All Screen record",
                color="Green",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Screenrecord",
            )
            @all_screen_record_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                init_cam_type = self.camera_type.value

                output_dir = self.task_output_dir()
                os.makedirs(output_dir, exist_ok=True)

                for camera_type in self.render_targets:
                    self.clean_all_camera_frustum()
                    self.camera_type.value = camera_type
                    cameras = self.get_cameras_by_type()

                    fov, quat, position = camera_to_fov_quat_position(cameras[0])

                    f_min, f_max = self.frustum_range.value
                    f_min = int(f_min)
                    f_max = int(f_max)
                    cam_min = f_min
                    cam_max = min(f_max, len(cameras))

                    start_color = np.array(self.frustum_start_color.value)
                    end_color = np.array(self.frustum_end_color.value)
                    t = np.linspace(0, 1, (cam_max - cam_min))[:, None]  # (N, 1)
                    colors = (1 - t) * start_color + t * end_color  # (N, 3)

                    screenrecord_list = []
                    for i in tqdm(range(cam_min, cam_max, 1)):
                        if i % self.frustum_interval.value == 0:
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
                        image = client.camera.get_render(width=1280, height=720)
                        screenrecord_list.append(image)
                    save_path = os.path.join(output_dir, f"cam_frustum_vis_{camera_type}.mp4")
                    save_video_imageio(np.array(screenrecord_list), save_path, fps=self.fps.value)
                    print(f"Video Saved: {save_path}")

                self.camera_type.value = init_cam_type

    def _setup_pcd_controls(self):
        """Point cloud display plus client-camera save/load and screen record."""
        # ===== PCD

        self.pcd_type = self.server.gui.add_dropdown(
            "Point Cloud Type",
            ["colmap", "ply", "npy", "torch"],
            initial_value="ply",
        )

        self.pcd_scale = self.server.gui.add_number(
                "PCD Scale",
                min=0.001,
                max=1.0,
                step=0.001,
                initial_value=0.01,
            )

        self.pcd_sample_checkbox = self.server.gui.add_checkbox(
            "Sample PCD",
            initial_value=True,
        )

        self.pcd_sample_num = self.server.gui.add_number(
                "Frustum Scale",
                min=10000,
                max=10000000,
                step=1,
                initial_value=1000000,
            )

        vis_point_cloud_button = self.server.gui.add_button(
            "Visualize Point Cloud",
            color="Red",
            icon=viser.Icon.PLAYER_PLAY,
            hint="Visualize Colmap.",
        )
        @vis_point_cloud_button.on_click
        def _(event: viser.GuiEvent) -> None:
            self.vis_pointcloud()

        save_client_camera_button = self.server.gui.add_button(
                "Save Client Camera",
                color="Brown",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Save camera state",
            )
        @save_client_camera_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client

            camera_state = {
                "position": client.camera.position.tolist(),
                "wxyz": client.camera.wxyz.tolist(),
                "look_at": client.camera.look_at.tolist(),
            }

            with open(self.client_camera_save_path, "w") as f:
                json.dump(camera_state, f, indent=4)
            print(f"Saved Camera: {self.client_camera_save_path}")

        load_client_camera_button = self.server.gui.add_button(
                "Load Client Camera",
                color="Blue",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Load camera state",
            )
        @load_client_camera_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            if not os.path.exists(self.client_camera_save_path):
                print("Saved camera doesn't exist", self.client_camera_save_path)
                return

            with open(self.client_camera_save_path, "r") as f:
                camera_stat = json.load(f)

            client.camera.position = np.array(camera_stat["position"])
            client.camera.wxyz = np.array(camera_stat["wxyz"])
            client.camera.look_at = np.array(camera_stat["look_at"])

        if self.mode is PanelMode.TASK:   # screen record over every prediction model
            screenrecord_button = self.server.gui.add_button(
                "Screenrecord",
                color="Green",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Screenrecord",
            )
            @screenrecord_button.on_click
            def _(event: viser.GuiEvent) -> None:
                client = event.client
                assert client is not None

                init_cam_type = self.camera_type.value
                init_frustum_interval = self.frustum_interval.value
                init_interp_checkbox = self.interpolate_cam_checkbox.value

                output_dir = self.task_output_dir()

                os.makedirs(output_dir, exist_ok=True)

                for cam_type in tqdm(self.render_targets):
                    self.camera_type.value = cam_type
                    self.frustum_interval.value = 10
                    self.interpolate_cam_checkbox.value = True

                    self.clean_all_camera_frustum()
                    renders = self.render_frustums(client)

                    timestamp = int(time.time())
                    save_path = os.path.join(output_dir, f"{cam_type}_frustum_{timestamp}.mp4")
                    # imageio.mimsave(save_path, renders, fps=self.fps.value)
                    save_video_imageio(np.array(renders), save_path, fps=self.fps.value)

                self.camera_type.value = init_cam_type
                self.frustum_interval.value = init_frustum_interval
                self.interpolate_cam_checkbox.value = init_interp_checkbox

                print("Saving video done.")

    def _setup_attention_controls(self):
        """Attention-weighted point colouring."""
        # ===== Attention Viz

        self.attention_threshold = self.server.gui.add_number(
                "Attention Threshold",
                min=0.01,
                max=1.0,
                step=0.01,
                initial_value=0.2,
            )

        color_points_button = self.server.gui.add_button(
                "Color World Points",
                color="Red",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Color Points",
            )
        @color_points_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if self.points is None:
                print("no points")
                return

            if self.attention is None:
                print("no attention")
                return

            attention = self.attention

            cameras = self.get_cameras_by_type()

            f_min, f_max = self.frustum_range.value
            f_min = int(f_min)
            f_max = int(f_max)
            cam_min = f_min
            cam_max = min(f_max, len(cameras))

            cam_max = min(attention.shape[0], cam_max)

            start_color = np.array(self.frustum_start_color.value)
            end_color = np.array(self.frustum_end_color.value)
            t = np.linspace(0, 1, (cam_max - cam_min))[:, None]  # (T, 1)
            colors = (1 - t) * start_color + t * end_color  # (T, 3)

            if self.vis_single_frustum.value:
                i = min(self.vis_frustum_idx.value, cam_max)
                i = max(i, cam_min)
                cam_indices = [i]
            else:
                cam_indices = []
                for i in range(cam_min, cam_max, self.frustum_interval.value):
                    cam_indices.append(i)

            attention = attention[cam_indices]
            colors = colors[[i - cam_min for i in cam_indices]]

            new_rgb = color_points_by_attention(attention, torch.from_numpy(colors).to("cuda"), threshold=self.attention_threshold.value)

            self.server.scene.add_point_cloud(
                name="/pcd",
                points=self.points[:, :3].cpu().numpy(),
                colors=new_rgb.cpu().numpy(),
                point_size=0.01,
            )

            print("Colored Points")

    def load_tasks_from_csv(self, csv_path):
        """Fill the separate csv_task_list used by the CSV Idx slider/buttons.
        This does NOT touch the .txt-based task_list / Task Idx."""
        try:
            csv_tasks, orig_idxs = task_parser.load_csv_tasks(csv_path)
        except FileNotFoundError:
            print("CSV task file doesn't exist:", csv_path)
            return

        if not csv_tasks:
            print("No tasks parsed from CSV:", csv_path)
            return

        self.csv_task_list = csv_tasks
        self.csv_orig_index = orig_idxs
        self._task_index_map = None  # rebuilt on demand against the current task_list
        try:
            self.csv_slider.max = max(len(self.csv_task_list) - 1, 0)
        except Exception as e:
            print("Could not update CSV Idx slider max:", e)
        self.csv_slider.value = 0
        print(f"Loaded {len(self.csv_task_list)} CSV tasks from {csv_path}")

    def get_csv_task(self, csv_idx, client=None):
        """Load the CSV task at csv_idx and sync the Task Idx slider to it."""
        if not self.csv_task_list:
            print("No CSV tasks loaded")
            return

        csv_idx = max(0, min(csv_idx, len(self.csv_task_list) - 1))
        self.csv_slider.value = csv_idx

        task_str = self.csv_task_list[csv_idx]
        orig_index = self.csv_orig_index[csv_idx] if csv_idx < len(self.csv_orig_index) else -1
        if self._task_index_map is None:
            self._task_index_map = {t: i for i, t in enumerate(self.task_list)}
        task_idx = task_parser.resolve_task_idx(task_str, orig_index, self.task_list,
                                                self._task_index_map)
        if task_idx is None:
            print(f"CSV task '{task_str}' not in current task list; Task Idx not synced")
            return

        self.load_task(task_idx, client)  # get_task keeps Task Idx in sync

    def task_output_dir(self, *parts):
        """Output directory for the current task, results/<idx>_<task>, falling back
        to results/ when no task is loaded. Extra components are appended."""
        if self.current_task_idx is not None:
            base = f"{self.output_dir}/{self.current_task_idx}_{self.task_list[self.current_task_idx]}"
        else:
            base = self.output_dir
        return os.path.join(base, *parts) if parts else base

    def load_task(self, idx, client=None):
        """Load the task at idx and draw its frustums. The single entry point for
        both the Task Idx buttons and the CSV navigation."""
        self.get_task(idx)
        self.visualize_task(client)

    def visualize_task(self, client):
        """Draw whatever frustums the current task type calls for."""
        if client is None:
            return
        if task_parser.is_seg_task(self.task_type):
            self.visualize_all_frustums(client)
        elif self.task_type == task_parser.GT_TASK_TYPE:
            self.camera_type.value = "GT"
            self.visualize_camera_frustum(client)

    def get_task(self, idx):
        print("Load Task")
        idx = max(0, min(idx, len(self.task_list) - 1))
        task = self.task_list[idx]
        self.current_task_idx = idx
        self.task_slider.value = idx

        split, scene_name, seg_idx_str = task_parser.parse_task(task, self.task_type)
        self.scene_root_path, self.pcd_path, pcd_type = task_parser.resolve_scene(split, scene_name)
        self.pcd_type.value = pcd_type
        print("pcd :", self.pcd_path)

        self.client_camera_save_path = os.path.join(self.scene_root_path, "saved_client_camera.json")

        self._load_task_text(task, seg_idx_str)

        if self.pcd is not None:
            self.pcd.remove()
            self.pcd = None
            self.points = None
        self.vis_pointcloud()

        with open(os.path.join(self.output_dir, f"{self.task_type}_task_checkpoint.json"), "w") as f:
            json.dump(self.current_task_idx, f)

        self.clean_all_camera_frustum()
        self.interpolate_cam_checkbox.value = False
        self.frustum_interval.value = 5

    def _load_task_text(self, task, seg_idx_str):
        """Print the task's caption/tags, and for segment tasks set frame_idx from
        the segment's frame range."""
        if task_parser.is_seg_task(self.task_type):
            prompt_path = os.path.join(self.scene_root_path, "prompts.json")
            with open(prompt_path, "r") as f:
                prompt_data = json.load(f)
            s, e = prompt_data[seg_idx_str]["frame_idx"]
            self.frame_idx = s

            model_type = self.camera_type.value
            if model_type not in self.valid_camera_pred_types:
                model_type = self.valid_camera_pred_types[0]

            text_path = os.path.join(self.data_root_path, model_type, "test", f"{task}_caption.json")
            with open(text_path, "r") as f:
                text_data = json.load(f)
                print(text_data)
        elif self.task_type == task_parser.GT_TASK_TYPE:
            tag_path = os.path.join(self.scene_root_path, "viz", "camera_tags_per_seg.json")
            with open(tag_path, "r") as f:
                tag_data = json.load(f)
                print("camera tag:", tag_data["0"]["description"])

            text_path = os.path.join(self.scene_root_path, "prompts.json")
            with open(text_path, "r") as f:
                text_data = json.load(f)
                print("camera text:", text_data["0"]["prompt_camera"])
                print("final text:", text_data["0"]["prompt_camera_with_scene_video_inpainted"])
        else:
            raise ValueError("Invalid task type", self.task_type)

    def visualize_all_frustums(self, client):
        s_r, s_g, s_b = self.frustum_start_color.value
        e_r, e_g, e_b = self.frustum_end_color.value

        def set_frustum_color(color):
            self.frustum_start_color.value = (color[0], color[1], color[2])
            self.frustum_end_color.value = (color[0], color[1], color[2])

        for pred_cam_type in self.auto_frustum_targets:
            self.camera_type.value = pred_cam_type
            set_frustum_color(self.preset_frustum_colors[pred_cam_type])
            self.visualize_camera_frustum(client)

        self.camera_type.value = self.valid_camera_pred_types[-1]
        self.frustum_start_color.value = (s_r, s_g, s_b)
        self.frustum_end_color.value = (e_r, e_g, e_b)

        # cameras = self.get_cameras_by_type()
        # first_camera = cameras[0]
        # print(first_camera.camera_to_world)
        # first_camera.camera_to_world = move_camera_back_and_up(first_camera.camera_to_world, 1.0, 1.0)
        # print(first_camera.camera_to_world)
        # fov, quat, t = camera_to_fov_quat_position(first_camera)

    def visualize_camera_frustum(self, client):
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

        if self.vis_single_frustum.value:
            i = min(self.vis_frustum_idx.value, cam_max)
            i = max(i, cam_min)

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
                color=colors[i - cam_min],
            )

            self.frustum_handles[camera_type].append(handle)

        else:
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
                    color=colors[i - cam_min],
                )

                self.frustum_handles[camera_type].append(handle)

    def render_frustums(self, client):
        camera_type = self.camera_type.value

        self.is_frustum_visualized[camera_type] = not self.is_frustum_visualized[camera_type]
        cameras = self.get_cameras_by_type()

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

        renders = []
        for i in tqdm(range(cam_min, cam_max, 1)):
            if i % self.frustum_interval.value == 0:
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
                image = client.camera.get_render(width=1280, height=720)
            renders.append(image)
        
        return renders

    def get_pcd_by_type(self, sample=False):
        return pointcloud_parser.get_pcd_by_type(
            self.pcd_type.value,
            root_path=self.root_path,
            pcd_path=self.pcd_path,
            recon=self.recon,
            sample_num=self.pcd_sample_num.value if sample else None,
        )

    def vis_pointcloud(self):
        if self.pcd is not None:
            self.pcd.remove()
            self.pcd = None
            self.points = None
            return
        
        print("Visualizing Pointcloud")
        points, colors = self.get_pcd_by_type(
            sample=self.pcd_sample_checkbox.value,
        )

        self.pcd = self.server.scene.add_point_cloud(
            name="/pcd",
            points=points,
            colors=colors,
            point_size=self.pcd_scale.value,
        )

        self.points = torch.from_numpy(np.concatenate([points, colors], axis=1)).to("cuda")

        print("Pointcloud Loaded")

    def clean_all_camera_frustum(self):
        print("Clear all camera")
        for camera_type in self.valid_camera_types + self.valid_camera_pred_types:
            for handle_dict in self.frustum_handles[camera_type]:
                    for key, handle in handle_dict.items():
                        handle.remove()
            self.frustum_handles[camera_type].clear()
            self.is_frustum_visualized[camera_type] = False
            
    def get_cameras_by_type(self, sample=False):
        return camera_parser.get_cameras_by_type(
            self.camera_type.value,
            root_camera_path=self.root_camera_path,
            scene_root_path=self.scene_root_path,
            root_path=self.root_path,
            recon=self.recon,
            custom_camera_path=getattr(self, "custom_camera_path", None),
            use_custom_camera=self.custom_camera_checkbox.value,
            custom_camera_data_path=self.custom_camera_data_path,
            data_root_path=self.data_root_path,
            task_list=self.task_list,
            current_task_idx=self.current_task_idx,
            valid_camera_pred_types=self.valid_camera_pred_types,
            task_type=self.task_type,
            frame_idx=self.frame_idx,
            scale_factor=self.scale_factor,
            sample_range=self.frustum_range.value if sample else None,
            sample_interval=self.frustum_interval.value,
            interpolate_num=self.interpolate_cam_num.value if self.interpolate_cam_checkbox.value else None,
        )

    def render_with_client(self, client, cameras: Cameras):
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
        
        print("Rendering done.")

        return images

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

        # timestamp = int(time.time())
        # save_path = os.path.join(self.output_dir, f"{self.current_task_idx}_{self.task_list[self.current_task_idx]}_{self.camera_type.value}_scene_viz_{timestamp}.mp4")
        # imageio.mimsave(save_path, renders, fps=fps)

        # if optimizable is False:
        #     renders = np.array(renders)
        # else:
        #     renders = torch.stack(renders)
        
        # print("Rendering done. Saved to", save_path)

        return renders

    def render_pointcloud_multi_cam(self, cameras: Cameras, H=512, W=512):
        if self.points is None:
            print("no points")
            return
        points_torch = torch.tensor(self.points, device="cuda", dtype=torch.float32)
        points = points_torch[:, :3]
        colors = points_torch[:, 3:]
        device = points.device
        M = points.shape[0]   # number of points

        # (M, 4) homogeneous points
        points_h = torch.cat(
            [points, torch.ones(M, 1, device=device)],
            dim=1
        )

        images = []
        masks = []
        index_maps = []

        extrinsics = cameras.world_to_camera.to(device)

        # print(extrinsics[0])

        for cam_idx, camera in enumerate(cameras):
            fx = camera.fx
            fy = camera.fy
            cx = camera.cx
            cy = camera.cy
            K = torch.tensor([
                [fx, 0, cx, 0],
                [0, fy, cy, 0],
                [0,  0,  1, 0],
                [0,  0,  0, 1],
            ], device=device, dtype=torch.float32)

            H = int(2 * cy)
            W = int(2 * cx)

            # R = camera.R
            # T = camera.T
            # E = torch.eye(4, device=device)
            # E[:3, :3] = R
            # E[:3, 3] = T
            E = extrinsics[cam_idx]

            # World -> Camera
            # cam_pts = (E @ points_h.T).T   # (M, 4)
            cam_pts = points_h @ E

            # Filter
            z = cam_pts[:, 2]
            valid = z > 1e-5
            cam_pts = cam_pts[valid]
            z = z[valid]
            point_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)

            # if cam_idx == 0:
            #     print(K)

            # Camera -> Pixel
            pix = (K @ cam_pts.T).T
            u = pix[:, 0] / pix[:, 2]
            v = pix[:, 1] / pix[:, 2]

            u = u.long()
            v = v.long()

            # img = torch.zeros(H, W, 3, device=device)
            # depth = torch.full((H, W), float("inf"), device=device)
            # idx_map = torch.full((H, W), -1, dtype=torch.long, device=device)

            # # Naive rasterization
            # for i in range(len(u)):
            #     x, y = u[i], v[i]
            #     if 0 <= x < W and 0 <= y < H:
            #         if z[i] < depth[y, x]:
            #             depth[y, x] = z[i]
            #             img[y, x] = colors[point_indices[i]]
            #             idx_map[y, x] = point_indices[i]

            img = torch.ones((H * W, 3), device=device, dtype=colors.dtype)
            depth = torch.full((H * W,), float("inf"), device=device, dtype=z.dtype)
            idx_map = torch.full((H * W,), -1, device=device, dtype=torch.long)
            
            valid = (
                (u >= 0) & (u < W) &
                (v >= 0) & (v < H)
            )

            u = u[valid]
            v = v[valid]
            z = z[valid]
            pidx = point_indices[valid]

            lin_idx = v * W + u   # (Nv,)

            sort_key = lin_idx.to(torch.float64) * 1e10 + z.to(torch.float64)
            order = torch.argsort(sort_key)

            lin_idx = lin_idx[order]
            z = z[order]
            pidx = pidx[order]

            keep = torch.ones_like(lin_idx, dtype=torch.bool)
            keep[1:] = lin_idx[1:] != lin_idx[:-1]

            lin_idx = lin_idx[keep]
            z = z[keep].to(depth.dtype)
            pidx = pidx[keep]

            depth[lin_idx] = z
            idx_map[lin_idx] = pidx
            # img[lin_idx] = colors[pidx].to(img.dtype) / 255
            img[lin_idx] = colors[pidx].to(img.dtype)

            img = img.view(H, W, 3)
            depth = depth.view(H, W)
            idx_map = idx_map.view(H, W)

            images.append(img)
            index_maps.append(idx_map)
            masks.append(idx_map != -1)
        
        renders = torch.stack(images).detach().cpu().numpy()
        print("Rendering Done")

        return renders

    def _handle_new_client(self, client):
        print("hello")
        if self.mode is PanelMode.PCD:
            self.pcd_type.value = self.input_pcd_type
            self.pcd_path = self.ply_path
            self.vis_pointcloud()
        elif self.mode is PanelMode.TASK:
            idx = 0
            ckpt_path = os.path.join(self.output_dir, f"{self.task_type}_task_checkpoint.json")
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    idx = json.load(f)
                print("Task Checkpoint Loaded", idx)

            self.load_task(idx, client)
        else:
            self.scene_root_path = self.root_path



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

def move_camera_back_and_up(c2w: torch.Tensor, back: float, up: float, forward_is_plus_z: bool = True):
    """
    w2c: (4,4) world-to-camera
    back: 뒤로 얼마나 (world unit)
    up: 위로 얼마나 (world unit)
    forward_is_plus_z:
        - True: 카메라 forward가 camera +Z (OpenCV 스타일에서 종종 이렇게 씀)
        - False: 카메라 forward가 camera -Z (OpenGL/graphics 스타일에서 종종 이렇게 씀)
    """
    assert c2w.shape == (4, 4)

    R_c2w = c2w[:3, :3]     # camera axes in world
    t_c2w = c2w[:3, 3]      # camera position in world (C)

    # camera basis in world
    right_w = R_c2w[:, 0]
    up_w    = R_c2w[:, 1]
    z_w     = R_c2w[:, 2]   # camera z-axis in world

    # choose forward direction in world
    forward_w = z_w if forward_is_plus_z else (-z_w)

    # move opposite to forward (back) and along up
    t_c2w_new = t_c2w + (-forward_w) * back + up_w * up

    # rebuild c2w and w2c (keep rotation, change translation)
    c2w_new = c2w.clone()
    c2w_new[:3, 3] = t_c2w_new
    w2c_new = torch.linalg.inv(c2w_new)
    return w2c_new

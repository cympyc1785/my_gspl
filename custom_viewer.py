import os
from pathlib import Path

import viser
import argparse
import numpy as np
import time
import json

import torch

from custom_utils.custom_panel import CustomPanel, PanelMode
from internal.viewer.ui import populate_render_tab, TransformPanel, EditPanel
from internal.viewer.ui.up_direction_folder import UpDirectionFolder

from internal.viewer.viewer import Viewer

DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE = "@Direct"

class CustomViewer(Viewer):
    def __init__(
        self,
        model_paths: list[str],
        root_path: str = None,
        scale_factor: float = 1.0,
        scale_factor_path: str = None,
        custom_camera_path: str = None,
        colmap_path:str = None,
        **kwargs):
        kwargs['model_paths'] = model_paths
        super().__init__(**kwargs)

        self.root_path = root_path
        self.scale_factor = scale_factor
        if scale_factor_path is not None:
            self.scale_factor = json.load(open(scale_factor_path, "r"))
            print("Scale Factor:", self.scale_factor)
        self.custom_camera_path = custom_camera_path
        self.colmap_path = colmap_path
    
    def start(self, block: bool = True, server_config_fun=None, tab_config_fun=None, enable_renderer_options: bool = True):
        # create viser server
        server = viser.ViserServer(host=self.host, port=self.port)
        server.scene.set_up_direction(self.up_direction)
        self._server = server
        server.gui.configure_theme(
            control_layout="collapsible",
            show_logo=False,
        )

        if server_config_fun is not None:
            server_config_fun(self, server)

        tabs = server.gui.add_tab_group()

        if tab_config_fun is not None:
            tab_config_fun(self, server, tabs)

        with tabs.add_tab("General"):
            # add render options
            with server.gui.add_folder("Render", visible=not self.demo_mode):
                self.max_res_when_static = server.gui.add_slider(
                    "Max Res",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1920,
                    disabled=self.demo_mode,
                )
                self.max_res_when_static.on_update(self._handle_option_updated)
                self.jpeg_quality_when_static = server.gui.add_slider(
                    "JPEG Quality",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=100,
                    disabled=self.demo_mode,
                )
                self.jpeg_quality_when_static.on_update(self._handle_option_updated)

                self.max_res_when_moving = server.gui.add_slider(
                    "Max Res when Moving",
                    min=128,
                    max=3840,
                    step=128,
                    initial_value=1280,
                    disabled=self.demo_mode,
                )
                self.jpeg_quality_when_moving = server.gui.add_slider(
                    "JPEG Quality when Moving",
                    min=0,
                    max=100,
                    step=1,
                    initial_value=60,
                    disabled=self.demo_mode,
                )

            if not self.demo_mode:
                self.viewer_renderer.setup_options(self, server)

            with server.gui.add_folder("Model", visible=not self.demo_mode):
                self.scaling_modifier = server.gui.add_slider(
                    "Scaling Modifier",
                    min=0.,
                    max=1.,
                    step=0.01,
                    initial_value=1.,
                    disabled=self.demo_mode,
                )
                self.scaling_modifier.on_update(self._handle_option_updated)

                if self.viewer_renderer.gaussian_model.max_sh_degree > 0:
                    self.active_sh_degree_slider = server.gui.add_slider(
                        "Active SH Degree",
                        min=0,
                        max=self.viewer_renderer.gaussian_model.max_sh_degree,
                        step=1,
                        initial_value=self.viewer_renderer.gaussian_model.max_sh_degree,
                        disabled=self.demo_mode,
                    )
                    self.active_sh_degree_slider.on_update(self._handle_activate_sh_degree_slider_updated)

                if self.available_appearance_options is not None:
                    # find max appearance id
                    max_input_id = 0
                    available_option_values = list(self.available_appearance_options.values())
                    if isinstance(available_option_values[0], list) or isinstance(available_option_values[0], tuple):
                        for i in available_option_values:
                            if i[0] > max_input_id:
                                max_input_id = i[0]
                    else:
                        # convert to tuple, compatible with previous version
                        for i in self.available_appearance_options:
                            self.available_appearance_options[i] = (0, self.available_appearance_options[i])
                    self.available_appearance_options[DROPDOWN_USE_DIRECT_APPEARANCE_EMBEDDING_VALUE] = None

                    self.appearance_id = server.gui.add_slider(
                        "Appearance Direct",
                        min=0,
                        max=max_input_id,
                        step=1,
                        initial_value=0,
                        visible=max_input_id > 0,
                        disabled=self.demo_mode,
                    )

                    self.normalized_appearance_id = server.gui.add_slider(
                        "Normalized Appearance Direct",
                        min=0.,
                        max=1.,
                        step=0.01,
                        initial_value=0.,
                        disabled=self.demo_mode,
                    )

                    appearance_options = list(self.available_appearance_options.keys())

                    self.appearance_group_dropdown = server.gui.add_dropdown(
                        "Appearance Group",
                        options=appearance_options,
                        initial_value=appearance_options[0],
                        disabled=self.demo_mode,
                    )
                    self.appearance_id.on_update(self._handle_appearance_embedding_slider_updated)
                    self.normalized_appearance_id.on_update(self._handle_appearance_embedding_slider_updated)
                    self.appearance_group_dropdown.on_update(self._handel_appearance_group_dropdown_updated)

                self.time_slider = server.gui.add_slider(
                    "Time",
                    min=0.,
                    max=1.,
                    step=0.01,
                    initial_value=0.,
                    disabled=self.demo_mode,
                )
                self.time_slider.on_update(self._handle_option_updated)

            # add cameras
            if self.show_cameras is True:
                self.add_cameras_to_scene(server)

            if not self.demo_mode:
                UpDirectionFolder(self, server)

            go_to_scene_center = server.gui.add_button(
                "Go to scene center",
            )

            @go_to_scene_center.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                event.client.camera.position = self.camera_center + np.asarray([2., 0., 0.])
                event.client.camera.look_at = self.camera_center

        if self.show_edit_panel is True:
            with tabs.add_tab("Edit") as edit_tab:
                self.edit_panel = EditPanel(server, self, edit_tab)

        self.transform_panel: TransformPanel = None
        if self.enable_transform is True:
            with tabs.add_tab("Transform"):
                self.transform_panel = TransformPanel(server, self, self.loaded_model_count)

        if self.show_render_panel is True:
            with tabs.add_tab("Render"):
                populate_render_tab(
                    server,
                    self,
                    self.model_paths,
                    Path("./"),
                    orientation_transform=torch.linalg.inv(self.camera_transform).cpu().numpy(),
                    enable_transform=self.enable_transform,
                    background_color=self.background_color,
                    sh_degree=self.sh_degree,
                    extra_args=self.extra_video_render_args,
                )

        if self.enable_measurement:
            with tabs.add_tab("Measure") as tab:
                from internal.viewer.ui.distance_measurement import DistanceMeasurementPanel
                self.measure_panel = DistanceMeasurementPanel(
                    viewer=self,
                    server=server,
                    tab=tab,
                )

        if enable_renderer_options is True and not self.demo_mode:
            self.viewer_renderer.renderer.setup_web_viewer_tabs(self, server, tabs)

        recon = None
        if self.colmap_path is not None:
            print("Colmap path exists")
            import pycolmap
            sparse_model_dir = self.colmap_path
            if not os.path.exists(os.path.join(sparse_model_dir, "images.bin")) and os.path.exists(os.path.join(sparse_model_dir, "sparse")):
                sparse_model_dir = os.path.join(sparse_model_dir, "sparse")
            if not os.path.exists(os.path.join(sparse_model_dir, "images.bin")) and os.path.exists(os.path.join(sparse_model_dir, "0")):
                sparse_model_dir = os.path.join(sparse_model_dir, "0")
            recon = pycolmap.Reconstruction(sparse_model_dir)

        with tabs.add_tab("Custom"):
            self.custom_panel = CustomPanel(self, self._server, self.root_path, self.viewer_renderer, recon=recon,
                                            mode=PanelMode.GS)

        # register hooks
        server.on_client_connect(self._handle_new_client)
        server.on_client_disconnect(self._handle_client_disconnect)

        if block is True:
            while True:
                time.sleep(999)


def cli():
    # define arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("root_path", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--colmap_path", type=str, default=None)
    parser.add_argument("--scale_factor_path", type=str, default=None)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--custom_camera_path", type=str, default=None)
    parser.add_argument("--host", "-a", type=str, default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=9999)
    parser.add_argument("--background_color", "--background_color", "--bkg_color", "-b",
                        type=str, nargs="+", default=["black"],
                        help="e.g.: white, black, 0 0 0, 1 1 1")
    parser.add_argument("--image_format", "--image-format", "-f", type=str, default="jpeg")
    parser.add_argument("--reorient", "-r", type=str, default="auto",
                        help="whether reorient the scene, available values: auto, enable, disable")
    parser.add_argument("--sh_degree", "--sh-degree", "--sh",
                        type=int, default=3)
    parser.add_argument("--enable_transform", "--enable-transform",
                        action="store_true", default=False,
                        help="Enable transform options on Web UI. May consume more memory")
    parser.add_argument("--enable_measurement", "--enable-measurement", "--measure",
                        action="store_true", default=False)
    parser.add_argument("--show_cameras", "--show-cameras",
                        action="store_true")
    parser.add_argument("--cameras-json", "--cameras_json", type=str, default=None)
    parser.add_argument("--vanilla_deformable", action="store_true", default=False)
    parser.add_argument("--vanilla_gs4d", action="store_true", default=False)
    parser.add_argument("--vanilla_gs2d", action="store_true", default=False)
    parser.add_argument("--up", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--default_camera_position", "--dcp", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--default_camera_look_at", "--dcla", nargs=3, required=False, type=float, default=None)
    parser.add_argument("--no_edit_panel", action="store_true", default=False)
    parser.add_argument("--no_render_panel", action="store_true", default=False)
    parser.add_argument("--demo_mode", action="store_true", default=False)
    parser.add_argument("--gsplat", action="store_true", default=False,
                        help="Use gsplat v1 renderer for ply file")
    parser.add_argument("--gsplat_aa", action="store_true", default=False,
                        help="Enable gsplat's anti-aliasing for ply file")
    parser.add_argument("--gsplat_v1_example", action="store_true", default=False,
                        help="Load checkpoint generated by the gsplat repo.")
    parser.add_argument("--gsplat_v1_example_aa", action="store_true", default=False,
                        help="Enable anti-aliased for the gsplat repo.'s model")
    parser.add_argument("--seganygs", type=str, default=None,
                        help="Path to a SegAnyGaussian model output directory or checkpoint file")
    parser.add_argument("--vanilla_seganygs", action="store_true", default=False)
    parser.add_argument("--vanilla_mip", action="store_true", default=False)
    parser.add_argument("--vanilla_pvg", action="store_true", default=False)
    parser.add_argument("--difix", action="store_true", default=False)
    parser.add_argument("--float32_matmul_precision", "--fp", type=str, default=None)
    args = parser.parse_args()

    # set torch float32_matmul_precision
    if args.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
    del args.float32_matmul_precision

    # arguments post process
    if len(args.background_color) == 1 and isinstance(args.background_color[0], str):
        if args.background_color[0] == "white":
            args.background_color = (1., 1., 1.)
        else:
            args.background_color = (0., 0., 0.)
    else:
        args.background_color = tuple([float(i) for i in args.background_color])

    # create viewer
    viewer_init_args = {key: getattr(args, key) for key in vars(args) if key != "model_path"}
    
    if args.model_path is not None:
        viewer_init_args['model_paths'] = [args.model_path]
    else:
        viewer_init_args['model_paths'] = [os.path.join(args.root_path, "scene.ply")] if args.root_path is not None else []

    viewer = CustomViewer(**viewer_init_args)

    # start viewer server
    viewer.start()

if __name__ == "__main__":
    cli()
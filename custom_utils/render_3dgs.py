import glob
import os, sys
import queue
import subprocess
import argparse
import json
import threading
import traceback

import numpy as np
import lightning
import torch
import torchvision
import mediapy
import imageio
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from internal.cameras.cameras import Cameras
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_model_editor import MultipleGaussianModelEditor
from internal.viewer.renderer import ViewerRenderer

def initializer_viewer_renderer(
        model_paths: list[str],
        enable_transform: bool = False,
        renderer_override = None,
        device = "cpu",
        background_color = [0, 0, 0],
        difix: bool = False,
) -> ViewerRenderer:
    if len(model_paths) == 1 and model_paths[0].endswith(".yaml"):
        import yaml
        from internal.models.vanilla_gaussian import VanillaGaussian
        model = VanillaGaussian().instantiate()
        model.setup_from_number(0)
        model.pre_activate_all_properties()
        model.eval()
        from internal.renderers.partition_lod_renderer import PartitionLoDRenderer
        with open(model_paths[0], "r") as f:
            lod_config = yaml.safe_load(f)
        renderer = PartitionLoDRenderer(**lod_config).instantiate()
        renderer.setup("validation")

        model_manager = model
    else:
        model_list = []
        renderer = None

        load_device = torch.device("cuda") if len(model_paths) == 1 or enable_transform is False else torch.device("cpu")
        # TODO: pick the renderer from the first model so that it is consistent with the viewer
        for model_path in model_paths:
            model, renderer = GaussianModelLoader.search_and_load(model_path, load_device)
            model.freeze()
            model_list.append(model)

        if renderer_override is not None:
            print(f"Renderer: {renderer_override.__class__}")
            renderer = renderer_override

        model_manager = MultipleGaussianModelEditor(model_list, device)

    renderer = ViewerRenderer(
        model_manager,
        renderer,
        torch.tensor(background_color, dtype=torch.float, device=device),
        difix=difix,
    )
    # Enable difix
    if difix:
        renderer.difix_enabled = True
        renderer._set_output_type("rgb", renderer_output_info=renderer.renderer.get_available_outputs()["rgb"])

    return renderer

def load_cameras(camera_path:str):
    if camera_path.endswith(".npz"):
        return load_cameras_from_npz(camera_path)
    elif camera_path.endswith(".json"):
        return load_cameras_from_json(camera_path)
    else:
        raise ValueError("Invalid Camera File Type", camera_path)

def load_cameras_from_npz(npz_path: str):
    data = np.load(npz_path)

    w2c = torch.tensor(data["extrinsics"], dtype=torch.float32)  # (N,4,4) c2w
    intrinsics = torch.tensor(data["intrinsics"], dtype=torch.float32)  # (N,3,3)

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    # assume all frames share same resolution
    width = torch.full_like(fx, int(cx[0] * 2), dtype=torch.int16)
    height = torch.full_like(fy, int(cy[0] * 2), dtype=torch.int16)

    return Cameras(
        R=w2c[:, :3, :3],
        T=w2c[:, :3, 3],
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        appearance_id=torch.zeros_like(fx, dtype=torch.long),
        normalized_appearance_id=torch.zeros_like(fx),
        distortion_params=None,
        camera_type=torch.zeros_like(fx),
    )

def load_cameras_from_json(json_path: str):
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

    cameras = build_cameras(extrinsics, intrinsics)

    return cameras

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

def render(
        cameras: Cameras,
        viewer_renderer: ViewerRenderer,
        output_path: str = "./scene_viz.mp4",
        fps: int = 10,
        start_frame=0,
        end_frame=-1,
):
    renders = []
    if end_frame < 0:
        end_frame = len(cameras)
    end_frame = min(len(cameras), end_frame)
    print(f"Rendering frame {start_frame} to {end_frame}")

    for cam_idx in tqdm(range(start_frame, end_frame), desc="rendering frames"):
        camera = cameras[cam_idx]
        render = viewer_renderer.get_outputs(camera.to_device("cuda")).detach().cpu().numpy()
        # Clipping
        render = np.clip(render, 0.0, 1.0)
        render = (render * 255).astype(np.uint8).transpose(1, 2, 0)
        renders.append(render)
    
    save_path = os.path.join(output_path)
    imageio.mimsave(save_path, renders, fps=fps)
        
def get_renders(
        model_path,
        cameras: Cameras,
):
    device = torch.device("cuda")

    viewer_renderer = initializer_viewer_renderer(
        [model_path],
        device=device
    )

    renders = []
    for camera in cameras:
        render = viewer_renderer.get_outputs(camera.to_device("cuda")).detach().cpu().numpy()
        # Clipping
        render = np.clip(render, 0.0, 1.0)
        render = (render * 255).astype(np.uint8).transpose(1, 2, 0)
        renders.append(render)

    return renders

def render_3dgs(
        model_path,
        camera_path,
        output_path,
        fps=10,
        type=None,
        save_images=False,
        image_save_batch=8,
        disable_transform=False,
        vanilla_gs2d=False,
        difix=False,
        start_frame=0,
        end_frame=-1,
):
    device = torch.device("cuda")

    # whether a 2DGS model
    renderer_override = None
    if vanilla_gs2d is True:
        from internal.renderers.vanilla_2dgs_renderer import Vanilla2DGSRenderer

        renderer_override = Vanilla2DGSRenderer()

    # instantiate renderer
    # TODO: set output type
    renderer = initializer_viewer_renderer(
        [model_path],
        renderer_override=renderer_override,
        device=device,
        difix=difix,
    )

    if type is not None:
        renderer._set_output_type(type, renderer.renderer.get_available_outputs()[type])

    # load cameras
    cameras = load_cameras(camera_path)

    # create output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame_output_path = None
    if save_images is True:
        frame_output_path = output_path + "_frames"
        os.makedirs(frame_output_path, exist_ok=True)
        for i in glob.glob(os.path.join(frame_output_path, "*.png")):
            os.unlink(i)

    # Sequential Rendering 
    render(cameras, renderer, output_path, start_frame=start_frame, end_frame=end_frame)

    if frame_output_path is not None:
        print(f"Video frames saved to '{frame_output_path}'")
    print(f"Video saved to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("model_paths", type=str, nargs="+")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--camera-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--type", type=str, default=None)
    parser.add_argument("--save-images", "--save-image", "--save_image", "--save-frames", action="store_true",
                        help="Whether save each frame to an image file")
    parser.add_argument("--image-save-batch", "-b", type=int, default=8,
                        help="increase this to speedup rendering, but more memory will be consumed")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--disable-transform", action="store_true", default=False)
    parser.add_argument("--vanilla_gs2d", action="store_true", default=False)
    parser.add_argument("--difix", action="store_true", default=False)
    args = parser.parse_args()

    render_3dgs(
        model_path=args.model_path,
        camera_path=args.camera_path,
        output_path=args.output_path,
        fps=args.fps,
        save_images=args.save_images,
        image_save_batch=args.image_save_batch,
        disable_transform=args.disable_transform,
        vanilla_gs2d=args.vanilla_gs2d,
        difix=args.difix,
    )

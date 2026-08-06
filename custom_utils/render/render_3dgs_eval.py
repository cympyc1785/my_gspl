import glob
import os, sys
import queue
import subprocess
import argparse
import json
import threading
import traceback
import csv

import numpy as np
import lightning
import torch
import torchvision
import mediapy
import imageio
from tqdm import tqdm
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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

def load_cameras(camera_path:str, gt_camera_path=None, model_type=None):
    if camera_path.endswith(".npz"):
        return load_cameras_from_npz(camera_path)
    elif camera_path.endswith(".json"):
        return load_cameras_from_json(camera_path, gt_camera_path, model_type)
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

def load_cameras_from_json(json_path: str, gt_camera_path=None, model_type=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if gt_camera_path is not None:
        with open(gt_camera_path, 'r') as f:
            gt_camera_data = json.load(f)
        gt_camera_c2w_0 = np.array(gt_camera_data['frames'][0]['transform_matrix'])
        pred_camera_c2w_0 = np.array(data['frames'][0]['transform_matrix'])
        T = gt_camera_c2w_0 @ np.linalg.inv(pred_camera_c2w_0)

    
    if model_type == 'director' or model_type == 'director3d':
        width = gt_camera_data['w']
        height = gt_camera_data['h']
        fx = gt_camera_data['fl_x']
        fy = gt_camera_data['fl_y']
        cx = gt_camera_data['cx']
        cy = gt_camera_data['cy']
    if model_type == 'ours':
        fx = data['fl_x'] / data['w'] * gt_camera_data['fl_x']
        fy = data['fl_y'] / data['h'] * gt_camera_data['fl_y']
        cx = data['cx']
        cy = data['cy']
    else:
        fx = gt_camera_data['fl_x']
        fy = gt_camera_data['fl_y']
        cx = gt_camera_data['cx']
        cy = gt_camera_data['cy']

    extrinsic_list = []
    intrinsic_list = []
    for params in data['frames']:
        c2w = np.array(params['transform_matrix'])
        if gt_camera_path is not None:
            c2w = T @ c2w
        c2w[:3, 1:3] *= -1 # Conversion from clatr to viser
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]

        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_w2c
        extrinsic[:3, 3] = t_w2c
        

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

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    width = intrinsics[:, 0, 2] * 2
    height = intrinsics[:, 1, 2] * 2
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

    for cam_idx in range(start_frame, end_frame):
        camera = cameras[cam_idx]
        render = viewer_renderer.get_outputs(camera.to_device("cuda")).detach().cpu().numpy()
        # Clipping
        render = np.clip(render, 0.0, 1.0)
        render = (render * 255).astype(np.uint8).transpose(1, 2, 0)
        renders.append(render)
    
    save_path = os.path.join(output_path)
    imageio.mimsave(save_path, renders, fps=fps)
    return renders

def npy_to_pil_list(arr):
    # arr: (N, 3, H, W)
    arr = np.array(arr)
    arr = arr.transpose(0, 2, 3, 1)  # (N, H, W, 3)
    images = [Image.fromarray(img) for img in arr]
    return images

def render_3dgs(
        renderer,
        camera_path,
        gt_camera_path,
        output_path,
        fps=10,
        type=None,
        save_images=False,
        save_depths=False,
        image_save_batch=8,
        disable_transform=False,
        vanilla_gs2d=False,
        difix=False,
        start_frame=0,
        end_frame=-1,
        model_type=None,
):
    if type is not None:
        renderer._set_output_type(type, renderer.renderer.get_available_outputs()[type])

    # load cameras
    cameras = load_cameras(camera_path, gt_camera_path, model_type)

    # create output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame_output_path = None
    if save_images is True:
        frame_output_path = output_path + "_frames"
        os.makedirs(frame_output_path, exist_ok=True)
        for i in glob.glob(os.path.join(frame_output_path, "*.png")):
            os.unlink(i)

    # Sequential Rendering 
    renders = render(cameras, renderer, output_path, start_frame=start_frame, end_frame=end_frame, fps=fps)
    return renders

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, default='/data1/cympyc1785/SceneData/DL3DV/scenes')
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--clip_version', type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=None)
    args = parser.parse_args()

    device = "cuda:0"
    clip = CLIPScore(model_name_or_path=args.clip_version).to(device)
    # whether a 2DGS model
    renderer_override = None
    # instantiate renderer
    # TODO: set output type

    model_type = args.model_type
    data_dir_path = args.data_dir_path
    pred_dir_path = os.path.join(data_dir_path, 'test')
     
    dataset_dir = args.dataset_dir
    data_name_list = [
        f.removesuffix("_transforms_pred.json")
        for f in os.listdir(pred_dir_path)
        if f.endswith("_transforms_pred.json") and f[1] == 'K' and int(f[0]) > args.chunk * 2 and int(f[0]) < (args.chunk + 1) * 2
    ]
    result_path = os.path.join(data_dir_path, f'clip_scores_{args.chunk}.csv')
    metric_path = os.path.join(data_dir_path, f'clip_score_{args.chunk}.json')
    results = []
    for data_name in tqdm(data_name_list):
        dataset_name, data_dir, data_idx = data_name.split('_')
        data_path = os.path.join(dataset_dir, dataset_name, data_dir)
        scene_path = os.path.join(data_path, 'scene.ply')
        camera_path = os.path.join(pred_dir_path, data_name + "_transforms_pred.json")
        gt_camera_path = os.path.join(pred_dir_path, data_name + "_transforms_ref.json")
        caption_path = os.path.join(pred_dir_path, data_name + "_caption.json")
        render_video_path = os.path.join(pred_dir_path, data_name + "_render.mp4")

        renderer = initializer_viewer_renderer(
            [scene_path],
            renderer_override=renderer_override,
            device=device,
        )

        video = render_3dgs(
            renderer,
            camera_path=camera_path,
            gt_camera_path=gt_camera_path,
            output_path=render_video_path,
            fps=args.fps,
            model_type=args.model_type,
        )
        with open(caption_path, 'r') as f:
            # text = json.load(f)['Concise Interaction']
            text = json.load(f)['scene_text']
        video = torch.tensor(np.array(video)).permute(0, 3, 1, 2).to(device)
        clip_score = clip(video, [text] * video.shape[0]).item()
        results.append((os.path.basename(render_video_path), clip_score, text))

    avg_clip_score = sum([out[1] for out in results]) / len(results)
    
    with open(result_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "clip_score", "text"])  # header
        writer.writerows(results)

    with open(metric_path, 'w') as f:
        json.dump({
            "clip_score": avg_clip_score
        }, f, indent=4)
    print(f"CLIP Score per data: {result_path}")
    print(f"Average CLIP Score: {metric_path}")
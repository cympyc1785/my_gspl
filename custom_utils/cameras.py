from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

class CameraType:
    PERSPECTIVE: int = 0
    FISHEYE: int = 1


@dataclass
class Camera:
    idx: Tensor
    R: Tensor  # [3, 3]
    T: Tensor  # [3]
    fx: Tensor
    fy: Tensor
    fov_x: Tensor
    fov_y: Tensor
    cx: Tensor
    cy: Tensor
    width: Tensor
    height: Tensor
    appearance_id: Tensor
    normalized_appearance_id: Tensor
    time: Tensor

    distortion_params: Optional[Tensor]
    """
    NOTE: this should be None or a zero tensor currently
        
    For perspective: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    For fisheye: (k1, k2, k3, k4)
    """

    camera_type: Tensor

    world_to_camera: Tensor
    camera_to_world: Tensor
    projection: Tensor
    full_projection: Tensor
    camera_center: Tensor

    def to_device(self, device):
        for field in Camera.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                setattr(self, field, value.to(device))

        return self

    def get_K(self):
        K = torch.eye(4, dtype=torch.float, device=self.device)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy

        return K

    def get_full_perspective_projection(self):
        K = self.get_K()

        # full.transpose() = (K[R T]).transpose() = [R T].transpose() K.transpose()

        return self.world_to_camera @ K.T        

    @property
    def device(self):
        return self.R.device

@dataclass
class CamerasInterface:
    R: Tensor  # [n_cameras, 3, 3]
    T: Tensor  # [n_cameras, 3]
    fx: Tensor  # [n_cameras]
    fy: Tensor  # [n_cameras]
    fov_x: Tensor = field(init=False)  # [n_cameras]
    fov_y: Tensor = field(init=False)  # [n_cameras]
    cx: Tensor  # [n_cameras]
    cy: Tensor  # [n_cameras]
    width: Tensor  # [n_cameras]
    height: Tensor  # [n_cameras]
    appearance_id: Tensor  # [n_cameras]
    normalized_appearance_id: Optional[Tensor]  # [n_cameras]
    distortion_params: Optional[Union[Tensor, list[Tensor]]]
    camera_type: Tensor  # Int[n_cameras]

@dataclass
class Cameras:
    """
    Y down, Z forward
    world-to-camera
    """

    R: Tensor  # [n_cameras, 3, 3]
    T: Tensor  # [n_cameras, 3]
    fx: Tensor  # [n_cameras]
    fy: Tensor  # [n_cameras]
    fov_x: Tensor = field(init=False)  # [n_cameras]
    fov_y: Tensor = field(init=False)  # [n_cameras]
    cx: Tensor  # [n_cameras]
    cy: Tensor  # [n_cameras]
    width: Tensor  # [n_cameras]
    height: Tensor  # [n_cameras]
    appearance_id: Tensor  # [n_cameras]
    normalized_appearance_id: Optional[Tensor]  # [n_cameras]

    distortion_params: Optional[Union[Tensor, list[Tensor]]]
    """
    NOTE: this should be None or zero tensors currently

    For perspective: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    For fisheye: (k1, k2, k3, k4)
    """

    camera_type: Tensor  # Int[n_cameras]

    world_to_camera: Tensor = field(init=False)  # [n_cameras, 4, 4], transposed
    camera_to_world: Tensor = field(init=False)
    projection: Tensor = field(init=False)
    full_projection: Tensor = field(init=False)
    camera_center: Tensor = field(init=False)

    intrinsics: Tensor = field(init=False)

    time: Optional[Tensor] = None  # [n_cameras]

    idx: Tensor = None  # [N_cameras]

    def _set_intrinsit(self):
        N = self.R.shape[0]
        intrinsics = torch.zeros((N, 3, 3), device=self.R.device)
        intrinsics[:, 0, 0] = self.fx
        intrinsics[:, 1, 1] = self.fy
        intrinsics[:, 0, 2] = self.cx
        intrinsics[:, 1, 2] = self.cy
        intrinsics[:, 2, 2] = 1
        self.intrinsics = intrinsics

    def _calculate_fov(self):
        # calculate fov
        self.fov_x = 2 * torch.atan((self.width / 2) / self.fx)
        self.fov_y = 2 * torch.atan((self.height / 2) / self.fy)

    def _calculate_w2c(self):
        # build world-to-camera transform matrix
        self.world_to_camera = torch.zeros((self.R.shape[0], 4, 4))
        self.world_to_camera[:, :3, :3] = self.R
        self.world_to_camera[:, :3, 3] = self.T
        self.world_to_camera[:, 3, 3] = 1.
        self.world_to_camera = torch.transpose(self.world_to_camera, 1, 2)
    
    def _calculate_c2w(self):
        # build camera-to-world transform matrix
        R_c2w = self.R.transpose(-1, -2)
        t_c2w = (- R_c2w @ self.T.unsqueeze(-1)).squeeze(-1)
        
        self.camera_to_world = torch.zeros((R_c2w.shape[0], 4, 4))
        self.camera_to_world[:, :3, :3] = R_c2w
        self.camera_to_world[:, :3, 3] = t_c2w
        self.camera_to_world[:, 3, 3] = 1.

    def _calculate_ndc_projection_matrix(self):
        """
        calculate ndc projection matrix
        http://www.songho.ca/opengl/gl_projectionmatrix.html

        TODO:
            1. support colmap refined principal points
            2. the near and far here are ignored in diff-gaussian-rasterization
        """
        zfar = 100.0
        znear = 0.01

        tanHalfFovY = torch.tan((self.fov_y / 2))
        tanHalfFovX = torch.tan((self.fov_x / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(self.fov_y.shape[0], 4, 4)

        z_sign = 1.0

        P[:, 0, 0] = 2.0 * znear / (right - left)  # = 1 / tanHalfFovX = 2 * fx / width
        P[:, 1, 1] = 2.0 * znear / (top - bottom)  # = 2 * fy / height
        P[:, 0, 2] = (right + left) / (right - left)  # = 0, right + left = 0
        P[:, 1, 2] = (top + bottom) / (top - bottom)  # = 0, top + bottom = 0
        P[:, 3, 2] = z_sign
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 2, 3] = -(zfar * znear) / (zfar - znear)

        self.projection = torch.transpose(P, 1, 2)

        self.full_projection = self.world_to_camera.bmm(self.projection)

    def _calculate_camera_center(self):
        self.camera_center = torch.linalg.inv(self.world_to_camera)[:, 3, :3]

    def __post_init__(self):
        self._calculate_fov()
        self._calculate_w2c()
        self._calculate_c2w()
        self._calculate_ndc_projection_matrix()
        self._calculate_camera_center()
        self._set_intrinsit()

        self.idx = torch.arange(self.R.shape[0], dtype=torch.int32)

        if self.time is None:
            self.time = torch.zeros(self.R.shape[0])
        if self.distortion_params is None:
            self.distortion_params = torch.zeros(self.R.shape[0], 4)

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, index) -> Camera:
        return Camera(
            idx=self.idx[index],
            R=self.R[index],
            T=self.T[index],
            fx=self.fx[index],
            fy=self.fy[index],
            fov_x=self.fov_x[index],
            fov_y=self.fov_y[index],
            cx=self.cx[index],
            cy=self.cy[index],
            width=self.width[index],
            height=self.height[index],
            appearance_id=self.appearance_id[index],
            normalized_appearance_id=self.normalized_appearance_id[index],
            distortion_params=self.distortion_params[index],
            time=self.time[index],
            camera_type=self.camera_type[index],
            world_to_camera=self.world_to_camera[index],
            camera_to_world=self.camera_to_world[index],
            projection=self.projection[index],
            full_projection=self.full_projection[index],
            camera_center=self.camera_center[index],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

def make_list_to_cameras(cameras:list[Camera])->Cameras:
    if len(cameras) == 0:
        raise ValueError("cameras list is empty")

    device = cameras[0].device
    # dtype은 R 기준으로 맞추는 게 안전
    dtype = cameras[0].R.dtype

    def _stack(name: str):
        vals = [getattr(c, name) for c in cameras]
        # Optional 텐서는 None이 섞일 수 있으니 여기선 stack 대상만
        if any(v is None for v in vals):
            return None
        # 스칼라 텐서/벡터/행렬 모두 torch.stack 가능
        return torch.stack([v.to(device=device) for v in vals], dim=0)

    R = _stack("R").to(dtype=dtype)
    T = _stack("T").to(dtype=dtype)

    fx = _stack("fx").to(dtype=dtype)
    fy = _stack("fy").to(dtype=dtype)
    cx = _stack("cx").to(dtype=dtype)
    cy = _stack("cy").to(dtype=dtype)

    width = _stack("width").to(dtype=dtype)
    height = _stack("height").to(dtype=dtype)

    appearance_id = _stack("appearance_id")
    if appearance_id is None:
        appearance_id = torch.zeros(len(cameras), device=device, dtype=torch.int64)
    else:
        appearance_id = appearance_id.to(device=device)

    normalized_appearance_id = _stack("normalized_appearance_id")
    # None allowed 그대로 둠

    distortion_params = _stack("distortion_params")
    # Cameras.__post_init__에서 None이면 zeros 만들어주니 None 허용

    time = _stack("time")
    # Cameras.__post_init__에서 None이면 zeros 만들어주니 None 허용

    camera_type = _stack("camera_type")
    if camera_type is None:
        camera_type = torch.zeros(len(cameras), device=device, dtype=torch.int32)

    cams = Cameras(
        R=R,
        T=T,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        appearance_id=appearance_id,
        normalized_appearance_id=normalized_appearance_id,
        distortion_params=distortion_params,
        camera_type=camera_type,
        time=time,
    )

    # idx는 Cameras.__post_init__에서 arange로 만들지만,
    # 입력 Camera.idx를 보존하고 싶으면 덮어쓰기
    idx = _stack("idx")
    if idx is not None:
        cams.idx = idx.to(device=device)

    return cams


@dataclass
class BatchCameras:
    """
    Batch of Cameras

    Y down, Z forward
    world-to-camera
    """

    R: Tensor  # [B, n_cameras, 3, 3]
    T: Tensor  # [B, n_cameras, 3]
    fx: Tensor  # [B, n_cameras]
    fy: Tensor  # [B, n_cameras]
    cx: Tensor  # [B, n_cameras]
    cy: Tensor  # [B, n_cameras]
    width: Tensor  # [B, n_cameras]
    height: Tensor  # [B, n_cameras]
    appearance_id: Tensor  # [B, n_cameras]
    normalized_appearance_id: Optional[Tensor]  # [B, n_cameras]

    distortion_params: Optional[Union[Tensor, list[Tensor]]]
    """
    NOTE: this should be None or zero tensors currently

    For perspective: (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements
    For fisheye: (k1, k2, k3, k4)
    """

    camera_type: Tensor  # Int[B, n_cameras]

    def __len__(self):
        return self.R.shape[0]

    def __getitem__(self, index) -> Cameras:
        return Cameras(
            R=self.R[index],
            T=self.T[index],
            fx=self.fx[index],
            fy=self.fy[index],
            cx=self.cx[index],
            cy=self.cy[index],
            width=self.width[index],
            height=self.height[index],
            appearance_id=self.appearance_id[index],
            normalized_appearance_id=self.normalized_appearance_id[index],
            distortion_params=self.distortion_params[index],
            camera_type=self.camera_type[index],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

def build_cameras(extrinsics, intrinsics):
    """
    Input:
        extrinsics: [N, 4, 4] numpy array
        intrinsics: [N, 3, 3] numpy array
    """
    if isinstance(extrinsics, np.ndarray):
        extrinsics = torch.from_numpy(extrinsics)
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.from_numpy(intrinsics)
    
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
        R=R_w2c,
        T=T_w2c,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        appearance_id=appearance_id,
        normalized_appearance_id=normalized_appearance_id,
        distortion_params=distortion_params,
        camera_type=camera_type,
    )
    return cameras


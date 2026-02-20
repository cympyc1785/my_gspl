import numpy as np


# ============================================================
# 1. Compute far-plane corners from FOV + aspect
# ============================================================
def far_plane_corners_local_from_fov(fov_y, aspect, far):
    """
    Compute far-plane corners in CAMERA-LOCAL coordinates.
    """
    half_h = np.tan(fov_y / 2) * far
    half_w = half_h * aspect

    return np.array([
        [-half_w,  half_h, far],   # TL
        [ half_w,  half_h, far],   # TR
        [ half_w, -half_h, far],   # BR
        [-half_w, -half_h, far],   # BL
    ], dtype=float)

def add_gradient_spline_segments(
    scene,
    name,
    p0,                     # (3,) start point
    p1,                     # (3,) end point
    wxyz,                   # camera/world rotation quaternion
    position,               # camera/world translation
    n_segments=20,          # 원하는 segment 개수
    color_start=(1, 0, 0),  # 시작 색 (R,G,B)
    color_end=(0, 0, 1),    # 끝 색 (R,G,B)
    thickness=3.0,
    handles={},
    visible=True,
):
    """
    p0 → p1 방향으로 n개의 segment spline을 만들고,
    색을 color_start → color_end로 그라데이션시키며 추가한다.
    """
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)

    color_start = np.array(color_start, float)
    color_end   = np.array(color_end, float)

    for i in range(n_segments):
        t0 = i / n_segments
        t1 = (i + 1) / n_segments

        seg_p0 = p0 * (1 - t0) + p1 * t0
        seg_p1 = p0 * (1 - t1) + p1 * t1

        # linear color interpolation
        seg_color = color_start * (1 - t0) + color_end * t0
        seg_color = tuple(seg_color.tolist())

        seg_name = f"{name}_seg_{i}"

        handles[seg_name] = scene.add_spline_catmull_rom(
            name=seg_name,
            positions=np.array([seg_p0, seg_p1]),
            line_width=float(thickness),
            color=seg_color,
            wxyz=wxyz,
            position=position,
            visible=visible,
        )

    return handles



# ============================================================
# 2. Add frustum splines to the Viser scene
# ============================================================
def add_gradation_frustum_spline(
    scene,
    name,
    fov_y,          # vertical FOV (radians)
    aspect,         # width / height
    wxyz,           # (4,) quaternion (w, x, y, z)
    position,       # (3,) camera position
    far=1.0,
    scale=0.1,
    thickness=3.0,
    color=(1.0, 0, 0),
    visible=True,
):
    """
    Draw a frustum using:
        - fov_y
        - aspect
        - wxyz, position (camera world frame)
    ONLY far-plane (no near plane).
    """

    # camera origin in LOCAL coordinates
    cam_local = np.zeros(3, dtype=float)

    # far-plane corners (camera-local)
    far_pts = far_plane_corners_local_from_fov(fov_y, aspect, far)

    # scaling
    far_pts *= float(scale)

    handles = {}

    # 1) Rays
    for i in range(4):
        spline_name = f"{name}/ray_{i}"
        handles = add_gradient_spline_segments(
            scene=scene,
            name=spline_name,
            p0=cam_local,
            p1=far_pts[i],
            wxyz=wxyz,
            position=position,
            n_segments=5,
            color_start=color * 0.0,
            color_end=color,
            thickness=float(thickness),
            handles=handles,
            visible=visible,
        )

    # 2) Far-plane rectangle
    for i in range(4):
        p0 = far_pts[i]
        p1 = far_pts[(i + 1) % 4]
        spline_name = f"{name}/rect_{i}"
        handles[spline_name] = scene.add_spline_catmull_rom(
            name=spline_name,
            positions=np.array([p0, p1], dtype=float),
            line_width=float(thickness),
            color=color,
            wxyz=wxyz,
            position=position,
            visible=visible,
        )

    return handles

def add_frustum_spline(
    scene,
    name,
    fov_y,          # vertical FOV (radians)
    aspect,         # width / height
    wxyz,           # (4,) quaternion (w, x, y, z)
    position,       # (3,) camera position
    far=1.0,
    scale=0.1,
    thickness=3.0,
    color=(1.0, 0, 0),
    visible=True,
):
    """
    Draw a frustum using:
        - fov_y
        - aspect
        - wxyz, position (camera world frame)
    ONLY far-plane (no near plane).
    """

    # camera origin in LOCAL coordinates
    cam_local = np.zeros(3, dtype=float)

    # far-plane corners (camera-local)
    far_pts = far_plane_corners_local_from_fov(fov_y, aspect, far)

    # scaling
    far_pts *= float(scale)

    handles = {}

    # 1) Rays
    for i in range(4):
        p0 = cam_local
        p1 = far_pts[i]
        spline_name = f"{name}/ray_{i}"
        handles[spline_name] = scene.add_spline_catmull_rom(
            name=spline_name,
            positions=np.array([p0, p1], dtype=float),
            line_width=float(thickness),
            color=color,
            wxyz=wxyz,
            position=position,
            visible=visible,
        )

    # 2) Far-plane rectangle
    for i in range(4):
        p0 = far_pts[i]
        p1 = far_pts[(i + 1) % 4]
        spline_name = f"{name}/rect_{i}"
        handles[spline_name] = scene.add_spline_catmull_rom(
            name=spline_name,
            positions=np.array([p0, p1], dtype=float),
            line_width=float(thickness),
            color=color,
            wxyz=wxyz,
            position=position,
            visible=visible,
        )

    return handles
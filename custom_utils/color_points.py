import numpy as np
import torch

def color_points_by_attention(attention, cam_rgb, threshold=0.1, gray: torch.Tensor | None = None):
    """
    attention : (T x P)
    cam_rgb : (T, 3)
    """
    print(attention.shape)
    print(cam_rgb.shape)
    assert attention.ndim == 2, attention.shape
    assert cam_rgb.ndim == 2 and cam_rgb.shape[1] == 3, cam_rgb.shape
    T, P = attention.shape
    assert cam_rgb.shape[0] == T

    device = attention.device
    dtype = cam_rgb.dtype

    if gray is None:
        gray = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=dtype)
    else:
        gray = gray.to(device=device, dtype=dtype)
        assert gray.shape == (3,)
    
    # threshold = torch.quantile(normalized_attention, q=0.9)

    mask = attention > threshold  # (T, P)
    
    mask3 = mask.unsqueeze(-1) # (T, P, 1)
    cam_colors = cam_rgb[:, None, :]  # (T, 1, 3)

    # masked sum
    color_sum = (cam_colors * mask3).sum(dim=0) # (P,3)

    # how many cameras selected per point
    count = mask.sum(dim=0).unsqueeze(-1) # (P,1)

    # avoid divide-by-zero
    count_clamped = count.clamp(min=1)

    out_rgb = color_sum / count_clamped # (P,3)

    # gray for invisible point
    no_hit = (count == 0).squeeze(-1) # (P,)
    out_rgb[no_hit] = gray

    return out_rgb


if __name__ == "__main__":
    attention_path = "/data1/cympyc1785/SceneData/DL3DV/scenes/3K/3fa142f449c51c3eb581311a0c09212ba0bc32be92e8d5fe0c24790a95af5b4f/ray_sims_T_P.pt"

    attention = torch.load(attention_path) # (T, P)

    start_color = np.array(self.frustum_start_color.value)
    end_color = np.array(self.frustum_end_color.value)
    t = np.linspace(0, 1, (cam_max - cam_min))[:, None]  # (T, 1)
    colors = (1 - t) * start_color + t * end_color  # (T, 3)

    



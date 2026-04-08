#!/usr/bin/env python3
# subsample_colmap_pycolmap_interval.py
#
# Requires: pycolmap==3.13.0
#
# Keep images by interval (deterministic) instead of random sampling.
# Example: interval=5 -> keep every 5th image in a sorted order.
#
# Usage:
#   python subsample_colmap_pycolmap_interval.py \
#     --colmap_path /path/to/colmap \
#     --output_model /path/to/colmap_sub \
#     --interval 5
#
# Optional:
#   --offset 0..interval-1  (start index)
#   --order name|id         (how to order images before subsampling)
#   --min_track_len 2
#
import os
import argparse
import pycolmap
import shutil


def prune_points3D_by_track_len(recon: pycolmap.Reconstruction, min_track_len: int) -> int:
    """Delete all points with track length < min_track_len. Returns number deleted."""
    if min_track_len <= 1:
        min_track_len = 1
    pids = list(recon.point3D_ids())
    deleted = 0
    for pid in pids:
        if not recon.exists_point3D(pid):
            continue
        p = recon.point3D(pid)
        if p.track.length() < min_track_len:
            recon.delete_point3D(pid)
            deleted += 1
    return deleted


def resolve_sparse_dir(colmap_path: str) -> str:
    d = colmap_path
    if not os.path.exists(os.path.join(d, "images.bin")) and os.path.exists(os.path.join(d, "sparse")):
        d = os.path.join(d, "sparse")
    if not os.path.exists(os.path.join(d, "images.bin")) and os.path.exists(os.path.join(d, "0")):
        d = os.path.join(d, "0")
    return d

def drop_cameras_from_colmap(
    colmap_path,
    drop_type, # ["interval", "truncation_front", "truncation_back", "truncation_mid", "truncation_front_back"]
    output_path=None,
    keep_frame_num=2,
    interval=None,
    offset=0,
    order="name",
    link_mode="symlink",
    ):

    if drop_type == "interval":
        assert interval is not None
        assert interval >= 1, "--interval must be >= 1"
        assert 0 <= offset and offset < interval, "--offset must be in [0, interval-1]"

    sparse_model_dir = resolve_sparse_dir(colmap_path)
    recon = pycolmap.Reconstruction(sparse_model_dir)

    image_ids = list(recon.images.keys())
    if len(image_ids) == 0:
        raise RuntimeError("No images found in reconstruction (is the model registered?).")

    # Build ordered list of image_ids
    if order == "id":
        ordered = sorted(image_ids)
    else:
        # order by image name
        ordered = sorted(image_ids, key=lambda iid: recon.images[iid].name)

    # Make Camera Sparse
    if drop_type == "interval":
        # Interval selection: keep indices offset, offset+interval, ...
        keep_ids = set(ordered[offset::interval])
        drop_ids = [iid for iid in image_ids if iid not in keep_ids]

    elif drop_type.startswith("truncation"):
        if keep_frame_num < 2:
            keep_frame_num = 2

        if drop_type == "truncation_front":
            drop_ids = ordered[:-keep_frame_num]
        elif drop_type == "truncation_back":
            drop_ids = ordered[keep_frame_num:]
        elif drop_type == "truncation_mid":
            drop_ids = ordered[keep_frame_num//2:-keep_frame_num//2]
        elif drop_type == "truncation_front_back":
            img_len = len(ordered)
            mid_point = img_len//2
            drop_ids = ordered[:mid_point-keep_frame_num//2] + ordered[mid_point+keep_frame_num//2:]
        else:
            raise ValueError("Invalid Truncation", drop_type)

        drop_set = set(drop_ids)
        keep_ids = [i for i in ordered if i not in drop_set]
    
    # Remove images (pycolmap 3.13: use deregister_frame)
    for iid in drop_ids:
        recon.deregister_frame(iid)

    # # Prune 3D points (이거 하면 point 다 사라져서 일단 제외)
    # deleted_pts = prune_points3D_by_track_len(recon, min_track_len)

    # Output paths
    postfix = drop_type

    output_path = output_path or os.path.join(os.path.dirname(colmap_path), f"colmap_sparse_{postfix}")
    sparse_out = os.path.join(output_path, "sparse")
    os.makedirs(sparse_out, exist_ok=True)
    recon.write(sparse_out)

    print(f"Input images : {len(image_ids)}")
    print(f"Kept images  : {len(keep_ids)}")
    print(f"Dropped imgs : {len(drop_ids)}")
    # print(f"Deleted pts  : {deleted_pts} (min_track_len={min_track_len})")
    print(f"Saved model  : {output_path}")

    # Populate images directory (link or copy)
    images_dir = os.path.join(colmap_path, "images")
    out_img_dir = os.path.join(output_path, "images")

    if os.path.exists(out_img_dir):
        shutil.rmtree(out_img_dir)
    os.makedirs(out_img_dir, exist_ok=True)

    if not os.path.isdir(images_dir):
        print(f"Warning: images dir not found: {images_dir} (skip linking/copying)")
        return

    if link_mode == "copy":
        import shutil as _sh
        for iid in sorted(keep_ids, key=lambda x: (recon.images[x].name if order == "name" else x)):
            name = recon.images[iid].name
            src = os.path.join(images_dir, name)
            dst = os.path.join(out_img_dir, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            _sh.copy2(src, dst)
    else:
        # symlink
        for iid in sorted(keep_ids, key=lambda x: (recon.images[x].name if order == "name" else x)):
            name = recon.images[iid].name
            src = os.path.join(images_dir, name)
            dst = os.path.join(out_img_dir, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst)

    print(f"Prepared images at: {out_img_dir} ({link_mode})")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_path", required=True, help="COLMAP root or sparse dir (e.g., <colmap>/sparse/0)")
    parser.add_argument("--output_path", type=str, default=None, help="Output COLMAP root dir")
    # Truncation Option
    parser.add_argument("--truncation", type=str, choices=["front", "back", "mid", "front_back"], default="front", help="Truncation Position")
    
    # Subsampling Option
    parser.add_argument("--interval", type=int, default=None, help="Keep every N-th image (N>=1)")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in [0, interval-1]")
    
    # Others
    parser.add_argument("--order", choices=["name", "id"], default="name",
                        help="Ordering used before interval sampling")
    parser.add_argument("--min_track_len", type=int, default=2, help="Prune points with track length < this")
    parser.add_argument("--link_mode", choices=["symlink", "copy"], default="symlink",
                        help="How to populate output images/ (default: symlink)")
    args = parser.parse_args()

    drop_cameras_from_colmap(
        colmap_path=args.colmap_path,
        output_path=args.output_path,
        truncation=args.truncation,
        interval=args.interval,
        offset=args.offset,
        order=args.order,
        min_track_len=args.min_track_len,
        link_mode=args.link_mode,
    )


if __name__ == "__main__":
    main()

ROOT_PATH="/data1/cympyc1785/gaussian-splatting-lightning/SceneData/NVS/scene_recon/colmap/pure_GS"
python custom_viewer.py "$ROOT_PATH" \
    --model_path $ROOT_PATH/scene.ply \

# youtube_vis/0ae1ff65a5
# DAVIS/blackswan


# # DA3 FF 3DGS
# ROOT_PATH="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/boat"
# python custom_viewer.py "$ROOT_PATH" \
#     --model_path $ROOT_PATH/da3/inpainted/gs_ply/0000.ply \
#     --custom_camera_path $ROOT_PATH/da3/inpainted/camera_params.npz \
#     --colmap_path $ROOT_PATH/da3/inpainted/colmap/sparse \
#     --scale_factor_path $ROOT_PATH/da3/inpainted/gs_ply/scale_factor.json \

# Vanilla 3DGS
# ROOT_PATH="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/kite-surf/da3/inpainted"
# python custom_viewer.py "$ROOT_PATH" \
#     --model_path $ROOT_PATH/colmap_sparse_truncation/pure_GS/checkpoints/epoch=1000-step=2000.ply \
#     --custom_camera_path $ROOT_PATH/camera_params.npz \
#     --colmap_path $ROOT_PATH/colmap/sparse \
#     # --scale_factor_path $ROOT_PATH/gs_ply/scale_factor.json \

# Colmap?
# ROOT_PATH="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/blackswan/da3/no_inpainting_sparse_with_mask"
# python custom_viewer.py "$ROOT_PATH/colmap/mask_and_depth_GS" \
#     --colmap_path $ROOT_PATH/colmap/sparse

# # GenFusion
# python custom_viewer.py "" \
#     --model_path /data1/cympyc1785/3dgs_completion/GenFusion/Reconstruction/output_ours/custom_completion/scene.ply \
#     --colmap_path /data1/cympyc1785/3dgs_completion/GenFusion/data/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/sparse


# ROOT_PATH="/data1/cympyc1785/SceneData/DL3DV/scenes/7K/009c9b366a3ceadd2e36ebe5c55e768541377822be253387d7d91eb88a52ea96"
# python custom_viewer.py "$ROOT_PATH" \
#     --model_path $ROOT_PATH/scene.ply \
#     --custom_camera_path /data1/cympyc1785/camera_gen/Director3D/exps/tmp/009c9b366a3ceadd2e36ebe5c55e768541377822be253387d7d91eb88a52ea96_transforms_pred.json \
#     # --scale_factor_path $ROOT_PATH/gs_ply/scale_factor.json \

# /data1/cympyc1785/SceneData/DL3DV/scenes/7K/014c07518eb04355edf72c55db0da41022c1550f33e04b69de02f8cd6afe1d6b

ROOT_PATH="/data1/cympyc1785/gaussian-splatting-lightning/SceneData/DL3DV/scenes/3K/19573ed4a93e9b29a30333697de386cdd04bb47b23d0e3bc123957f3cd52c9a8"
python custom_viewer.py "$ROOT_PATH" \
    --model_path $ROOT_PATH/scene.ply \
    --custom_camera_path $ROOT_PATH/camera_params.npz \
#     # --colmap_path $ROOT_PATH/colmap/sparse \
#     # --scale_factor_path $ROOT_PATH/gs_ply/scale_factor.json \

# # monst3r
# ROOT_PATH="/data1/cympyc1785/SceneData/DL3DV/scenes/7K/009c9b366a3ceadd2e36ebe5c55e768541377822be253387d7d91eb88a52ea96"
# python custom_viewer.py $ROOT_PATH \
#     --model_path $ROOT_PATH/scene.ply \


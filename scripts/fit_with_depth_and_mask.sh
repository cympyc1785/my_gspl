# export VID_NAME=parkour_1
# export ROOT_PATH="/data1/cympyc1785/Depth-Anything-3/SceneData/$VID_NAME/scene_recon/colmap"

# youtube_vis/0ae1ff65a5
# DAVIS/blackswan

ROOT_PATH="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/blackswan/da3/no_inpainting_sparse_with_mask/colmap"

python main.py fit \
    --data.path $ROOT_PATH \
    --data.parser Colmap \
    --data.parser.mask_dir "$ROOT_PATH/masks" \
    --output $ROOT_PATH/mask_and_depth_GS \
    --config configs/depth_regularization/estimated_inverse_depth-l1.yaml \
    -n blackswan_no_inpainting_sparse_with_mask
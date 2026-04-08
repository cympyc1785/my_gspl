# export VID_NAME=parkour_1
# export ROOT_PATH="/data1/cympyc1785/Depth-Anything-3/SceneData/$VID_NAME/scene_recon/colmap"

# youtube_vis/0ae1ff65a5
# DAVIS/blackswan

ROOT_PATH="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/cows/da3/inpainted/colmap_sparse_truncation_back"

python main.py fit \
    --data.path $ROOT_PATH \
    --data.parser Colmap \
    --data.parser.mask_dir "$ROOT_PATH/masks" \
    --output $ROOT_PATH/mask_GS \
    --config configs/gsplat_v1-accel.yaml \
    -n fit_with_mask \
    --max_steps 2000 \
    --save_iterations "[250, 500, 1000]"
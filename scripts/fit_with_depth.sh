# export VID_NAME=pavilion_1
# export ROOT_PATH="/data1/cympyc1785/Depth-Anything-3/SceneData/$VID_NAME/scene_recon/colmap"

# python main.py fit \
#     --data.path $ROOT_PATH \
#     --config configs/depth_regularization/estimated_inverse_depth-l1.yaml \
#     --data.parser Colmap \
#     -n $VID_NAME


# youtube_vis/0ae1ff65a5
# DAVIS/blackswan

ROOT_PATH="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/boat/da3/inpainted/colmap"

python main.py fit \
    --data.path $ROOT_PATH \
    --data.parser Colmap \
    --output $ROOT_PATH/depth_GS \
    --config configs/depth_regularization/estimated_inverse_depth-l1.yaml \
    -n 0ae1ff65a5_with_depth \
    --max_steps 90000 \
    --model.save_ply false \
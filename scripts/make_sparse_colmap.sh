ROOT_PATH="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/kite-surf/da3/inpainted/colmap"

# python utils/drop_colmap_cameras.py \
#     --colmap_path $ROOT_PATH \
#     --interval 10 \

python utils/drop_colmap_cameras.py \
    --colmap_path $ROOT_PATH \
    --truncation mid
    # --interval 10 \
    
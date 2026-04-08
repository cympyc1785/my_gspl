VID_NAME=pavilion_1
# ROOT_PATH=/data1/cympyc1785/Depth-Anything-3/SceneData/$VID_NAME/scene_recon
# ROOT_PATH=/data1/cympyc1785/gaussian-splatting-lightning/motion_dataset/DL3DV/FCGS/dataset/data/imgs_undist/1K/0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3

ROOT_PATH="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/boat/da3/inpainted/colmap"
# OUTPUT_PATH="/data1/cympyc1785/data/SceneData/DynamicVerse/youtube_vis/0ae1ff65a5"
# --data.path "/data1/cympyc1785/Depth-Anything-3/SceneData/$VID_NAME/scene_recon/colmap" \
# --data.path "$ROOT_PATH/colmap" \

# python main.py fit \
#     --data.path $ROOT_PATH \
#     --data.parser Colmap \
#     --data.image_on_cpu false \
#     --output $ROOT_PATH/pure_GS \
#     --config configs/gsplat_v1-accel.yaml \
#     -n boat_pure_GS \
#     --max_steps 90000 \
#     --model.save_ply false \
#     # --ckpt_path last \

ROOT_PATH="/data1/cympyc1785/gaussian-splatting-lightning/SceneData/NVS/scene_recon/colmap"

python main.py fit \
    --data.path $ROOT_PATH \
    --data.parser Colmap \
    --data.image_on_cpu false \
    --output $ROOT_PATH/pure_GS \
    --config configs/gsplat_v1-accel.yaml \
    -n pure_GS \
    --max_steps 7000 \
    --model.save_ply false \
    # --ckpt_path last \
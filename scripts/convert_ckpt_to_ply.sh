# example
# python utils/ckpt2ply.py outputs/lego/checkpoints/epoch=300-step=30000.ckpt

# youtube_vis/0ae1ff65a5

ROOT_PATH="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/boat/da3/inpainted/colmap/da3_depth_GS/checkpoints"

python utils/ckpt2ply.py "$ROOT_PATH/epoch=139-step=6999.ckpt"
# python utils/ckpt2ply.py "$ROOT_PATH/epoch=599-step=29999.ckpt"
# python utils/ckpt2ply.py "$ROOT_PATH/epoch=1800-step=90000.ckpt"
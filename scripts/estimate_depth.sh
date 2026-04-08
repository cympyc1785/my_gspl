# export VID_NAME=parkour_1
# export ROOT_PATH="/data1/cympyc1785/Depth-Anything-3/SceneData/$VID_NAME/scene_recon/colmap"

# youtube_vis/0ae1ff65a5
# DAVIS/blackswan

export ROOT_PATH="/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/boat/da3/inpainted/colmap"

python utils/estimate_dataset_depths.py $ROOT_PATH
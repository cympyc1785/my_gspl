# # for task
# python view_camera_outputs.py \
#     --task_list_path /data1/cympyc1785/gaussian-splatting-lightning/SceneData/valid_task_list.txt \
#     --data_root_path /data2/ckd248/SCVideo/camera_generation/evaluation

# for task
python view_camera_outputs.py \
    --task_list_path /data1/cympyc1785/SCVideo/camera_generation/my_eval/cfg/ours_mild10_unguided/test_valid.txt \
    --data_root_path /data1/cympyc1785/SCVideo/camera_generation/my_eval/cfg \
    --task_offset 2

# # # Point Cloud Scene
# ROOT_DIR=/data1/cympyc1785/gaussian-splatting-lightning/SceneData/DynamicVerse/scenes/DAVIS/blackswan
# python view_camera_outputs.py \
#     --root_path $ROOT_DIR \
#     --ply_path $ROOT_DIR/scene.ply \
#     --pcd

# # GS Point Cloud Scene (.pt)
# ROOT_DIR=/data1/cympyc1785/caption/GenDoP/DataDoP/tmp
# python view_camera_outputs.py $ROOT_DIR \
#     --ply_path $ROOT_DIR/scene.pt \
#     --gs

# # Only Camera
# ROOT_DIR=/data1/cympyc1785/gaussian-splatting-lightning/SceneData/DL3DV/scenes/1K/0032cd2f169847864c28e5e190c2496c03ddd1a5e68d52145634164ebe57d3ac
# python view_camera_outputs.py $ROOT_DIR
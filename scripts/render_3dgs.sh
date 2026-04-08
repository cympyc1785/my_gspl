ROOT_PATH="/data1/cympyc1785/SceneData/DL3DV/scenes/7K/930305bf3aa55158d78b62504c0b2a355be0d0d30533431cc0c4feb9f7e93e21"

python utils/render_3dgs.py $ROOT_PATH/scene.ply \
    --camera-path "$ROOT_PATH/cameras.json" \
    --output-path "./hi.mp4"
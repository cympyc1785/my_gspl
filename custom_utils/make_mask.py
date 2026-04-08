import cv2
import os
import numpy as np
import random
import time

def resize_mask(mask, ref_img):
    h, w = ref_img.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def make_mask(ref_img, mask_type=0):
    h, w = ref_img.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8)

    half_h = h // 2
    half_w = w // 2

    if mask_type == 0:
        mask[:half_h, :half_w] = 0
    elif mask_type == 1:
        mask[:half_h, half_w: ] = 0
    elif mask_type == 2:
        mask[half_h:, half_w: ] = 0
    elif mask_type == 3:
        mask[half_h:, :half_w] = 0
    else:
        raise ValueError("Invalid Mask Type", mask_type)
    
    mask = mask * 255
    
    return mask

def make_random_masks(src_dir, export_dir):
    os.makedirs(export_dir, exist_ok=True)
    random.seed(time.time())
    mask_type = random.randint(0, 3)
    print("\n[Mask Type]", mask_type)
    for name in sorted(os.listdir(src_dir)):
        img_path = os.path.join(src_dir, name)
        # mask_path = os.path.join(masks_dir, name)

        img = cv2.imread(img_path)
        mask = make_mask(img, mask_type)
        cv2.imwrite(os.path.join(export_dir, name), mask)

if __name__ == "__main__":
    ROOT_PATH = "/data1/cympyc1785/SceneData/DynamicVerse/scenes/DAVIS/cows/da3/inpainted/colmap_sparse_truncation_back"

    images_dir = f"{ROOT_PATH}/images"
    # masks_dir = f"{ROOT_PATH}/mask"
    out_dir = f"{ROOT_PATH}/masks"
    
    make_random_masks(images_dir, out_dir)


    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # mask_resized = resize_mask(mask, img)

    # cv2.imwrite(os.path.join(out_dir, name), mask_resized)

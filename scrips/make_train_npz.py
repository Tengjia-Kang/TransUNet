import os
import numpy as np
from PIL import Image
import h5py

def create_npz_files(image_folder, mask_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

    # 确保图像和掩码数量相同
    if len(image_files) != len(mask_files):
        raise ValueError("The number of images and masks must be the same.")

    # 遍历图像和掩码，创建npz文件
    for image_file, mask_file in zip(image_files, mask_files):
        # 构建完整的文件路径
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)

        # 读取图像和掩码
        image = np.array(Image.open(image_path), dtype=np.float32) / 255.0
        label = np.array(Image.open(mask_path), dtype=np.float32) / 255.0

        # 构建输出文件名
        output_filename = os.path.splitext(image_file)[0] + '.npz'

        # 保存为npz文件
        np.savez(os.path.join(output_folder, output_filename), image=image, label =label)

def create_npz_h5_files(image_folder, mask_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

    # 确保图像和掩码数量相同
    if len(image_files) != len(mask_files):
        raise ValueError("The number of images and masks must be the same.")

    # 遍历图像和掩码，创建npz.h5文件
    for image_file, mask_file in zip(image_files, mask_files):
        # 构建完整的文件路径
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)

        # 读取图像和掩码
        # 读取图像和掩码
        image = np.array(Image.open(image_path), dtype=np.float32) / 255.0
        mask = np.array(Image.open(mask_path), dtype=np.float32) / 255.0

        # 构建输出文件名
        output_filename = os.path.splitext(image_file)[0] + '.npz.h5'

        # 保存为npz.h5文件
        with h5py.File(os.path.join(output_folder, output_filename), 'w') as h5f:
            h5f.create_dataset('image', data=image)
            h5f.create_dataset('label', data=mask)

if __name__ == "__main__":
    image_folder = '../datasets/CSP/TDI/cross_fold/train/fold_0'
    mask_folder = '../datasets/CSP/TDI/cross_fold/val/fold_0'
    output_folder = '../datasets/CSP/TDI/cross_fold/train_npz/fold_0'
    create_npz_files(image_folder, mask_folder, output_folder)
    # create_npz_h5_files(image_folder, mask_folder, output_folder)
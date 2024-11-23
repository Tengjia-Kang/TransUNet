
import numpy as np

file_path = "/home/kang/Hard/Medical_Seg/TransUNet-main/datasets/CSP/TDI/train_npz/09.npz"
with np.load(file_path) as data:
    # 假设.npz文件中包含两个数组，名为'image'和'mask'
    image = data['image']
    mask = data['label']


    pass
import os
import shutil
import numpy as np
import h5py
from sklearn.model_selection import KFold

# 设置路径
npz_folder = '../datasets/CSP/TDI/train_npz'
save_folder = '../datasets/CSP/TDI/cross_fold'

# 获取所有 .npz 文件
npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]

# 确保文件数量可以被5整除
if len(npz_files) % 5 != 0:
    raise ValueError("The number of .npz files must be divisible by 5.")

# 创建交叉验证折叠
kf = KFold(n_splits=5)

# 遍历每个交叉验证折叠
for fold, (train_index, val_index) in enumerate(kf.split(npz_files)):
    # 获取训练集和验证集的 .npz 文件
    train_npzs = [npz_files[i] for i in train_index]
    val_npzs = [npz_files[i] for i in val_index]

    # 创建训练集和验证集的文件夹
    train_folder = f'{save_folder}/train/fold_{fold}'
    val_folder = f'{save_folder}/val/fold_{fold}'
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 将训练集的 .npz 文件复制到相应的文件夹
    for npz in train_npzs:
        shutil.copy(os.path.join(npz_folder, npz), os.path.join(train_folder, npz))

    # 将验证集的 .npz 文件转换为 .npz.h5 格式并复制到相应的文件夹
    for npz in val_npzs:
        # 读取 .npz 文件
        with np.load(os.path.join(npz_folder, npz)) as data:
            # 获取 .npz 文件中的所有键名和数据
            keys = data.files
            arrays = [data[key] for key in keys]

        # 创建 .h5 文件
        h5_file_path = os.path.join(val_folder, os.path.splitext(npz)[0] + '.npz.h5')
        with h5py.File(h5_file_path, 'w') as h5f:
            # 将数据写入 .h5 文件
            for key, array in zip(keys, arrays):
                h5f.create_dataset(key, data=array)

import os
import h5py
import numpy as np


def convert_h5_to_npz(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件扩展名是否为 .h5
        if filename.endswith('.h5'):
            # 获取完整的文件路径
            input_path = os.path.join(input_folder, filename)

            # 构造输出文件路径，去掉 .h5 扩展名并添加 .npz 扩展名
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.npz')

            # 打开 HDF5 文件并读取数据
            with h5py.File(input_path, 'r') as h5file:
                # 创建一个字典来存储数据集
                datasets = {}
                # 遍历 HDF5 文件中的所有数据集
                for key in h5file.keys():
                    # 读取数据集并存储到字典中
                    datasets[key] = h5file[key][()]

            # 将数据集保存到 .npz 文件
            np.savez(output_path, **datasets)
            print(f"Converted {input_path} to {output_path}")



def convert_npz_h5_to_npy_h5(folder_path):
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件扩展名是否为 .npz.h5
        if filename.endswith('.npz.h5'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # 提取原始文件名和去除扩展名的部分
            base_name = os.path.splitext(filename)[0]
            new_filename = base_name + '.npy.h5'
            new_file_path = os.path.join(folder_path, new_filename)

            # 使用 h5py 打开文件
            with h5py.File(file_path, 'r') as f_in:
                # 创建一个新的 HDF5 文件
                with h5py.File(new_file_path, 'w') as f_out:
                    # 遍历原始文件中的每个键（组或数据集）
                    for key in f_in.keys():
                        # 将每个对象复制到新文件中
                        f_out.create_group(key) if isinstance(f_in[key], h5py.Group) else f_out.create_dataset(
                            key, data=f_in[key][:], compression="gzip" if 'compression' in f_in[key].attrs else None
                        )
                        # 复制属性（如果有）
                        for attr_name, attr_value in f_in[key].attrs.items():
                            f_out[key].attrs[attr_name] = attr_value

            print(f"Converted {file_path} to {new_file_path}")


def rename_files(folder_path):
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否以 .npz.h5 结尾
        if filename.endswith('.npz.h5'):
            # 生成旧文件和新文件的完整路径
            old_file_path = os.path.join(folder_path, filename)
            # 去掉 .npz 部分，改为 .npy.h5
            new_filename = filename[:-7] + '.npy.h5'
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} -> {new_file_path}')



folder_path = '/home/kang/Hard/Medical_Seg/TransUNet-main/datasets/CSP/TDI/cross_fold/val/fold_0'
# convert_h5_to_npz(folder_path, folder_path)
# convert_npz_h5_to_npy_h5(folder_path)
rename_files(folder_path)
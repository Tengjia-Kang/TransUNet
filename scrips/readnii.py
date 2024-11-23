import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    nii_file_path = '../../data/nii/images/2D/245.nii'  # 替换为你的“.nii”文件路径
    nii_img = nib.load(nii_file_path)
    data = nii_img.get_fdata()  # 获取图像数据，返回一个NumPy数组
    output_dir = "demo_image"
    # 检查数据的维度
    if data.ndim == 4:
        # 假设数据是四维的，其中最后一个维度是颜色通道（对于彩色图像）
        # 如果数据是灰度图像，则可能没有颜色通道维度，或者颜色通道维度大小为1并被squeeze掉了
        num_slices = data.shape[0]  # 切片数量（假设第一个维度是切片索引）
        slice_shape = data.shape[1:]  # 每个切片的形状（高度，宽度，[颜色通道]）


        os.makedirs(output_dir, exist_ok=True)

        for i in range(num_slices):
            # 如果数据是彩色的，则直接显示；如果是灰度的，则可能需要取一个颜色通道或转换为灰度图
            if data.shape[-1] == 3:  # 彩色图像
                slice_data = data[i, :, :, :]  # 选择一个切片
                slice_data_squeezed = np.squeeze(slice_data) if slice_data.shape[1] == 1 and slice_data.shape[2] == 1 else slice_data
                # 注意：这里不需要squeeze如果切片本身就是三维的（高度x宽度x颜色通道）
                # 但是，如果切片在某个维度上是1（比如单通道彩色图像被错误地保存为四维），则需要squeeze

                # 显示和保存切片
                plt.imshow(slice_data_squeezed.transpose((1, 0, 2)), origin='lower')  # 注意transpose来正确显示图像（高度x宽度x颜色通道）
                plt.axis('off')
                output_path = os.path.join(output_dir, f'slice_{i:03d}.png')
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()
            else:  # 灰度图像或单通道彩色图像（被当作灰度处理）
                slice_data = data[i, :, :, 0] if data.shape[-1] == 1 else data[i, :, :]  # 选择一个切片并可能去掉颜色通道维度
                plt.imshow(slice_data, cmap='gray', origin='lower')  # 使用灰度色图显示
                plt.axis('off')
                output_path = os.path.join(output_dir, f'slice_{i:03d}.png')
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()

        print(f"Saved {num_slices} slices to the directory '{output_dir}'.")
    else:
        squeezed_data = np.squeeze(data)
        plt.imshow(squeezed_data, origin='lower')  # 使用RGB色图显示
        plt.axis('off')  # 关闭坐标轴
        output_path = os.path.join(output_dir, 'image_demo.png')  # 只有一个切片，所以命名为000
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
import os
import re
def extract_numbers(s):
    """从字符串中提取所有数字"""
    return int(''.join(re.findall(r'\d+', s)))

def make_train_list(folder_path, output_file):
    bash_folder = os.path.dirname(output_file)
    if not os.path.exists(bash_folder):
        os.makedirs(bash_folder)
    # 获取文件夹下所有文件名
    files = os.listdir(folder_path)
    # 提取不含后缀的文件名
    filenames = [os.path.splitext(f)[0] for f in files]
    # 按字典顺序排序
    sorted_filenames = sorted(filenames)
    # 保存到外部txt文件
    with open(output_file, 'w') as f:
        for name in sorted_filenames:
            f.write(name + '\n')


def make_test_list(folder_path, output_file):
    # 获取文件夹下所有 .npz.h5 文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz.h5')]
    # 提取不含后缀的文件名
    filenames = [os.path.splitext(f)[0].replace('.npz', '') for f in files]
    # 按数字顺序排序
    sorted_filenames = sorted(filenames, key=extract_numbers)

    # 保存到外部txt文件
    with open(output_file, 'w') as f:
        for name in sorted_filenames:
            f.write(name + '\n')

if __name__ == '__main__':

    folder_path = '/home/kang/Hard/Medical_Seg/TransUNet-main/datasets/CSP/TDI/cross_fold/val/fold_4'
    output_file = '/home/kang/Hard/Medical_Seg/TransUNet-main/lists/lists_CSP/fold_4/test_vol.txt'
    # make_train_list(folder_path, output_file)
    make_test_list(folder_path, output_file)

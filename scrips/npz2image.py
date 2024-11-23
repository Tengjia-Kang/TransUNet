import numpy as np
import h5py
import matplotlib.pyplot as plt
if __name__ == "__main__":
    expmle_path = "../datasets/project_TransUNet/data/Synapse/train_npz/case0005_slice001.npz"
    emp1 = np.load(expmle_path, allow_pickle=True)
    img = emp1['image']
    label = emp1['label']
    # plt.imshow(img, cmap='gray')
    plt.imshow(label, cmap='gray')
    plt.show()
    pass
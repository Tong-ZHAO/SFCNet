import os
import numpy as np

from torch.utils.data import Dataset
from distance import *

class DGCNNDataset(Dataset):
    """
    Training / validation / test dataset class to load ABC Dataset for DGCNN backbone

    Params:
        - root: root path to the dataset
        - file_list: file list
        - sharp_thresh: threshold for filtering distances from point to sharp features
        - point_noise_level: noise level for data augmentation
        - flag_test: a flag indicating if training or testing 
        - seed: random seed to use
    """

    def __init__(self, 
                 root, 
                 file_list, 
                 sharp_thresh = 0.03, 
                 point_noise_level = 0.01, 
                 seed = None,
                 flag_test = False):

        self.root = root
        self.file_list = file_list
        self.sharp_thresh = sharp_thresh
        self.point_noise_level = point_noise_level if not flag_test else 0.
        self.random_engine = np.random.RandomState(seed)
        self.flag_test = flag_test

        with open(file_list, "r") as fp:
            self.file_list = fp.read().split()[:-1]

    #----------------------------  Dataset ----------------------------#
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        """
        Returns:
            - samples: point cloud of size (10000, 3)
            - labels: 0/1 labels of size (10000,)
            - offsets: offset of current point to sharp features (10000, 3)
        """

        file_name = self.file_list[index]

        if not self.flag_test:
            # read files: xyz, obj and yml
            m_verts, center, sacle = read_obj_and_normalize(os.path.join(self.root, file_name + ".obj"))
            m_primitives = read_yml(os.path.join(self.root, file_name + ".yml"), center, sacle)
            samples = read_xyz(os.path.join(self.root, file_name + ".xyz"), center, sacle, self.with_normal)
            # add noise if training
            if abs(self.point_noise_level) > 1e-10:
                samples = samples + self.point_noise_level * self.random_engine.uniform(-1., 1., size = samples.shape)
            # compute groundtruth
            labels, offsets = process_dists_to_primitives(samples, m_primitives, m_verts, self.sharp_thresh)
            return samples, labels, offsets
        else:
            # read file: xyz
            samples, center, scale = read_xyz_with_scaling(os.path.join(self.root, file_name + ".xyz"))
            return samples, file_name, center, scale



if __name__ == '__main__':

    data = DGCNNDataset('../datasets/ABC_dataset', '../datasets/ABC_dataset/train_models.csv')
    
    X, Y, offsets = data[0]
    print("Samples shape: ", X.shape)
    print("Labels shape: ", Y.shape)
    print("Offset shape: ", offsets.shape)
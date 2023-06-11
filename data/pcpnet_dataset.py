from __future__ import print_function
import torch
from torch.utils.data import Dataset
import numpy as np
from pcpnet_dataset_utils import *


class PCPDataset(Dataset):
    """
    Training / validation / test dataset class to load ABC Dataset for DGCNN backbone

    Params:
        - root: root path to the dataset
        - file_list: file list
        - patch_radius: the sphere size defining the neighborhood
        - points_per_patch: the number of sampled points in the neighborhood
        - sharp_thresh: threshold for filtering distances from point to sharp features
        - noise_level: noise level for data augmentation
        - seed: random seed to use
        - use_pca: a flag indicating if apply pca before feeding to the neural network
        - flag_test: a flag indicating if training or testing 
    """

    # patch radius as fraction of the bounding box diagonal of a shape
    def __init__(self, 
                 root, 
                 file_list, 
                 patch_radius, 
                 points_per_patch, 
                 sharp_thresh, 
                 noise_level,
                 seed = None, 
                 use_pca = False,
                 flag_test = False):

        # initialize parameters
        self.root = root
        self.file_list = file_list
        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.sharp_thresh = sharp_thresh
        self.noise_level = noise_level
        self.use_pca = use_pca
        self.seed = seed
        self.flag_test = flag_test

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.randint(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)
        self.shape_cache = Cache(self.sharp_thresh, self.noise_level, self.rng, self.flag_test)

        # get all shape names in the dataset
        with open(file_list, "r") as fp:
            self.shape_names = fp.read().split()[:-1]
        print("Found %d files!" % len(self.shape_names))

        # get basic information for each shape in the dataset
        print("Begin loading data...")
        self.shape_patch_count = []
        self.patch_radius_absolute = []
        self.shape_patch_count, self.patch_radius_absolute = self.shape_cache.read_files(self.root, self.shape_names, self.patch_radius)


    def reload_data_for_epoch(self):
        print("Reload dataset!")
        # randomize noises for a new epoch
        self.shape_patch_count = []
        self.patch_radius_absolute = []
        self.shape_patch_count, self.patch_radius_absolute = self.shape_cache.read_files(self.root, self.shape_names, self.patch_radius)


    # returns a patch centered at the point with the given global index
    # and the ground truth normal the the patch center
    def __getitem__(self, index):

        # find shape that contains the point with given global index
        shape_ind, center_point_ind = self.shape_index(index)
        shape = self.shape_cache.get(shape_ind)

        # get neighboring points (within euclidean distance patch_radius)
        rad = self.patch_radius_absolute[shape_ind]
        patch_point_inds = np.array(shape.kdtree.query_ball_point(shape.pts[center_point_ind, :], rad))
        patch_point_inds = patch_point_inds[self.rng.choice(len(patch_point_inds), self.points_per_patch, replace=True)]

        # convert points to torch tensors
        patch_pts = torch.from_numpy(shape.pts[patch_point_inds, :])
        # center patch (central point at origin - but avoid changing padded zeros)
        patch_pts = patch_pts - torch.from_numpy(shape.pts[center_point_ind, :])

        if self.use_pca:
            # compute pca of points in the patch:
            # center the patch around the mean:
            pts_mean = patch_pts.mean(0)
            patch_pts = patch_pts - pts_mean

            trans, _, _ = torch.svd(torch.t(patch_pts))
            patch_pts = torch.mm(patch_pts, trans)

            cp_new = -pts_mean # since the patch was originally centered, the original cp was at (0,0,0)
            cp_new = torch.matmul(cp_new, trans)

            # re-center on original center point
            patch_pts = patch_pts - cp_new
        else:
            trans = torch.eye(3).float()

        if not self.flag_test:
            patch_cla = torch.from_numpy(shape.cla[center_point_ind, :])
            patch_disp = torch.from_numpy(shape.disp[center_point_ind, :])
            return (patch_pts, patch_cla, patch_disp, trans)
        else:
            return (patch_pts, shape.pts[center_point_ind], trans)


    def __len__(self):
        return sum(self.shape_patch_count)


    # translate global (dataset-wide) point index to shape index & local (shape-wide) point index
    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind



if __name__ == '__main__':

    data = PCPDataset(  root = '../data/ABC_dataset', 
                        file_list = '../data/ABC_dataset/test_models.csv', 
                        patch_radius = 0.05, 
                        points_per_patch = 500, 
                        sharp_thresh = 0.03, 
                        noise_level = 0.01,
                        seed = None,
                        use_pca = True,
                        flag_test = False)
    print("length data: ", len(data))
    
    pts, cla, disp, trans = data[0]
    print("Points shape: ", pts.shape)
    print("Labels shape: ", cla.shape)
    print("Offset shape: ", disp.shape)
    print("Trans shape: ", trans.shape)
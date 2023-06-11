from __future__ import print_function
import os, sys
import torch.utils.data as data
import numpy as np
import scipy.spatial as spatial
from distance import *


def load_shape(file_root, file_name, sharp_thresh, noise_level, random_engine, flag_test):

    if flag_test:
        pts , _ , _ = read_xyz_with_scaling(os.path.join(file_root, file_name + ".xyz"))
        sys.setrecursionlimit(int(max(1000, round(pts.shape[0] / 10)))) # otherwise KDTree construction may run out of recursions
        kdtree = spatial.cKDTree(pts, 10)
        return Shape(file_name, pts=pts, kdtree=kdtree, cla=None, disp=None)
    else:
        m_verts, center, scale = read_obj_and_normalize(os.path.join(file_root, file_name + ".obj"))
        m_primitives = read_yml(os.path.join(file_root, file_name + ".yml"), center, scale)
        pts = read_xyz(os.path.join(file_root, file_name + ".xyz"), center, scale, False)
        pts = pts + noise_level * random_engine.uniform(-1., 1., size = pts.shape)
        labels, offsets = process_dists_to_primitives(pts, m_primitives, m_verts, sharp_thresh)
        labels = np.hstack((labels.reshape((-1, 1)), (1. - labels).reshape((-1, 1))))

        sys.setrecursionlimit(int(max(1000, round(pts.shape[0] / 10)))) # otherwise KDTree construction may run out of recursions
        kdtree = spatial.cKDTree(pts, 10)

        return Shape(file_name, pts=pts, kdtree=kdtree, cla=labels, disp=offsets)


class SequentialPointcloudPatchSampler(data.sampler.Sampler):
    """
    A sampler that returns the points in each point cloud of the dataset sequentially.
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + self.data_source.shape_patch_count[shape_ind]

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count
    

class SequentialShapeRandomPointcloudPatchSampler(data.sampler.Sampler):
    """
    A sampler that returns a permutation of the points in the dataset where all points in the same shape are adjacent.
    (for performance reasons)
    """

    def __init__(self, data_source, patches_per_shape, seed = None, sequential_shapes = False, identical_epochs = False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.sequential_shapes = sequential_shapes
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.shape_patch_inds = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0

        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        # global point index offset for each shape
        shape_patch_offset = list(np.cumsum(self.data_source.shape_patch_count))
        shape_patch_offset.insert(0, 0)
        shape_patch_offset.pop()

        shape_inds = range(len(self.data_source.shape_names))
        if not self.sequential_shapes:
            shape_inds = self.rng.permutation(shape_inds)

        self.shape_patch_inds = [[]] * len(self.data_source.shape_names)
        point_permutation = []

        for shape_ind in shape_inds:
            start = shape_patch_offset[shape_ind]
            end = shape_patch_offset[shape_ind] + self.data_source.shape_patch_count[shape_ind]
            global_patch_inds = self.rng.choice(range(start, end), size = min(self.patches_per_shape, end - start), replace = False)
            point_permutation.extend(global_patch_inds)
            # save indices of shape point subset
            self.shape_patch_inds[shape_ind] = global_patch_inds - start

        return iter(point_permutation)

    def __len__(self):
        return self.total_patch_count

class RandomPointcloudPatchSampler(data.sampler.Sampler):
    """
    A sampler that returns a permutation of the points in the dataset randomly.
    """

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape, self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):
        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class Shape():

    def __init__(self, name, pts, kdtree, cla = None, disp = None):
        self.pts = pts
        self.kdtree = kdtree
        self.cla = cla
        self.disp = disp
        self.name = name


class Cache():
    
    def __init__(self, sharp_thresh, noise_level, random_engine, flag_test = False):
        self.elements = {}
        self.noise_level = noise_level
        self.sharp_thresh = sharp_thresh
        self.random_engine = random_engine
        self.flag_test = flag_test

    def read_files(self, root, file_list, patch_radius):
        self.elements.clear()
        shape_patch_count = []
        patch_radius_absolute = []

        for shape_ind, name in enumerate(file_list):
            shape = load_shape(root, name, self.sharp_thresh, self.noise_level, self.random_engine, self.flag_test)
            self.elements[shape_ind] = shape
            shape_patch_count.append(shape.pts.shape[0])
            bbdiag = float(np.linalg.norm(shape.pts.max(0) - shape.pts.min(0), 2))
            patch_radius_absolute.append(bbdiag * patch_radius)

        return shape_patch_count, patch_radius_absolute

    def get(self, element_id):
        return self.elements[element_id]


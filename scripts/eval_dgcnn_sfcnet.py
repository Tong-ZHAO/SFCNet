import os, sys
sys.path.append('../data')
sys.path.append('../models')
sys.path.append('../utils')

import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from scipy import spatial

import torch
import argparse

from distance import *
from dgcnn_dataset import DGCNNDataset
from dgcnn_sfcnet import SFCNetWithDGCNN


def compute_chamfer_distance(preds, gts):

    tree_pred_to_gt = spatial.cKDTree(gts, copy_data = False, balanced_tree = False, compact_nodes = False)
    dist_pred_to_gt = tree_pred_to_gt.query(preds, k = 1)[0]
    dist_pred_to_gt = dist_pred_to_gt.mean()

    tree_gt_to_pred = spatial.cKDTree(preds, copy_data = False, balanced_tree = False, compact_nodes = False)
    dist_gt_to_pred = tree_gt_to_pred.query(gts, k = 1)[0]
    dist_gt_to_pred = dist_gt_to_pred.mean()

    return dist_pred_to_gt, dist_pred_to_gt + dist_gt_to_pred


if __name__ == '__main__':

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataRoot', type = str, default = '../datasets/ABC_test/', help = 'file root')
    parser.add_argument('--testList', type = str, default = '../datasets/ABC_test/test_list_new.txt', help = 'evaluation list')
    parser.add_argument('--modelPath', type = str, default = '../pretrained_models/pretrained_sfcnet_dgcnn/pretrained_sfcnet_dgcnn_new.pth', help = 'pretrained model')
    parser.add_argument('--numNb', type = int, help = 'number of neighbors used in DGCNN (should be the same for training)', default = 20)
    parser.add_argument('--batchSize', type = int, help = 'batch size', default = 2)

    opt = parser.parse_args()
    print (opt)

    # Check Device (CPU / GPU)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # Create Dataset
    print("Loading test dataset...")
    test_data = DGCNNDataset(root=opt.dataRoot, file_list = opt.testList, flag_test = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = opt.batchSize, shuffle = False)
    print("  Number of test files: ", len(test_data))
    print("  Number of test batches: ", len(test_loader))

    # Create Model
    print("Loading model...")
    model = SFCNetWithDGCNN(input_channels=3, nn_nb = opt.numNb, device=device)
    model.to(device)

    checkpoint = torch.load(opt.modelPath) if use_cuda else torch.load(opt.modelPath, map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print("Load pretrained model: ", opt.modelPath)

    # Evaluation
    print("Begin testing...")
    model.eval()

    with torch.no_grad():

        total_samples = 0
        recall_total = 0
        precision_total = 0
        iou_total = 0
        f1_total = 0
        accuracy_total = 0
        average_distance_curve_total = 0
        average_chamfer_total = 0

        for samples, file_names, centers, scales in test_loader:
            # model prediction
            samples = samples.float().to(device)
            y_pred, y_offsets = model(samples)
            # get results
            samples = samples.cpu().data.numpy()
            y_pred = y_pred.cpu().data.numpy()
            y_offsets = y_offsets.cpu().data.numpy()
            centers = centers.cpu().data.numpy()
            scales = scales.cpu().data.numpy()
            # compute metrics
            for i in range(samples.shape[0]):
                # classification
                labels = np.loadtxt(os.path.join(opt.dataRoot, file_names[i].split('_')[0] + '.cla'))[:, 0]
                preds = np.squeeze(y_pred[i])
                preds_tag = (preds >= 0.)

                recall_total += recall_score(labels, preds_tag, average = 'binary')
                precision_total += precision_score(labels, preds_tag, average = 'binary')
                iou_total += jaccard_score(labels, preds_tag, average = 'binary')
                f1_total += f1_score(labels, preds_tag, average = 'binary')
                accuracy_total += accuracy_score(labels, preds_tag)

                # distance
                sharp_preds = samples[i, preds_tag] + y_offsets[i, preds_tag]
                sharp_gt = np.loadtxt(os.path.join(opt.dataRoot, file_names[i].split('_')[0] + '_gt.xyz'))
                sharp_gt = (sharp_gt - centers[i].reshape((1, -1))) / scales[i]
                distance_curve, chamfer = compute_chamfer_distance(sharp_preds, sharp_gt)
                average_distance_curve_total += distance_curve
                average_chamfer_total += chamfer

            # print progress
            total_samples += samples.shape[0]
            if total_samples > 0 and np.abs(total_samples % 100) == 0:
                print("  Tested on %d samples!" % total_samples)

        # compute mean metrics
        recall_total /= len(test_data)
        precision_total /= len(test_data)
        iou_total /= len(test_data)
        f1_total /= len(test_data)
        accuracy_total /= len(test_data)
        average_distance_curve_total /= len(test_data)
        average_chamfer_total /= len(test_data)

        # print metrics
        print("Results:")
        print("  - recall_total: %f" % recall_total)
        print("  - precision_total: %f" % precision_total)
        print("  - iou_total: %f" % iou_total)
        print("  - f1_total: %f" % f1_total)
        print("  - accuracy_total: %f\n" % accuracy_total)

        print("  - average_distance_curve_total: %f" % average_distance_curve_total)
        print("  - average_chamfer_total: %f" % average_chamfer_total)
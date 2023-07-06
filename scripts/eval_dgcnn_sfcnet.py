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

import torch
import argparse

from distance import *
from dgcnn_dataset import DGCNNDataset
from dgcnn_sfcnet import SFCNetWithDGCNN


def get_distance_to_sharp_cruve(samples, preds, offsets, m_primitives, m_verts):

    preds_tag = (preds >= 0.)
    sharp_pts = samples[preds_tag] + offsets[preds_tag]
    ldists = []
    if sharp_pts.shape[0]==0:
        return 0

    for i in range(len(m_primitives)):
        dists = dist_from_points_to_primitive(m_primitives[i], sharp_pts, m_verts)[0]
        ldists.append(dists)

    ldists = np.array(ldists).T
    indices = np.argmin(ldists, axis = 1)
    dists = ldists[np.arange(len(sharp_pts)), indices]

    return np.mean(dists)


if __name__ == '__main__':

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataRoot', type = str, default = '../datasets/ABC_dataset/', help = 'file root')
    parser.add_argument('--testList', type = str, default = 'test_models.csv', help = 'test list')
    parser.add_argument('--modelPath', type = str, default = '../log/18-01-22_18-57/model_epoch5.pth', help = 'pretrained model')
    parser.add_argument('--numNb', type = int, help = 'number of neighbors used in DGCNN (should be the same for training)', default = 20)
    parser.add_argument('--batchSize', type = int, help = 'batch size', default = 8)
    parser.add_argument('--sharpThresh', type=float, help='threshold for sharp features (should be the same for training)', default=0.03)

    opt = parser.parse_args()
    print (opt)

    # Check Device (CPU / GPU)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # Create Dataset
    print("Loading test dataset...")
    test_data = DGCNNDataset(root=opt.dataRoot, file_list=opt.trainList, sharp_thresh=opt.sharpThresh, flag_test=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)
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
    print("Begin evaluation...")
    model.eval()

    with torch.no_grad():

        total_samples = 0
        recall_total = 0
        precision_total = 0
        iou_total = 0
        f1_total = 0
        accuracy_total = 0
        average_distance_curve_total = 0

        for samples, file_name, labels, offsets in test_loader:
            # model prediction
            samples = samples.float().to(device)
            y_pred, y_offsets = model(samples)
            # get results
            samples = samples.cpu().data.numpy()
            labels = labels.cpu().data.numpy()
            y_pred = y_pred.cpu().data.numpy()
            y_offsets = y_offsets.cpu().data.numpy()
            # compute metrics
            for i in range(samples.shape[0]):
                m_verts, center, scale = read_obj_and_normalize(os.path.join(opt.dataRoot, file_name[i] + ".obj"))
                m_primitives = read_yml(os.path.join(opt.dataRoot, file_name[i] + ".yml"), center, scale)
                preds = np.squeeze(y_pred[i])
                preds_tag = (preds >= 0.)
                recall_total += recall_score(np.squeeze(labels[i]), preds_tag, average = 'binary')
                precision_total += precision_score(np.squeeze(labels[i]), preds_tag, average = 'binary')
                iou_total += jaccard_score(np.squeeze(labels[i]), preds_tag, average = 'binary')
                f1_total += f1_score(np.squeeze(labels[i]), preds_tag, average = 'binary')
                accuracy_total += accuracy_score(np.squeeze(labels[i]), preds_tag)
                average_distance_curve_total += get_distance_to_sharp_cruve(samples[i], preds,y_offsets[i], m_primitives, m_verts)
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

        # print metrics
        print("average_distance_curve_total: %f\n" % average_distance_curve_total)
        print("recall_total: %f\n" % recall_total)
        print("precision_total: %f\n" % precision_total)
        print("iou_total: %f\n" % iou_total)
        print("f1_total: %f\n" % f1_total)
        print("accuracy_total: %f\n" % accuracy_total)
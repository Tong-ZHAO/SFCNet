import os, sys
sys.path.append('../data')
sys.path.append('../models')
sys.path.append('../utils')

import numpy as np

import torch
import argparse

from distance import read_xyz_with_scaling
from dgcnn_sfcnet import SFCNetWithDGCNN


if __name__ == '__main__':

    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--testFile', type = str, default = '../datasets/examples/00000006.xyz', help = 'test file')
    parser.add_argument('--modelPath', type = str, default = '../pretrained_models/pretrained_sfcnet_dgcnn/pretrained_sfcnet_dgcnn_new.pth', help = 'pretrained model')
    parser.add_argument('--outPath', type = str, default = '../test', help = 'file root for saving results')
    parser.add_argument('--numNb', type = int, help = 'number of neighbors used in DGCNN (should be the same for training)', default = 20)

    opt = parser.parse_args()
    print (opt)

    # Check Device (CPU / GPU)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # Read file
    samples, center, scale = read_xyz_with_scaling(opt.testFile)

    # Create Model
    print("Loading model...")
    model = SFCNetWithDGCNN(input_channels=3, nn_nb = opt.numNb, device=device)
    model.to(device)

    checkpoint = torch.load(opt.modelPath) if use_cuda else torch.load(opt.modelPath, map_location = torch.device('cpu'))
    model.load_state_dict(checkpoint)
    print("Load pretrained model: ", opt.modelPath)

    # Test
    print("Begin testing...")
    model.eval()

    with torch.no_grad():    
        samples = torch.from_numpy(samples.reshape((1, -1, 3))).float().to(device)
        y_pred, y_offsets = model(samples)
        # get results
        samples = np.squeeze(samples.cpu().data.numpy())
        y_pred = y_pred.cpu().data.numpy()
        y_offsets = y_offsets.cpu().data.numpy()

    # save results
    out_name = opt.testFile.split('/')[-1].rstrip('.xyz')
    preds_tag = np.squeeze(y_pred >= 0.)
    sharp_pts = samples[preds_tag] + y_offsets[0, preds_tag]
    sharp_pts = sharp_pts * scale + center.reshape((1, -1))

    # consolidated sharp feature points
    with open(os.path.join(opt.outPath, out_name + '_sharp_offset.xyz'), 'w') as fp:
        fp.write('\n'.join([" ".join(line) for line in sharp_pts.astype(str)]))

    # raw points that recognized as sharp feature points
    pred_pts = samples[preds_tag] * scale + center.reshape((1, -1))
    with open(os.path.join(opt.outPath, out_name + '_sharp.xyz'), 'w') as fp:
        fp.write('\n'.join([" ".join(line) for line in pred_pts.astype(str)]))

    print("Finish!")

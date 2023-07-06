import os, sys
sys.path.append('../data')
sys.path.append('../models')
sys.path.append('../utils')

import numpy as np
import argparse
import random
import shutil
import datetime

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from pcpnet_dataset import PCPDataset
from pcpnet_dataset_utils import RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from pcpnet_sfcnet import SFCNetWithPCP
import torch.nn as nn

def compute_loss(pred, target, output_loss_weight, patch_rot, device):

    loss = 0
    tensor_weights = torch.from_numpy(np.ones(2)).to(device)
    CE_loss = nn.CrossEntropyLoss(weight = tensor_weights.float())
    
    # classification
    cla_pred = pred[0]
    cla_target = target[0]
    cla_loss = CE_loss(cla_pred, torch.max(cla_target.long(), 1)[1])
    loss += output_loss_weight[0] * cla_loss

    # displacement
    disp_pred = pred[1]
    disp_target = target[1]
    if patch_rot is not None:
        # transform predictions with inverse transform
        # since we know the transform to be a rotation (QSTN), the transpose is the inverse
        disp_pred = torch.bmm(disp_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)

    reg_loss = cla_target[:, 0] * (disp_target - disp_pred).pow(2).sum(1)
    reg_loss = reg_loss.sum() / (cla_target[:, 0].sum() + 1)
    loss += output_loss_weight[1] * reg_loss

    return cla_loss, reg_loss, loss


if __name__ == '__main__':

    # Parameters
    parser = argparse.ArgumentParser()
    # naming / file handlingcd
    parser.add_argument('--dataRoot', type=str, default='../datasets/ABC_dataset/', help='file root')
    parser.add_argument('--logDir', type=str, default='./logs', help='training log folder')
    parser.add_argument('--trainList', type=str, default='train_models.csv', help='train list')
    parser.add_argument('--valList', type=str, default='val_models.csv', help='val list')
    # training parameters
    parser.add_argument('--workers', type = int, help = 'number of data loading workers', default = 4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--sharpThresh', type = float, help = 'threshold for sharp features', default = 0.015)
    parser.add_argument('--pointNoise', type = float, help = 'noise level for point cloud', default = 0.01)
    parser.add_argument('--nEpoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--checkPoint', type = str, default = "", help = 'pretrained model for finetuning')
    parser.add_argument('--dropRate', type = float, default = 0.3, help = 'dropout rate')
    parser.add_argument('--posWeight', type = float, default = 1, help = 'positive weight for bce loss (for unbalanced classification)')
    parser.add_argument('--seed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--trainingOrder', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--numFirstStage', type = int, help = 'number of epochs in the first stage', default = 10)
    parser.add_argument('--coeffMSE_f', type = float, help = 'coefficient for mse loss for the first stage', default = 1.)
    parser.add_argument('--coeffCEL_f', type = float, help = 'coefficient for bce loss for the first stage', default = 20.)
    parser.add_argument('--coeffMSE_s', type=float, help='coefficient for mse loss for the second stage', default = 1.)
    parser.add_argument('--coeffCEL_s', type=float, help='coefficient for bce loss for the second stage', default = 20.)
    # model parameters
    parser.add_argument('--patchRadius', type=float, default=0.05, help='patch radius')
    parser.add_argument('--patchesPerShape', type=int, default=100, help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--pointsPerPatch', type=int, default=500, help='max. number of points per patch')
    parser.add_argument('--usePca', type=int, default=False, help='Give both inputs and ground truth in local PCA coordinate frame')
    parser.add_argument('--usePointStn', type=int, default=True, help='use point spatial transformer')
    parser.add_argument('--useFeatStn', type=int, default=True, help='use feature spatial transformer')
    parser.add_argument('--symOp', type=str, default='max', help='symmetry operation')
    
    opt = parser.parse_args()
    print(opt)

    # Check Device (CPU / GPU)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create Log Path
    dir_name = os.path.join(opt.logDir, datetime.datetime.now().strftime("%d-%m-%y_%H-%M"))
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    log_name = os.path.join(dir_name, "log.txt")

    # Create Dataset
    train_dataset = PCPDataset( root = opt.dataRoot,
                                file_list = opt.trainList,
                                patch_radius = opt.patchRadius,
                                points_per_patch = opt.pointsPerPatch,
                                sharp_thresh = opt.sharpThresh,
                                noise_level = opt.pointNoise,
                                seed = None, # random seed for train
                                use_pca = opt.usePca)
    if opt.trainingOrder == 'random':
        train_datasampler = RandomPointcloudPatchSampler(
                                train_dataset,
                                patches_per_shape = opt.patchesPerShape,
                                seed = opt.seed,
                                identical_epochs = False)
    elif opt.trainingOrder == 'random_shape_consecutive':
        train_datasampler = SequentialShapeRandomPointcloudPatchSampler(
                                train_dataset,
                                patches_per_shape = opt.patchesPerShape,
                                seed = opt.seed,
                                identical_epochs = False)
    else:
        raise ValueError('Unknown training order: %s' % (opt.trainingOrder))
    
    train_dataloader = torch.utils.data.DataLoader( train_dataset,
                                                    sampler = train_datasampler,
                                                    batch_size = opt.batchSize,
                                                    num_workers = int(opt.workers))
    
    val_dataset = PCPDataset(   root = opt.dataRoot,
                                file_list = opt.valList,
                                patch_radius = opt.patchRadius,
                                points_per_patch = opt.pointsPerPatch,
                                sharp_thresh = opt.sharpThresh,
                                noise_level = opt.pointNoise,
                                seed = 0, 
                                use_pca = opt.use_pca)
    
    if opt.trainingOrder == 'random':
        val_datasampler = RandomPointcloudPatchSampler( 
                                val_dataset,
                                patches_per_shape = opt.patchesPerShape,
                                seed = opt.seed,
                                identical_epochs = False)
    elif opt.trainingOrder == 'random_shape_consecutive':
        val_datasampler = SequentialShapeRandomPointcloudPatchSampler(
                                val_dataset,
                                patches_per_shape = opt.patchesPerShape,
                                seed = opt.seed,
                                identical_epochs = opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.trainingOrder))

    val_dataloader = torch.utils.data.DataLoader(
                                val_dataset,
                                sampler = val_datasampler,
                                batch_size = opt.batchSize,
                                num_workers = int(opt.workers))


    # Create Model
    print("Loading model...")
    pcpnet = SFCNetWithPCP( num_points = opt.pointsPerPatch,
                            use_point_stn = opt.usePointStn,
                            use_feat_stn = opt.useFeatStn,
                            sym_op = opt.symOp)
    pcpnet.to(device)

    # Load Checkpoint
    if os.path.isfile(opt.checkPoint):
        checkpoint = torch.load(opt.checkPoint) if use_cuda else torch.load(opt.checkPoint, map_location = torch.device('cpu'))
        pcpnet.load_state_dict(checkpoint)
        print("Load pretrained model: ", opt.checkPoint)

    # Create Optimizer
    optimizer = optim.SGD(pcpnet.parameters(), lr = opt.lr, momentum = opt.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [], gamma = 0.1) # milestones in number of optimizer iterations

    train_num_batch = len(train_dataloader)
    val_num_batch = len(val_dataloader)

    # save parameters
    params_filename = os.path.join(dir_name, 'model_params.pth')
    torch.save(opt, params_filename)

    # Training
    print("Begin training...")

    for epoch in range(opt.nepoch):

        print("Begin %d epoch: " % epoch)
        pcpnet.train()

        trainloss = []
        valloss = []
        output_loss_weight = [opt.coeffCEL_f, opt.coeffMSE_f] if epoch < opt.numFirstStage else [opt.coeffCEL_s, opt.coeffMSE_s]

        for train_batchind, data in enumerate(train_dataloader, 0):

            # set to training mode
            pcpnet.train()

            # get trainingset batch and upload to GPU
            samples = data[0].transpose(2, 1).to(device)
            target = tuple(t.to(device) for t in data[1:-1])

            # forward pass
            y_pred, trans, y_offsets, _, _= pcpnet(samples)

            # zero gradients
            optimizer.zero_grad()
            cla_loss, reg_loss, loss = compute_loss(
                        pred = (y_pred, y_offsets), target = target,
                        output_loss_weight = output_loss_weight,
                        patch_rot = trans if opt.usePointStn else None,
                        device = device)

            # backpropagate through entire network to compute gradients of loss w.r.t. parameters
            loss.backward()

            # parameter optimization step
            optimizer.step()

            train_fraction_done = (train_batchind+1) / train_num_batch

            # print info and update log file
            acc = (torch.max(pred, 1)[1] == torch.max(target[0].long(), 1)[1]).sum() / float(pred.shape[0])
            print('[%s %d: %d/%d] %s pred loss: %f, disp loss: %f, loss: %f, acc: %f' % (opt.name, epoch, train_batchind, train_num_batch-1, green('train'), cla_loss.item(), reg_loss.item(), loss.item(), acc))
            trainloss.append(loss.item())
            
        for val_batchind, data in enumerate(val_dataloader, 0):

            # set to evaluation mode
            pcpnet.eval()

            # get testset batch and upload to GPU
            samples = data[0].transpose(2, 1).to(device)
            target = tuple(t.to(device) for t in data[1:-1])

            # forward pass
            with torch.no_grad():
                y_pred, trans, y_offsets, _ , _= pcpnet(samples)

            cla_loss, reg_loss, loss = compute_loss(
                pred=(y_pred, y_offsets), target = target,
                output_loss_weight = output_loss_weight,
                patch_rot = trans if opt.use_point_stn else None,
                device = device)

            # print info and update log file
            acc = (torch.max(y_pred.long(), 1)[1] == torch.max(target[0].long(), 1)[1]).sum() / float(pred.shape[0])
            print('[%s %d: %d/%d] %s pred loss: %f, disp loss: %f, loss: %f, acc: %f' % (opt.name, epoch, val_batchind, val_num_batch-1, blue('test'), cla_loss.item(), reg_loss.item(), loss.item(), acc))
            valloss.append(loss.item())

        # update learning rate
        scheduler.step()

        # save model, overwriting the old model
        torch.save(pcpnet.state_dict(), os.path.join(opt.outdir, '%s_model_%d.pth' % (opt.name, epoch)))
        np.savetxt(os.path.join('./results', 'valloss'+str(epoch)+'.txt'), valloss)
        np.savetxt(os.path.join('./results', 'trainloss'+str(epoch)+'.txt'), trainloss)
 
        train_dataset.reload_data_for_epoch()

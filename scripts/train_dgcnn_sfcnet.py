import os, sys
sys.path.append('../data')
sys.path.append('../models')
sys.path.append('../utils')

import numpy as np
import argparse
import datetime

import torch
import torch.optim as optim
import torch.nn as nn

from dgcnn_dataset import DGCNNDataset
from dgcnn_sfcnet import SFCNetWithDGCNN
from train_utils import MaskedMSELoss


if __name__ == '__main__':

    # Parameters
    parser = argparse.ArgumentParser()
    # naming / file handlingcd
    parser.add_argument('--dataRoot', type = str, default = '../datasets/ABC_dataset/', help = 'file root')
    parser.add_argument('--logDir', type=str, default='../logs', help='training log folder')
    parser.add_argument('--trainList', type = str, default = '../datasets/ABC_dataset/train_models.csv', help = 'train list')
    parser.add_argument('--valList', type = str, default = '../datasets/ABC_dataset/val_models.csv', help = 'val list')
    parser.add_argument('--workers', type = int, help = 'number of data loading workers', default = 4)
    # training parameters
    parser.add_argument('--numNb', type = int, help = 'number of neighbors used in DGCNN', default = 20)
    parser.add_argument('--batchSize', type = int, help = 'batch size', default = 2)
    parser.add_argument('--sharpThresh', type = float, help = 'threshold for sharp features', default = 0.03)
    parser.add_argument('--pointNoise', type = float, help = 'noise level for point cloud', default = 0.01)
    parser.add_argument('--numFirstStage', type = int, help = 'number of epochs in the first stage', default = 10)
    parser.add_argument('--coeffMSE_f', type = float, help = 'coefficient for mse loss for the first stage', default = 0.01)
    parser.add_argument('--coeffBCE_f', type = float, help = 'coefficient for bce loss for the first stage', default = 1.)
    parser.add_argument('--coeffMSE_s', type=float, help='coefficient for mse loss for the second stage', default = 100)
    parser.add_argument('--coeffBCE_s', type=float, help='coefficient for bce loss for the second stage', default = 0.01)
    parser.add_argument('--nEpoch', type = int, default = 50, help = 'number of epochs to train for')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'initial learning rate')
    parser.add_argument('--checkPoint', type = str, default = "", help = 'pretrained model for finetuning')
    parser.add_argument('--dropRate', type = float, default = 0.1, help = 'dropout rate')
    parser.add_argument('--posWeight', type = float, default = 1, help = 'positive weight for bce loss (for unbalanced classification)')

    opt = parser.parse_args()
    print(opt)

    # Check Device (CPU / GPU)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    # Create Log Path
    dir_name = os.path.join(opt.logDir, datetime.datetime.now().strftime("%d-%m-%y_%H-%M"))
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    log_name = os.path.join(dir_name, "log.txt")

    # Create Dataset
    print("Loading training and validation dataset...")
    train_data = DGCNNDataset(root=opt.dataRoot, file_list=opt.trainList, sharp_thresh=opt.sharpThresh, point_noise_level=opt.pointNoise)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = opt.batchSize, shuffle = True)
    print("  Number of train files: ", len(train_data))
    print("  Number of train batches: ", len(train_loader))

    val_data = DGCNNDataset(root=opt.dataRoot, file_list=opt.valList, sharp_thresh=opt.sharpThresh, point_noise_level=opt.pointNoise, seed=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = opt.batchSize, shuffle = True)
    print("  Number of val files: ", len(val_data))
    print("  Number of val batches: ", len(val_loader))

    # Create Model
    print("Loading model...")
    num_channels = 3
    model = SFCNetWithDGCNN(input_channels = num_channels, nn_nb = opt.numNb)
    model.to(device)

    # Load Checkpoint
    if os.path.isfile(opt.checkPoint):
        checkpoint = torch.load(opt.checkPoint) if use_cuda else torch.load(opt.checkPoint, map_location = torch.device('cpu'))
        model.load_state_dict(checkpoint)
        print("Load pretrained model: ", opt.checkPoint)

    # Create Optimizer
    curr_lr = opt.lr
    optimizer = optim.Adam(model.parameters(), lr = curr_lr)

    # save parameters
    params_filename = os.path.join(dir_name, 'model_params.pth')
    torch.save(opt, params_filename)

    # Create Loss
    loss_weight = torch.tensor([opt.posWeight]).to(device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight = loss_weight)
    mse_loss = MaskedMSELoss()

    # Training
    print("Begin training...")

    for epoch in range(opt.nEpoch):

        print("Begin %d epoch: " % epoch)
        model.train()
        
        total_acc = 0.
        total_samples = 0.
        total_pred_loss = 0.
        total_offset_loss = 0.
        total_loss = 0.

        for samples, labels, offsets in train_loader:
            samples = samples.float().to(device)
            offsets = offsets.float().to(device)
            labels = labels.float().unsqueeze(2).to(device)
            # model prediction
            y_pred, y_offsets = model(samples)
            # compute loss
            loss_pred = 0.
            loss_offset = 0.
            if epoch < opt.numFirstStage:
                loss_pred = opt.coeffBCE_f * bce_loss(y_pred, labels)
                loss_offset = opt.coeffMSE_f * mse_loss(labels, y_offsets, offsets)
            else:
                loss_pred = opt.coeffBCE_s * bce_loss(y_pred, labels)
                loss_offset = opt.coeffMSE_s * mse_loss(labels, y_offsets, offsets)
            loss = loss_pred + loss_offset
            # optimization
            optimizer.zero_grad()         
            loss.backward()               
            optimizer.step() 
            # compute loss
            total_loss = total_loss + loss.item()  
            total_pred_loss = total_pred_loss + loss_pred.item()
            total_offset_loss = total_offset_loss + loss_offset.item()
            # compute accuracy
            y_pred_tag = (y_pred >= 0)
            y_pred_num = np.prod(y_pred.detach().cpu().numpy().shape)
            curr_acc = float((y_pred_tag == labels).sum()) / float(y_pred_num)
            total_acc += curr_acc
            total_samples += samples.shape[0]
            # print evaluation progress
            if total_samples > 0 and np.abs(total_samples % 200) == 0:
                print("  Trained on %d samples!" % total_samples)
                print("    * curr train Loss: ", loss.item(), ", curr train pred loss:", loss_pred.item(), ", curr train offset loss: ", loss_offset.item(), ", curr train accuracy is: ", curr_acc)
            

        tacc = float(total_acc) / float(len(train_loader))
        tloss = total_loss / total_samples
        tpred = total_pred_loss / total_samples
        toffset = total_offset_loss / total_samples

        print("%d epoch training results: " % epoch)
        print("  * Total train Loss: ", tloss, ", total train pred loss:", tpred, ", total train offset loss: ", toffset, ", total train accuracy is: ", tacc)

        print("Save model!")
        torch.save(model.state_dict(), "%s/model_epoch%d.pth" % (dir_name, epoch))

        # Evaluation phase
        print("Begin evaluation...")
        model.eval()

        total_acc = 0.
        total_samples = 0.
        total_pred_loss = 0.
        total_offset_loss = 0.
        total_loss = 0.

        with torch.no_grad():
            for samples, labels, offsets in val_loader:
                samples = samples.float().to(device)
                offsets = offsets.float().to(device)
                labels = labels.float().unsqueeze(2).to(device)
                # model prediction
                y_pred, y_offsets = model(samples)    
                # compute loss
                loss_pred = 0.
                loss_offset = 0.
                if epoch < opt.numFirstStage:
                    loss_pred = opt.coeffBCE_f * bce_loss(y_pred, labels)
                    loss_offset = opt.coeffMSE_f * mse_loss(labels, y_offsets, offsets)
                else:
                    loss_pred = opt.coeffBCE_s * bce_loss(y_pred, labels)
                    loss_offset = opt.coeffMSE_s * mse_loss(labels, y_offsets, offsets)
                loss = loss_pred + loss_offset
                # save loss
                total_loss = total_loss + loss.item()  
                total_pred_loss = total_pred_loss + loss_pred.item()
                total_offset_loss = total_offset_loss + loss_offset.item()
                # compute accuracy
                y_pred_tag = (y_pred >= 0.)
                y_pred_num = np.prod(y_pred.detach().cpu().numpy().shape)
                curr_acc = float((y_pred_tag == labels).sum()) / float(y_pred_num)
                total_acc += curr_acc
                total_samples += samples.shape[0]
                # print evaluation progress
                if total_samples > 0 and np.abs(total_samples % 100) == 0:
                    print("  Evaluated on %d samples!" % total_samples)
                    print("    * curr val Loss: ", loss.item(), ", curr val pred loss:", loss_pred.item(), ", curr val offset loss: ", loss_offset.item(), ", curr val accuracy is: ", curr_acc)
            

            tacc = float(total_acc) / float(len(val_loader))
            tloss = total_loss / total_samples
            tpred = total_pred_loss / total_samples
            toffset = total_offset_loss / total_samples

            print("%d epoch eval results: " % epoch)
            print("  * Total val Loss: ", tloss, ", total val pred loss:", tpred, ", total val offset loss: ", toffset, ", total val accuracy is: ", tacc)

        # adjust learning rate
        if epoch == 19: 
            curr_lr = curr_lr / 10.
            optimizer = optim.Adam(model.parameters(), lr = curr_lr)
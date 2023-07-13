import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from dgcnn_backbone import DGCNNEncoderGn


class SFCNetWithDGCNN(nn.Module):
    """
    The implementation of the SFCNet with DGCNN backbone

    Params:
        - input_channels: number of input feature dimension, 3 (xyz) by default
        - nn_nb: number of the nearest neighbors
        - device: cpu/gpu
    """

    def __init__(self, input_channels = 3, nn_nb = 80, device = torch.device('cuda')):

        super(SFCNetWithDGCNN, self).__init__()
        self.encoder = DGCNNEncoderGn(input_channels = input_channels, nn_nb = nn_nb, device = device)

        # feature fusion
        self.conv1 = nn.Sequential( nn.Conv1d(1472, 512, 1),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU())
        self.conv2 = nn.Sequential( nn.Conv1d(512, 256, 1),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU())
        
        # regression branch (predicting offset)
        self.mlp_offset_prob1 = nn.Sequential( nn.Conv1d(256, 256, 1),
                                               nn.BatchNorm1d(256),
                                               nn.ReLU())
        self.mlp_offset_prob2 = nn.Conv1d(256, 3, 1)
        
        # classification branch (predicting label)
        self.mlp_boundary_prob1 = nn.Sequential( nn.Conv1d(256, 256, 1),
                                                 nn.BatchNorm1d(256),
                                                 nn.ReLU())
        self.mlp_boundary_prob2 = nn.Sequential( nn.Conv1d(256, 64, 1),
                                                 nn.BatchNorm1d(64),
                                                 nn.ReLU())
        self.mlp_boundary_prob3 = nn.Sequential( nn.Conv1d(64, 16, 1),
                                                 nn.BatchNorm1d(16),
                                                 nn.ReLU())
        self.mlp_boundary_prob4 = nn.Conv1d(16, 1, 1)



    def forward(self, points):

        batch_size, num_points, _ = points.shape
        points = points.permute(0, 2, 1)

        # encoder input point cloud
        feature, local_features = self.encoder(points)
        # concat global and local features
        feature = feature.view(batch_size, 1024, 1).repeat(1, 1, num_points)
        feature = torch.cat([feature, local_features], 1) # batch * 1472 * num_points
        # fusion features
        feature = self.conv1(feature)
        feature = self.conv2(feature) # batch * 256 * num_points
        # regression branch
        boundary = self.mlp_boundary_prob1(feature)
        boundary1 = self.mlp_boundary_prob2(boundary)
        boundary2 = self.mlp_boundary_prob3(boundary1)
        boundary3 = self.mlp_boundary_prob4(boundary2).permute(0, 2, 1)
        # classification branch
        offset = self.mlp_offset_prob1(feature)
        offset = self.mlp_offset_prob2(offset).transpose(1, 2)
    
        return boundary3, offset

        


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: ", device)

    model = SFCNetWithDGCNN(device = device, nn_nb = 18)
    model.to(device)
    #summary(model, input_size = (10000, 3))

    pts = torch.rand(1, 10000, 3).to(device)
    y_bd, y_off = model(pts)

    print(pts.shape)
    print(y_off.shape)
    print(y_bd.shape)
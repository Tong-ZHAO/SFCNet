import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary


def knn(x, k):
    """
    Find the nearest neighbors   
    Input: (batch_size, feature_dim, num_points)
    Output: (batch_size, num_points, k)
    """

    batch_size = x.shape[0]

    with torch.no_grad():
        distances = []
        for b in range(batch_size):
            inner = -2 * torch.matmul(x[b:b+1].transpose(2, 1), x[b:b+1])
            xx = torch.sum(x[b:b+1] ** 2, dim = 1, keepdim = True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            distances.append(pairwise_distance)
        distances = torch.stack(distances, 0)
        distances = distances.squeeze(1)
        idx = distances.topk(k=k, dim=-1)[1]

    return idx


def get_graph_feature(x, k, device):
    """
    Assemble graph features by finding the nearest neighbors in latent space

    Params:
        - x: input feature of dimension (batch_size, feature_dim, num_points)
        - k: number of the nearest neighbors
        - device: cpu/gpu

    Output: 
        - features: graph feature of dimension (batch_size, 2 * feature_dim, num_points, k)
    """

    batch_size = x.size(0)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)
    idx = knn(x, k = k) # batch * number_of_points * k
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    feature_dim = x.size()[1]
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]

    feature = feature.view(batch_size, num_points, k, feature_dim)
    x = x.view(batch_size, num_points, 1, feature_dim).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim = 3).permute(0, 3, 1, 2)

    return feature



class DGCNNEncoderGn(nn.Module):
    """
    The implementation of the DGCNN backbone

    Params:
        - input_channels: number of input feature dimension, 3 (xyz) by default
        - nn_nb: number of the nearest neighbors
        - device: cpu/gpu
    """

    def __init__(self, input_channels = 3, nn_nb = 80, device = torch.device('cuda')):

        super(DGCNNEncoderGn, self).__init__()
        self.k = nn_nb
        self.device = device
        
        # Define model layers
        self.conv1 = nn.Sequential( nn.Conv2d(input_channels * 2, 64, kernel_size = 1, bias = False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(negative_slope = 0.2) )
        self.conv2 = nn.Sequential( nn.Conv2d(64 * 2, 128, kernel_size = 1, bias = False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(negative_slope = 0.2) )
        self.conv3 = nn.Sequential( nn.Conv2d(128 * 2, 256, kernel_size = 1, bias = False),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(negative_slope = 0.2) )
        self.mlp1 = nn.Sequential(  nn.Conv1d(448, 1024, 1),
                                    nn.BatchNorm1d(1024),
                                    nn.functional.relu() )


    def forward(self, x):
        
        # First edge conv
        x = get_graph_feature(x, k = self.k, device = self.device) # on the Euclide space batch * 6 * num_points * k
        x = self.conv1(x)
        x1 = x.max(dim = -1, keepdim = False)[0] # pick the biggest value for each points from its k neighbors batch * 64 * num_points

        # Second edge conv
        x = get_graph_feature(x1, k = self.k, device = self.device) # on the feature map space batch * 128 * num_points * k
        x = self.conv2(x)
        x2 = x.max(dim = -1, keepdim = False)[0] # batch * 128 * num_points

        # Third edge conv
        x = get_graph_feature(x2, k = self.k, device = self.device) # on the feature map space batch * 256 * num_points * k
        x = self.conv3(x)
        x3 = x.max(dim = -1, keepdim = False)[0] # batch * 256 * num_points

        x_features = torch.cat((x1, x2, x3), dim = 1) # batch * (64+128+256) * num_points
        x = self.mlp1(x_features) # batch * 1024 * num_points
        x4 = x.max(dim = 2)[0]    # batch * 1024

        return x4, x_features # globle feature, local feature



if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: ", device)

    model = DGCNNEncoderGn()
    model.to(device)
    summary(model, input_size = (-1, 3, 10000))

    pts = torch.rand(1, 3, 10000).to(device)
    y, y_feats = model(pts)
    print(pts.shape)
    print(y.shape)
    print(y_feats.shape)
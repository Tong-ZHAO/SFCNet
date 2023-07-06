import os, sys
sys.path.append('../utils')

import torch
import torch.nn as nn
import torch.nn.functional as F

import train_utlis


class STN(nn.Module):
    """
    Implementation of feature transformation network

    Input: 
        - num_points: number of points
        - dim: the dimension of the feature
        - sym_op: the aggregation operator

    Output: 
        - x: a transformation matrix of size (batch_size, dim, dim)
    """

    def __init__(self, num_points = 500, dim = 3, sym_op = 'max'):
        super(STN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim * self.dim)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):

        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.dim, dtype = x.dtype, device = x.device).view(1, self.dim * self.dim).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.dim, self.dim)

        return x

class QSTN(nn.Module):
    """
    Implementation of point transformation network

    Input: 
        - num_points: number of points
        - dim: the dimension of the point cloud
        - sym_op: the aggregation operator

    Output: 
        - x: a transformation matrix of size (batch_size, dim, dim)
    """

    def __init__(self, num_points = 500, dim = 3, sym_op = 'max'):
        super(QSTN, self).__init__()

        self.dim = dim
        self.sym_op = sym_op
        self.num_points = num_points

        self.conv1 = torch.nn.Conv1d(self.dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # symmetric operation over all points
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # add identity quaternion (so the network can output 0 to leave the point cloud identical)
        iden = x.new_tensor([1, 0, 0, 0])
        x = x + iden

        # convert quaternion to rotation matrix
        x = train_utlis.batch_quat_to_rotmat(x)

        return x


class PCPEncoder(nn.Module):
    """
    Implementation of PCPNet backbone

    Input: 
        - num_points: number of points in a point patch
        - use_point_stn: flag indicates if we use the point transformer network
        - use_feat_stn: flag indicates if we use the feature transformer network
        - sym_op: the aggregation operator
        - get_pointfvals: flag indicates if we return pointwise feature in the patch

    Output: 
        - x: the feature of the point patch of size (batch_size, 1024)
        - trans: the point transformation matrix of size (batch_size, 3, 3)
        - trans2: the feature transformation matrix of size (batch_size, 64, 64)
        - pointfvals: the pointwise feature of size (batch_size, 1024, 500)
    """

    def __init__(self, num_points = 500, use_point_stn = True, use_feat_stn = True, sym_op = 'max', get_pointfvals = False):
        super(PCPEncoder, self).__init__()
        self.num_points = num_points
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals

        if self.use_point_stn:
            self.stn1 = QSTN(num_points = num_points, dim = 3, sym_op = self.sym_op)

        if self.use_feat_stn:
            self.stn2 = STN(num_points = num_points, dim = 64, sym_op = self.sym_op)

        self.conv0a = torch.nn.Conv1d(3, 64, 1)
        self.conv0b = torch.nn.Conv1d(64, 64, 1)
        self.bn0a = nn.BatchNorm1d(64)
        self.bn0b = nn.BatchNorm1d(64)
        self.conv1 = torch.nn.Conv1d(64, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.mp1 = None
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

    def forward(self, x):

        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3, -1)
        else:
            trans = None

        # mlp (64,64)
        x = F.relu(self.bn0a(self.conv0a(x)))
        x = F.relu(self.bn0b(self.conv0b(x)))

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        # mlp (64, 128, 1024)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        pointfvals = x if self.get_pointfvals else None

        # symmetric max operation over all points
        if self.sym_op == 'max':
            x = self.mp1(x)
        elif self.sym_op == 'sum':
            x = torch.sum(x, 2, keepdim = True)
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

        x = x.view(-1, 1024)

        return x, trans, trans2, pointfvals
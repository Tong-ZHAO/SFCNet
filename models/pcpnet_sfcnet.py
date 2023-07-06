import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from pcpnet_backbone import PCPEncoder

class SFCNetWithPCP(nn.Module):
    """
    The implementation of the SFCNet with PCPNet backbone

    Params:
        - num_points: number of points in a patch
        - dropout: dropout rate
        - use_point_stn: flag indicates if we use the point transformer network
        - use_feat_stn: flag indicates if we use the feature transformer network
        - sym_op: the aggregation operator
        - get_pointfvals: flag indicates if we return pointwise feature in the patch

    Output:
        - x: classification results of size (batch_size, 2)
        - trans: point transformation matrix of size (batch_size, 3, 3)
        - off: regression results of size (batch_size, 3)
        - trans2: feature transformation matrix of size (batch_size, 64, 64)
        - pointfvals: pointwise features of size (batch_size, 1024, 500)
    """

    def __init__(self, num_points = 500, dropout = 0.3, use_point_stn = True, use_feat_stn = True, sym_op = 'max', get_pointfvals = False):

        super(SFCNetWithPCP, self).__init__()
        self.num_points = num_points
        self.feat = PCPEncoder( num_points = num_points,
                                use_point_stn = use_point_stn,
                                use_feat_stn = use_feat_stn,
                                sym_op = sym_op,
                                get_pointfvals = get_pointfvals )

        # classification branch
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.do1 = nn.Dropout(p = dropout)
        self.do2 = nn.Dropout(p = dropout)

        # regression branch
        self.fc1o = nn.Linear(1024, 512)
        self.fc2o = nn.Linear(512, 256)
        self.fc3o = nn.Linear(256, 3)
        self.bn1o = nn.BatchNorm1d(512)
        self.bn2o = nn.BatchNorm1d(256)
        self.do1o = nn.Dropout(p = dropout)
        self.do2o = nn.Dropout(p = dropout)

    def forward(self, x):

        x_feature, trans, trans2, pointfvals = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x_feature)))
        x = self.do1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.do2(x)
        x = self.fc3(x)

        off = F.relu(self.bn1o(self.fc1o(x_feature)))
        off = self.do1o(off)
        off = F.relu(self.bn2o(self.fc2o(off)))
        off = self.do2o(off)
        off = self.fc3o(off)

        return x, trans, off, trans2, pointfvals
    

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Using device: ", device)

    model = SFCNetWithPCP()
    model.to(device)
    summary(model, input_size = (-1, 3, 500))

    pts = torch.rand(64, 3, 500).to(device)
    cla, _, disp, _ , _ = model(pts)
    print(pts.shape)
    print(cla.shape)
    print(disp.shape)
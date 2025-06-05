from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

# log = True
log = False

class PointNetfeat(nn.Module):
    '''
        The feature extractor in PointNet, corresponding to the left MLP in the pipeline figure.
        Args:
        d: the dimension of the global feature, default is 1024.
        segmentation: whether to perform segmentation, default is True.
    '''
    def __init__(self, segmentation = False, d=1024):
        super(PointNetfeat, self).__init__()
        # nn.Linear
        self.flatten = nn.Flatten()
        self.segmentation = segmentation
        self.d = d
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, d)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
            If segmentation == True
                return the concatenated global feature and local feature. # (B, d+64, N)
            If segmentation == False
                return the global feature, and the per point feature for cruciality visualization in question b). # (B, d), (B, N, d)
            Here, B is the batch size, N is the number of points, d is the dimension of the global feature.
        '''
        x = self.relu(self.bn1(self.fc1(x).permute(0, 2, 1))).permute(0,2,1)  # (B, N, 64)
        feature1 = x
        x = self.relu(self.bn2(self.fc2(x).permute(0, 2, 1))).permute(0,2,1)  # (B, N, 128)
        x = self.bn3(self.fc3(x).permute(0,2,1)).permute(0,2,1)  # (B, N, 256)
        points_feat = x
        global_feat, _ = torch.max(x, 1) # (B, d)
        if self.segmentation:
            global_feat = global_feat.unsqueeze(-1)
            global_feat = global_feat.repeat(1, 1, x.size(1)).permute(0, 2, 1)
            
            return torch.cat((feature1, global_feat), -1) 
        else:
            return global_feat, points_feat


class PointNetCls1024D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the middle right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2):
        super(PointNetCls1024D, self).__init__()

        self.feat = PointNetfeat(segmentation=False, d=1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=1024)
        '''

        x0, points_feat = self.feat(x)
        x1 = self.relu(self.bn1(self.fc1(x0)))
        x2 = self.relu(self.bn2(self.fc2(x1)))
        x3 = self.fc3(x2)

        return self.logsoftmax(x3), points_feat

class PointNetCls256D(nn.Module):
    '''
        The classifier in PointNet, corresponding to the upper right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''
    def __init__(self, k=2 ):
        super(PointNetCls256D, self).__init__()

        self.feat = PointNetfeat(segmentation=False, d=256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        '''
            return the log softmax of the classification result and the per point feature for cruciality visualization in question b). # (B, k), (B, N, d=256)
        '''

        x0, point_feat = self.feat(x)
        x1 = self.relu(self.bn1(self.fc1(x0)))
        x2 = self.fc2(x1)

        return self.logsoftmax(x2), point_feat

class PointNetSeg(nn.Module):
    '''
        The segmentation head in PointNet, corresponding to the lower right MLP in the pipeline figure.
        Args:
        k: the number of classes, default is 2.
    '''

    def __init__(self, k = 2):
        super(PointNetSeg, self).__init__()
        self.k = k
        self.fc1 = nn.Linear(1024 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        # self.bn4 = nn.BatchNorm1d(k)
        self.relu = nn.ReLU()
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.pointfeat = PointNetfeat(segmentation=True, d=1024)

    def forward(self, x):
        '''
            Input:
                x: the input point cloud. # (B, N, 3)
            Output:
                the log softmax of the segmentation result. # (B, N, k)
        '''

        x0 = self.pointfeat(x)# .permute(0, 2, 1)
        x1 = self.relu(self.bn1(self.fc1(x0).permute(0, 2, 1))).permute(0, 2, 1)
        x2 = self.relu(self.bn2(self.fc2(x1).permute(0, 2, 1))).permute(0, 2, 1)
        x3 = self.relu(self.bn3(self.fc3(x2).permute(0, 2, 1))).permute(0, 2, 1)
        x4 = self.fc4(x3)
        return self.logsoftmax(x4)


if __name__ == '__main__':
    modal = PointNetSeg()
    x = torch.rand(10, 500, 3)
    y1 = modal(x)
    print(y1.shape)
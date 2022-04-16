import torch
import torch.nn as nn
import pointops.functions as pointops
import math

import numpy as np

class FeatureExtractionComponent(nn.Module):
    def __init__(self,pre_channel,num_neighbors,G=12):
        super(FeatureExtractionComponent, self).__init__()

        self.num_neighbors = num_neighbors
        self.pre = nn.Sequential(nn.Conv1d(pre_channel,24,1),
                                 nn.BatchNorm1d(24),
                                 nn.ReLU(inplace=True))

        self.grouper = pointops.QueryAndGroup(nsample=num_neighbors + 1,return_idx=True,use_xyz=False)

        in_channel = 48
        self.shared_mlp_1 = nn.Sequential(nn.Conv2d(in_channel,G,1,bias=False),
                                          nn.BatchNorm2d(G),
                                          nn.ReLU(inplace=True))
        in_channel += G
        self.shared_mlp_2 = nn.Sequential(nn.Conv2d(in_channel,G,1,bias=False),
                                      nn.BatchNorm2d(G),
                                      nn.ReLU(inplace=True))
        in_channel += G
        self.shared_mlp_3 = nn.Conv2d(in_channel, G,1,bias=False)

        self.maxpool = nn.MaxPool2d([1,self.num_neighbors])

    def forward(self,x,xyz=None):
        '''
        :param x: [B,C,N]
        :return: [B,C',N]
        '''
        if xyz == None:
            xyz = x.detach()
        # pre layer [B,24,N]
        pre_points = self.pre(x)
        # knn [B,24,N,K]
        grouped_feature,grouped_xyz,indices = self.grouper(xyz=xyz.permute(0,2,1),features=pre_points)
        grouped_feature = grouped_feature[...,1:]
        grouped_xyz = grouped_xyz[...,1:]
        indices = indices[...,1:]
        #print(grouped_feature.shape,grouped_xyz.shape,indices.shape)
        #torch.Size([2, 24, 256, 16]) torch.Size([2, 3, 256, 16]) torch.Size([2, 256, 16])

        y = torch.cat([pre_points.unsqueeze(-1).repeat(1,1,1,self.num_neighbors),grouped_feature],dim=1)
        # print(y.shape) #torch.Size([2, 48, 256, 15])

        y = torch.cat([self.shared_mlp_1(y),y],dim=1) #torch.Size([2, 60, 256, 15])
        # print(y.shape)

        y = torch.cat([self.shared_mlp_2(y),y],dim=1) #torch.Size([2, 72, 256, 15])
        # print(y.shape)

        y = torch.cat([self.shared_mlp_3(y),y],dim=1) #torch.Size([2, 84, 256, 15])
        # print(y.shape)

        y = self.maxpool(y) #torch.Size([2, 84, 256, 1])
        # print(y.shape)

        y = torch.cat([y.squeeze(),x],dim=1)
        return y


class FeatureExtractionUnit(nn.Module):
    def __init__(self,num_neighbors,G=12):
        super(FeatureExtractionUnit, self).__init__()

        self.feature_extractor_1 = FeatureExtractionComponent(pre_channel=3,num_neighbors=num_neighbors,G=G)

        self.feature_extractor_2 = FeatureExtractionComponent(pre_channel=84 + 3, num_neighbors=num_neighbors, G=G)

        self.feature_extractor_3 = FeatureExtractionComponent(pre_channel=84 + 84 + 3, num_neighbors=num_neighbors, G=G)

        self.feature_extractor_4 = FeatureExtractionComponent(pre_channel=84 + 84 + 84 + 3, num_neighbors=num_neighbors, G=G)

    def forward(self,x):
        '''
        :param x: [B,3,N]
        :return: [B,339,N]
        '''
        y = self.feature_extractor_1(x) # torch.Size([2, 87, 256])
        # print(f'FE 1 : {y.shape}')
        y = self.feature_extractor_2(y,x) # torch.Size([2, 171, 256])
        # print(f'FE 2 : {y.shape}')
        y = self.feature_extractor_3(y,x) # torch.Size([2, 255, 256])
        # print(f'FE 3 : {y.shape}')
        y = self.feature_extractor_4(y,x) # torch.Size([2, 339, 256])
        # print(f'FE 4 : {y.shape}')
        return y

class FeatureExpansionUnit(nn.Module):
    def __init__(self,use_cuda=True,upsacle_factor=4):
        super(FeatureExpansionUnit, self).__init__()
        self.use_cuda = use_cuda
        self.upsacle_factor = upsacle_factor

    def gen_grid(self,grid_size):
        """
        output [2, grid_size x grid_size]
        """
        x = torch.linspace(-0.2, 0.2, grid_size, dtype=torch.float32)
        # grid_sizexgrid_size
        x, y = torch.meshgrid(x, x)

        # 2xgrid_sizexgrid_size
        grid = torch.stack([x, y], dim=0).view(2,
                                               grid_size * grid_size)  # [2, grid_size, grid_size] -> [2, grid_size*grid_size]
        return grid

    def cat_code(self,x):
        B,C,N = x.shape


        # code_plus = torch.ones(B,1,N)
        # code_minus = -torch.ones(B,1,N)

        code = self.gen_grid(round(math.sqrt(self.upsacle_factor * N))).expand(B,-1,-1)

        if self.use_cuda:
            # code_plus = code_plus.cuda()
            # code_minus = code_minus.cuda()
            code = code.cuda()

        # plus = torch.cat([x,code_plus],dim=1)
        # minus = torch.cat([x,code_minus],dim=1)

        return torch.cat([x.repeat(1,1,self.upsacle_factor),code],dim=1)

    def forward(self,x):
        '''
        :param x:[B,C,N]
        :return: [B,C+2,4N]
        '''
        return self.cat_code(x)


class LocalRefinementUnit(nn.Module):
    def __init__(self,num_neighbors,in_channels):
        super(LocalRefinementUnit, self).__init__()

        self.coord_grouper = pointops.QueryAndGroup(nsample=num_neighbors + 1,return_idx=True,use_xyz=True)
        self.feature_grouper = pointops.QueryAndGroupFeature(nsample=num_neighbors + 1,return_idx=False,use_feature=False)

        self.feature_mlps = nn.Sequential(nn.Conv2d(in_channels + 3,2*in_channels,1,bias=False),
                                          nn.BatchNorm2d(2*in_channels),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(2*in_channels,in_channels,1),
                                          nn.BatchNorm2d(in_channels),
                                          nn.ReLU(inplace=True)
                                          )
        self.coord_mlps = nn.Sequential(nn.Conv2d(3,in_channels,1,bias=False),
                                          nn.BatchNorm2d(in_channels),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_channels,num_neighbors,1)
                                          )

        self.mlps_maxpool = nn.Sequential(nn.Conv2d(in_channels,2*in_channels,1,bias=False),
                                          nn.BatchNorm2d(2*in_channels),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(2*in_channels,in_channels,1,bias=False),
                                          nn.BatchNorm2d(in_channels),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d([1,num_neighbors])
                                          )

    def forward(self,x,xyz):
        '''
        :param x: features [B,C,4N]
        :param xyz: coord [B,4N,3]
        :return: [B,C,4N]
        '''

        grouped_diff, _, indices = self.coord_grouper(xyz=xyz.contiguous())

        grouped_diff = grouped_diff[...,1:]

        grouped_feature = self.feature_grouper(xyz=xyz.contiguous(), features=x.contiguous(), idx=indices.int())
        grouped_feature = grouped_feature[...,1:]
        #print(grouped_diff.shape,grouped_xyz.shape,grouped_feature.shape)
        #torch.Size([2, 3, 1024, 15]) torch.Size([2, 3, 1024, 15]) torch.Size([2, 32, 1024, 15])

        weights = self.coord_mlps(grouped_diff)
        #print(weights.shape) torch.Size([2, 15, 1024, 15])

        cat_feature = torch.cat([grouped_feature,grouped_diff],dim=1)
        #print(cat_feature.shape) torch.Size([2, 35, 1024, 15])

        features = self.feature_mlps(cat_feature)
        #print(features.shape) torch.Size([2, 32, 1024, 15])

        rNxC_feature = self.mlps_maxpool(features).squeeze()
        #print(rNxC_feature.shape) torch.Size([2, 32, 1024])

        # torch.Size([2,1024,15,15]) torch.Size([2,1024,15,32])
        rNxC_weight = torch.matmul(weights.permute(0,2,3,1),features.permute(0,2,3,1))
        rNxC_weight = rNxC_weight.sum(dim=-2).squeeze()
        # print(rNxC_weight.shape) torch.Size([2, 1024, 32])

        return rNxC_weight.permute(0,2,1) + rNxC_feature

class SelfAttentionUnit(nn.Module):
    def __init__(self,in_channels):
        super(SelfAttentionUnit, self).__init__()

        self.to_q = nn.Sequential(nn.Conv1d(3 + in_channels,2 * in_channels,1,bias=False),
                                  nn.BatchNorm1d(2 * in_channels),
                                  nn.ReLU(inplace=True))
        self.to_k = nn.Sequential(nn.Conv1d(3 + in_channels,2 * in_channels, 1, bias=False),
                                  nn.BatchNorm1d(2 * in_channels),
                                  nn.ReLU(inplace=True))
        self.to_v = nn.Sequential(nn.Conv1d(3 + in_channels,2 * in_channels, 1, bias=False),
                                  nn.BatchNorm1d(2 * in_channels),
                                  nn.ReLU(inplace=True))

        self.fusion = nn.Sequential(nn.Conv1d(2 * in_channels, in_channels, 1, bias=False),
                                  nn.BatchNorm1d(in_channels),
                                  nn.ReLU(inplace=True))

    def forward(self,x):
        '''
        :param x: [B,C + 3,4N]
        :return: [B,C,4N]
        '''

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        #print(q.shape,k.shape,v.shape)
        attention_map = torch.matmul(q.permute(0,2,1),k)
        #print(attention_map.shape) torch.Size([2, 1024, 1024])
        value = torch.matmul(attention_map,v.permute(0,2,1)).permute(0,2,1)

        value = self.fusion(value)
        #print(value.shape) torch.Size([2, 64, 1024])

        return value


class GlobalRefinementUnit(nn.Module):
    def __init__(self,in_channels):
        super(GlobalRefinementUnit, self).__init__()
        self.sa = SelfAttentionUnit(in_channels=in_channels)

    def forward(self,x,xyz):
        '''
        :param x: [B,C,4N]
        :param xyz: [B,4N,3]
        :return: [B,C,4N]
        '''
        cat = torch.cat([xyz.permute(0,2,1),x],dim=1)

        return self.sa(cat)

class OffsetRegression(nn.Module):
    def __init__(self,in_channels):
        super(OffsetRegression, self).__init__()
        self.coordinate_regression = nn.Sequential(nn.Conv1d(in_channels, 256, 1),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv1d(256, 64, 1),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv1d(64, 3, 1),
                                                   nn.Sigmoid())
        self.range_max = 0.5

    def forward(self,x):
        '''
        :param x: [B,C,4N]
        :return: [B,4N,3]
        '''

        offset = self.coordinate_regression(x)

        return (offset * self.range_max * 2 - self.range_max).permute(0,2,1)

class DenseGenerator(nn.Module):
    def __init__(self):
        super(DenseGenerator, self).__init__()

        self.feature_extractor = FeatureExtractionUnit(num_neighbors=16,G=12)

        self.feature_expansion = FeatureExpansionUnit(use_cuda=True,upsacle_factor=4)

        self.coordinate_regression = nn.Sequential(nn.Conv1d(339 + 2,256,1),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv1d(256,64,1),
                                                   nn.ReLU(inplace=True),
                                                   nn.Conv1d(64,3,1))

    def forward(self,x):
        '''
        :param x:[B,N,3]
        :return: [B,3,4*N]
        '''

        y = self.feature_extractor(x.permute(0,2,1))
        y = self.feature_expansion(y)

        return y,self.coordinate_regression(y).permute(0,2,1)

class SpatialRefiner(nn.Module):
    def __init__(self,num_neighbors,in_channels):
        super(SpatialRefiner, self).__init__()

        self.local_refinement =  LocalRefinementUnit(num_neighbors=num_neighbors,in_channels=in_channels)
        self.global_refinement = GlobalRefinementUnit(in_channels=in_channels)
        self.offset = OffsetRegression(in_channels=in_channels)

    def forward(self,x,xyz):
        '''
        :param x: [B,C,4N]
        :param xyz: [B,4N,3]
        :return: [B,4N,3]
        '''
        local_refine = self.local_refinement(x, xyz)
        global_refine = self.global_refinement(x, xyz)

        refine = local_refine + global_refine

        return self.offset(refine)


class DisPU(nn.Module):
    def __init__(self,num_neighbors=16):
        super(DisPU, self).__init__()
        self.dense = DenseGenerator()

        self.refine = SpatialRefiner(num_neighbors=num_neighbors,in_channels=341)

    def forward(self,x):
        '''
        :param x: [B,N,3]
        :return: [B,4N,3]
        '''

        feature, coord = self.dense(x)
        offset = self.refine(feature,coord)

        return coord,offset

xyz = torch.randn(2,256,3).cuda()

# f = DisPU().cuda()
# _1,_2 = f(xyz)
# print(_1.shape,_2.shape)


# x = torch.randn(2,64,1024).cuda()
# f = OffsetRegression(in_channels=64).cuda()
# print(f(x).shape)

# xyz = torch.randn(2,1024,3).cuda()
# feature = torch.randn(2,64,1024).cuda()
# f = GlobalRefinementUnit(in_channels=64).cuda()
# print(f(feature,xyz).shape)

# xyz = torch.randn(2,1024,3).cuda()
# feature = torch.randn(2,64,1024).cuda()
# f = LocalRefinementUnit(num_neighbors=15,in_channels=64).cuda()
# output = f(feature,xyz)
# print(output.shape)

# x = torch.randn(2,256,3).cuda()
# # f = FeatureExtractionComponent(pre_channel=3,num_neighbors=15,G=12).cuda()
# f = FeatureExtractionUnit(num_neighbors=15,G=12).cuda()
# print(f(x.permute(0,2,1)).shape)

# x = torch.randn(2,4,256).cuda()
# f = FeatureExpansionUnit().cuda()
# print(x.shape)
# print(f(x).shape)

# x = torch.randn(64,256,3).cuda()
# f = DenseGenerator().cuda()
# print(f(x)[0].shape)
import torch
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.self_attn import AttFusion

import pdb

class PointPillarOPV2V(nn.Module):
    def __init__(self, args):
        super(PointPillarOPV2V, self).__init__()

        self.max_cav = args['max_cav']
        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)
        # used to downsample the feature map for efficient computation
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
        self.compression = False

        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = AttFusion(256)

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        spatial_correction_matrix = data_dict['spatial_correction_matrix']

        # B, max_cav, 3(dt dv infra), 1, 1
        prior_encoding =\
            data_dict['prior_encoding'].unsqueeze(-1).unsqueeze(-1)

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        spatial_features = batch_dict['spatial_features']   # 2 64 192 704

        batch_dict = self.backbone(batch_dict)



        spatial_features_2d = batch_dict['spatial_features_2d']  #2 384 96 352
        

        # test ratio calc
        # spatial_features_2d = torch.tensor([
        #     1,0
        # ]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        ###TODO:baolu  calc the ratio of ego vehicle in feature maps
        Is_ratio = True
        if Is_ratio:
            # v1,v2 = torch.max(spatial_features_2d,dim=0,keepdim=True)
            # v3 = torch.where(spatial_features_2d[0].unsqueeze(0)==v1,1,0)
            # v4 = torch.sum(v3)
            # v5 = v3.shape[0]*v3.shape[1]*v3.shape[2]*v3.shape[3]
            # ratio = v4/v5

            #v1 = torch.max(spatial_features_2d)
            # v2 = spatial_features_2d#/v1
            # v3 = torch.ones_like(v2)*1
            # v4 = torch.where(v2>v3,1,0)
            # v5 = torch.sum(v4)
            # v6 = v3.shape[0]*v3.shape[1]*v3.shape[2]*v3.shape[3]
            # ratio = v5/v6

            # max = torch.max(spatial_features_2d)
            # median = torch.median(spatial_features_2d)
            # min = torch.min(spatial_features_2d)

            v1 = torch.zeros_like(spatial_features_2d)
            v2 = torch.where(spatial_features_2d==v1,1,0)
            v3 = torch.sum(v2)
            #v4 = v1.shape[0]*v1.shape[1]*v1.shape[2]*v1.shape[3]
            ratio = v3#/v4
        


        #ratio = 1
        # print(max,'    ',median,'     ',min)


        # downsample feature to reduce memory
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        # compressor
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)  # 2 256 48 176

        fused_feature = self.fusion_net(spatial_features_2d, record_len)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        # output_dict = {'psm': psm,
        #                'rm': rm}
        
        output_dict = {'psm': psm,
                       'rm': rm,
                       'spatial_features':spatial_features,
                       'before_feature': spatial_features_2d}
        
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if Is_ratio:
            output_dict['ratio'] = float(ratio)
        return output_dict
        # if self.training:
        #     feature = []
        #     spatial_features_2d_list = []
        #     for data_per in data_dict:#######[source_data_dict, target_data_dict]####
        #         voxel_features = data_per['processed_lidar']['voxel_features']
        #         voxel_coords = data_per['processed_lidar']['voxel_coords']
        #         voxel_num_points = data_per['processed_lidar']['voxel_num_points']
        #         record_len = data_per['record_len']

        #         batch_dict = {'voxel_features': voxel_features,
        #                     'voxel_coords': voxel_coords,
        #                     'voxel_num_points': voxel_num_points,
        #                     'record_len': record_len}
        #         # n, 4 -> n, c
        #         batch_dict = self.pillar_vfe(batch_dict)
        #         # n, c -> N, C, H, W
        #         batch_dict = self.scatter(batch_dict)
        #         batch_dict = self.backbone(batch_dict)

        #         spatial_features_2d = batch_dict['spatial_features_2d']

        #         # downsample feature to reduce memory
        #         if self.shrink_flag:
        #             spatial_features_2d = self.shrink_conv(spatial_features_2d)
        #         # compressor
        #         if self.compression:
        #             spatial_features_2d = self.naive_compressor(spatial_features_2d)

        #         fused_feature = self.fusion_net(spatial_features_2d, record_len)


        #         feature.append(fused_feature)####[source_feature, target_feature]####
        #         spatial_features_2d_list.append(spatial_features_2d)

        #     output_dict = {'psm': self.cls_head(feature[0]),#source_feature
        #                 'rm': self.reg_head(feature[0]),#source_feature
        #                 'source_feature': feature[0],#source_feature
        #                 'target_feature': feature[1],#target_feature
        #                 'source_multifea': spatial_features_2d_list[0],#source_feature
        #                 'target_multifea': spatial_features_2d_list[1],#target_feature
        #                 'target_psm': self.cls_head(feature[1]),#target_feature
        #                 'target_rm': self.reg_head(feature[1])#target_feature
        #                 }
        # else:
        #     voxel_features = data_dict['processed_lidar']['voxel_features']
        #     voxel_coords = data_dict['processed_lidar']['voxel_coords']
        #     voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        #     record_len = data_dict['record_len']

        #     batch_dict = {'voxel_features': voxel_features,
        #                 'voxel_coords': voxel_coords,
        #                 'voxel_num_points': voxel_num_points,
        #                 'record_len': record_len}
        #     # n, 4 -> n, c
        #     batch_dict = self.pillar_vfe(batch_dict)
        #     # n, c -> N, C, H, W
        #     batch_dict = self.scatter(batch_dict)
        #     batch_dict = self.backbone(batch_dict)

        #     spatial_features_2d = batch_dict['spatial_features_2d']

        #     # downsample feature to reduce memory
        #     if self.shrink_flag:
        #         spatial_features_2d = self.shrink_conv(spatial_features_2d)
        #     # compressor
        #     if self.compression:
        #         spatial_features_2d = self.naive_compressor(spatial_features_2d)

        #     fused_feature = self.fusion_net(spatial_features_2d, record_len)

        #     # pdb.set_trace()


        #     psm = self.cls_head(fused_feature)
        #     rm = self.reg_head(fused_feature)

        #     output_dict = {'psm': psm,
        #                 'rm': rm}

        # return output_dict
import torch
import torch.nn as nn   
from models.module_dcn import ModulatedDeformableConv2d

class PCDAlignment(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8, n_frames=3, total_levels=3, current_level=1):
        super(PCDAlignment, self).__init__()
        
        self.offset_conv1 = nn.Conv2d(num_feat*2, num_feat, 3, 1, 1)

        if current_level == total_levels:
            self.offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        else: 
            self.offset_conv2 = nn.Conv2d(num_feat*2, num_feat, 3, 1, 1)
            self.offset_conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.dcn_pack = ModulatedDeformableConv2d(num_feat,
                                                  num_feat,
                                                  3,
                                                  deformable_groups=deformable_groups,
                                                  offset_in_channel=num_feat)
        if current_level < total_levels:
            self.feat_conv = nn.Conv2d(num_feat*2, num_feat, 1, 1, 0)
        
        self.offset_shrink = nn.Conv2d(num_feat, num_feat//2, 1, 1, 0)
        self.feat_shrink = nn.Conv2d(num_feat, num_feat//2, 1, 1, 0)

        self.n_frames = n_frames
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.current_level = current_level
        self.total_levels = total_levels
        self.conv_fusion = nn.Conv2d(num_feat*n_frames, num_feat, 3, 1, 1)
        # self.conv_fusion = nn.Conv2d(num_feat*n_frames, num_feat, 3, 1, 1)
        # if current_level == total_levels:
        #     self.conv_stage2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.beta = nn.Parameter(torch.zeros(1, num_feat, 1, 1), requires_grad=True)

    def forward(self, one_stage_feats, transferred_feats, transferred_offset):

        aligned_feats = []
        transferred_new_offset = []
        transferred_new_feats = []
        upsampled_offset, upsampled_feat = None, None
        splitted_feats = torch.chunk(one_stage_feats, self.n_frames, dim=1)
        ref_feat = splitted_feats[self.n_frames//2]
        for i in range(self.n_frames):
            nbr_feat = splitted_feats[i]
            offset = torch.cat([nbr_feat, ref_feat], dim=1)
            offset = self.lrelu(self.offset_conv1(offset))
            if self.current_level == self.total_levels:
                offset = self.lrelu(self.offset_conv2(offset))
            else:
                offset = self.lrelu(self.offset_conv2(
                    torch.cat([offset, transferred_offset[i]], dim=1)))
                offset = self.lrelu(self.offset_conv3(offset))
            
            feat = self.dcn_pack([nbr_feat, offset]) + nbr_feat
            if self.current_level < self.total_levels:
                feat = self.lrelu(self.feat_conv(torch.cat([feat, transferred_feats[i]], dim=1)))

            else:
                feat = self.lrelu(feat)
            
            if self.current_level > 1:
                upsampled_offset = self.upsample(self.offset_shrink(offset)) * 2
                upsampled_feat = self.upsample(self.feat_shrink(feat))

            aligned_feats.append(feat)
            transferred_new_offset.append(upsampled_offset)
            transferred_new_feats.append(upsampled_feat)

        aligned_feats = torch.cat(aligned_feats, dim=1)
        aligned_feats = self.conv_fusion(aligned_feats)
        # if self.current_level == self.total_levels:
        #     aligned_feats = self.conv_stage2(self.lrelu(aligned_feats))
        aligned_feats = aligned_feats * self.beta + ref_feat

        return aligned_feats, transferred_new_feats, transferred_new_offset
    
class PCDAlignment_raw(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8, n_frames=3, total_levels=3, current_level=1):
        super(PCDAlignment_raw, self).__init__()
        
        self.offset_conv1 = nn.Conv2d(num_feat*2, num_feat, 3, 1, 1)

        if current_level == total_levels:
            self.offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.offset_conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        else: 
            self.offset_conv2 = nn.Conv2d(num_feat*2, num_feat, 3, 1, 1)
            self.offset_conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.offset_conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.dcn_pack = ModulatedDeformableConv2d(num_feat,
                                                  num_feat,
                                                  3,
                                                  deformable_groups=deformable_groups,
                                                  offset_in_channel=num_feat)
        if current_level < total_levels:
            self.feat_conv = nn.Conv2d(num_feat*2, num_feat, 1, 1, 0)
        
        self.offset_shrink = nn.Conv2d(num_feat, num_feat//2, 1, 1, 0)
        self.feat_shrink = nn.Conv2d(num_feat, num_feat//2, 1, 1, 0)

        self.n_frames = n_frames
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.current_level = current_level
        self.total_levels = total_levels
        self.extra_conv = nn.Conv2d(num_feat*n_frames, num_feat, 3, 1, 1)
        self.conv_fusion = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.conv_fusion = nn.Conv2d(num_feat*n_frames, num_feat, 3, 1, 1)
        # if current_level == total_levels:
        #     self.conv_stage2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.beta = nn.Parameter(torch.zeros(1, num_feat, 1, 1), requires_grad=True)

    def forward(self, one_stage_feats, transferred_feats, transferred_offset):

        aligned_feats = []
        transferred_new_offset = []
        transferred_new_feats = []
        upsampled_offset, upsampled_feat = None, None
        splitted_feats = torch.chunk(one_stage_feats, self.n_frames, dim=1)
        ref_feat = splitted_feats[self.n_frames//2]
        for i in range(self.n_frames):
            nbr_feat = splitted_feats[i]
            offset = torch.cat([nbr_feat, ref_feat], dim=1)
            offset = self.lrelu(self.offset_conv1(offset))
            if self.current_level == self.total_levels:
                offset = self.lrelu(self.offset_conv3(self.lrelu(self.offset_conv2(offset))))
            else:
                offset = self.lrelu(self.offset_conv2(
                    torch.cat([offset, transferred_offset[i]], dim=1)))
                offset = self.lrelu(self.offset_conv3(offset))
                offset = self.lrelu(self.offset_conv4(offset))
            
            feat = self.dcn_pack([nbr_feat, offset]) + nbr_feat
            if self.current_level < self.total_levels:
                feat = self.lrelu(self.feat_conv(torch.cat([feat, transferred_feats[i]], dim=1)))

            else:
                feat = self.lrelu(feat)
            
            if self.current_level > 1:
                upsampled_offset = self.upsample(self.offset_shrink(offset)) * 2
                upsampled_feat = self.upsample(self.feat_shrink(feat))

            aligned_feats.append(feat)
            transferred_new_offset.append(upsampled_offset)
            transferred_new_feats.append(upsampled_feat)

        aligned_feats = torch.cat(aligned_feats, dim=1)
        aligned_feats = self.lrelu(self.extra_conv(aligned_feats))
        aligned_feats = self.conv_fusion(aligned_feats)
        # if self.current_level == self.total_levels:
        #     aligned_feats = self.conv_stage2(self.lrelu(aligned_feats))
        aligned_feats = aligned_feats * self.beta + ref_feat

        return aligned_feats, transferred_new_feats, transferred_new_offset
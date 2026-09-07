import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU()

    def forward(self, x):
        xout = self.relu_s1(self.bn_s1(self.conv_s1(x)))
        return xout

def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

class MGMCB(nn.Module):
    def __init__(self, in_ch=1, out_ch=16, kernel_size=3):
        super(MGMCB, self).__init__()
        self.kernel_size = kernel_size
        self.num_directions = 24
        self.conv = nn.Conv2d(in_ch, out_ch * self.num_directions, kernel_size=kernel_size, padding=kernel_size//2, groups=1)
        self.bn = nn.BatchNorm2d(out_ch * self.num_directions)
        self.relu = nn.ReLU(inplace=True)
        self.fusion = nn.Conv2d(out_ch * self.num_directions, out_ch, kernel_size=1)
        self._freeze_kernel()
    
    def _freeze_kernel(self):
        for param in self.conv.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat_fused = self.fusion(feat)
        return feat_fused, feat

def get_gaussian_kernel(kernel_size=3, sigma=1.0, channels=1):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*np.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

class PriorExtractHead(nn.Module):
    def __init__(self, in_ch=48, mid_ch=8):
        super(PriorExtractHead, self).__init__()
        self.pke1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )
        for param in self.pke1.parameters():
            param.requires_grad = False
        self.pke2 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(mid_ch * 2, mid_ch * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch * 4, 8, kernel_size=1)
        )
    
    def forward(self, x):
        cp1 = self.pke1(x)
        cp2 = self.pke2(x)
        return cp1, cp2

class RSU7_Aug(nn.Module):
    def __init__(self, in_ch=1, mid_ch=12, out_ch=3, mode='encoder'):
        super(RSU7_Aug, self).__init__()
        self.mode = mode
        if self.mode == 'encoder':
            self.mgmcbs = nn.ModuleList([
                MGMCB(in_ch=in_ch, out_ch=16, kernel_size=3),
                MGMCB(in_ch=in_ch, out_ch=16, kernel_size=5),
                MGMCB(in_ch=in_ch, out_ch=16, kernel_size=7)
            ])
            self.prior_extract = PriorExtractHead(in_ch=16*3, mid_ch=8)
            self.rebnconvin = REBNCONV(in_ch + 1, out_ch, dirate=1)
        else:
            self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        if self.mode == 'encoder':
            self.cp2_fuse = REBNCONV(in_ch=mid_ch + 8, out_ch=mid_ch, dirate=1)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)  
        self.cp1 = None    
        self.cp2 = None      
        self.hx6_with_cp2 = None
        self.mid_ch = mid_ch

    def forward(self, x):
        B, C, H, W = x.shape
        if self.mode == 'encoder':
            mgm_feats = []
            for mgmcb in self.mgmcbs:
                feat, _ = mgmcb(x)
                mgm_feats.append(feat)
            mgm_cat = torch.cat(mgm_feats, dim=1)
            self.cp1, self.cp2 = self.prior_extract(mgm_cat)
            x_with_cp1 = torch.cat([x, self.cp1], dim=1)
            hxin = self.rebnconvin(x_with_cp1)
        else:
            hxin = self.rebnconvin(x)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        if self.mode == 'encoder':
            cp2_upsampled = _upsample_like(self.cp2, hx6)
            self.hx6_with_cp2 = self.cp2_fuse(torch.cat([hx6, cp2_upsampled], dim=1))
            hx7 = self.rebnconv7(self.hx6_with_cp2)
        else:
            hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        output_feat = hx1d + hxin  
        return output_feat

class DBPEA(nn.Module):
    def __init__(self, feat_ch: int, prior_ch: int = 9):
        super(DBPEA, self).__init__()
        self.feat_attention = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 4, kernel_size=1),
            nn.BatchNorm2d(feat_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 4, prior_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.prior_attention = nn.Sequential(
            nn.Conv2d(prior_ch, prior_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(prior_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(prior_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feat_ch + prior_ch, feat_ch, kernel_size=1),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        feat_att = self.feat_attention(feat)  
        refined_prior = prior * feat_att       
        prior_att = self.prior_attention(refined_prior)  
        weighted_feat = feat * prior_att                  
        fused = torch.cat([weighted_feat, refined_prior], dim=1)  
        enhanced_feat = self.fusion_conv(fused)                    
        return enhanced_feat

class RSU_DBPEA_Base(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, num_layers: int, variant: str = "default"):
        super(RSU_DBPEA_Base, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch
        self.num_layers = num_layers 
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        encoder_input_chs = [out_ch] + [mid_ch]*(num_layers-1)
        encoder_output_chs = [mid_ch]*num_layers
        self.encoder_layers = self._build_layers(encoder_input_chs, encoder_output_chs, [1]*num_layers)
        self.pools = nn.ModuleList([nn.MaxPool2d(2, stride=2, ceil_mode=True) for _ in range(num_layers-1)])
        self.rebnconv_deep = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.dbpea = DBPEA(feat_ch=mid_ch, prior_ch=9)
        if num_layers == 6: 
            if out_ch == 128: 
                decoder_input_chs = [64, 64, 64, 64, 64, 160]
                self.rebnconv_out = REBNCONV(128*2, out_ch, dirate=1) 
            elif out_ch == 64: 
                decoder_input_chs = [64, 64, 64, 64, 64, 96]
                self.rebnconv_out = REBNCONV(64*2, out_ch, dirate=1) 
            else:
                raise ValueError(f"Unsupported out_ch for RSU6_DBPEA: {out_ch}")
            decoder_output_chs = [mid_ch, mid_ch, mid_ch, mid_ch, mid_ch, out_ch]
        elif num_layers == 5: 
            if out_ch == 256:
                decoder_input_chs = [128, 128, 128, 128, 320]
                self.rebnconv_out = REBNCONV(256*2, out_ch, dirate=1)  
            elif out_ch == 128:  
                decoder_input_chs = [128, 128, 128, 128, 192]
                self.rebnconv_out = REBNCONV(128*2, out_ch, dirate=1) 
            else:
                raise ValueError(f"Unsupported out_ch for RSU5_DBPEA: {out_ch}")
            decoder_output_chs = [mid_ch, mid_ch, mid_ch, mid_ch, out_ch]
        elif num_layers == 4:  
            if out_ch == 512: 
                decoder_input_chs = [256, 256, 256, 640]
                self.rebnconv_out = REBNCONV(512*2, out_ch, dirate=1) 
            elif out_ch == 256: 
                decoder_input_chs = [256, 256, 256, 384]
                self.rebnconv_out = REBNCONV(256*2, out_ch, dirate=1)  
            else:
                raise ValueError(f"Unsupported out_ch for RSU4_DBPEA: {out_ch}")
            decoder_output_chs = [mid_ch, mid_ch, mid_ch, out_ch]
        else:
            raise ValueError(f"Unsupported num_layers: {num_layers} (only 4/5/6 allowed)")
        self.decoder_layers = self._build_layers(decoder_input_chs, decoder_output_chs, [1]*num_layers)

    def _build_layers(self, input_chs: List[int], output_chs: List[int], dirates: List[int]) -> nn.ModuleList:
        layers = nn.ModuleList()
        for in_ch, out_ch, dirate in zip(input_chs, output_chs, dirates):
            layers.append(REBNCONV(in_ch, out_ch, dirate=dirate))
        return layers

    def forward(self, x: torch.Tensor, cp1: torch.Tensor, cp2: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)
        encoder_feats = [hxin]
        feat = hxin
        for i in range(self.num_layers):
            feat = self.encoder_layers[i](feat)
            encoder_feats.append(feat)
            if i < self.num_layers - 1:
                feat = self.pools[i](feat)
        deep_feat = encoder_feats[-1]
        cp1_adapted = _upsample_like(cp1, deep_feat)
        cp2_adapted = _upsample_like(cp2, deep_feat)
        dual_prior = torch.cat([cp1_adapted, cp2_adapted], dim=1)
        deep_feat_enhanced = self.dbpea(deep_feat, dual_prior)
        deep_feat = self.rebnconv_deep(deep_feat_enhanced)
        feat = deep_feat
        for i in range(self.num_layers-1, -1, -1):
            feat_upsampled = _upsample_like(feat, encoder_feats[i])
            concat_ch = feat_upsampled.shape[1] + encoder_feats[i].shape[1]
            decoder_layer_idx = (self.num_layers - 1) - i
            feat_concat = torch.cat([feat_upsampled, encoder_feats[i]], dim=1)
            feat = self.decoder_layers[decoder_layer_idx](feat_concat)
        out = self.rebnconv_out(torch.cat([feat, hxin], dim=1))
        return out

class RSU6_DBPEA(RSU_DBPEA_Base):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(RSU6_DBPEA, self).__init__(in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch, num_layers=6)

class RSU5_DBPEA(RSU_DBPEA_Base):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(RSU5_DBPEA, self).__init__(in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch, num_layers=5)

class RSU4_DBPEA(RSU_DBPEA_Base):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super(RSU4_DBPEA, self).__init__(in_ch=in_ch, mid_ch=mid_ch, out_ch=out_ch, num_layers=4)

class RSU4F_DBPEA(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F_DBPEA, self).__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)  
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)  
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4) 
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)  
        self.dbpea1 = DBPEA(feat_ch=mid_ch, prior_ch=9)  
        self.dbpea2 = DBPEA(feat_ch=mid_ch, prior_ch=9) 
        self.dbpea3 = DBPEA(feat_ch=mid_ch, prior_ch=9)  
        self.dbpea4 = DBPEA(feat_ch=mid_ch, prior_ch=9)  
        decode_in_ch = mid_ch * 2  
        self.rebnconv3d = REBNCONV(decode_in_ch, mid_ch, dirate=4)  
        self.rebnconv2d = REBNCONV(decode_in_ch, mid_ch, dirate=2)  
        self.rebnconv1d = REBNCONV(decode_in_ch, out_ch, dirate=1)  

    def forward(self, x: torch.Tensor,cp1: torch.Tensor, cp2: torch.Tensor) -> torch.Tensor:
        hxin = self.rebnconvin(x)  
        prior = torch.cat([cp1, cp2], dim=1) 
        hx1 = self.rebnconv1(hxin)
        hx1 = self.dbpea1(feat=hx1, prior=prior)  
        hx2 = self.rebnconv2(hx1)   
        hx2 = self.dbpea2(feat=hx2, prior=prior)
        hx3 = self.rebnconv3(hx2)  
        hx3 = self.dbpea3(feat=hx3, prior=prior)  
        hx4 = self.rebnconv4(hx3)   
        hx4 = self.dbpea4(feat=hx4, prior=prior) 
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), dim=1))  
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), dim=1)) 
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), dim=1))  
        return hx1d + hxin
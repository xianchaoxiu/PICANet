import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ==========================================
# Helper Functions & Lightweight Modules
# ==========================================
def _upsample_like(src, tar):
    """Bilinear upsample src to the spatial size of tar."""
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)

class DSConv2d(nn.Module):
    """Depthwise Separable Convolution for drastic parameter reduction."""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(DSConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# ==========================================
# Contribution 1: Multi-scale Deterministic Prior Extraction
# ==========================================
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
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        feat_fused = self.fusion(feat)
        return feat_fused, feat

class PriorExtractHead(nn.Module):
    def __init__(self, in_ch=48, mid_ch=8):
        super(PriorExtractHead, self).__init__()
        
        # Shallow structural prior (1 channel)
        self.pke1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )
            
        # Deep semantic prior (8 channels)
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

# ==========================================
# Contribution 2: DBPEA (Lightweight Version)
# ==========================================
class DBPEA(nn.Module):
    def __init__(self, feat_ch: int, prior_ch: int):
        super(DBPEA, self).__init__()
        
        # Feature attention: Lightweight 1x1 Convolution
        self.feat_attention = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch // 4, prior_ch, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Prior attention: Depthwise Separable Convolution
        self.prior_attention = nn.Sequential(
            nn.Conv2d(prior_ch, prior_ch, kernel_size=3, padding=1, groups=prior_ch, bias=False), # Depthwise
            nn.Conv2d(prior_ch, prior_ch, kernel_size=1, bias=False), # Pointwise
            nn.BatchNorm2d(prior_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(prior_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Fusion layer: 1x1 Convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feat_ch + prior_ch, feat_ch, kernel_size=1, bias=False),
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

# ==========================================
# Contribution 3: Asymmetric Bidirectional Cross-fusion
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x

class AsymBiChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=256, r=4):
        super(AsymBiChaFuseReduce, self).__init__()
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),
            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),
            SpatialAttention(kernel_size=3),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

        self.hook_input_xh = nn.Identity()      
        self.hook_aligned_xh = nn.Identity()     
        self.hook_topdown_wei = nn.Identity()   
        
        self.hook_input_xl = nn.Identity()       
        self.hook_modulated_xl = nn.Identity()   
        self.hook_bottomup_wei = nn.Identity()   
        
        self.hook_out1 = nn.Identity()           
        self.hook_out2 = nn.Identity()           

    def forward(self, xh, xl):
        xh_in = self.hook_input_xh(xh)
        

        xh_aligned = self.feature_high(xh_in)
        xh_aligned_hook = self.hook_aligned_xh(xh_aligned)
        

        topdown_wei = self.topdown(xh_aligned_hook)
        topdown_wei_hook = self.hook_topdown_wei(topdown_wei)


        xl_in = self.hook_input_xl(xl)
        

        xl_modulated = xl_in * topdown_wei_hook
        xl_modulated_hook = self.hook_modulated_xl(xl_modulated)
        

        bottomup_wei = self.bottomup(xl_modulated_hook)
        bottomup_wei_hook = self.hook_bottomup_wei(bottomup_wei)


        xs1 = 2 * xl_modulated_hook  
        out1 = self.post(xs1)
        out1_hook = self.hook_out1(out1)


        xs2 = 2 * xh_aligned_hook * bottomup_wei_hook    
        out2 = self.post(xs2)
        out2_hook = self.hook_out2(out2)
        
        return out1_hook, out2_hook

# ==========================================
# Main Network: PICA-ResNet50-FPN
# ==========================================
class PICA_ResNet_FPN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, fpn_ch=256):
        super(PICA_ResNet_FPN, self).__init__()
        
        # 1. Multi-scale frozen prior extraction
        self.mgmcbs = nn.ModuleList([
            MGMCB(in_ch=in_ch, out_ch=16, kernel_size=3),
            MGMCB(in_ch=in_ch, out_ch=16, kernel_size=5),
            MGMCB(in_ch=in_ch, out_ch=16, kernel_size=7)
        ])
        self.prior_extract = PriorExtractHead(in_ch=16*3, mid_ch=8)

        # 2. Backbone: ResNet50
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        
        # --- High-Resolution Stem Surgery & Weight Inheritance ---
        if in_ch != 3:
            old_weight = resnet.conv1.weight.data
            # Sum the weights of the 3 channels to retain pre-trained edge detection capacity
            new_weight = old_weight.sum(dim=1, keepdim=True)
            # Remove stride=2 to prevent initial aggressive downsampling
            resnet.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=1, padding=3, bias=False)
            resnet.conv1.weight.data = new_weight
        else:
            resnet.conv1.stride = (1, 1)
            
        # Remove MaxPool to maintain high spatial resolution aligned with U-Net
        resnet.maxpool = nn.Identity()
            
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # C1: 1/1
        self.layer1 = resnet.layer1 # C2: 1/1
        self.layer2 = resnet.layer2 # C3: 1/2
        self.layer3 = resnet.layer3 # C4: 1/4
        self.layer4 = resnet.layer4 # C5: 1/8

        # 3. Lateral dimensionality reduction convolutions
        self.lat_c5 = nn.Conv2d(2048, fpn_ch, kernel_size=1)
        self.lat_c4 = nn.Conv2d(1024, fpn_ch, kernel_size=1)
        self.lat_c3 = nn.Conv2d(512, fpn_ch, kernel_size=1)
        self.lat_c2 = nn.Conv2d(256, fpn_ch, kernel_size=1)

        # --- Scale-Aware DBPEA Initialization ---
        # Deep features (C5, C4) interact with 8-channel deep semantic prior (cp2)
        self.dbpea5 = DBPEA(feat_ch=fpn_ch, prior_ch=8)
        self.dbpea4 = DBPEA(feat_ch=fpn_ch, prior_ch=8)
        # Shallow features (C3, C2) interact with 1-channel shallow spatial prior (cp1)
        self.dbpea3 = DBPEA(feat_ch=fpn_ch, prior_ch=1)
        self.dbpea2 = DBPEA(feat_ch=fpn_ch, prior_ch=1)

        # 5. Asymmetric feature fusion
        self.fuse4 = AsymBiChaFuseReduce(in_high_channels=fpn_ch, in_low_channels=fpn_ch, out_channels=fpn_ch)
        self.fuse3 = AsymBiChaFuseReduce(in_high_channels=fpn_ch, in_low_channels=fpn_ch, out_channels=fpn_ch)
        self.fuse2 = AsymBiChaFuseReduce(in_high_channels=fpn_ch, in_low_channels=fpn_ch, out_channels=fpn_ch)

        # --- FPN Smoothing layers with DSConv2d ---
        self.smooth_p4 = nn.Sequential(DSConv2d(fpn_ch * 2, fpn_ch, kernel_size=3, padding=1), nn.BatchNorm2d(fpn_ch), nn.ReLU(True))
        self.smooth_p3 = nn.Sequential(DSConv2d(fpn_ch * 2, fpn_ch, kernel_size=3, padding=1), nn.BatchNorm2d(fpn_ch), nn.ReLU(True))
        self.smooth_p2 = nn.Sequential(DSConv2d(fpn_ch * 2, fpn_ch, kernel_size=3, padding=1), nn.BatchNorm2d(fpn_ch), nn.ReLU(True))

        # 6. Deep supervision and side output heads
        self.side5 = nn.Conv2d(fpn_ch, out_ch, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(fpn_ch, out_ch, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(fpn_ch, out_ch, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(fpn_ch, out_ch, kernel_size=3, padding=1)
        
        self.outconv = nn.Conv2d(4 * out_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # --- A. Extract prior ---
        mgm_feats = [mgmcb(x)[0] for mgmcb in self.mgmcbs]
        mgm_cat = torch.cat(mgm_feats, dim=1)
        cp1, cp2 = self.prior_extract(mgm_cat)

        # --- B. ResNet Encoding Stage ---
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # --- C. Scale-Aware Lateral Connections ---
        cp2_for_c5 = F.interpolate(cp2, size=c5.shape[2:], mode='bilinear', align_corners=False)
        l5 = self.dbpea5(self.lat_c5(c5), cp2_for_c5)
        
        cp2_for_c4 = F.interpolate(cp2, size=c4.shape[2:], mode='bilinear', align_corners=False)
        l4 = self.dbpea4(self.lat_c4(c4), cp2_for_c4)

        cp1_for_c3 = F.interpolate(cp1, size=c3.shape[2:], mode='bilinear', align_corners=False)
        l3 = self.dbpea3(self.lat_c3(c3), cp1_for_c3)
        
        cp1_for_c2 = F.interpolate(cp1, size=c2.shape[2:], mode='bilinear', align_corners=False)
        l2 = self.dbpea2(self.lat_c2(c2), cp1_for_c2)

        # --- D. Top-down Asymmetric Cross-fusion ---
        p5 = l5
        
        p5_up = _upsample_like(p5, l4)
        fuse4_1, fuse4_2 = self.fuse4(xh=p5_up, xl=l4)
        p4 = self.smooth_p4(torch.cat([fuse4_1, fuse4_2], dim=1))

        p4_up = _upsample_like(p4, l3)
        fuse3_1, fuse3_2 = self.fuse3(xh=p4_up, xl=l3)
        p3 = self.smooth_p3(torch.cat([fuse3_1, fuse3_2], dim=1))

        p3_up = _upsample_like(p3, l2)
        fuse2_1, fuse2_2 = self.fuse2(xh=p3_up, xl=l2)
        p2 = self.smooth_p2(torch.cat([fuse2_1, fuse2_2], dim=1))

        # --- E. Multi-scale Deep Supervision Outputs ---
        d5 = _upsample_like(self.side5(p5), x)
        d4 = _upsample_like(self.side4(p4), x)
        d3 = _upsample_like(self.side3(p3), x)
        d2 = _upsample_like(self.side2(p2), x)

        d0 = self.outconv(torch.cat([d5, d4, d3, d2], dim=1))

        if self.training:
            return d0, d2, d3, d4, d5, cp1, cp2, p5
        else:
            return d0

if __name__ == '__main__':
    model = PICA_ResNet_FPN(in_ch=1, out_ch=1)
    # Batch=2 recommended for high-resolution mode memory constraints
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print(f"Network built successfully. Output shape: {out[0].shape if isinstance(out, tuple) else out.shape}")
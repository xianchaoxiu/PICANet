import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from modules.RSU import RSU4_DBPEA, RSU5_DBPEA, RSU6_DBPEA, RSU7_Aug, RSU4F_DBPEA, _upsample_like 
from modules.fusion import AsymBiChaFuseReduce 

class PICA_UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, image_size=512):
        super(PICA_UNet, self).__init__()        
        self.image_size = image_size
        self.out_ch = out_ch
        
        # Encoder modules
        self.stage1 = RSU7_Aug(in_ch, 32, 64, mode='encoder')
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6_DBPEA(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5_DBPEA(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4_DBPEA(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # Decoder modules
        self.stage5 = RSU4F_DBPEA(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F_DBPEA(512, 256, 512)
        self.stage5d = RSU4F_DBPEA(1024, 256, 512)
        self.stage4d = RSU4_DBPEA(1024, 128, 256)
        self.stage3d = RSU5_DBPEA(512, 64, 128)
        self.stage2d = RSU6_DBPEA(256, 32, 64)
        self.stage1d = RSU7_Aug(128, 16, 64, mode='decoder')
        
        # Feature fusion modules
        self.fuse5 = AsymBiChaFuseReduce(512, 512, 512)
        self.fuse4 = AsymBiChaFuseReduce(512, 512, 512)
        self.fuse3 = AsymBiChaFuseReduce(256, 256, 256)
        self.fuse2 = AsymBiChaFuseReduce(128, 128, 128)
        
        # Output modules
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

        self.out_e2 = nn.Identity()
        self.out_e3 = nn.Identity()
        self.out_e4 = nn.Identity()
        self.out_e5 = nn.Identity()
        
        self.out_d4 = nn.Identity()
        self.out_d3 = nn.Identity()
        self.out_d2 = nn.Identity()

    def forward(self, x, mode='backbone_DBPEA'):
        # --- Encoder ---
        hx1 = self.stage1(x) 
        cp1 = self.stage1.cp1  
        cp2 = self.stage1.cp2 
        hx6_with_cp2 = self.stage1.hx6_with_cp2 
        
        hx = self.pool12(hx1)
        
        def get_cp_at_size(size):
            c1 = F.interpolate(cp1, size=size, mode='bilinear', align_corners=False)
            c2 = F.interpolate(cp2, size=size, mode='bilinear', align_corners=False)
            return c1, c2



        def process_stage(stage_module, x_in, identity_layer):
            size = x_in.shape[2:]
            c1_in, c2_in = get_cp_at_size(size)
            
            if mode == 'backbone_DBPEA':
               
                return identity_layer(stage_module(x_in, c1_in, c2_in))

            zero_c1 = torch.zeros_like(c1_in)
            zero_c2 = torch.zeros_like(c2_in)
            out = stage_module(x_in, zero_c1, zero_c2)
            target_ch = out.shape[1]

            def align(prior):
                return F.conv2d(prior, torch.ones(target_ch, prior.shape[1], 1, 1, device=prior.device) / prior.shape[1])

            if mode == 'only_backbone':
                final_out = out
            elif mode == 'backbone_PCP1':
                final_out = out + align(c1_in)
            elif mode == 'backbone_PCP2':
                final_out = out + align(c2_in)
            elif mode == 'backbone_PCP1_PCP2':
                final_out = out + align(c1_in) + align(c2_in)
            else:
                raise ValueError(f"Unknown mode: {mode}")
                

            return identity_layer(final_out)


        
        # --- Encoder ---
        hx2 = process_stage(self.stage2, hx, self.out_e2)
        hx = self.pool23(hx2)
        
        hx3 = process_stage(self.stage3, hx, self.out_e3)
        hx = self.pool34(hx3)
        
        hx4 = process_stage(self.stage4, hx, self.out_e4)
        hx = self.pool45(hx4)

        hx5 = process_stage(self.stage5, hx, self.out_e5)
        hx = self.pool56(hx5)
        
        hx6 = process_stage(self.stage6, hx, nn.Identity()) # stage6不提取，临时给个
        hx6up = _upsample_like(hx6, hx5)
              
        # --- Decoder ---
        fusec51, fusec52 = self.fuse5(hx6up, hx5)
        hx5d = process_stage(self.stage5d, torch.cat((fusec51, fusec52), 1), nn.Identity())
        hx5dup = _upsample_like(hx5d, hx4)

        fusec41, fusec42 = self.fuse4(hx5dup, hx4)
        hx4d = process_stage(self.stage4d, torch.cat((fusec41, fusec42), 1), self.out_d4)
        hx4dup = _upsample_like(hx4d, hx3)

        fusec31, fusec32 = self.fuse3(hx4dup, hx3)
        hx3d = process_stage(self.stage3d, torch.cat((fusec31, fusec32), 1), self.out_d3)
        hx3dup = _upsample_like(hx3d, hx2)

        fusec21, fusec22 = self.fuse2(hx3dup, hx2)
        hx2d = process_stage(self.stage2d, torch.cat((fusec21, fusec22), 1), self.out_d2)
        hx2dup = _upsample_like(hx2d, hx1)


        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # --- Side Outputs & Fusion ---
        d1 = self.side1(hx1d)
        d2 = _upsample_like(self.side2(hx2d), d1)
        d3 = _upsample_like(self.side3(hx3d), d1)
        d4 = _upsample_like(self.side4(hx4d), d1)
        d5 = _upsample_like(self.side5(hx5d), d1)
        d6 = _upsample_like(self.side6(hx6), d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))         

        if self.training:
            return d0, d1, d2, d3, d4, d5, d6, cp1, cp2, hx6_with_cp2
        else:
            return d0
import torch
from thop import profile
from thop import clever_format


from models.PICA_ResNet_FPN import PICA_ResNet_FPN  


def count_model_complexity():
    
    model = PICA_ResNet_FPN(in_ch=1, out_ch=1, fpn_ch=256)
    
    
    input_tensor = torch.randn(1, 1, 256, 256)

    
    model.eval()

    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")

    print("=" * 60)
    print("        PICA-ResNet50-FPN Complexity Statistics")
    print("=" * 60)
    print(f"MACs (Equivalent FLOPs): {macs}")
    print(f"Total Parameters: {params}")
    print("=" * 60)


if __name__ == "__main__":
    count_model_complexity()
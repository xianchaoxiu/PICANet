# models/__init__.py

# Assuming you saved the new code in a file named `resnet_fpn.py` inside the `models` folder.
# If you saved it in a different file (e.g., `model.py`), change the import accordingly.
from .PICA_ResNet_FPN import *

# You can keep your old model import here if you want to switch back later
# from .rsu_unet import PICA_UNet 

def get_model(net_name):
    if net_name == 'pica_resnet_fpn':
        # Ensure the in_ch matches your dataset (e.g., 3 for RGB, 1 for Grayscale)
        return PICA_ResNet_FPN(in_ch=1, out_ch=1)
        
    # elif net_name == 'pica_unet':
    #     return PICA_UNet(in_ch=3, out_ch=1)
        
    else:
        raise NotImplementedError(f"Network {net_name} is not registered in models/__init__.py")
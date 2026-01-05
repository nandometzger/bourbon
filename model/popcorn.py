
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from types import SimpleNamespace

# Import form local networks file
try:
    from .networks import DualStreamUNet, LoRAConv2d # LoRAConv2d might need to be moved to networks if reused, or defined here.
    # Actually LoRAConv2d was in popcorn.py originally. I should keep it here.
except ImportError:
    # Try importing from model.networks if run from root
    from model.networks import DualStreamUNet

# Hardcoded Config matching 'utils.constants'
stage1feats = 8
stage2feats = 16
MODEL = SimpleNamespace(TYPE='dualstreamunet', OUT_CHANNELS=1, IN_CHANNELS=6, TOPOLOGY=[stage1feats, stage2feats,] )
# CONSISTENCY_TRAINER = SimpleNamespace(LOSS_FACTOR=0.5) # Not needed for usage
# PATHS = SimpleNamespace(OUTPUT='.') # Not needed
DATALOADER = SimpleNamespace(SENTINEL1_BANDS=['VV', 'VH'], SENTINEL2_BANDS=['B02', 'B03', 'B04', 'B08'])
# TRAINER = SimpleNamespace(LR=1e5)
dda_cfg = SimpleNamespace(MODEL=MODEL, DATALOADER=DATALOADER)


class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer, rank=4, alpha=1.0):
        super().__init__()
        self.conv = conv_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original weights
        for param in self.conv.parameters():
            param.requires_grad = False
            
        # LoRA weights
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size[0] # Assuming square 
        
        self.lora_A = nn.Parameter(torch.zeros((rank, in_channels, kernel_size, kernel_size)))
        self.lora_B = nn.Parameter(torch.zeros((out_channels, rank, 1, 1)))
        
        # Init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.scaling = alpha / rank

    def forward(self, x):
        original_out = self.conv(x)
        
        # LoRA path
        x_a = F.conv2d(
            x, 
            self.lora_A, 
            stride=self.conv.stride, 
            padding=self.conv.padding, 
            dilation=self.conv.dilation, 
            groups=self.conv.groups
        )
        
        lora_out = F.conv2d(
            x_a, 
            self.lora_B, 
            stride=1, 
            padding=0
        )
        
        return original_out + lora_out * self.scaling


class POPCORN(nn.Module):
    '''
    POPCORN model (Hub Version)
    '''
    def __init__(self, input_channels, feature_extractor="DDA",
                occupancymodel=False, pretrained=False, biasinit=0.75,
                sentinelbuildings=False, activation="ReLU", output_activation="ReLU",
                lora_rank=0, lora_alpha=1.0, 
                dropout=0.0, trainable_backbone=False): # Added extra args to match hubconf signature
        super(POPCORN, self).__init__()

        self.occupancymodel = occupancymodel 
        self.sentinelbuildings = sentinelbuildings
        self.feature_extractor = feature_extractor
        self.output_activation = output_activation
 
        head_input_dim = 0
        
        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        self.parent = None
        
        self.S1, self.S2 = True, True
        if input_channels==0:
            self.S1, self.S2 = False, False
        elif input_channels==2:
            self.S1, self.S2 = True, False
        elif input_channels==4:
            self.S1, self.S2 = False, True
        
        ## INSTANTIATE UNET DIRECTLY (No loading of DDA weights)
        self.unetmodel = DualStreamUNet(dda_cfg)

        if not pretrained:
            # Initialize weights randomly (Standard Kaiming)
            for m in self.unetmodel.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
        # unet_out = 8*2
        unet_out = self.S1*stage1feats + self.S2*stage1feats
        head_input_dim += unet_out
                    
        # num_params_sar = sum(p.numel() for p in self.unetmodel.sar_stream.parameters() if p.requires_grad)  
        # num_params_opt = sum(p.numel() for p in self.unetmodel.optical_stream.parameters() if p.requires_grad) 

        self.unetmodel.num_params = sum(p.numel() for p in self.unetmodel.parameters() if p.requires_grad)

        # Inject LoRA if requested
        if lora_rank > 0:
            self.inject_lora(self.unetmodel, lora_rank, lora_alpha)

        # Build the head
        h = 64
        
        if activation == "GELU":
            act_layer = nn.GELU()
        else:
            act_layer = nn.ReLU(inplace=True)
            
        self.head = nn.Sequential(
            nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), act_layer,
            nn.Conv2d(h, h, kernel_size=1, padding=0), act_layer,
            nn.Conv2d(h, h, kernel_size=1, padding=0), act_layer,
            nn.Conv2d(h, 2, kernel_size=1, padding=0)
        )
        
        # Apply bias init
        if self.output_activation == "Softplus":
             y = max(biasinit, 1e-6)
             b = torch.tensor(y).expm1().log().item()
             self.head[-1].bias.data = b * torch.ones(2)
        else:
             self.head[-1].bias.data = biasinit * torch.ones(2)

        # define urban extractor
        # INSTANTIATE DIRECTLY
        self.building_extractor = DualStreamUNet(dda_cfg)
        self.building_extractor = self.building_extractor.cuda()


    def inject_lora(self, module, rank, alpha):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                if child.groups == 1:
                    setattr(module, name, LoRAConv2d(child, rank, alpha))
            else:
                self.inject_lora(child, rank, alpha)

    def get_output_activation(self, out):
        if self.output_activation == "ReLU":
            return F.relu(out)
        elif self.output_activation == "Softplus":
            return F.softplus(out)
        elif self.output_activation == "Identity":
            return out
        else:
            return F.relu(out)

    def forward(self, inputs, train=False, padding=True, return_features=True,
                encoder_no_grad=False, unet_no_grad=False, sparse=False):
        
        X = inputs["input"]

        # create building score
        if "building_counts" not in inputs.keys() or self.sentinelbuildings:
            with torch.no_grad():
                inputs["building_counts"]  = self.create_building_score(inputs)
            # torch.cuda.empty_cache() # Removed for efficiency in packaged model
        
        aux = {}
        middlefeatures = []

        if sparse:
            sparsity_mask, ratio = self.get_sparsity_mask(inputs)

        # Forward the main model
        if self.unetmodel is not None: 
            X, (px1,px2,py1,py2) = self.add_padding(X, padding) 
            self.unetmodel.freeze_bn_layers()
            if self.S1 and self.S2:
                X = torch.cat([
                    X[:, 4:6], # S1
                    torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                    X[:, 3:4]], # S2_NIR
                dim=1)
            elif self.S1 and not self.S2:
                X = torch.cat([
                    X, # S1
                    torch.zeros(X.shape[0], 4, X.shape[2], X.shape[3], device=X.device)], # S2
                dim=1)
            elif not self.S1 and self.S2:
                X = torch.cat([
                    torch.zeros(X.shape[0], 2, X.shape[2], X.shape[3], device=X.device), # S1
                    torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                    X[:, 3:4]], # S2_NIR
                dim=1)
            
            if unet_no_grad:
                with torch.no_grad():
                    self.unetmodel.eval()
                    X = self.unetmodel(X, alpha=0, encoder_no_grad=encoder_no_grad, return_features=True, S1=self.S1, S2=self.S2)
            else:
                X = self.unetmodel(X, alpha=0, encoder_no_grad=encoder_no_grad, return_features=True, S1=self.S1, S2=self.S2)

            # revert padding
            X = self.revert_padding(X, (px1,px2,py1,py2))
            middlefeatures.append(X)

        headin = torch.cat(middlefeatures, dim=1)

        # forward the head 
        if sparse and hasattr(self, 'sparse_module_forward'): # Check if sparse logic is needed/included
             # I should include sparse_module_forward if I want feature parity.
             # Or just skip it if sparse=False default.
             pass 
        
        out = self.head(headin)[:,0] # Assuming non-sparse for now

        # Population map and total count
        if self.occupancymodel:
            scale = self.get_output_activation(out)
            aux["scale"] = scale
            popdensemap = scale * inputs["building_counts"][:,0]
        else:
            popdensemap = self.get_output_activation(out)
            aux["scale"] = None
        
        popcount = popdensemap.sum((1,2))

        return {"popcount": popcount, "popdensemap": popdensemap, **aux }

    def add_padding(self, data: torch.Tensor, force=True) -> torch.Tensor:
        px1,px2,py1,py2 = None, None, None, None
        if force:
            data  = nn.functional.pad(data, self.p2d, mode='reflect')
            px1,px2,py1,py2 = self.p, self.p, self.p, self.p
        else:
            # pad to make sure it is divisible by 32
            if (data.shape[2] % 32) != 0:
                px1 = (64 - data.shape[2] % 64) //2
                px2 = (64 - data.shape[2] % 64) - px1
                data = nn.functional.pad(data, (0,0,px1,px2,), mode='reflect') 
            if (data.shape[3] % 32) != 0:
                py1 = (64 - data.shape[3] % 64) //2
                py2 = (64 - data.shape[3] % 64) - py1
                data = nn.functional.pad(data, (py1,py2,0,0), mode='reflect')

        return data, (px1,px2,py1,py2)
    

    def revert_padding(self, data: torch.tensor, padding: tuple) -> torch.Tensor:
        px1,px2,py1,py2 = padding
        if px1 is not None or px2 is not None:
            data = data[:,:,px1:-px2,:]
        if py1 is not None or py2 is not None:
            data = data[:,:,:,py1:-py2]
        return data


    def create_building_score(self, inputs: dict) -> torch.Tensor:
        self.building_extractor.eval()
        self.unetmodel.freeze_bn_layers()
 
        X, (px1,px2,py1,py2) = self.add_padding(inputs["input"], True)

        if self.S1 and self.S2:
            X = torch.cat([
                X[:, 4:6], # S1
                torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                X[:, 3:4]], # S2_NIR
            dim=1)
            _, _, logits, _, _ = self.building_extractor(X, alpha=0, return_features=False, S1=self.S1, S2=self.S2)
        elif self.S1 and not self.S2:
            X = torch.cat([
                X, # S1
                torch.zeros(X.shape[0], 4, X.shape[2], X.shape[3], device=X.device)], # S2
            dim=1)
            logits = self.building_extractor(X, alpha=0, return_features=False, S1=self.S1, S2=self.S2)
        elif not self.S1 and self.S2:
            X = torch.cat([
                torch.zeros(X.shape[0], 2, X.shape[2], X.shape[3], device=X.device), # S1
                torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                X[:, 3:4]], # S2_NIR
                dim=1)
            logits = self.building_extractor(X, alpha=0, return_features=False, S1=self.S1, S2=self.S2)
            
        score = torch.sigmoid(logits)
        score = self.revert_padding(score, (px1,px2,py1,py2))

        return score


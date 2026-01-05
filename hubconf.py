dependencies = ['torch', 'numpy']

import torch
import os
import sys

# Ensure local imports work when loaded via hub
try:
    from model.popcorn import POPCORN
except ImportError:
    # Handle relative import if loaded as module
    from .model.popcorn import POPCORN

def bourbon(pretrained=True, checkpoint_path=None, **kwargs):
    """
    Load the Bourbon Model 🥃 (Distilled POPCORN with ResNet backbone).
    
    Args:
        pretrained (bool): If True, loads the specific checkpoint trained on Rwanda (seed 42).
        checkpoint_path (str): Optional path to a specific checkpoint to load. Overrides default if provided.
        **kwargs: Additional arguments to override default configuration.
    """
    
    # Default Args matching the trained model
    config = {
        "input_channels": 4, # S2 (RGB+NIR) 
        "feature_extractor": "DDA",
        "occupancymodel": False,
        "pretrained": False, # Backbone pretrained, irrelevant for inference if loading full weights
        "biasinit": 0.05,
        "sentinelbuildings": False,
        "activation": "ReLU",
        "output_activation": "ReLU",
        "lora_rank": 0,
        "lora_alpha": 1.0,
        "dropout": 0.0,
        "trainable_backbone": False
    }
    
    # Override defaults with kwargs
    config.update(kwargs)
    
    # Instantiate Core Model
    core_model = POPCORN(**config)
    
    if pretrained:
        if checkpoint_path is None:
            # TODO: USER MUST REPLACE THIS WITH PUBLIC URL AFTER UPLOAD
            default_ckpt = "/scratch3/metzgern/random/Popcorn/distill_outputs/ensemble_resnet_10members_S2/member_seed42/distill_POPCORN_ResNet_treg-rwa_20260103-223532/checkpoint_epoch_39.pth"
            
            if os.path.exists(default_ckpt):
                checkpoint_path = default_ckpt
            else:
                 print(f"Warning: Default checkpoint not found.")
        
        if checkpoint_path is not None:
             if checkpoint_path.startswith("http"):
                 state_dict = torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu")
                 if 'model' in state_dict: state_dict = state_dict['model']
             else:
                 checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                 if 'model' in checkpoint: state_dict = checkpoint['model']
                 else: state_dict = checkpoint
            
             core_model.load_state_dict(state_dict)
             print(f"Loaded weights from {checkpoint_path}")

    # Wrap in Bourbon Interface
    try:
        from model.bourbon import BourbonModel
        model = BourbonModel(core_model)
    except ImportError:
        from .model.bourbon import BourbonModel
        model = BourbonModel(core_model)

    return model

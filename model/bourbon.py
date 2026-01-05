
import torch
import torch.nn as nn
import numpy as np

# Try to import utils relatively or absolute
try:
    from .utils import fetch_mpc, fetch_gee, normalize_s2
except ImportError:
    # If loaded as root module
    from model.utils import fetch_mpc, fetch_gee, normalize_s2

class BourbonModel(nn.Module):
    """
    Bourbon Model Wrapper.
    Wraps the POPCORN/ResNet model and adds convenient inference methods.
    """
    def __init__(self, core_model):
        super().__init__()
        self.core_model = core_model
        
    def forward(self, x):
        """
        Standard forward pass.
        Args:
            x (dict or Tensor): Input dict {"input": Tensor} or raw Tensor.
        """
        if isinstance(x, torch.Tensor):
            return self.core_model({"input": x})
        return self.core_model(x)
        
    def predict(self, image):
        """
        Run inference on a numpy image (C, H, W) or batch (B, C, H, W).
        Handles normalization automatically.
        
        Args:
            image (np.ndarray): Sentinel-2 image. Channels: [R, G, B, N].
                               Values: Raw reflection (approx 0-10000).
        Returns:
            dict: {
                'pop_map': np.ndarray (H, W),
                'pop_count': float
                'pop_maps': list (if ensemble input)
                'std_map': np.ndarray (if ensemble input)
            }
        """
        self.eval()
        
        # 1. Normalize
        img_norm = normalize_s2(image)
        
        # 2. To Tensor
        input_tensor = torch.from_numpy(img_norm).float()
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            self.cuda()
            
        # 3. Handle Ensemble / Batch
        # If input was (T, C, H, W), treat as Ensemble? Or Batch?
        # If we want Ensemble behavior (average predictions):
        is_ensemble = (image.ndim == 4)
        
        if is_ensemble:
             # Loop through stack
             preds = []
             valid_imgs = []
             with torch.no_grad():
                 for i in range(input_tensor.shape[0]):
                     x = input_tensor[i].unsqueeze(0) # (1, C, H, W)
                     
                     # 1. Filter out empty/masked tiles (Coverage check)
                     raw_slice = image[i]
                     nz = np.count_nonzero(np.isfinite(raw_slice) & (raw_slice > 0)) / raw_slice.size
                     if nz < 0.6: continue
                     
                     # 2. Patch-level Cloud Filter
                     # Filter out cloudy patches based on Blue band reflectance (Band 2).
                     # Threshold: If > 20% of pixels exceed 2200 reflectance unit.
                     cloud_mask = raw_slice[2] > 2200
                     cloud_ratio = np.count_nonzero(cloud_mask) / cloud_mask.size
                     if cloud_ratio > 0.2: continue
                     
                     valid_imgs.append(raw_slice)
                     
                     out = self.core_model({"input": x})
                     pmap = out["popdensemap"].squeeze().cpu().numpy()
                     preds.append(np.maximum(pmap, 0))
             
             if not preds:
                 return {'pop_count': 0.0, 'pop_map': None, 'clean_image': None}
                 
             avg_map = np.mean(preds, axis=0)
             std_map = np.std(preds, axis=0)
             count = np.sum(avg_map)
             
             return {
                 'pop_map': avg_map,
                 'pop_count': float(count),
                 'std_map': std_map,
                 'ensemble_count': len(preds),
                 'clean_image': np.nanmedian(valid_imgs, axis=0)
             }
        else:
            # Single
             input_tensor = input_tensor.unsqueeze(0) # (1, C, H, W)
             with torch.no_grad():
                 out = self.core_model({"input": input_tensor})
                 pmap = out["popdensemap"].squeeze().cpu().numpy()
                 pmap = np.maximum(pmap, 0)
                 
             return {
                 'pop_map': pmap,
                 'pop_count': float(np.sum(pmap)),
                 'clean_image': image
             }

    def predict_coords(self, lat, lon, provider='mpc', size=None, size_meters=None, ensemble=0, date_start="2020-01-01", date_end="2020-12-31"):
        """
        End-to-End Prediction from Coordinates.
        Fetches imagery, normalizes, and runs inference.
        
        Args:
            lat, lon (float): Location.
            provider (str): 'mpc' or 'gee'.
            size (int): Crop size in pixels (10m/px).
            size_meters (int): Crop size in meters (overrides size).
            ensemble (int): Number of images (0/1=Off).
            
        Returns:
            dict: Prediction result (see predict()) + 'profile' (GeoMetadata) + 'image' (Raw Input).
        """
        if size_meters is not None:
            size = int(size_meters // 10)
        
        if size is None:
            size = 96 # Default back to 96 pixels (~1km) if nothing specified

        if provider == 'mpc':
            img, profile = fetch_mpc(lat, lon, date_start, date_end, size, ensemble)
        else:
            img, profile = fetch_gee(lat, lon, date_start, date_end, size, ensemble)
            
        result = self.predict(img)
        result['profile'] = profile
        result['image'] = img # Raw image/stack
        
        return result

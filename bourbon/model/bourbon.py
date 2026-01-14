
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
        
    def predict(self, image, mask=None):
        """
        Run inference on a numpy image (C, H, W) or batch (B, C, H, W).
        Handles normalization automatically.
        
        Args:
            image (np.ndarray): Sentinel-2 image. Channels: [R, G, B, N].
                               Values: Raw reflection (approx 0-10000).
            mask (np.ndarray): Optional cloud mask (0=Clear, 1=Cloudy). Same shape as H,W.
                               If ensemble input, shape (N, H, W).
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
        device = next(self.core_model.parameters()).device
        input_tensor = torch.from_numpy(img_norm).float().to(device)
            
        # 3. Handle Ensemble / Batch
        # If input was (T, C, H, W), treat as Ensemble? Or Batch?
        # If we want Ensemble behavior (average predictions):
        is_ensemble = (image.ndim == 4)
        
        if is_ensemble:
             # Loop through stack
             preds = []
             valid_imgs = []
             weights = [] # For weighted average
             
             with torch.no_grad():
                 for i in range(input_tensor.shape[0]):
                     x = input_tensor[i].unsqueeze(0) # (1, C, H, W)
                     
                     # Check Mask (pixel-wise)
                     if mask is not None:
                         m = mask[i] # (H, W) boolean or int
                         # Weight map: 1.0 where clear, 0.0 where cloudy
                         # Mask: 1=Cloudy. 
                         w = 1.0 - m.astype(np.float32)
                         # If image is entirely cloudy (weight sum 0 or bad coverage), skip?
                         # Only skip if virtually no valid pixels to save compute.
                         if np.sum(w) < 10: # Arbitrary small number
                             continue 
                     else:
                         w = np.ones((x.shape[2], x.shape[3]), dtype=np.float32)

                     # Coverage check (Standard invalid/zero pixels in image)
                     raw_slice = image[i]
                     nz = np.count_nonzero(np.isfinite(raw_slice) & (raw_slice > 0)) / raw_slice.size
                     if nz < 0.6: continue
                     
                     # Simple Cloud Filter (fallback if no mask provided, or supplementary)
                     if mask is None:
                         # Filter out cloudy patches based on Blue band reflectance (Band 2).
                         cloud_mask = raw_slice[2] > 2200
                         cloud_ratio = np.count_nonzero(cloud_mask) / cloud_mask.size
                         if cloud_ratio > 0.2: continue
                     
                     # Prediction
                     valid_imgs.append(raw_slice)
                     out = self.core_model({"input": x})
                     pmap = out["popdensemap"].squeeze().cpu().numpy()
                     pmap = np.maximum(pmap, 0)
                     
                     preds.append(pmap)
                     weights.append(w)
             
             if not preds:
                 return {'pop_count': 0.0, 'pop_map': None, 'clean_image': None}
             
             # Stack predictions and weights
             preds_stack = np.stack(preds, axis=0) # (N, H, W)
             weights_stack = np.stack(weights, axis=0) # (N, H, W)
             
             # Robust Weighted Average
             sum_weights = np.sum(weights_stack, axis=0)
             sum_weighted_preds = np.sum(preds_stack * weights_stack, axis=0)
             
             # Avoid division by zero
             # Pixels with 0 weight (all cloudy in all images) -> NaN or 0?
             # Let's fallback to unweighted mean for those pixels to be safe, or 0.
             # If all pixels are cloud, likely we want 0 or average of cloudy predis.
             # Fallback: Where sum_weights < epsilon, set weights to 1/N (uniform)
             fallback_mask = sum_weights < 1e-6
             if np.any(fallback_mask):
                 # For these pixels, use uniform average of available predictions
                 # (assumes preds are somewhat valid even if masked)
                 # Or better: Just ignore mask for these pixels?
                 # Let's just set the denominator to 1 to avoid error, and resultant is 0?
                 # No, better to average.
                 pass
             
             # Safe Division
             avg_map = np.divide(sum_weighted_preds, sum_weights, out=np.zeros_like(sum_weighted_preds), where=sum_weights>1e-6)
             
             # If we have pixels where ALL images were cloudy (sum_weights=0),
             # avg_map is 0. This might be wrong. 
             # Let's fill holes with the simple mean of predictions (best guess).
             simple_mean = np.mean(preds_stack, axis=0)
             avg_map[fallback_mask] = simple_mean[fallback_mask]

             std_map = np.std(preds, axis=0)
             count = np.sum(avg_map)
             
             # Create Clean Image Composite
             # Weighted average of RGBs? Or Median.
             # Median is robust. But weighted mean is cleaner for transitions.
             # Let's stick to Median of Valid Images for visualization to avoid blur.
             clean_image = np.nanmedian(np.stack(valid_imgs), axis=0)
             
             return {
                 'pop_map': avg_map,
                 'pop_count': float(count),
                 'std_map': std_map,
                 'ensemble_count': len(preds),
                 'clean_image': clean_image
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
            img, profile, mask = fetch_mpc(lat, lon, date_start, date_end, size, ensemble)
        else:
            img, profile, mask = fetch_gee(lat, lon, date_start, date_end, size, ensemble)
            
        result = self.predict(img, mask=mask)
        result['profile'] = profile
        result['image'] = img # Raw image/stack
        
        return result

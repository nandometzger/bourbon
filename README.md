# Bourbon 🥃: Distilled Population Prediction

**Bourbon** is a lightweight, distilled version of the [POPCORN](https://github.com/google-research/google-research/tree/master/popcorn) model for population estimation from Sentinel-2 satellite imagery. By "distilling" the knowledge from the larger POPCORN model into a compact ResNet backbone, Bourbon delivers fast, accurate inference suitable for large-scale mapping.

> **Fun Fact:** Why "Bourbon"? Because it's what you get when you distill (Pop)Corn! 🌽➡️🥃

## Contents

*   `model/`: Contains the model architecture definitions.
*   `hubconf.py`: Configuration for `torch.hub`.
*   `predict_from_coords.py`: Command-line tool for fetching imagery and generating population maps.
*   `requirements.txt`: Dependencies.

## Installation

```bash
pip install torch numpy matplotlib requests rasterio
# For MPC Support:
pip install pystac-client planetary-computer stackstac rioxarray
# For GEE Support:
pip install earthengine-api
```

## Quick Start (Command Line)

Generate a population map for Kigali, Rwanda:

```bash
python predict_from_coords.py \
  --lat -1.9441 --lon 30.0619 \
  --size 512 \
  --provider mpc \
  --ensemble 10 \
  --output kigali_bourbon.png
```

## Nowcasting & Time Series (`predict_timeseries.py`) 📈

Bourbon can track population trends over time by "nowcasting" across satellite archives at regular intervals (default: every 6 months).

**Example (2016-2025 growth analysis):**
```bash
python predict_timeseries.py \
  --lat -1.9441 --lon 30.0619 \
  --size_meters 2000 \
  --ensemble 3 \
  --out_dir kigali_growth \
  --vmax 1.5
```

**Note**: Use `--vmax` to set a constant scale for the population maps. This is highly recommended for time-series GIFs to ensure the density intensities are visually comparable across years.

**Outputs:**
*   `population_growth.gif`: An animated time-lapse of population expansion.
*   `growth_curve.png`: A plot of the population trend line.
*   `population_timeseries.csv`: Data spreadsheet (Date, Count).
*   `frame_XXX.png`: Individual snapshots for each time step.

## Python Usage (TorchHub)

Bourbon is designed for simplicity. You can load it and run inference in 3 lines of code.

### 1. Load the Model

```python
import torch

# Load Bourbon (pretrained on Rwanda)
# Replace 'username/repo' with your GitHub repository
model = torch.hub.load('username/repo', 'bourbon', pretrained=True)
if torch.cuda.is_available(): model.cuda()
```

### 2. Run Inference (Two Ways)

**Option A: End-to-End from Coordinates**
Automatically fetches imagery (from Microsoft Planetary Computer) and predicts.

```python
# Predict for Kigali (returns dict with maps and counts)
# Note: Requires MPC dependencies installed
result = model.predict_coords(lat=-1.9441, lon=30.0619, size_meters=5000, ensemble=10)

print(f"Population: {result['pop_count']:.2f}")

# Access results
pop_map = result['pop_map'] # (H, W) array
if 'std_map' in result:
    uncert = result['std_map'] # Uncertainty
```

**Option B: From Satellite Image**
If you have your own Sentinel-2 image (Numpy array, 4 channels: R,G,B,N).

```python
import numpy as np
# Your image: (4, H, W) float32 array (Rough range 0-10000)
img = ... 

# Predict (Handles normalization automatically!)
result = model.predict(img)

print(f"Population: {result['pop_count']:.2f}")
```


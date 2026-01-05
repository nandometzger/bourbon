
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.hub
import os
import sys
import rasterio
import requests
import zipfile
import io

# Define Providers
PROVIDER_MPC = 'mpc'
PROVIDER_GEE = 'gee'

def fetch_mpc(lat, lon, date_start, date_end, crop_size=96, ensemble=0):
    """
    Fetch Sentinel-2 L2A locally using Microsoft Planetary Computer and pystac_client.
    Returns: (numpy_array, profile_dict)
    """
    try:
        from pystac_client import Client
        import planetary_computer
        import stackstac
        import rioxarray # For profile extraction
    except ImportError as e:
        raise ImportError(f"MPC dependencies missing: {e}")

    # 1. Search
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    
    # Buffer: Approx 50m extra
    # We need a proper bbox for the crop.
    # 0.01 deg is ~1km.
    # crop_size=512 is ~5km.
    # We need approx crop_size * 10m.
    # Dynamic buffer
    meters = (crop_size * 10 / 2) + 100
    deg = meters / 111320.0 
    bbox = [lon - deg, lat - deg, lon + deg, lat + deg]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": 20}},
        sortby="properties.eo:cloud_cover" # Ascending
    )
    
    items = search.item_collection()
    if len(items) == 0:
        raise ValueError("No items found.")
        
    print(f"Found {len(items)} items.")
    
    # Select items
    limit = ensemble if ensemble > 1 else 10
    selected_items = items[:limit]
    
    if ensemble > 1:
        print(f"Using top {limit} items for Ensemble Stack.")
    else:
        print(f"Using top {limit} items for Median Composite.")
        
    ref_item = selected_items[0]
    print(f"Reference Item: {ref_item.id} ({ref_item.datetime})")
    
    # EPSG Detection
    epsg = ref_item.properties.get("proj:epsg")
    if epsg is None:
        pcode = ref_item.properties.get("proj:code")
        if pcode and pcode.startswith("EPSG:"):
             try: epsg = int(pcode.split(":")[1])
             except: pass
    if epsg is None and "B04" in ref_item.assets:
         epsg = ref_item.assets["B04"].extra_fields.get("proj:epsg")
         
    print(f"Detected EPSG: {epsg}")
    
    stack = stackstac.stack(
        selected_items,
        assets=["B04", "B03", "B02", "B08"],
        resolution=10, 
        bounds_latlon=bbox,
        epsg=epsg 
    )

    # Median Composite
    if ensemble:
         print("Downloading data (Ensemble Stack)...")
         data = stack.compute()
    else:
         print("Downloading data (Median Composite)...")
         data = stack.compute()
         if 'time' in data.dims:
             data = data.median(dim="time", skipna=True)
         
    # Convert to numpy for slicing
    arr = data.values
    
    # Crop
    if arr.ndim == 4: # (T, C, H, W)
         h, w = arr.shape[2], arr.shape[3]
         cy, cx = h//2, w//2
         r = crop_size//2
         if cx - r < 0 or cy - r < 0: raise ValueError("Fetched crop too small.")
         patch = arr[:, :, cy-r:cy+r, cx-r:cx+r]
    else: # (C, H, W)
         h, w = arr.shape[1], arr.shape[2]
         cy, cx = h//2, w//2
         r = crop_size//2
         if cx - r < 0 or cy - r < 0: raise ValueError("Fetched crop too small.")
         patch = arr[:, cy-r:cy+r, cx-r:cx+r]

    # Normalize
    patch_np = patch.astype(np.float32)
    patch_np = np.nan_to_num(patch_np, nan=0.0)
    
    # Metadata for GeoTIFF
    profile = None
    try:
         from rasterio.transform import from_origin
         # Center index of original data
         if arr.ndim == 4:
             c_idx_x = arr.shape[3]//2
             c_idx_y = arr.shape[2]//2
         else:
             c_idx_x = arr.shape[2]//2
             c_idx_y = arr.shape[1]//2
             
         cx_val = float(data.x[c_idx_x])
         cy_val = float(data.y[c_idx_y])
         
         # Top Left of CROP
         west = cx_val - (crop_size * 10 / 2)
         north = cy_val + (crop_size * 10 / 2)
         transform = from_origin(west, north, 10, 10)
         
         profile = {
             'driver': 'GTiff',
             'width': crop_size, 'height': crop_size,
             'count': 1,
             'dtype': 'float32',
             'crs': f"EPSG:{epsg}",
             'transform': transform,
             'nodata': 0
         }
    except Exception as e:
         print(f"Warning: Could not determine GeoProfile: {e}")

    return patch_np, profile


def fetch_gee(lat, lon, date_start, date_end, crop_size=96, ensemble=0):
    """
    Fetch using Google Earth Engine.
    Requires `earthengine authenticate` to have been run.
    """
    try:
        import ee
    except ImportError:
        raise ImportError("Please install 'earthengine-api' for GEE support.")
    
    try:
        ee.Initialize()
    except Exception as e:
        print(f"GEE Initialize failed: {e}")
        print("Try running `earthengine authenticate` in your terminal.")
        sys.exit(1)

    print("Connected to Google Earth Engine.")
    point = ee.Geometry.Point([lon, lat])
    
    # Filter S2
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
        .filterBounds(point) \
        .filterDate(date_start, date_end) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
        .sort("CLOUDY_PIXEL_PERCENTAGE") \
        .first()
    
    if s2 is None:
        raise ValueError("No images found in GEE.")
        
    s2 = s2.select(['B4', 'B3', 'B2', 'B8']) # R, G, B, N
    

    
    print("Downloading crop via GEE URL...")
    
    # Use native projection from B4
    proj = s2.select('B4').projection()
    crs = proj.getInfo()['crs']

    # Define ROI
    radius = (crop_size * 10 / 2) + 50
    roi = point.buffer(radius).bounds()
    
    url = s2.getDownloadURL({
        'name': 's2_patch',
        'scale': 10,
        'crs': crs, 
        'region': roi 
    })
    
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"Failed to download GEE image: {resp.text}")
        
    z = zipfile.ZipFile(io.BytesIO(resp.content))
    file_list = z.namelist()
    
    def read_band(bname):
        candidates = [f for f in file_list if f".{bname}." in f or f.endswith(f"_{bname}.tif")]
        if not candidates: 
             # Fallback: maybe just 'B4.tif' if single? But we requested 4 bands.
             # Check if names are just band names
             if f"{bname}.tif" in file_list: return read_band_file(f"{bname}.tif")
             raise ValueError(f"Band {bname} not found in zip: {file_list}")
        return read_band_file(candidates[0])

    def read_band_file(fname):
        with z.open(fname) as f:
            with rasterio.MemoryFile(f.read()) as memfile:
                with memfile.open() as src:
                    return src.read(1)

    r = read_band('B4')
    g = read_band('B3')
    b = read_band('B2')
    n = read_band('B8')
    
    # Ensure exact size
    def center_crop_pad(arr, size):
        h, w = arr.shape
        cy, cx = h//2, w//2
        rad = size//2
        y1 = max(0, cy-rad)
        y2 = min(h, cy+rad)
        x1 = max(0, cx-rad)
        x2 = min(w, cx+rad)
        out = arr[y1:y2, x1:x2]
        
        oh, ow = out.shape
        if oh < size or ow < size:
             # simple zero padding to bottom/right
             out = np.pad(out, ((0, size-oh), (0, size-ow)), mode='constant', constant_values=0)
        return out
        
    r = center_crop_pad(r, crop_size)
    g = center_crop_pad(g, crop_size)
    b = center_crop_pad(b, crop_size)
    n = center_crop_pad(n, crop_size)
    
    img = np.stack([r, g, b, n], axis=0).astype(np.float32)
    # Return raw for main to normalize
    
    return img, None


def main():
    parser = argparse.ArgumentParser(description="Run POPCORN inference on Satellite Imagery")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--date_start", type=str, default="2020-01-01", help="YYYY-MM-DD")
    parser.add_argument("--date_end", type=str, default="2020-12-31", help="YYYY-MM-DD")
    parser.add_argument("--size", type=int, default=512, help="Crop size in pixels (10m/px, default 512)")
    parser.add_argument("--size_meters", type=int, default=None, help="Crop size in meters (overrides --size)")
    parser.add_argument("--provider", type=str, default=PROVIDER_MPC, choices=[PROVIDER_MPC, PROVIDER_GEE], help="Imagery Provider")
    parser.add_argument("--ensemble", type=int, default=0, help="Number of images for ensemble (0/1=Off, >1=Count)")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed max for population colorbar")
    parser.add_argument("--output", type=str, default="prediction.png", help="Output image path")
    
    args = parser.parse_args()
    
    # 1. Load Model (Bourbon Interface)
    print("Loading Model (Bourbon)...")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from hubconf import bourbon
        model = bourbon(pretrained=True)
        if torch.cuda.is_available(): model.cuda()
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Predict (End-to-End)
    print(f"Running inference for {args.lat}, {args.lon} using {args.provider}...")
    
    try:
        result = model.predict_coords(
            lat=args.lat, 
            lon=args.lon, 
            provider=args.provider, 
            size=args.size, 
            size_meters=args.size_meters,
            ensemble=args.ensemble,
            date_start=args.date_start, 
            date_end=args.date_end
        )
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Extract Results
    pop_count = result.get('pop_count', 0)
    avg_pop_map = result.get('pop_map')
    img = result.get('image') # Raw Input
    profile = result.get('profile')
    std_map = result.get('std_map')
    
    if avg_pop_map is None:
        print("No valid prediction output.")
        return

    print(f"Predicted Population Count: {pop_count:.2f}")
    if result.get('ensemble_count'):
         print(f"Ensemble Members Used: {result['ensemble_count']}")
    
    # 4. Visualize
    vis_img_raw = result.get('clean_image')
    if vis_img_raw is None:
        print("No valid imagery for visualization.")
        return
        
    vis_img = np.clip(vis_img_raw[:3] / 3000.0, 0, 1)
    ens_count = result.get('ensemble_count', 1)
         
    rgb = vis_img.transpose(1, 2, 0)
    
    cols = 3 if std_map is not None else 2
    fig, ax = plt.subplots(1, cols, figsize=(5*cols, 5), constrained_layout=True)
    if cols==2: ax=[ax[0], ax[1]]

    # RGB
    ax[0].set_title(f"Input ({'Ensemble' if ens_count>1 else 'Single'})")
    ax[0].imshow(rgb)
    ax[0].axis("off")
    
    # Pop Map
    ax[1].set_title(f"Population (Count: {pop_count:.0f})")
    vmax_val = args.vmax if args.vmax is not None else max(0.05, avg_pop_map.max())
    im = ax[1].imshow(avg_pop_map, cmap="inferno", vmin=0, vmax=vmax_val)
    ax[1].axis("off")
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label="People / Pixel")
    
    # Uncertainty
    if std_map is not None:
        ax[2].set_title("Uncertainty (Std Dev)")
        im2 = ax[2].imshow(std_map, cmap="viridis", vmin=0)
        ax[2].axis("off")
        
        divider2 = make_axes_locatable(ax[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax2, label="Std Dev")
        
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Visualization saved to {args.output}")

    # 5. GeoTIFF
    if profile:
        out_tif = args.output.replace(".png", ".tif")
        if out_tif == args.output: out_tif += ".tif"
        profile.update(count=1, dtype='float32')
        with rasterio.open(out_tif, 'w', **profile) as dst:
            dst.write(avg_pop_map, 1)
        print(f"GeoTIFF saved to {out_tif}")

if __name__ == "__main__":
    main()

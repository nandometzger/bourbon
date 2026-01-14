
import numpy as np
import sys

# Sentinel-2 Normalization Constants (Rwanda Dataset)
S2_MEAN = np.array([1460.46, 1468.30, 1383.46, 2226.68]).reshape(4, 1, 1)
S2_STD  = np.array([1130.79, 1129.03, 1053.32, 1724.32]).reshape(4, 1, 1)

def normalize_s2(img_np):
    """
    Normalize Sentinel-2 image (C, H, W) or (T, C, H, W) using dataset stats.
    Input should be float32 in approximate range 0-10000.
    """
    # Fill NaNs with channel means to prevent NaN-explosion in Conv layers
    # (T, C, H, W) or (C, H, W)
    if img_np.ndim == 4:
        for c in range(4):
             img_np[:, c] = np.nan_to_num(img_np[:, c], nan=S2_MEAN[c,0,0])
        mean = S2_MEAN.reshape(1, 4, 1, 1)
        std = S2_STD.reshape(1, 4, 1, 1)
        return (img_np - mean) / std
    else:
        for c in range(4):
             img_np[c] = np.nan_to_num(img_np[c], nan=S2_MEAN[c,0,0])
        return (img_np - S2_MEAN) / S2_STD

def fetch_mpc(lat, lon, date_start, date_end, crop_size=96, ensemble=0):
    """
    Fetch Sentinel-2 L2A locally using Microsoft Planetary Computer.
    Returns: (numpy_array, profile_dict)
    """
    try:
        from pystac_client import Client
        import planetary_computer
        import stackstac
        from rasterio.transform import from_origin
    except ImportError as e:
        raise ImportError(f"MPC dependencies missing: {e}. Install 'pystac-client planetary-computer stackstac rioxarray'.")

    # 1. Search
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    
    # Buffer: Approx 50m extra
    meters = (crop_size * 10 / 2) + 100
    deg = meters / 111320.0 
    bbox = [lon - deg, lat - deg, lon + deg, lat + deg]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": 10}},
        sortby="properties.eo:cloud_cover" # Ascending
    )
    
    items = search.item_collection()
    if len(items) == 0:
        raise ValueError("No items found.")
        
    # Group items by date to handle tile boundaries
    from collections import defaultdict
    date_to_items = defaultdict(list)
    for item in items:
        date_to_items[item.datetime.date()].append(item)
    
    # Sort dates by average cloud cover
    sorted_dates = sorted(date_to_items.keys(), 
                         key=lambda d: np.mean([it.properties.get("eo:cloud_cover", 0) for it in date_to_items[d]]))
    
    # Take top N dates
    limit = ensemble if ensemble > 0 else 1
    selected_dates = sorted_dates[:limit]
    
    final_items = []
    for d in selected_dates:
        final_items.extend(date_to_items[d])
    
    print(f"Found {len(items)} items across {len(date_to_items)} dates. Selected {len(selected_dates)} best dates ({len(final_items)} items).")
    items = final_items
    
    # Select items and handle tile seams
    # If a location straddles 2 tiles, multiple items exist for the same date.
    # Grouping by date and taking the median merges these tiles into one valid frame.
    print(f"Found {len(items)} items. Detecting Projection...")
    
    # EPSG Detection
    ref_item = items[0]
    epsg = ref_item.properties.get("proj:epsg")
    if epsg is None:
        pcode = ref_item.properties.get("proj:code")
        if pcode and pcode.startswith("EPSG:"):
             try: epsg = int(pcode.split(":")[1])
             except: pass
    if epsg is None and "B04" in ref_item.assets:
         epsg = ref_item.assets["B04"].extra_fields.get("proj:epsg")
    
    print(f"Mosaicking tiles by date (EPSG:{epsg})...")
    
    stack = stackstac.stack(
        items,
        assets=["B04", "B03", "B02", "B08"],
        resolution=10, 
        bounds_latlon=bbox,
        epsg=epsg,
        fill_value=np.nan,
        dtype="float32",
        rescale=True # Apply STAC scale/offset (Handles baseline 04.00+ shift)
    )

    # Group by exact date (YYYY-MM-DD) to merge adjacent tiles
    stack = stack.groupby("time.date").median(dim="time")
    
    # After grouping, 'date' is the new dimension instead of 'time'
    if len(stack.date) == 0:
        raise ValueError("No valid observations after grouping.")

    limit = ensemble if ensemble > 1 else 10
    selected_stack = stack[:limit]
    
    # Compute
    processing_ensemble = (ensemble > 1)
    
    if processing_ensemble:
         print(f"Downloading data (Ensemble Mosaicked Days N={len(selected_stack.date)})...")
         data = selected_stack.compute()
    else:
         print("Downloading data (Median Composite)...")
         data = selected_stack.median(dim="date", skipna=True).compute()

    # Rescale to 0-10000 range (Model Expectation)
    # stackstac(rescale=True) returns reflectance 0-1.
    # We map this back to the "Old DN" domain (0-10000) to match training stats.
    data = data * 10000.0
         
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

    # Pre-Normalize (Cast and NaN fix)
    patch_np = patch.astype(np.float32)
    patch_np = np.nan_to_num(patch_np, nan=0.0)
    
    # Metadata for GeoTIFF
    profile = None
    try:
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
    """
    try:
        import ee
        import requests
        import zipfile
        import io
        import rasterio
    except ImportError:
        raise ImportError("Please install 'earthengine-api requests rasterio' for GEE support.")
    
    try:
        ee.Initialize()
    except Exception as e:
        print(f"GEE Initialize failed: {e}")
        print("Try running `earthengine authenticate` in your terminal.")
        sys.exit(1)

    point = ee.Geometry.Point([lon, lat])
    
    # Filter S2
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
        .filterBounds(point) \
        .filterDate(date_start, date_end) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)) \
        .sort("CLOUDY_PIXEL_PERCENTAGE") \
        .first()
    
    if s2 is None:
        raise ValueError("No images found in GEE.")
        
    s2 = s2.select(['B4', 'B3', 'B2', 'B8']) # R, G, B, N
    
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
    
    def read_band_file(fname):
        with z.open(fname) as f:
            with rasterio.MemoryFile(f.read()) as memfile:
                with memfile.open() as src:
                    return src.read(1)

    def read_band(bname):
        candidates = [f for f in file_list if f".{bname}." in f or f.endswith(f"_{bname}.tif")]
        if not candidates: 
             if f"{bname}.tif" in file_list: return read_band_file(f"{bname}.tif")
             raise ValueError(f"Band {bname} not found in zip: {file_list}")
        return read_band_file(candidates[0])

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
    
    return img, None

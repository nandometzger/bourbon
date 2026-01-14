
import numpy as np
import sys
import xarray as xr
import requests
import zipfile
import io

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
        assets=["B04", "B03", "B02", "B08", "SCL"], # Include SCL
        resolution=10, 
        bounds_latlon=bbox,
        epsg=epsg,
        fill_value=np.nan,
        rescale=False 
    )

    # --- Radiometric Harmonization ---
    # Apply offset: Subtract 1000 where baseline >= "04.00"
    if 's2:processing_baseline' in stack.coords:
        baseline = stack.coords['s2:processing_baseline']
        mask_offset = (baseline >= "04.00")
        
        # Only apply to spectral bands, exclude SCL (which is categorical)
        spectral_bands = stack.sel(band=["B04", "B03", "B02", "B08"])
        scl_band = stack.sel(band=["SCL"])
        
        spectral_bands = spectral_bands.where(~mask_offset, spectral_bands - 1000)
        
        # Re-combine
        stack = xr.concat([spectral_bands, scl_band], dim="band")

    # Group by exact date (YYYY-MM-DD) to merge adjacent tiles
    # For SCL, median is risky. We use 'first' or 'mode' ideally, but median of ints 
    # might give float. stackstac returns float output by default with fill_value=nan.
    # We will assume SCL is robust enough for median or we explicitly strictly ensure it.
    # Actually, for SCL, let's just take the first valid pixel if mosaicking.
    # But stackstac doesn't support mixed reducers easily in one call.
    # Let's stick to median for simplicity, but round the SCL result or check threshold.
    # A cleaner way: Process SCL separately? No, keep it simple.
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
    
    # Split Data and Mask
    # SCL values: 0=NoData, 1=Sat, 3=Shadow, 8=Cloud Med, 9=Cloud High, 10=Cirrus
    # Mask = 1 if Cloud/Shadow (Bad), 0 if Clear (Good)
    # Median might produce non-integers, so we check ranges.
    
    arr = data.values # (C, H, W) or (T, C, H, W)
    
    if arr.ndim == 4: # (T, C, H, W)
         img_bands = arr[:, 0:4]
         scl_band = arr[:, 4]
    else:
         img_bands = arr[0:4]
         scl_band = arr[4]

    # Create Mask (Broad cloud types: 3, 8, 9, 10, maybe 1)
    # Since SCL was median-ed, values like 8.5 might exist. 
    # We define bad ranges or nearest integer.
    # Let's be aggressive: > 7.5 is Cloud (8,9,10,11). 
    # Shadow is 3. range [2.5, 3.5].
    # Saturation is 1.
    
    # Simple thresholding logic for "Bad" pixels
    bad_mask = (scl_band >= 2.5) & (scl_band <= 3.5) # Shadows
    bad_mask |= (scl_band >= 7.5) # Clouds (8,9,10) and Snow(11)? Snow is 11. Keep snow?
    # Maybe masking snow is good too?
    # Let's mask 1 (Saturated), 3 (Shadow), 8-10 (Cloud).
    bad_mask |= (scl_band >= 0.5) & (scl_band <= 1.5) # Saturated
    
    # Also handle standard NaNs in image as Mask
    # (Already handled by nodata, but good to be explicit)
    
    # Crop
    def crop_center(img, csize):
        if img.ndim == 3: # (C, H, W)
            h, w = img.shape[1], img.shape[2]
            cy, cx = h//2, w//2
            r = csize//2
            return img[:, cy-r:cy+r, cx-r:cx+r]
        elif img.ndim == 4: # (T, C, H, W)
            h, w = img.shape[2], img.shape[3]
            cy, cx = h//2, w//2
            r = csize//2
            return img[:, :, cy-r:cy+r, cx-r:cx+r]
        elif img.ndim == 2: # (H, W) for mask
            h, w = img.shape[0], img.shape[1]
            cy, cx = h//2, w//2
            r = csize//2
            return img[cy-r:cy+r, cx-r:cx+r]
        elif img.ndim == 3: # (T, H, W) for mask stack
            h, w = img.shape[1], img.shape[2]
            cy, cx = h//2, w//2
            r = csize//2
            return img[:, cy-r:cy+r, cx-r:cx+r]

    patch = crop_center(img_bands, crop_size)
    mask_patch = crop_center(bad_mask, crop_size) # Boolean array
    
    # Pre-Normalize (Cast and NaN fix)
    patch_np = patch.astype(np.float32)
    patch_np = np.nan_to_num(patch_np, nan=0.0)
    
    # Metadata for GeoTIFF (from utils.py original)
    profile = None
    try:
         # ... existing profile logic ...
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

    return patch_np, profile, mask_patch


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
    
    # Filter S2: Use SR Harmonized for SCL band
    # COPERNICUS/S2_SR_HARMONIZED: Surface Reflectance (L2A) + Harmonized
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(point) \
        .filterDate(date_start, date_end) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)) \
        .sort("CLOUDY_PIXEL_PERCENTAGE") \
        .first()
    
    if s2 is None:
        raise ValueError("No images found in GEE.")
        
    s2 = s2.select(['B4', 'B3', 'B2', 'B8', 'SCL']) # R, G, B, N, SCL
    
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
    scl = read_band('SCL')
    
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
    scl = center_crop_pad(scl, crop_size)
    
    img = np.stack([r, g, b, n], axis=0).astype(np.float32)

    # Make Mask
    # SCL(GEE): 0=NoData, 1=Sat, 3=Shadow, 8=Cloud Med, 9=Cloud High, 10=Cirrus
    bad_mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 1)
    
    return img, None, bad_mask


import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# Load Bourbon
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hubconf import bourbon

def create_date_intervals(start_year=2016, end_year=2025, months_step=12):
    """Generate 12-month intervals (Yearly)."""
    intervals = []
    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    while current_date < end_date:
        next_date = current_date + timedelta(days=months_step * 30.5)
        if next_date > end_date:
            next_date = end_date
        
        intervals.append({
            'start': current_date.strftime('%Y-%m-%d'),
            'end': next_date.strftime('%Y-%m-%d'),
            'label': current_date.strftime('%Y') # Just Year
        })
        current_date = next_date
    return intervals

def main():
    parser = argparse.ArgumentParser(description="Bourbon Nowcasting: Population Time Series")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--name", type=str, default="", help="City/Location name for title")
    parser.add_argument("--size_meters", type=int, default=2000, help="Patch size in meters")
    parser.add_argument("--ensemble", type=int, default=5, help="Images per time step")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed max for population colorbar (e.g. 1.0)")
    parser.add_argument("--provider", type=str, default="mpc", choices=["mpc", "gee"])
    parser.add_argument("--out_dir", type=str, default="timeseries_output", help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. Load Model
    print("🥃 Loading Bourbon...")
    model = bourbon(pretrained=True)
    if torch.cuda.is_available(): model.cuda()
    
    # 2. Get Intervals
    intervals = create_date_intervals()
    results_vals = [np.nan] * len(intervals) # Aligned with intervals
    frames = []

    print(f"🥃 Starting Nowcast for ({args.lat}, {args.lon}) across {len(intervals)} years...")
    
    for i, interval in enumerate(intervals):
        print(f"\nStep {i+1}/{len(intervals)}: {interval['label']} ({interval['start']} to {interval['end']})")
        
        try:
            res = model.predict_coords(
                lat=args.lat, 
                lon=args.lon, 
                provider=args.provider, 
                size_meters=args.size_meters,
                ensemble=args.ensemble,
                date_start=interval['start'],
                date_end=interval['end']
            )
            
            pop_count = res.get('pop_count', 0)
            pop_map = res.get('pop_map')
            
            # RGB Input
            rgb_raw = res.get('clean_image')
            if rgb_raw is None:
                 print(f"  ⚠️ Skipping interval {interval['label']}: No valid imagery found (likely too cloudy).")
                 continue
            
            if pop_map is None:
                 print(f"  ⚠️ Skipping interval {interval['label']}: Model failed to produce map.")
                 continue

            # Update stats
            results_vals[i] = pop_count
            
            # Create Frame
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            
            fig = plt.figure(figsize=(12, 11))
            if args.name:
                fig.suptitle(f"{args.name}", fontsize=24, fontweight='bold', y=0.96)
                
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.7])
            plt.subplots_adjust(top=0.9) # Make room for suptitle
            
            ax_rgb = fig.add_subplot(gs[0, 0])
            ax_map = fig.add_subplot(gs[0, 1])
            ax_curve = fig.add_subplot(gs[1, :])
            
            # 1. RGB
            rgb = np.clip(rgb_raw[:3].transpose(1, 2, 0) / 3000.0, 0, 1)
            ax_rgb.imshow(rgb)
            ax_rgb.set_title(f"Satellite Input ({interval['label']})", fontsize=14, fontweight='bold')
            ax_rgb.axis('off')
            
            # 2. Pop Map
            # Use fixed vmax if provided, otherwise dynamic
            vmax_val = args.vmax if args.vmax is not None else max(0.1, pop_map.max())
            im = ax_map.imshow(pop_map, cmap='inferno', vmin=0, vmax=vmax_val)
            ax_map.set_title(f"Population Density (Total: {pop_count:.0f})", fontsize=14, fontweight='bold')
            ax_map.axis('off')
            
            divider = make_axes_locatable(ax_map)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label='People / Pixel')
            
            # 3. Growth Curve (X-axis aligned with range(len(intervals)))
            ax_curve.plot(range(len(intervals)), results_vals, marker='o', color='brown', lw=2, alpha=0.6)
            ax_curve.scatter(i, pop_count, color='red', s=120, zorder=5, edgecolors='black')
            
            ax_curve.set_xlim(-0.5, len(intervals) - 0.5)
            ax_curve.set_xticks(range(len(intervals)))
            ax_curve.set_xticklabels([intv['label'] for intv in intervals], rotation=45, ha='right')
            
            ax_curve.set_ylabel("Estimated Population", fontsize=12)
            ax_curve.set_title("Population Growth Timeline", fontsize=14, fontweight='bold')
            ax_curve.grid(True, linestyle='--', alpha=0.5)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            frame_path = os.path.join(args.out_dir, f"frame_{i:03d}.png")
            plt.savefig(frame_path, dpi=120, bbox_inches='tight')
            plt.close()
            frames.append(frame_path)
            
            print(f"  Result: {pop_count:.0f} people")
            
        except Exception as e:
            print(f"  ⚠️ Skipping interval {interval['label']}: {e}")

    # 3. Save CSV
    df = pd.DataFrame([{ 'date': intv['label'], 'population': val } for intv, val in zip(intervals, results_vals)])
    csv_path = os.path.join(args.out_dir, "population_timeseries.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n📈 Timeseries saved to {csv_path}")

    # 4. Attempt GIF creation
    try:
        import imageio
        from PIL import Image
        print("\n🎬 Creating GIF...")
        gif_path = os.path.join(args.out_dir, "population_growth.gif")
        
        processed_images = []
        target_size = None
        
        for f in frames:
            img_pil = Image.open(f).convert('RGB')
            if target_size is None:
                target_size = img_pil.size
            else:
                img_pil = img_pil.resize(target_size, resample=Image.Resampling.LANCZOS)
            processed_images.append(np.array(img_pil))
            
        imageio.mimsave(gif_path, processed_images, fps=1)
        print(f"🎬 Animation saved to {gif_path}")
    except Exception as e:
        print(f"⚠️ GIF generation failed: {e}")

    # 5. Plot growth curve
    has_results = any(not np.isnan(val) for val in results_vals)
    if has_results:
        plt.figure(figsize=(10, 5))
        plt.plot([intv['label'] for intv in intervals], results_vals, marker='o', color='brown')
        plt.title(f"Population Growth Nowcast: {args.name if args.name else (str(args.lat) + ', ' + str(args.lon))}")
        plt.xlabel("Year")
        plt.ylabel("Total Population")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "growth_curve.png"))
        print(f"📊 Final growth curve saved to {args.out_dir}/growth_curve.png")

if __name__ == "__main__":
    main()


import os
import imageio
import numpy as np
from PIL import Image

def create_teaser(out_path="assets/bourbon_showcase.gif", showcases_dir="showcases"):
    print("🎬 Generating Teaser GIF...")
    
    gifs = []
    # Find all GIFs
    for root, dirs, files in os.walk(showcases_dir):
        for f in files:
            if f == "population_growth.gif":
                gifs.append(os.path.join(root, f))
    
    gifs.sort() # Consistent order
    print(f"Found {len(gifs)} GIFs.")
    
    all_frames = []
    target_size = None
    
    for g_path in gifs:
        print(f"  Reading {g_path}...")
        try:
            reader = imageio.get_reader(g_path)
            # Read all frames from this GIF
            frames_list = list(reader)
            
            # Subsample? Use all?
            # Let's use all to show full timeline.
            
            for frame in frames_list:
                # Resize if needed
                if target_size is None:
                    target_size = frame.shape[:2] # H, W
                
                if frame.shape[:2] != target_size:
                    img_pil = Image.fromarray(frame)
                    img_pil = img_pil.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
                    frame = np.array(img_pil)
                
                all_frames.append(frame)
                
        except Exception as e:
            print(f"  ⚠️ Failed to read {g_path}: {e}")
            
    if not all_frames:
        print("No frames found!")
        return

    print(f"saving {len(all_frames)} frames to {out_path}...")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    imageio.mimsave(out_path, all_frames, fps=2, loop=0) 
    print("✅ Teaser generated!")

if __name__ == "__main__":
    create_teaser()

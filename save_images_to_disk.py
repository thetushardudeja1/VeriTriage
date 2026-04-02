"""
Convert base64 images from notebook_images.json to actual PNG files in results/plots/
This ensures GitHub can render images using relative file paths.
"""

import json
import base64
from pathlib import Path

ROOT = Path(r"C:\Users\TUSHAR\2026-27\PROJECTS\VeriTriage")

# Create plots directory if it doesn't exist
plots_dir = ROOT / "results" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Load notebook images
notebook_json = ROOT / "notebook_images.json"
if not notebook_json.exists():
    print("❌ notebook_images.json not found")
    exit(1)

with open(notebook_json, 'r') as f:
    notebook_imgs = json.load(f)

print(f"Found {len(notebook_imgs)} images to save...")

# Map notebook image keys to file names
image_mapping = {
    '05_shap_analysis_img1': 'shap_feature_importance.png',
    '05_shap_analysis_img2': 'shap_waterfall_example.png',
    '03_model_training_img1': 'model_performance_confusion.png',
    '03_model_training_img2': 'model_training_curves.png',
    '04_chip_family_analysis_img1': 'chip_family_failure_rates.png',
    '04_chip_family_analysis_img2': 'cross_architecture_generalization.png',
}

# Extract and save images
for notebook_key, filename in image_mapping.items():
    if notebook_key not in notebook_imgs:
        print(f"  ⚠ Skipping {notebook_key}: not in notebook_images.json")
        continue
    
    b64_data = notebook_imgs[notebook_key]
    
    # Remove data URI prefix if present
    if b64_data.startswith('data:image/png;base64,'):
        b64_data = b64_data.replace('data:image/png;base64,', '')
    
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(b64_data)
        
        # Save to PNG file
        filepath = plots_dir / filename
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        print(f"  ✓ {filename} ({len(image_bytes)} bytes)")
    except Exception as e:
        print(f"  ✗ Error saving {filename}: {e}")

print(f"\n✅ Images saved to {plots_dir}")
print("\nYou can now reference them in README with relative paths:")
print("  ![SHAP Importance](results/plots/shap_feature_importance.png)")

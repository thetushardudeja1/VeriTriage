#!/usr/bin/env python3
"""
Extract images directly from Jupyter notebooks and save as base64 for README.
This ensures images are always fresh and prevents corruption issues on GitHub.
"""

import json
import base64
from pathlib import Path

def extract_images_from_notebooks():
    """Extract all PNG images from notebooks and return as base64 strings."""
    notebook_dir = Path('notebooks')
    images = {}
    
    # Priority notebooks to extract from
    priority_notebooks = [
        '05_shap_analysis.ipynb',
        '03_model_training.ipynb',
        '04_chip_family_analysis.ipynb',
    ]
    
    for nb_name in priority_notebooks:
        nb_path = notebook_dir / nb_name
        if not nb_path.exists():
            print(f"⚠️  {nb_name} not found, skipping...")
            continue
            
        print(f"\n📓 Extracting images from {nb_name}...")
        
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        image_num = 0
        for cell_idx, cell in enumerate(nb['cells']):
            if 'outputs' not in cell:
                continue
                
            for output in cell['outputs']:
                if output.get('output_type') in ['display_data', 'execute_result']:
                    if 'data' in output and 'image/png' in output['data']:
                        image_num += 1
                        b64_data = output['data']['image/png']
                        
                        # Store with notebook context
                        key = f"{nb_name.replace('.ipynb', '')}_img{image_num}"
                        images[key] = f"data:image/png;base64,{b64_data}"
                        print(f"  ✓ Image {image_num} extracted from cell {cell_idx}")
    
    return images

def update_readme_with_notebook_images():
    """Update README with images extracted directly from notebooks."""
    
    # Extract images
    notebook_images = extract_images_from_notebooks()
    
    if not notebook_images:
        print("❌ No images found in notebooks!")
        return False
    
    print(f"\n✅ Total images extracted: {len(notebook_images)}")
    
    # Print image keys for reference
    for key in notebook_images.keys():
        print(f"   - {key}: {len(notebook_images[key])} bytes")
    
    # Save to a JSON file for use by generate_industry_readme.py
    with open('notebook_images.json', 'w') as f:
        json.dump(notebook_images, f, indent=2)
    
    print(f"\n✅ Saved {len(notebook_images)} images to notebook_images.json")
    print("🔄 Run: python generate_industry_readme.py")
    
    return True

if __name__ == '__main__':
    print("="*60)
    print("📸 Jupyter Notebook Image Extractor")
    print("="*60)
    
    success = update_readme_with_notebook_images()
    
    if success:
        print("\n✅ Ready to regenerate README with notebook images!")
    else:
        print("\n❌ Failed to extract images from notebooks")

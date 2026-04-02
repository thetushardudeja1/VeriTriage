# VeriTriage Setup Guide

Complete instructions to set up VeriTriage locally.

## Prerequisites

- Python 3.10+ ([Download](https://www.python.org/downloads/))
- Git ([Download](https://git-scm.com/))
- pip (comes with Python)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/dudeja-tushar/veritriage.git
cd veritriage
```

### 2. Create Virtual Environment (Recommended)

```bash
# On macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import pandas, numpy, xgboost, shap; print('✅ All dependencies installed successfully!')"
```

## Data Setup

### CircuitNet-N28 Dataset

Download the dataset from [CircuitNet GitHub](https://github.com/circuitnet/CircuitNet-N28):

```bash
# Create data directory structure
mkdir -p data/raw/circuitnet/IR_drop
mkdir -p data/raw/circuitnet/routability/{cell_density,macro_region,DRC,congestion}
mkdir -p data/processed

# Place CircuitNet files in data/raw/circuitnet/
# The directory structure should be:
# data/
#   raw/
#     circuitnet/
#       IR_drop/
#         <design folders>
#       routability/
#         cell_density/
#         macro_region/
#         DRC/
#         congestion/
```

### Feature CSV Files

Processed feature CSVs should be in `data/processed/`:

- `veritriage_features.csv` — Extracted features + labels
- `model_results.csv` — Model performance metrics
- `cross_family_results.csv` — Cross-validation results

## Using Jupyter Notebooks

### Launch JupyterLab

```bash
jupyter lab
```

### Run Analysis Pipeline (In Order)

1. **01_data_exploration.ipynb** — Load and explore CircuitNet data
2. **02_feature_engineering.ipynb** — Extract statistical features
3. **03_model_training.ipynb** — Train XGBoost models
4. **04_chip_family_analysis.ipynb** — Analyze RISCY vs zero-riscy
5. **05_shap_analysis.ipynb** — SHAP interpretability analysis
6. **06_triage_pipeline.ipynb** — Deployment-ready pipeline

## Quick Start

### Predict on New Design

```python
from src.triage import TriagePipeline
import numpy as np

# Load trained pipeline
pipeline = TriagePipeline.load("models/")

# Create feature tensor (50 statistical features)
features = np.random.randn(50)  # Replace with actual placement features

# Get predictions
results = pipeline.predict(features, architecture="RISCY")
print(results)
```

### Generate README

```bash
python generate_enhanced_readme.py
```

This regenerates README.md with latest data from CSVs.

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'xgboost'

**Solution:**
```bash
pip install xgboost==1.7.6
```

### Issue: Jupyter kernel not found

**Solution:**
```bash
pip install ipykernel
python -m ipykernel install --user --name veritriage
```

### Issue: SHAP plots not displaying

**Solution:**
```bash
pip install --upgrade shap matplotlib
```

## GPU Acceleration (Optional)

For faster training on GPU-enabled XGBoost:

```bash
# Install GPU version (requires CUDA toolkit)
pip install xgboost-gpu

# Update notebook GPU parameters:
# params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# .env
PROJECT_ROOT=C:\Users\TUSHAR\2026-27\PROJECTS\VeriTriage
DATA_PATH=data/
MODELS_PATH=models/
RESULTS_PATH=results/
```

## Next Steps

1. ✅ Complete setup
2. 📊 Run `01_data_exploration.ipynb` to verify CircuitNet data
3. 🔧 Run feature engineering pipeline
4. 🤖 Train models
5. 📈 Generate visualizations
6. 🚀 Deploy triage pipeline

## Support

- 📖 [README.md](README.md)
- 🤝 [CONTRIBUTING.md](CONTRIBUTING.md)
- 📧 Email: tushar.dudeja@gmail.com
- 💬 GitHub Issues

---

**Happy analyzing! 🎯**
